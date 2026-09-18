from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from smolagents import Model, OpenAIModel

from config import (
    COHERE_BASE_URL,
    COHERE_FALLBACK_API_KEY,
    COHERE_FALLBACK_MODEL,
    COHERE_FALLBACK_TRANSCRIPTION_MODEL,
    COHERE_PRIMARY_API_KEY,
    COHERE_PRIMARY_MODEL,
    COHERE_PRIMARY_TRANSCRIPTION_MODEL,
    FAILOVER_ATTEMPTS,
    FAILOVER_COOLDOWN_SECONDS,
    MODEL_MAX_RETRIES,
    MODEL_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class Route:
    key_slot: str
    model_id: str
    api_key: str


def get_chat_routes() -> list[Route]:
    key_pairs = [
        ("primary_key", COHERE_PRIMARY_API_KEY),
        ("fallback_key", COHERE_FALLBACK_API_KEY),
    ]
    model_pairs = [
        ("primary_model", COHERE_PRIMARY_MODEL),
        ("fallback_model", COHERE_FALLBACK_MODEL),
    ]

    routes = []
    for model_slot, model_id in model_pairs:
        if not model_id:
            continue
        for key_slot, api_key in key_pairs:
            if api_key:
                routes.append(
                    Route(
                        key_slot=key_slot,
                        model_id=model_id,
                        api_key=api_key,
                    )
                )

    if not routes:
        raise RuntimeError(
            "Configure COHERE_PRIMARY_API_KEY and/or COHERE_FALLBACK_API_KEY."
        )

    return routes


def get_transcription_routes() -> list[Route]:
    key_pairs = [
        ("primary_key", COHERE_PRIMARY_API_KEY),
        ("fallback_key", COHERE_FALLBACK_API_KEY),
    ]
    model_pairs = [
        ("primary_transcription_model", COHERE_PRIMARY_TRANSCRIPTION_MODEL),
        ("fallback_transcription_model", COHERE_FALLBACK_TRANSCRIPTION_MODEL),
    ]

    routes = []
    for model_slot, model_id in model_pairs:
        if not model_id:
            continue
        for key_slot, api_key in key_pairs:
            if api_key:
                routes.append(
                    Route(
                        key_slot=key_slot,
                        model_id=model_id,
                        api_key=api_key,
                    )
                )

    return routes


class FailoverModel(Model):
    """smolagents model with automatic key/model failover."""

    def __init__(self):
        super().__init__(model_id=COHERE_PRIMARY_MODEL)
        self.routes = get_chat_routes()
        self.cooldowns: dict[tuple[str, str], float] = {}
        self.failed_routes: set[tuple[str, str]] = set()
        self.route_order = deque(range(len(self.routes)))
        self.last_route: Route | None = None
        self.failover_count = 0
        self.total_attempts = 0

    def _build_model(self, route: Route) -> OpenAIModel:
        return OpenAIModel(
            model_id=route.model_id,
            api_base=COHERE_BASE_URL,
            api_key=route.api_key,
            client_kwargs={
                "max_retries": MODEL_MAX_RETRIES,
                "timeout": MODEL_TIMEOUT_SECONDS,
            },
            retry=False,
            temperature=0,
            reasoning_effort="high",
            max_tokens=4096,
        )

    def _ordered_routes(self) -> list[Route]:
        now = time.monotonic()
        ready: list[Route] = []
        cooling: list[Route] = []

        for index in list(self.route_order):
            route = self.routes[index]
            identity = (route.key_slot, route.model_id)

            if identity in self.failed_routes:
                continue

            if self.cooldowns.get(identity, 0) > now:
                cooling.append(route)
            else:
                ready.append(route)

        return ready or cooling or list(self.routes)

    def _record_failure(self, route: Route, exc: Exception) -> None:
        identity = (route.key_slot, route.model_id)
        if self._is_transient(exc):
            self.cooldowns[identity] = (
                time.monotonic() + FAILOVER_COOLDOWN_SECONDS
            )
        else:
            self.failed_routes.add(identity)

        try:
            self.route_order.remove(self.routes.index(route))
        except ValueError:
            pass
        self.route_order.append(self.routes.index(route))

    def _record_success(self, route: Route) -> None:
        self.last_route = route
        index = self.routes.index(route)
        try:
            self.route_order.remove(index)
        except ValueError:
            pass
        self.route_order.appendleft(index)

        self.cooldowns.pop((route.key_slot, route.model_id), None)

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ):
        routes = self._ordered_routes()
        attempts = min(FAILOVER_ATTEMPTS, len(routes))
        errors = []

        for route in routes[:attempts]:
            self.total_attempts += 1
            try:
                delegate = self._build_model(route)
                result = delegate.generate(
                    messages,
                    stop_sequences=stop_sequences,
                    response_format=response_format,
                    tools_to_call_from=tools_to_call_from,
                    **kwargs,
                )
                self._record_success(route)
                return result
            except Exception as exc:
                self.failover_count += 1
                errors.append(
                    f"{route.key_slot}/{route.model_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._record_failure(route, exc)

        raise RuntimeError(
            "All Cohere model routes failed. " + " | ".join(errors)
        )

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(
            marker in text
            for marker in (
                "429",
                "rate limit",
                "rate_limit",
                "timeout",
                "timed out",
                "connection",
                "temporarily",
                "502",
                "503",
                "504",
                "server error",
            )
        )

    def status(self) -> dict[str, Any]:
        return {
            "routes": [
                {
                    "key_slot": route.key_slot,
                    "model_id": route.model_id,
                }
                for route in self.routes
            ],
            "active_route": (
                {
                    "key_slot": self.last_route.key_slot,
                    "model_id": self.last_route.model_id,
                }
                if self.last_route
                else None
            ),
            "failover_count": self.failover_count,
            "total_attempts": self.total_attempts,
            "cooling_routes": len(self.cooldowns),
            "failed_routes": len(self.failed_routes),
        }


class CohereFailoverClient:
    """Direct Cohere client used by multimodal tools."""

    def __init__(self):
        self.routes = get_chat_routes()
        self.last_route: Route | None = None
        self.failover_count = 0
        self._cooldowns: dict[tuple[str, str], float] = {}

    def _ordered_routes(self) -> list[Route]:
        now = time.monotonic()
        ready = []
        cooling = []

        for route in self.routes:
            identity = (route.key_slot, route.model_id)
            if self._cooldowns.get(identity, 0) > now:
                cooling.append(route)
            else:
                ready.append(route)

        return ready or cooling or list(self.routes)

    def chat(self, **kwargs):
        import cohere

        errors = []
        for route in self._ordered_routes()[:FAILOVER_ATTEMPTS]:
            identity = (route.key_slot, route.model_id)
            try:
                client = cohere.ClientV2(route.api_key)
                response = client.chat(
                    model=route.model_id,
                    **kwargs,
                )
                self.last_route = route
                self._cooldowns.pop(identity, None)
                return response
            except Exception as exc:
                self.failover_count += 1
                errors.append(
                    f"{route.key_slot}/{route.model_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if FailoverModel._is_transient(exc):
                    self._cooldowns[identity] = (
                        time.monotonic() + FAILOVER_COOLDOWN_SECONDS
                    )

        raise RuntimeError(
            "All Cohere multimodal routes failed. " + " | ".join(errors)
        )

    def transcribe(self, file_path, language="en"):
        import cohere

        routes = get_transcription_routes()
        if not routes:
            raise RuntimeError("No Cohere transcription route is configured.")

        errors = []
        for route in routes[:FAILOVER_ATTEMPTS]:
            try:
                client = cohere.ClientV2(route.api_key)
                with open(file_path, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model=route.model_id,
                        language=language,
                        file=audio_file,
                    )
                self.last_route = route
                return response
            except Exception as exc:
                self.failover_count += 1
                errors.append(
                    f"{route.key_slot}/{route.model_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

        raise RuntimeError(
            "All Cohere transcription routes failed. " + " | ".join(errors)
        )
