from __future__ import annotations

import json

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from smolagents import Model, OpenAIModel, REMOVE_PARAMETER

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


def get_vision_routes(model_id: str | None = None) -> list[Route]:
    """Return routes for a vision capable Cohere model."""
    if model_id is None:
        model_id = COHERE_PRIMARY_MODEL
    key_pairs = [
        ("primary_key", COHERE_PRIMARY_API_KEY),
        ("fallback_key", COHERE_FALLBACK_API_KEY),
    ]
    routes = [
        Route(
            key_slot=key_slot,
            model_id=model_id,
            api_key=api_key,
        )
        for key_slot, api_key in key_pairs
        if api_key
    ]
    if not routes:
        raise RuntimeError("No vision capable Cohere route is configured.")
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
            reasoning_effort=self._reasoning_effort(route.model_id),
            max_tokens=4096,
            tool_choice=REMOVE_PARAMETER,
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
                result = self._repair_malformed_final_answer(
                    delegate=delegate,
                    result=result,
                    messages=messages,
                    final_answer_tool=self._find_final_answer_tool(
                        tools_to_call_from
                    ),
                    stop_sequences=stop_sequences,
                    response_format=response_format,
                    kwargs=kwargs,
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
    def _find_final_answer_tool(tools):
        return next(
            (tool for tool in (tools or []) if tool.name == "final_answer"),
            None,
        )

    @staticmethod
    def _final_answer_arguments(result):
        """Return parsed arguments only for a single final_answer-only call."""
        calls = result.tool_calls or []
        if len(calls) != 1 or calls[0].function.name != "final_answer":
            return None

        arguments = calls[0].function.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                return None
        return arguments if isinstance(arguments, dict) else None

    @classmethod
    def _has_valid_final_answer(cls, result) -> bool:
        arguments = cls._final_answer_arguments(result)
        if arguments is None or "answer" not in arguments:
            return False
        answer = arguments["answer"]
        if answer is None:
            return False
        return not isinstance(answer, str) or bool(answer.strip())

    @classmethod
    def _has_malformed_final_answer(cls, result) -> bool:
        return any(
            call.function.name == "final_answer"
            for call in (result.tool_calls or [])
        ) and not cls._has_valid_final_answer(result)

    @staticmethod
    def _reasoning_content(result) -> str:
        """Keep reasoning retained by OpenAIModel in the raw provider response."""
        candidates = [getattr(result, "reasoning_content", None)]
        raw = getattr(result, "raw", None)
        try:
            message = raw.choices[0].message
            candidates.extend(
                [
                    getattr(message, "reasoning_content", None),
                    getattr(message, "reasoning", None),
                ]
            )
            model_extra = getattr(message, "model_extra", None) or {}
            candidates.extend(
                [model_extra.get("reasoning_content"), model_extra.get("reasoning")]
            )
        except (AttributeError, IndexError, TypeError):
            pass

        for candidate in candidates:
            if candidate is not None and str(candidate).strip():
                return str(candidate).strip()
        return str(result.content or "").strip()

    def _repair_malformed_final_answer(
        self,
        *,
        delegate,
        result,
        messages,
        final_answer_tool,
        stop_sequences,
        response_format,
        kwargs,
    ):
        """Repair Cohere compatibility responses that omit final_answer.answer."""
        if final_answer_tool is None or not self._has_malformed_final_answer(result):
            return result

        prior_reasoning = self._reasoning_content(result)
        repair_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Your preceding response selected final_answer but omitted its "
                    "required answer argument. Do not research or reconsider the "
                    "task. Using the conclusion already reached, call final_answer "
                    "now with exactly one non-empty argument named answer. The "
                    "answer value must contain only what the original task requests."
                    + (
                        f"\n\nPreceding reasoning:\n{prior_reasoning}"
                        if prior_reasoning
                        else ""
                    )
                ),
            }
        ]
        try:
            repaired = delegate.generate(
                repair_messages,
                stop_sequences=stop_sequences,
                response_format=response_format,
                tools_to_call_from=[final_answer_tool],
                **kwargs,
            )
        except Exception:
            return result

        if not self._has_valid_final_answer(repaired):
            return result
        return repaired
    @staticmethod
    def _reasoning_effort(model_id: str) -> str:
        model_name = model_id.lower()
        if "reasoning" in model_name or "plus" in model_name:
            return "high"
        return "none"

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

    def __init__(self, model_id: str | None = None):
        self.routes = get_vision_routes(model_id)
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
