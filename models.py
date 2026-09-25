from __future__ import annotations

import json
import threading
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
    COHERE_VISION_MODEL,
    COHERE_REQUEST_DELAY_SECONDS,
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


_cohere_rate_lock = threading.Lock()
_last_cohere_request_at = 0.0


def _wait_for_cohere_request():
    global _last_cohere_request_at

    with _cohere_rate_lock:
        now = time.monotonic()
        wait = COHERE_REQUEST_DELAY_SECONDS - (
            now - _last_cohere_request_at
        )

        if wait > 0:
            time.sleep(wait)

        _last_cohere_request_at = time.monotonic()


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
    seen = set()

    for model_slot, model_id in model_pairs:
        if not model_id:
            continue

        for key_slot, api_key in key_pairs:
            if not api_key:
                continue

            identity = (api_key, model_id)
            if identity in seen:
                continue

            seen.add(identity)
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


def get_vision_routes() -> list[Route]:
    """Build dedicated multimodal routes without text-only reasoning models."""
    routes = [
        Route(key_slot, COHERE_VISION_MODEL, api_key)
        for key_slot, api_key in (
            ("primary_key", COHERE_PRIMARY_API_KEY),
            ("fallback_key", COHERE_FALLBACK_API_KEY),
        )
        if api_key and COHERE_VISION_MODEL
    ]
    if not routes:
        raise RuntimeError("No vision-capable Cohere route is configured.")
    return routes


def get_verification_routes() -> list[Route]:
    """Use Command A+ for independent image verification, never text-only models."""
    routes = [
        route
        for route in get_chat_routes()
        if "command-a-plus" in route.model_id.lower()
    ]
    if not routes:
        raise RuntimeError("No Command A+ multimodal verification route is configured.")
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
                _wait_for_cohere_request()
                result = delegate.generate(
                    messages,
                    stop_sequences=[],
                    response_format=response_format,
                    tools_to_call_from=tools_to_call_from,
                    **kwargs,
                )
                self._repair_empty_web_search_call(result, messages)
                self._repair_pdf_webpage_call(result, messages)
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

    @classmethod
    def _repair_pdf_webpage_call(cls, result, messages):
        """Convert unusable remote PDF webpage calls into targeted web searches."""
        task = cls._original_task_text(messages)
        if not task:
            return result

        for call in (result.tool_calls or []):
            function = getattr(call, "function", None)
            if not function or function.name != "visit_webpage":
                continue

            arguments = function.arguments
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except (TypeError, ValueError):
                    continue
            else:
                parsed = arguments

            if not isinstance(parsed, dict):
                continue

            url = parsed.get("url", "")
            if not isinstance(url, str) or ".pdf" not in url.lower():
                continue

            function.name = "web_search"
            function.arguments = json.dumps(
                {
                    "query": f"{' '.join(task.split())} {url}"
                },
                ensure_ascii=False,
            )

        return result

    @classmethod
    def _original_task_text(cls, messages) -> str:
        for message in messages or []:
            role = (
                message.get("role")
                if isinstance(message, dict)
                else getattr(message, "role", None)
            )
            content = (
                message.get("content")
                if isinstance(message, dict)
                else getattr(message, "content", None)
            )
            if role == "user" and isinstance(content, str) and content.strip():
                return content.strip()
        return cls._message_text(messages).strip()

    @classmethod
    def _repair_empty_web_search_call(cls, result, messages):
        """Fill malformed empty web_search calls with the original task query."""
        task = cls._original_task_text(messages)
        if not task:
            return result

        for call in (result.tool_calls or []):
            function = getattr(call, "function", None)
            if not function or function.name != "web_search":
                continue

            arguments = function.arguments
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except (TypeError, ValueError):
                    continue
            else:
                parsed = arguments

            if parsed == {}:
                function.arguments = json.dumps(
                    {"query": " ".join(task.split())},
                    ensure_ascii=False,
                )

        return result


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

    @classmethod
    def _answer_value(cls, result):
        arguments = cls._final_answer_arguments(result)
        return arguments.get("answer") if arguments else None

    @staticmethod
    def _message_text(messages) -> str:
        parts = []
        for message in messages or []:
            content = (
                message.get("content")
                if isinstance(message, dict)
                else getattr(message, "content", None)
            )
            if isinstance(content, str):
                parts.append(content)
        return "\n".join(parts).lower()

    @classmethod
    def _needs_canonical_answer_repair(cls, result, messages) -> bool:
        """Detect verbose serialization only for clearly short exact-answer tasks."""
        answer = cls._answer_value(result)
        if not isinstance(answer, str) or len(answer.split()) <= 1:
            return False
        task_text = cls._message_text(messages)
        # Some exact-answer tasks encode their instruction by reversing the full
        # string. Decode only for task-shape classification; the answer itself is
        # still produced by the model's existing conclusion and one repair call.
        task_variants = (task_text, task_text[::-1])
        simple_markers = (
            "opposite of",
            "one word",
            "single word",
            "answer with the word",
            "what word",
            "which word",
        )
        matching_variants = [
            variant
            for variant in task_variants
            if any(marker in variant for marker in simple_markers)
        ]
        if not matching_variants:
            return False
        # Explanatory tasks legitimately need prose even when they mention a word.
        return not any(
            marker in variant
            for variant in matching_variants
            for marker in ("explain", "describe", "justify", "show your work")
        )

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
        """Repair Cohere compatibility responses that omit final_answer.answer.

        Command A can select the correct final tool while serializing its arguments
        as ``{}``.  Letting that reach ToolCallingAgent turns a completed task into
        a tool error and another research step.  Re-ask the same model to serialize
        only the already-derived answer, exposing only the final tool.  This stays
        at the model boundary, where malformed compatibility responses belong.
        """
        malformed = self._has_malformed_final_answer(result)
        verbose_simple_answer = self._needs_canonical_answer_repair(result, messages)
        if final_answer_tool is None or not (malformed or verbose_simple_answer):
            return result

        prior_reasoning = self._reasoning_content(result)
        current_answer = self._answer_value(result)
        repair_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Your preceding response selected final_answer but its argument "
                    "was missing, invalid, or unnecessarily verbose for the original "
                    "exact-answer task. Do not research or reconsider the task. Using "
                    "only the conclusion already reached, call final_answer now with "
                    "exactly one non-empty argument named answer. Return the minimal "
                    "answer value requested by the original task, without explanation."
                    + (
                        f"\n\nCurrent answer value:\n{current_answer}"
                        if current_answer is not None
                        else ""
                    )
                    + (
                        f"\n\nPreceding reasoning:\n{prior_reasoning}"
                        if prior_reasoning
                        else ""
                    )
                ),
            }
        ]
        try:
            _wait_for_cohere_request()
            repaired = delegate.generate(
                repair_messages,
                stop_sequences=[],
                response_format=response_format,
                tools_to_call_from=[final_answer_tool],
                **kwargs,
            )
        except Exception:
            # The original generation succeeded. A best-effort serialization
            # repair must not turn that route success into a failover event.
            return result

        # Never replace the original malformed call with another malformed call;
        # the normal agent error remains visible if the provider cannot repair it.
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

    def __init__(self):
        # Multimodal requests must never fail over to text-only reasoning models.
        # Text-agent FailoverModel continues to use all four normal routes.
        self.routes = get_vision_routes()
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
                client = cohere.ClientV2(
                    route.api_key,
                    log_warning_experimental_features=False,
                )
                _wait_for_cohere_request()
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
                client = cohere.ClientV2(
                    route.api_key,
                    log_warning_experimental_features=False,
                )
                with open(file_path, "rb") as audio_file:
                    _wait_for_cohere_request()
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


class CohereTextFailoverClient(CohereFailoverClient):
    """Direct text client retaining the normal four model/key routes."""

    def __init__(self):
        self.routes = get_chat_routes()
        self.last_route: Route | None = None
        self.failover_count = 0
        self._cooldowns: dict[tuple[str, str], float] = {}


class CohereVerificationClient(CohereFailoverClient):
    """Independent multimodal client backed only by Command A+ routes."""

    def __init__(self):
        self.routes = get_verification_routes()
        self.last_route: Route | None = None
        self.failover_count = 0
        self._cooldowns: dict[tuple[str, str], float] = {}
