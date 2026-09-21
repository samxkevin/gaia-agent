from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smolagents import Tool

from config import (
    ADVERSAL_CLI_PACKAGE,
    ADVERSAL_ENABLED,
    ADVERSAL_TIMEOUT_SECONDS,
    ADVERSAL_POLL_SECONDS,
)
from tools.video import (
    AnalyzeYouTubeVideoTool as NativeAnalyzeYouTubeVideoTool,
    FrameObservation,
    VideoCountingPlan,
    build_video_counting_plan,
    normalize_youtube_url,
)

_REQUEST_ID = re.compile(r"request_id\s*:\s*([A-Za-z0-9_.:-]+)", re.I)
_MINUTES = re.compile(r"([\d.]+)\s*minutes?", re.I)
_TIME = re.compile(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$")
_TIMESTAMP_KEYS = ("timestamp", "timestamp_seconds", "time", "start", "seconds", "ts")
_PATH_KEYS = ("frame", "path", "file", "filename", "image")
_OCR_KEYS = ("ocr", "ocr_text", "text")


@dataclass(frozen=True)
class AdversalFrame:
    timestamp: float
    path: Path
    ocr_text: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class AdversalEvidence:
    status: str
    message: str
    request_id: str | None
    remaining_minutes: float | None
    estimated_required_minutes: int | None
    frames: tuple[AdversalFrame, ...]
    notes: str
    output_dir: str | None


def _result_text(result: Any) -> str:
    for attribute in ("data", "content"):
        value = getattr(result, attribute, None)
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [getattr(item, "text", None) for item in value]
            joined = "\n".join(item for item in parts if isinstance(item, str))
            if joined:
                return joined
    return str(result)


def parse_request_id(text: str) -> str | None:
    match = _REQUEST_ID.search(text or "")
    return match.group(1) if match else None


def parse_remaining_minutes(text: str) -> float | None:
    value = text or ""
    patterns = (
        r"(?:remaining|left|available)\D{0,48}([\d.]+)\s*minutes?",
        r"([\d.]+)\s*minutes?\D{0,24}(?:remaining|left|available)",
    )
    for pattern in patterns:
        match = re.search(pattern, value, re.I)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None

    matches = _MINUTES.findall(value)
    if len(matches) == 1:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


def parse_time_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    match = _TIME.match(text)
    if match:
        hours, minutes, seconds = match.groups()
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    try:
        return float(text)
    except ValueError:
        return None


def classify_response(text: str) -> str:
    stripped = (text or "").strip()
    upper = stripped.upper()
    if not stripped:
        return "unknown"
    if upper.startswith("AUTHENTICATION REQUIRED"):
        return "auth_required"
    if upper.startswith("QUOTA EXHAUSTED"):
        return "quota_exhausted"
    if upper.startswith("COMPLETED") or upper.startswith("SUCCESS"):
        return "completed"
    if upper.startswith("RUNNING") or upper.startswith("NOT READY"):
        return "running"
    if upper.startswith("FAILED"):
        return "failed"
    if upper.startswith("UNAVAILABLE"):
        return "unavailable"
    if upper.startswith("AUTHENTICATED"):
        return "ok"
    return "unknown"


def resolve_adversal_command() -> tuple[str, list[str]] | None:
    configured = os.getenv("ADVERSAL_CLI", "").strip()
    if configured:
        return configured, []

    installed = shutil.which("adversal-cli")
    if installed:
        return installed, []

    uvx = shutil.which("uvx")
    if uvx:
        return uvx, ["--python", "3.13", ADVERSAL_CLI_PACKAGE]

    return None


def estimate_youtube_duration_seconds(url: str) -> float | None:
    try:
        import yt_dlp

        options = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        duration = info.get("duration") if isinstance(info, dict) else None
        return float(duration) if duration is not None else None
    except Exception:
        return None


def _load_frames_json(directory: Path) -> list[AdversalFrame]:
    manifest_candidates = [directory / "frames.json", directory / "frames" / "frames.json"]
    manifest = next((path for path in manifest_candidates if path.is_file()), None)
    if manifest is None:
        return []

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (
                payload[key]
                for key in ("frames", "items", "results")
                if isinstance(payload.get(key), list)
            ),
            [],
        )
    else:
        rows = []

    frames: list[AdversalFrame] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp = parse_time_value(
            next((row[key] for key in _TIMESTAMP_KEYS if key in row), None)
        )
        name = next((row[key] for key in _PATH_KEYS if row.get(key) is not None), None)
        if timestamp is None or not isinstance(name, str):
            continue

        raw_path = Path(name)
        candidates = [
            directory / raw_path,
            directory / "frames" / raw_path.name,
            directory / raw_path.name,
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            continue

        ocr = next(
            (row[key] for key in _OCR_KEYS if isinstance(row.get(key), str)),
            None,
        )
        frames.append(
            AdversalFrame(
                timestamp=timestamp,
                path=path,
                ocr_text=ocr,
                raw=row,
            )
        )

    dedup: dict[tuple[float, str], AdversalFrame] = {}
    for frame in frames:
        dedup[(round(frame.timestamp, 3), str(frame.path.resolve()))] = frame
    return sorted(dedup.values(), key=lambda item: (item.timestamp, str(item.path)))


async def _call_tool(session, name: str, arguments: dict[str, Any]) -> str:
    result = await asyncio.wait_for(
        session.call_tool(name, arguments),
        timeout=max(30.0, min(float(ADVERSAL_TIMEOUT_SECONDS), 120.0)),
    )
    return _result_text(result)


async def _run_adversal_async(
    url: str,
    output_dir: Path,
) -> AdversalEvidence:
    command = resolve_adversal_command()
    if command is None:
        return AdversalEvidence(
            status="unavailable",
            message="adversal-cli and uvx are not installed",
            request_id=None,
            remaining_minutes=None,
            estimated_required_minutes=None,
            frames=(),
            notes="",
            output_dir=None,
        )

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:
        return AdversalEvidence(
            status="unavailable",
            message=f"MCP client unavailable: {type(exc).__name__}: {exc}",
            request_id=None,
            remaining_minutes=None,
            estimated_required_minutes=None,
            frames=(),
            notes="",
            output_dir=None,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_url = normalize_youtube_url(url)
    duration = estimate_youtube_duration_seconds(source_url)
    required_minutes = math.ceil(duration / 60.0) if duration and duration > 0 else None

    command_name, args = command
    transport = StdioServerParameters(command=command_name, args=args)

    try:
        async with stdio_client(transport) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=max(30.0, min(float(ADVERSAL_TIMEOUT_SECONDS), 120.0)),
                )

                quota_text = await _call_tool(session, "check_remaining_quota", {})
                remaining = parse_remaining_minutes(quota_text)
                quota_status = classify_response(quota_text)

                if quota_status == "auth_required":
                    return AdversalEvidence(
                        status="auth_required",
                        message=quota_text,
                        request_id=None,
                        remaining_minutes=remaining,
                        estimated_required_minutes=required_minutes,
                        frames=(),
                        notes="",
                        output_dir=None,
                    )

                if remaining is None:
                    return AdversalEvidence(
                        status="unavailable",
                        message="Could not parse remaining Adversal quota; refusing provider call for free-tier safety.",
                        request_id=None,
                        remaining_minutes=None,
                        estimated_required_minutes=required_minutes,
                        frames=(),
                        notes="",
                        output_dir=None,
                    )

                if required_minutes is None:
                    return AdversalEvidence(
                        status="unavailable",
                        message="Could not determine video duration; refusing provider call for free-tier quota safety.",
                        request_id=None,
                        remaining_minutes=remaining,
                        estimated_required_minutes=None,
                        frames=(),
                        notes="",
                        output_dir=None,
                    )

                if remaining + 1e-9 < required_minutes:
                    return AdversalEvidence(
                        status="quota_insufficient",
                        message=(
                            f"Adversal free-tier quota is insufficient: "
                            f"{remaining:g} minutes remaining, approximately "
                            f"{required_minutes} required."
                        ),
                        request_id=None,
                        remaining_minutes=remaining,
                        estimated_required_minutes=required_minutes,
                        frames=(),
                        notes="",
                        output_dir=None,
                    )

                process_text = await _call_tool(
                    session,
                    "process_video",
                    {
                        "video_url": source_url,
                        "output_path": str(output_dir),
                    },
                )
                request_id = parse_request_id(process_text)
                process_status = classify_response(process_text)
                if process_status in {"auth_required", "quota_exhausted", "failed", "unavailable"}:
                    return AdversalEvidence(
                        status=process_status,
                        message=process_text,
                        request_id=request_id,
                        remaining_minutes=remaining,
                        estimated_required_minutes=required_minutes,
                        frames=(),
                        notes="",
                        output_dir=str(output_dir),
                    )
                if not request_id:
                    return AdversalEvidence(
                        status="unavailable",
                        message=f"Adversal did not return a request_id: {process_text}",
                        request_id=None,
                        remaining_minutes=remaining,
                        estimated_required_minutes=required_minutes,
                        frames=(),
                        notes="",
                        output_dir=str(output_dir),
                    )

                deadline = asyncio.get_running_loop().time() + float(ADVERSAL_TIMEOUT_SECONDS)
                last_status = process_text

                while asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(max(1.0, float(ADVERSAL_POLL_SECONDS)))
                    status_text = await _call_tool(
                        session,
                        "check_video_status",
                        {"request_id": request_id},
                    )
                    last_status = status_text
                    status = classify_response(status_text)
                    if status == "completed":
                        break
                    if status in {"auth_required", "quota_exhausted", "failed", "unavailable"}:
                        return AdversalEvidence(
                            status=status,
                            message=status_text,
                            request_id=request_id,
                            remaining_minutes=remaining,
                            estimated_required_minutes=required_minutes,
                            frames=(),
                            notes="",
                            output_dir=str(output_dir),
                        )
                else:
                    return AdversalEvidence(
                        status="timeout",
                        message=f"Adversal processing exceeded {ADVERSAL_TIMEOUT_SECONDS}s: {last_status}",
                        request_id=request_id,
                        remaining_minutes=remaining,
                        estimated_required_minutes=required_minutes,
                        frames=(),
                        notes="",
                        output_dir=str(output_dir),
                    )

                await _call_tool(
                    session,
                    "extract_frames",
                    {
                        "request_id": request_id,
                        "output_path": str(output_dir),
                    },
                )

        frames = _load_frames_json(output_dir)
        notes_path = output_dir / "notes.md"
        notes = notes_path.read_text(encoding="utf-8", errors="replace") if notes_path.is_file() else ""
        status = "completed" if frames else "completed_without_frames"
        message = (
            f"Adversal completed request {request_id} and returned "
            f"{len(frames)} timestamped frame artifacts."
        )
        return AdversalEvidence(
            status=status,
            message=message,
            request_id=request_id,
            remaining_minutes=remaining,
            estimated_required_minutes=required_minutes,
            frames=tuple(frames),
            notes=notes,
            output_dir=str(output_dir),
        )
    except Exception as exc:
        return AdversalEvidence(
            status="unavailable",
            message=f"{type(exc).__name__}: {exc}",
            request_id=None,
            remaining_minutes=None,
            estimated_required_minutes=required_minutes,
            frames=(),
            notes="",
            output_dir=str(output_dir),
        )


def run_adversal_video(url: str, output_dir: Path) -> AdversalEvidence:
    try:
        return asyncio.run(_run_adversal_async(url, output_dir))
    except RuntimeError as exc:
        if "cannot be called from a running event loop" in str(exc).lower():
            return AdversalEvidence(
                status="unavailable",
                message="Adversal integration cannot run inside an active event loop.",
                request_id=None,
                remaining_minutes=None,
                estimated_required_minutes=None,
                frames=(),
                notes="",
                output_dir=None,
            )
        raise


def _parse_native_evidence(result: str) -> dict[str, Any]:
    marker = "Structured video evidence:"
    start = result.find(marker)
    if start < 0:
        return {}
    payload = result[start + len(marker):].lstrip()
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(payload)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class AdversalAugmentedYouTubeVideoTool(Tool):
    name = "analyze_youtube_video"
    description = (
        "Analyze YouTube visual questions with Adversal as an independent "
        "candidate-frame ingestion layer plus the native FFmpeg/Cohere "
        "pipeline and independent visual verification."
    )
    inputs = {
        "url": {"type": "string", "description": "YouTube URL, including Markdown links."},
        "question": {
            "type": "string",
            "description": "The visual question; preserve its exact counting and temporal semantics.",
        },
        "max_frames": {
            "type": "integer",
            "description": "Optional native coarse frame limit (12-120).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, original_question: str | None = None):
        super().__init__()
        self.original_question = original_question

    def forward(self, url: str, question: str, max_frames: int | None = None) -> str:
        with tempfile.TemporaryDirectory(prefix="gaia-adversal-") as temporary:
            root = Path(temporary)
            return self._forward_impl(url, question, max_frames, root)

    def _forward_impl(
        self,
        url: str,
        question: str,
        max_frames: int | None,
        root: Path,
    ) -> str:
        question = self.original_question or question
        plan: VideoCountingPlan = build_video_counting_plan(question)

        if not ADVERSAL_ENABLED or not plan.is_numeric_maximum:
            return NativeAnalyzeYouTubeVideoTool(original_question=question).forward(
                url, question, max_frames
            )

        job_dir = root / re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id(url))
        job_dir.mkdir(parents=True, exist_ok=True)

        adversal = run_adversal_video(url, job_dir)
        native = NativeAnalyzeYouTubeVideoTool(original_question=question)

        if not adversal.frames:
            native_result = native.forward(url, question, max_frames)
            return (
                native_result
                + "\nAdversal auxiliary ingestion:\n"
                + json.dumps(
                    {
                        "status": adversal.status,
                        "message": adversal.message,
                        "remaining_minutes": adversal.remaining_minutes,
                        "estimated_required_minutes": adversal.estimated_required_minutes,
                    },
                    indent=2,
                )
            )

        adversal_frames = [(frame.timestamp, frame.path) for frame in adversal.frames]
        try:
            adversal_observations = native._observe_frames(
                adversal_frames,
                plan,
                "adversal",
            )
            adversal_candidates = native._select_candidates(adversal_observations)
            adversal_verified = native._verify_candidates(
                adversal_candidates,
                plan,
            )
        except Exception as exc:
            adversal_observations = []
            adversal_verified = []
            native.verification_diagnostics.append(
                {
                    "timestamp": None,
                    "status": "failed",
                    "stage": "adversal_verification",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )

        native_result = native.forward(url, question, max_frames)
        native_evidence = _parse_native_evidence(native_result)

        native_max = _safe_int(native_evidence.get("programmatic_maximum_observed"))
        adversal_max = max(
            (item.count for item in adversal_verified),
            default=0,
        )
        combined_max = max(native_max, adversal_max)

        supporting = []
        for item in adversal_verified:
            if item.count == combined_max:
                supporting.append(
                    {
                        "source": "adversal_verified",
                        "timestamp": item.timestamp,
                        "species": list(item.species),
                        "count": item.count,
                        "confidence": item.confidence,
                    }
                )
        for item in native_evidence.get("same_frame_candidates", []):
            if _safe_int(item.get("count")) == combined_max:
                supporting.append(
                    {
                        "source": f"native_{item.get('pass_name', 'evidence')}",
                        "timestamp": item.get("timestamp"),
                        "species": item.get("species", []),
                        "count": item.get("count"),
                        "confidence": item.get("confidence"),
                    }
                )

        combined = {
            "question_semantics": plan.entity,
            "strategy": {
                "adversal_primary_candidate_ingestion": True,
                "native_visual_fallback": True,
                "independent_verification": True,
                "global_maximum_proven": False,
                "evidence_type": "sampled",
            },
            "adversal": {
                "status": adversal.status,
                "message": adversal.message,
                "request_id": adversal.request_id,
                "remaining_minutes_before_submission": adversal.remaining_minutes,
                "estimated_required_minutes": adversal.estimated_required_minutes,
                "extracted_frame_count": len(adversal.frames),
                "observed_maximum": max(
                    (item.count for item in adversal_observations),
                    default=0,
                ),
                "verified_maximum": adversal_max,
                "verified_observations": [_public_observation(item) for item in adversal_verified],
            },
            "native": native_evidence,
            "combined_programmatic_maximum_observed": combined_max,
            "supporting_same_frame_evidence": supporting,
        }

        return (
            "Combined video evidence:\n"
            + json.dumps(combined, indent=2)
            + "\nNative synthesis:\n"
            + native_result.split("\nSynthesis:\n", 1)[-1]
        )


def source_id(url: str) -> str:
    normalized = normalize_youtube_url(url)
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{6,})", normalized)
    if match:
        return match.group(1)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized)[-80:]


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _public_observation(item: FrameObservation) -> dict[str, Any]:
    return {
        "timestamp": item.timestamp,
        "visible_subjects": list(item.visible_subjects),
        "species": list(item.species),
        "count": item.count,
        "confidence": item.confidence,
        "uncertain": list(item.uncertain),
        "note": item.note,
        "pass_name": item.pass_name,
    }
