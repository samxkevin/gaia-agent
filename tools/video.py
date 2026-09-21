from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from smolagents import Tool

from models import (
    CohereFailoverClient,
    CohereTextFailoverClient,
    CohereVerificationClient,
)

COARSE_MAX_FRAMES = 120
COARSE_INTERVAL_SECONDS = 1.0
REFINE_CANDIDATES = 4
REFINE_REGION_INTERVAL_SECONDS = 1.0
CANDIDATE_MIN_SEPARATION_SECONDS = 5.0
VERIFICATION_LIMIT = 4
VISION_BATCH_SIZE = 4
MAX_FRAME_WIDTH = 1280

OBSERVATION_RESPONSE_FORMAT = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "number"},
                        "visible_subjects": {"type": "array", "items": {"type": "string"}},
                        "species": {"type": "array", "items": {"type": "string"}},
                        "count": {"type": "integer"},
                        "confidence": {"type": "number"},
                        "uncertain": {"type": "array", "items": {"type": "string"}},
                        "note": {"type": "string"},
                    },
                    "required": [
                        "timestamp", "visible_subjects", "species", "count",
                        "confidence", "uncertain", "note",
                    ],
                },
            }
        },
        "required": ["observations"],
    },
}


def find_javascript_runtime(which=shutil.which) -> tuple[str, str] | None:
    for runtime, executables in (("deno", ("deno",)), ("node", ("node", "nodejs")), ("quickjs", ("qjs", "quickjs")), ("bun", ("bun",))):
        for executable in executables:
            if path := which(executable):
                return runtime, path
    return None


@dataclass(frozen=True)
class VideoCountingPlan:
    is_numeric_maximum: bool
    entity: str
    distinct_categories: bool
    simultaneous: bool
    instruction: str


@dataclass(frozen=True)
class FrameObservation:
    timestamp: float
    visible_subjects: tuple[str, ...]
    species: tuple[str, ...]
    count: int
    confidence: float
    uncertain: tuple[str, ...]
    note: str = ""
    pass_name: str = "coarse"
    frame_path: str = ""


def build_video_counting_plan(question: str) -> VideoCountingPlan:
    text = question.lower()
    maximum = any(word in text for word in ("highest", "maximum", "most"))
    simultaneous = any(p in text for p in ("simultaneously", "at once", "visible together", "at the same time"))
    species = "species" in text
    entity = "distinct species" if species else "individual visible entities"
    unit = ("Count distinct species, not individual animals; repeated individuals of one species count once." if species else "Count visible individuals separately; do not collapse them by species.")
    temporal = ("Simultaneous means visible in the same frame. Never combine subjects from different timestamps." if simultaneous else "Preserve the temporal relationship in the original question.")
    return VideoCountingPlan(maximum and simultaneous, entity, species, simultaneous, f"Original question: {question}\n{unit}\n{temporal}")


def normalize_youtube_url(value: str) -> str:
    markdown = re.search(r"\[[^]]*\]\((https?://[^)]+)\)", value)
    if markdown:
        value = markdown.group(1)
    match = re.search(r"https?://(?:www\.)?(?:youtube\.com/[^\s)>]+|youtu\.be/[^\s)>]+)", value)
    return match.group(0).rstrip(".,;") if match else value.strip()


class AnalyzeYouTubeVideoTool(Tool):
    name = "analyze_youtube_video"
    description = "Analyze timestamped YouTube frames with a bounded coarse-to-fine visual pipeline. Use for visible actions, objects, species, and simultaneous counts."
    inputs = {
        "url": {"type": "string", "description": "YouTube URL, including Markdown links."},
        "question": {"type": "string", "description": "The visual question; preserve the user's counting unit and temporal semantics."},
        "max_frames": {"type": "integer", "description": "Optional coarse frame limit (12-120).", "nullable": True},
    }
    output_type = "string"

    def __init__(
        self,
        visual_client_factory: Callable | None = None,
        synthesis_client_factory: Callable | None = None,
        verification_client_factory: Callable | None = None,
        original_question: str | None = None,
    ):
        super().__init__()
        self.visual_client_factory = visual_client_factory or CohereFailoverClient
        # One injected factory keeps tests and custom deployments fully controlled.
        self.synthesis_client_factory = synthesis_client_factory or (
            visual_client_factory if visual_client_factory is not None else CohereTextFailoverClient
        )
        self.verification_client_factory = verification_client_factory or (
            visual_client_factory if visual_client_factory is not None else CohereVerificationClient
        )
        self.original_question = original_question
        self.last_observations: list[FrameObservation] = []
        self.observation_parse_diagnostics: dict[str, dict] = {}
        self.observation_batch_diagnostics: dict[str, list[dict]] = {}
        self.coarse_diagnostics: dict = {}
        self.refinement_diagnostics: list[dict] = []
        self.verification_diagnostics: list[dict] = []

    def forward(self, url: str, question: str, max_frames: int | None = None) -> str:
        question = self.original_question or question
        plan = build_video_counting_plan(question)
        budget = max(12, min(int(max_frames or COARSE_MAX_FRAMES), COARSE_MAX_FRAMES))
        try:
            with tempfile.TemporaryDirectory(prefix="gaia-video-") as directory:
                root = Path(directory)
                video = self._download(normalize_youtube_url(url), root)
                duration = self._duration(video)
                interval = max(COARSE_INTERVAL_SECONDS if plan.is_numeric_maximum else duration / budget, duration / budget, 0.25)
                coarse = self._extract_frames(video, root / "coarse", duration, budget, interval=interval)
                observations = self._observe_frames(coarse, plan, "coarse")
                self.coarse_diagnostics = {
                    **self._coverage_diagnostics(
                        observations, requested_frames=len(coarse)
                    ),
                    **self.observation_parse_diagnostics.get("coarse", {}),
                }
                if plan.is_numeric_maximum and observations:
                    candidates = self._select_candidates(observations, duration)
                    if candidates:
                        try:
                            refined_frames = self._extract_candidate_frames(
                                video, root / "refined", duration, candidates
                            )
                            if refined_frames:
                                observations.extend(
                                    self._observe_frames(refined_frames, plan, "refined")
                                )
                        except Exception as exc:
                            # Coarse evidence remains usable, but failure is never silent.
                            self.refinement_diagnostics.append({
                                "attempted": True,
                                "candidate_timestamp": None,
                                "start": None,
                                "end": None,
                                "extracted_frame_count": 0,
                                "status": "failed",
                                "exception_type": type(exc).__name__,
                                "exception_message": str(exc),
                                "fallback_coarse_retained": True,
                            })
                    try:
                        verification_candidates = self._select_candidates(
                            self._merge_observations(observations), duration
                        )
                        observations.extend(
                            self._verify_candidates(verification_candidates, plan)
                        )
                    except Exception:
                        # Independent verification is bounded and best effort.
                        pass
                self.last_observations = observations
                return self._synthesize(observations, plan, duration, len(coarse))
        except Exception as exc:
            return f"Video frame analysis failed: {type(exc).__name__}: {exc}"

    def _download(self, url: str, workdir: Path) -> Path:
        from imageio_ffmpeg import get_ffmpeg_exe
        runtime = find_javascript_runtime()
        if runtime is None:
            raise RuntimeError("No yt-dlp JavaScript runtime found (deno, node, quickjs, or bun).")
        runtime_name, runtime_path = runtime
        template = str(workdir / "video.%(ext)s")
        subprocess.run([sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "--ffmpeg-location", get_ffmpeg_exe(), "--js-runtimes", f"{runtime_name}:{runtime_path}", "-f", "bestvideo[height<=720]/best[height<=720]/worst", "--merge-output-format", "mp4", "-o", template, url], check=True, capture_output=True, text=True, timeout=300)
        files = [p for p in workdir.glob("video.*") if p.is_file()]
        if not files:
            raise RuntimeError("yt-dlp produced no video file")
        return files[0]

    def _duration(self, video: Path) -> float:
        from imageio_ffmpeg import count_frames_and_secs
        return float(count_frames_and_secs(str(video))[1])

    def _extract_frames(self, video: Path, output_dir: Path, duration: float, max_frames: int, interval: float | None = None, offset: float = 0.0):
        from imageio_ffmpeg import get_ffmpeg_exe
        output_dir.mkdir(parents=True, exist_ok=True)
        interval = interval or max(duration / max_frames, 0.25)
        command = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error"]
        if offset:
            command += ["-ss", f"{offset:.6f}"]
        command += ["-i", str(video), "-vf", f"fps=1/{interval},scale='min({MAX_FRAME_WIDTH},iw)':-2,showinfo", "-frames:v", str(max_frames), "-q:v", "2", str(output_dir / "%04d.jpg")]
        completed = subprocess.run(command, check=True, capture_output=True, timeout=180)
        paths = sorted(output_dir.glob("*.jpg"))
        stderr = getattr(completed, "stderr", b"") or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        # showinfo reports timestamps after seeking, relative to the selected input.
        # Prefer these actual presentation timestamps over index-derived estimates.
        reported = [float(value) for value in re.findall(r"pts_time:([0-9.+-]+)", stderr)]
        timestamps = [min(offset + value, duration) for value in reported[:len(paths)]]
        if len(timestamps) < len(paths):
            timestamps.extend(
                min(offset + index * interval, duration)
                for index in range(len(timestamps), len(paths))
            )
        return list(zip(timestamps, paths))

    def _extract_candidate_frames(self, video: Path, output_dir: Path, duration: float, candidates: list[FrameObservation]):
        """Refine complete temporal regions at a bounded one-frame/second rate."""
        frames = []
        seen = set()
        seen_regions = set()
        self.refinement_diagnostics = []
        region_width = duration / REFINE_CANDIDATES
        for candidate_index, candidate in enumerate(candidates[:REFINE_CANDIDATES]):
            region = min(
                int(candidate.timestamp / region_width),
                REFINE_CANDIDATES - 1,
            )
            if region in seen_regions:
                continue
            seen_regions.add(region)
            start = region * region_width
            end = duration if region == REFINE_CANDIDATES - 1 else (region + 1) * region_width
            count = int((end - start) / REFINE_REGION_INTERVAL_SECONDS) + 1
            diagnostic = {
                "attempted": True,
                "region_index": region,
                "candidate_timestamp": candidate.timestamp,
                "start": start,
                "end": end,
                "interval_seconds": REFINE_REGION_INTERVAL_SECONDS,
                "extracted_frame_count": 0,
                "status": "pending",
                "exception_type": None,
                "exception_message": None,
                "fallback_coarse_retained": True,
            }
            try:
                batch = self._extract_frames(
                    video,
                    output_dir / str(candidate_index),
                    duration,
                    count,
                    interval=REFINE_REGION_INTERVAL_SECONDS,
                    offset=start,
                )
                diagnostic["extracted_frame_count"] = len(batch)
                diagnostic["status"] = "extracted" if batch else "empty"
                for timestamp, path in batch:
                    key = round(timestamp, 3)
                    if key not in seen:
                        seen.add(key)
                        frames.append((timestamp, path))
            except Exception as exc:
                diagnostic.update(
                    status="failed",
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
            self.refinement_diagnostics.append(diagnostic)
        return sorted(frames)

    def _observe_frames(self, frames, plan: VideoCountingPlan, pass_name: str) -> list[FrameObservation]:
        client = self.visual_client_factory()
        observations = []
        parsed_count = 0
        placeholder_count = 0
        batch_diagnostics = []
        for start in range(0, len(frames), VISION_BATCH_SIZE):
            batch = frames[start:start + VISION_BATCH_SIZE]
            diagnostic = {
                "batch_index": start // VISION_BATCH_SIZE,
                "requested_observations": len(batch),
                "returned_observations": 0,
                "status": "pending",
                "exception_type": None,
                "exception_message": None,
            }
            content = [{"type": "text", "text": f"{plan.instruction}\nReturn ONLY a JSON object with an observations array, one object per image, in the same order. Each observation has timestamp, visible_subjects, species, count, confidence, uncertain, and note. Inspect each image independently. Never merge evidence across images. Use timestamp labels exactly."}]
            try:
                for timestamp, path in batch:
                    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                    content += [{"type": "text", "text": f"timestamp={timestamp:.3f}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}}]
                response = self._chat_observations(
                    client, messages=[{"role": "user", "content": content}]
                )
                text = self._response_text(response)
                parsed = self._parse_observations(text, batch, pass_name)
                payload = self._json_payload(text) if text.strip() else None
                raw_values = (
                    payload.get("observations", [])
                    if isinstance(payload, dict)
                    else payload if isinstance(payload, list) else None
                )
                diagnostic["returned_observations"] = len(parsed)
                if not text.strip():
                    diagnostic["status"] = "empty_response"
                elif payload is None:
                    diagnostic["status"] = "response_parsing_failure"
                elif not isinstance(raw_values, list):
                    diagnostic["status"] = "malformed_observations_container"
                elif len(raw_values) != len(batch):
                    diagnostic["status"] = "wrong_observation_count"
                    diagnostic["provider_observation_count"] = len(raw_values)
                elif len(parsed) != len(batch):
                    diagnostic["status"] = "malformed_observation"
                else:
                    diagnostic["status"] = "complete"
            except Exception as exc:
                parsed = []
                diagnostic.update(
                    status="request_failure",
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
            batch_diagnostics.append(diagnostic)
            parsed_count += len(parsed)
            observations.extend(parsed)
            parsed_timestamps = {item.timestamp for item in parsed}
            for timestamp, path in batch:
                if float(timestamp) not in parsed_timestamps:
                    placeholder_count += 1
                    observations.append(FrameObservation(
                        float(timestamp),
                        (),
                        (),
                        0,
                        0.0,
                        ("provider observation missing",),
                        "No parseable observation was returned for this extracted frame.",
                        pass_name,
                        str(path),
                    ))
        self.observation_batch_diagnostics[pass_name] = batch_diagnostics
        self.observation_parse_diagnostics[pass_name] = {
            "requested_frame_count": len(frames),
            "parsed_observation_count": parsed_count,
            "missing_observation_count": placeholder_count,
            "complete_batches": sum(item["status"] == "complete" for item in batch_diagnostics),
            "failed_or_incomplete_batches": sum(item["status"] != "complete" for item in batch_diagnostics),
        }
        return sorted(observations, key=lambda item: item.timestamp)

    @staticmethod
    def _chat_observations(client, *, messages):
        """Prefer provider-enforced JSON, falling back only for schema incompatibility."""
        try:
            return client.chat(
                messages=messages,
                temperature=0,
                response_format=OBSERVATION_RESPONSE_FORMAT,
            )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}".lower()
            if not any(
                marker in detail
                for marker in ("response_format", "schema", "structured", "typeerror", "400")
            ):
                raise
            return client.chat(messages=messages, temperature=0)

    @staticmethod
    def _json_payload(text: str):
        stripped = text.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.I)
        if fenced:
            stripped = fenced.group(1).strip()
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            decoder = json.JSONDecoder()
            for index, character in enumerate(stripped):
                if character not in "[{":
                    continue
                try:
                    value, _ = decoder.raw_decode(stripped[index:])
                    return value
                except json.JSONDecodeError:
                    continue
        return None

    @staticmethod
    def _normalize_strings(value, *, uncertainty: bool = False) -> tuple[str, ...]:
        if value is None or value is False:
            return ()
        if value is True:
            return ("unspecified uncertainty",) if uncertainty else ()
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned or cleaned.lower() in {"none", "null", "false", "no", "n/a"}:
                return ()
            return (cleaned,)
        if isinstance(value, (list, tuple, set)):
            normalized = []
            for item in value:
                if item is None or isinstance(item, bool):
                    continue
                cleaned = str(item).strip()
                if cleaned and cleaned.lower() not in {"none", "null", "false", "n/a"}:
                    normalized.append(cleaned)
            return tuple(dict.fromkeys(normalized))
        cleaned = str(value).strip()
        return (cleaned,) if cleaned else ()

    @staticmethod
    def _safe_number(value, default: float = 0.0) -> float:
        try:
            number = float(value)
            return number if number == number and abs(number) != float("inf") else default
        except (TypeError, ValueError, OverflowError):
            return default

    @classmethod
    def _parse_observations(cls, text: str, batch, pass_name: str) -> list[FrameObservation]:
        payload = cls._json_payload(text)
        if isinstance(payload, dict):
            values = payload.get("observations", [])
        elif isinstance(payload, list):
            values = payload
        else:
            return []
        if not isinstance(values, list):
            return []

        result = []
        for index, value in enumerate(values[:len(batch)]):
            if not isinstance(value, dict):
                continue
            try:
                # FFmpeg extraction is authoritative; provider timestamps are advisory only.
                timestamp = float(batch[index][0])
                species = cls._normalize_strings(value.get("species"))
                subjects = cls._normalize_strings(value.get("visible_subjects"))
                uncertain = cls._normalize_strings(
                    value.get("uncertain"), uncertainty=True
                )
                raw_count = cls._safe_number(value.get("count"), 0.0)
                count = len(species) if species else max(0, int(raw_count))
                confidence = max(
                    0.0,
                    min(cls._safe_number(value.get("confidence"), 0.0), 1.0),
                )
                note = value.get("note", "")
                result.append(FrameObservation(
                    timestamp,
                    subjects,
                    species,
                    count,
                    confidence,
                    uncertain,
                    "" if note is None else str(note),
                    pass_name,
                    str(batch[index][1]),
                ))
            except Exception:
                # One malformed frame must not discard valid siblings in the batch.
                continue
        return result

    def _verify_candidates(
        self,
        candidates: list[FrameObservation],
        plan: VideoCountingPlan,
    ) -> list[FrameObservation]:
        """Independently reinspect a few actual candidate images with Command A+."""
        verified = []
        self.verification_diagnostics = []
        try:
            client = self.verification_client_factory()
        except Exception as exc:
            self.verification_diagnostics.append({
                "timestamp": None,
                "status": "failed",
                "stage": "initialization",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            })
            return verified
        for candidate in candidates[:VERIFICATION_LIMIT]:
            path = Path(candidate.frame_path)
            diagnostic = {
                "timestamp": candidate.timestamp,
                "status": "pending",
            }
            if not candidate.frame_path or not path.is_file():
                diagnostic.update(status="skipped", reason="candidate frame unavailable")
                self.verification_diagnostics.append(diagnostic)
                continue
            try:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                # Deliberately omit all prior labels, counts, and species claims.
                prompt = (
                    f"{plan.instruction}\nTimestamp: {candidate.timestamp:.3f} seconds. "
                    "Independently inspect only this frame. Never infer visibility from nearby "
                    "frames. Return ONLY a JSON object containing an observations array "
                    "with one object: timestamp, visible_subjects, species (distinct), "
                    "count, confidence (0..1), uncertain, and note with concise visual justification."
                )
                response = self._chat_observations(
                    client,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }],
                )
                parsed = self._parse_observations(
                    self._response_text(response),
                    [(candidate.timestamp, path)],
                    "verification",
                )
                if parsed:
                    verified.extend(parsed)
                    diagnostic["status"] = "verified"
                else:
                    diagnostic.update(status="failed", reason="malformed structured response")
            except Exception as exc:
                diagnostic.update(
                    status="failed",
                    stage="request_or_parse",
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
            self.verification_diagnostics.append(diagnostic)
        return verified

    @staticmethod
    def _coverage_diagnostics(
        observations: list[FrameObservation], requested_frames: int
    ) -> dict:
        timestamps = [item.timestamp for item in observations]
        return {
            "requested_frame_count": requested_frames,
            "coarse_observation_count": len(observations),
            "minimum_timestamp_observed": min(timestamps) if timestamps else None,
            "maximum_timestamp_observed": max(timestamps) if timestamps else None,
            "zero_count_observations": sum(item.count == 0 for item in observations),
            "positive_count_observations": sum(item.count > 0 for item in observations),
            "uncertain_observations": sum(bool(item.uncertain) for item in observations),
        }

    @staticmethod
    def _select_candidates(
        observations: list[FrameObservation], duration: float | None = None
    ) -> list[FrameObservation]:
        """Nominate one deterministic candidate from each full-video region.

        Zero-count observations remain eligible: refinement must not depend on the
        coarse model already recognizing the maximum. The strongest global frame is
        preserved because it wins its own region, while the other regions retain
        independent slots.
        """
        if not observations:
            return []
        duration = max(
            float(duration or 0),
            max(item.timestamp for item in observations),
            0.001,
        )
        region_width = duration / REFINE_CANDIDATES
        selected: list[FrameObservation] = []
        global_ranked = sorted(
            observations,
            key=lambda item: (
                -item.count,
                -item.confidence,
                -len(item.uncertain),
                item.timestamp,
            ),
        )
        # The strongest global candidate is authoritative for score while its
        # region's slot doubles as the global slot, leaving coverage elsewhere.
        selected.append(global_ranked[0])
        global_region = min(
            int(global_ranked[0].timestamp / region_width),
            REFINE_CANDIDATES - 1,
        )

        for region in range(REFINE_CANDIDATES):
            if region == global_region:
                continue
            if len(selected) >= REFINE_CANDIDATES:
                break
            lower = region * region_width
            upper = duration + 0.001 if region == REFINE_CANDIDATES - 1 else lower + region_width
            center = (lower + min(upper, duration)) / 2
            regional = [
                item for item in observations
                if lower <= item.timestamp < upper
            ]
            ranked = sorted(
                regional,
                key=lambda item: (
                    -item.count,
                    -item.confidence,
                    -len(item.uncertain),
                    abs(item.timestamp - center),
                    item.timestamp,
                ),
            )
            for item in ranked:
                if all(
                    abs(item.timestamp - other.timestamp)
                    >= CANDIDATE_MIN_SEPARATION_SECONDS
                    for other in selected
                ):
                    selected.append(item)
                    break

        # Sparse/missing regions can leave slots open. Fill deterministically while
        # retaining separation, but never let one region consume another's slot.
        if len(selected) < REFINE_CANDIDATES:
            ranked_all = sorted(
                observations,
                key=lambda item: (
                    -item.count,
                    -item.confidence,
                    -len(item.uncertain),
                    item.timestamp,
                ),
            )
            for item in ranked_all:
                if len(selected) >= REFINE_CANDIDATES:
                    break
                if all(
                    abs(item.timestamp - other.timestamp)
                    >= CANDIDATE_MIN_SEPARATION_SECONDS
                    for other in selected
                ):
                    selected.append(item)
        return selected

    @staticmethod
    def _merge_observations(
        observations: list[FrameObservation], timestamp_tolerance: float = 0.125
    ) -> list[FrameObservation]:
        """Deduplicate overlapping evidence without letting refinement erase it.

        Count and confidence are evidence strength. Refinement wins a true tie, but
        a lower-count or lower-confidence refined frame cannot replace stronger
        coarse evidence from the same moment.
        """
        merged: list[FrameObservation] = []

        def strength(item: FrameObservation):
            return (
                item.count,
                item.confidence,
                len(item.species),
                item.pass_name == "refined",
            )

        for item in sorted(observations, key=lambda value: value.timestamp):
            overlap = next(
                (
                    index
                    for index, existing in enumerate(merged)
                    if abs(existing.timestamp - item.timestamp) <= timestamp_tolerance
                ),
                None,
            )
            if overlap is None:
                merged.append(item)
            elif strength(item) > strength(merged[overlap]):
                merged[overlap] = item
        return sorted(merged, key=lambda value: value.timestamp)

    @staticmethod
    def _reconcile_observations(
        observations: list[FrameObservation], timestamp_tolerance: float = 0.125
    ) -> list[FrameObservation]:
        """Resolve same-frame evidence by consensus, then confidence.

        Two independent observations agreeing on a count form stronger evidence than
        a lone claim. Without consensus, confidence decides; source type only breaks
        exact ties. Observations from different timestamps are never combined.
        """
        groups: list[list[FrameObservation]] = []
        representative_timestamp: float | None = None
        for item in sorted(observations, key=lambda value: value.timestamp):
            if (
                not groups
                or representative_timestamp is None
                or abs(item.timestamp - representative_timestamp) > timestamp_tolerance
            ):
                groups.append([item])
                representative_timestamp = item.timestamp
            else:
                groups[-1].append(item)

        reconciled = []
        source_rank = {"coarse": 0, "refined": 1, "verification": 2}
        for group in groups:
            frequencies = {
                count: sum(item.count == count for item in group)
                for count in {item.count for item in group}
            }
            consensus_count, votes = max(
                frequencies.items(), key=lambda pair: (pair[1], pair[0])
            )
            eligible = (
                [item for item in group if item.count == consensus_count]
                if votes >= 2
                else group
            )
            reconciled.append(
                max(
                    eligible,
                    key=lambda item: (
                        item.confidence,
                        len(item.species),
                        item.count,
                        source_rank.get(item.pass_name, -1),
                    ),
                )
            )
        return sorted(reconciled, key=lambda value: value.timestamp)

    def _synthesize(self, observations: list[FrameObservation], plan: VideoCountingPlan, duration: float, coarse_count: int) -> str:
        if not observations:
            return "No parseable timestamped visual evidence was produced; no numeric maximum is verified."
        coarse = [item for item in observations if item.pass_name == "coarse"]
        refined = [item for item in observations if item.pass_name == "refined"]
        verification = [item for item in observations if item.pass_name == "verification"]
        reconciled = self._reconcile_observations(observations)
        maximum = max(item.count for item in reconciled)
        strongest = [item for item in reconciled if item.count == maximum]
        public = lambda item: {
            key: value
            for key, value in asdict(item).items()
            if key != "frame_path"
        }
        candidate_timestamps = [
            item.timestamp for item in self._select_candidates(coarse, duration)
        ]
        evidence = {
            "question_semantics": plan.entity,
            "duration_seconds": duration,
            "coverage": {
                "coarse_frames": coarse_count,
                "refined_frames": len(refined),
                "refinement_attempts": len(self.refinement_diagnostics),
                "refinement_failures": sum(
                    item.get("status") == "failed"
                    for item in self.refinement_diagnostics
                ),
                "verification_attempts": len(self.verification_diagnostics),
                "independently_verified_frames": len(verification),
                "verification_failures": sum(
                    item.get("status") == "failed"
                    for item in self.verification_diagnostics
                ),
                "exhaustive": False,
                "kind": "sampled",
            },
            "coarse_diagnostics": self.coarse_diagnostics or self._coverage_diagnostics(
                coarse, requested_frames=coarse_count
            ),
            "coarse_batch_diagnostics": self.observation_batch_diagnostics.get(
                "coarse", []
            ),
            "candidate_timestamps": candidate_timestamps,
            "refinement_diagnostics": self.refinement_diagnostics,
            "refined_observations": [public(item) for item in refined],
            "verification_observations": [public(item) for item in verification],
            "verification_diagnostics": self.verification_diagnostics,
            "programmatic_maximum_observed": maximum,
            "same_frame_candidates": [public(item) for item in strongest],
        }
        client = self.synthesis_client_factory()
        try:
            response = client.chat(messages=[{"role": "user", "content": "Explain and normalize the already-computed programmatic result using only this structured same-frame evidence. The programmatic maximum is authoritative. Do not recompute it. Do not add species from different timestamps or invent visual facts. Report its supporting timestamp/species and state that coverage is sampled.\n" + json.dumps(evidence)}], temperature=0)
            summary = self._response_text(response)
        except Exception:
            summary = "Secondary synthesis unavailable; preserving programmatic primary evidence."
        return "Structured video evidence:\n" + json.dumps(evidence, indent=2) + "\nSynthesis:\n" + summary

    @staticmethod
    def _response_text(response) -> str:
        blocks = response.message.content or []
        texts = [
            getattr(block, "text", "")
            for block in blocks
            if getattr(block, "text", None)
        ]
        return "\n".join(texts)
