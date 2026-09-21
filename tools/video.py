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
REFINE_RADIUS_SECONDS = 4.0
REFINE_FPS = 4
CANDIDATE_MIN_SEPARATION_SECONDS = 5.0
VERIFICATION_LIMIT = 4
VISION_BATCH_SIZE = 4
MAX_FRAME_WIDTH = 1280


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
                if plan.is_numeric_maximum and observations:
                    candidates = self._select_candidates(observations)
                    if candidates:
                        try:
                            refined_frames = self._extract_candidate_frames(
                                video, root / "refined", duration, candidates
                            )
                            observations.extend(
                                self._observe_frames(refined_frames, plan, "refined")
                            )
                        except Exception:
                            # Refinement is optional and cannot destroy coarse evidence.
                            pass
                    try:
                        verification_candidates = self._select_candidates(
                            self._merge_observations(observations)
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
        frames = []
        seen = set()
        for candidate_index, candidate in enumerate(candidates[:REFINE_CANDIDATES]):
            start = max(0.0, candidate.timestamp - REFINE_RADIUS_SECONDS)
            count = int(REFINE_RADIUS_SECONDS * 2 * REFINE_FPS) + 1
            batch = self._extract_frames(video, output_dir / str(candidate_index), duration, count, interval=1 / REFINE_FPS, offset=start)
            for timestamp, path in batch:
                key = round(timestamp, 3)
                if key not in seen:
                    seen.add(key)
                    frames.append((timestamp, path))
        return sorted(frames)

    def _observe_frames(self, frames, plan: VideoCountingPlan, pass_name: str) -> list[FrameObservation]:
        client = self.visual_client_factory()
        observations = []
        for start in range(0, len(frames), VISION_BATCH_SIZE):
            batch = frames[start:start + VISION_BATCH_SIZE]
            content = [{"type": "text", "text": f"{plan.instruction}\nReturn ONLY a JSON array, one object per image, in the same order. Schema: timestamp (number), visible_subjects (string array), species (distinct string array), count (integer), confidence (0..1), uncertain (string array), note (string). Inspect each image independently. Never merge evidence across images. Use timestamp labels exactly."}]
            for timestamp, path in batch:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                content += [{"type": "text", "text": f"timestamp={timestamp:.3f}"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}}]
            response = client.chat(messages=[{"role": "user", "content": content}], temperature=0)
            observations.extend(self._parse_observations(self._response_text(response), batch, pass_name))
        return observations

    @staticmethod
    def _parse_observations(text: str, batch, pass_name: str) -> list[FrameObservation]:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return []
        try:
            values = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        result = []
        for index, value in enumerate(values[:len(batch)]):
            if not isinstance(value, dict):
                continue
            # Frame identity comes from FFmpeg extraction, never model-generated JSON.
            timestamp = float(batch[index][0])
            species = tuple(dict.fromkeys(str(x).strip() for x in value.get("species", []) if str(x).strip()))
            subjects = tuple(str(x).strip() for x in value.get("visible_subjects", []) if str(x).strip())
            count = len(species) if species else max(0, int(value.get("count", 0)))
            result.append(
                FrameObservation(
                    timestamp,
                    subjects,
                    species,
                    count,
                    max(0.0, min(float(value.get("confidence", 0)), 1.0)),
                    tuple(str(x) for x in value.get("uncertain", [])),
                    str(value.get("note", "")),
                    pass_name,
                    str(batch[index][1]),
                )
            )
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
                "reason": f"verifier initialization failed: {type(exc).__name__}",
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
                    "frames. Return ONLY a JSON array with one object: timestamp, "
                    "visible_subjects, species (distinct), count, confidence (0..1), "
                    "uncertain, and note with concise visual justification."
                )
                response = client.chat(
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
                    temperature=0,
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
                    reason=f"verification request failed: {type(exc).__name__}",
                )
            self.verification_diagnostics.append(diagnostic)
        return verified

    @staticmethod
    def _select_candidates(observations: list[FrameObservation]) -> list[FrameObservation]:
        ranked = sorted(
            (item for item in observations if item.count > 0),
            key=lambda item: (item.count, item.confidence),
            reverse=True,
        )
        selected = []
        for item in ranked:
            if all(
                abs(item.timestamp - other.timestamp)
                >= CANDIDATE_MIN_SEPARATION_SECONDS
                for other in selected
            ):
                selected.append(item)
            if len(selected) == REFINE_CANDIDATES:
                break
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
        candidate_timestamps = [item.timestamp for item in self._select_candidates(coarse)]
        evidence = {
            "question_semantics": plan.entity,
            "duration_seconds": duration,
            "coverage": {
                "coarse_frames": coarse_count,
                "refined_frames": len(refined),
                "verification_attempts": len(self.verification_diagnostics),
                "independently_verified_frames": len(verification),
                "verification_failures": sum(
                    item.get("status") == "failed"
                    for item in self.verification_diagnostics
                ),
                "exhaustive": False,
                "kind": "sampled",
            },
            "candidate_timestamps": candidate_timestamps,
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
        return "\n".join(getattr(block, "text", "") for block in blocks if getattr(block, "text", None)) or str(blocks)
