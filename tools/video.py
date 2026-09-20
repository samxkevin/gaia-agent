from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from smolagents import Tool

from config import COHERE_VIDEO_MODEL
from models import CohereFailoverClient


def find_javascript_runtime(which=shutil.which) -> tuple[str, str] | None:
    """Find any JavaScript runtime supported by current yt-dlp EJS."""
    for runtime, executables in (
        ("deno", ("deno",)),
        ("node", ("node", "nodejs")),
        ("quickjs", ("qjs", "quickjs")),
        ("bun", ("bun",)),
    ):
        for executable in executables:
            path = which(executable)
            if path:
                return runtime, path
    return None


def normalize_youtube_url(value: str) -> str:
    """Extract a usable URL from plain text or a Markdown link."""
    markdown = re.search(r"\[[^]]*\]\((https?://[^)]+)\)", value)
    if markdown:
        value = markdown.group(1)
    match = re.search(
        r"https?://(?:www\.)?(?:youtube\.com/[^\s)>]+|youtu\.be/[^\s)>]+)",
        value,
    )
    return match.group(0).rstrip(".,;") if match else value.strip()


class AnalyzeYouTubeVideoTool(Tool):
    name = "analyze_youtube_video"
    description = (
        "Inspect actual YouTube video frames to answer a specific visual question. "
        "Use for visible objects, simultaneous counts, actions, scenes, or other "
        "on-camera evidence; transcripts and web descriptions are not substitutes."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "YouTube URL, including Markdown links.",
        },
        "question": {
            "type": "string",
            "description": "Specific question to answer from visible frames.",
        },
        "max_frames": {
            "type": "integer",
            "description": "Maximum frames to inspect. Temporal maximum, simultaneous, or counting questions are automatically sampled densely.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, visual_client_factory: Callable | None = None):
        super().__init__()
        self.visual_client_factory = visual_client_factory or (
            lambda: CohereFailoverClient(model_id=COHERE_VIDEO_MODEL)
        )

    @staticmethod
    def _needs_dense_temporal_sampling(question: str) -> bool:
        text = question.lower()
        return any(
            phrase in text
            for phrase in (
                "simultaneously",
                "at the same time",
                "highest number",
                "maximum number",
                "peak count",
            )
        )

    def forward(
        self,
        url: str,
        question: str,
        max_frames: int | None = 60,
    ) -> str:
        url = normalize_youtube_url(url)
        requested_frames = max(12, int(max_frames or 60))
        if self._needs_dense_temporal_sampling(question):
            max_frames = min(max(requested_frames, 180), 240)
        else:
            max_frames = min(requested_frames, 72)
        try:
            with tempfile.TemporaryDirectory(prefix="gaia-video-") as directory:
                workdir = Path(directory)
                video = self._download(url, workdir)
                duration = self._duration(video)
                frames = self._extract_frames(
                    video,
                    workdir / "frames",
                    duration,
                    max_frames,
                )
                if not frames:
                    return "Video frame analysis failed: no frames were extracted."
                return self._analyze_frames(frames, question, duration)
        except Exception as exc:
            return f"Video frame analysis failed: {type(exc).__name__}: {exc}"

    def _download(self, url: str, workdir: Path) -> Path:
        from imageio_ffmpeg import get_ffmpeg_exe

        runtime = find_javascript_runtime()
        if runtime is None:
            raise RuntimeError(
                "No yt-dlp JavaScript runtime found (deno, node, quickjs, or bun)."
            )
        runtime_name, runtime_path = runtime
        template = str(workdir / "video.%(ext)s")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--no-playlist",
                "--no-warnings",
                "--ffmpeg-location",
                get_ffmpeg_exe(),
                "--js-runtimes",
                f"{runtime_name}:{runtime_path}",
                "-f",
                "bestvideo[height<=480]/best[height<=480]/worst",
                "--merge-output-format",
                "mp4",
                "-o",
                template,
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        videos = [p for p in workdir.glob("video.*") if p.is_file()]
        if not videos:
            raise RuntimeError("yt-dlp produced no video file")
        return videos[0]

    def _duration(self, video: Path) -> float:
        from imageio_ffmpeg import count_frames_and_secs

        _, duration = count_frames_and_secs(str(video))
        return float(duration)

    def _extract_frames(
        self,
        video: Path,
        output_dir: Path,
        duration: float,
        max_frames: int,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        interval = max(duration / max_frames, 0.25)
        from imageio_ffmpeg import get_ffmpeg_exe

        subprocess.run(
            [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-vf",
                f"fps=1/{interval},scale='min(960,iw)':-2",
                "-frames:v",
                str(max_frames),
                "-q:v",
                "3",
                str(output_dir / "%04d.jpg"),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
        paths = sorted(output_dir.glob("*.jpg"))
        return [
            (min(index * interval, duration), path)
            for index, path in enumerate(paths)
        ]

    def _analyze_frames(self, frames, question: str, duration: float) -> str:
        client = self.visual_client_factory()
        observations = []
        batch_size = 8 if self._needs_dense_temporal_sampling(question) else 6
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]
            content = [
                {
                    "type": "text",
                    "text": (
                        f"Visual video question: {question}\n"
                        "Treat every attached image as an independent video frame. "
                        "Do not merge evidence between frames. Inspect small or distant birds carefully. "
                        "For every frame, report its supplied timestamp, each visibly identifiable bird "
                        "species, and the count of distinct species visible in that exact frame. "
                        "Keep the count tied to that single frame. Return one concise line per frame."
                    ),
                }
            ]
            for timestamp, path in batch:
                mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": f"Frame timestamp: {timestamp:.2f} seconds",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{encoded}",
                                "detail": "high"
                            },
                        },
                    ]
                )
            response = client.chat(
                messages=[{"role": "user", "content": content}],
                temperature=0,
            )
            observations.append(self._response_text(response))

        synthesis = CohereFailoverClient().chat(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Video duration: {duration:.2f} seconds.\n"
                        "Aggregate the timestamped visual observations below. Answer from same-frame "
                        "visibility only; never merge objects seen at different timestamps. State the "
                        "answer concisely and cite the strongest timestamp(s).\n\n"
                        + "\n\n".join(observations)
                    ),
                }
            ],
            temperature=0,
        )
        return self._response_text(synthesis)

    @staticmethod
    def _response_text(response) -> str:
        blocks = response.message.content or []
        texts = [getattr(block, "text", "") for block in blocks]
        return "\n".join(text for text in texts if text) or str(blocks)
