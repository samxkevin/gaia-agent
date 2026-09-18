import base64
import mimetypes
import re
from pathlib import Path

from smolagents import Tool

from models import CohereFailoverClient


class AnalyzeImageTool(Tool):
    name = "analyze_image"
    description = (
        "Analyze a local image with Cohere Command A+. Use this for visual "
        "reasoning, OCR like reading, charts, diagrams, chess positions, "
        "objects, spatial relationships, or other image based evidence."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Absolute or repository local path to the image.",
        },
        "question": {
            "type": "string",
            "description": "Specific visual question to answer about the image.",
        },
    }
    output_type = "string"

    def forward(self, path: str, question: str) -> str:
        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            return f"Image not found: {file_path}"

        mime_type, _ = mimetypes.guess_type(file_path.name)
        if mime_type not in {
            "image/png", "image/jpeg", "image/webp", "image/gif",
        }:
            return f"Unsupported image type: {mime_type}"

        data = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        client = CohereFailoverClient()
        response = client.chat(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{data}",
                            },
                        },
                    ],
                }
            ],
            temperature=0,
        )

        for block in response.message.content or []:
            text = getattr(block, "text", None)
            if text:
                return text

        return str(response.message.content)


class TranscribeAudioTool(Tool):
    name = "transcribe_audio"
    description = (
        "Transcribe a local MP3, WAV, OGG, FLAC, MPEG, or MPGA attachment "
        "with Cohere Transcribe using primary and fallback keys."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Absolute or repository local path to the audio file.",
        },
    }
    output_type = "string"

    def forward(self, path: str) -> str:
        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            return f"Audio not found: {file_path}"

        if file_path.suffix.lower() not in {
            ".mp3", ".wav", ".ogg", ".flac", ".mpeg", ".mpga",
        }:
            return f"Unsupported audio type: {file_path.suffix.lower()}"

        client = CohereFailoverClient()
        response = client.transcribe(
            file_path,
            language="en",
        )

        text = getattr(response, "text", None)
        if text:
            return text

        return str(response)


class YouTubeTranscriptTool(Tool):
    name = "get_youtube_transcript"
    description = (
        "Retrieve captions or an automatically generated transcript for a "
        "YouTube video. Use this for questions asking what was said in a video."
    )
    inputs = {
        "video_id": {
            "type": "string",
            "description": "The YouTube video ID, not the full URL.",
        },
    }
    output_type = "string"

    def forward(self, video_id: str) -> str:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            transcript = YouTubeTranscriptApi().fetch(
                video_id,
                languages=["en"],
            )
            return "\n".join(snippet.text for snippet in transcript)
        except Exception as exc:
            return f"Could not fetch YouTube transcript: {exc}"


class ExtractYouTubeIdTool(Tool):
    name = "extract_youtube_id"
    description = "Extract the video ID from a YouTube URL."
    inputs = {
        "url": {
            "type": "string",
            "description": "A standard YouTube watch URL or short URL.",
        },
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        match = re.search(
            r"(?:v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{6,})",
            url,
        )
        if not match:
            return "Could not extract a YouTube video ID."
        return match.group(1)
