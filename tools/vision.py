import base64
import mimetypes
from pathlib import Path

from smolagents import Tool

from config import COHERE_API_KEY, COHERE_MODEL


class AnalyzeImageTool(Tool):
    name = "analyze_image"
    description = (
        "Analyze a local image with Cohere Command A+. Use this when a GAIA "
        "task requires visual understanding, OCR-like reading, object "
        "identification, spatial relationships, charts, or image reasoning."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Absolute or repository-local path to the image.",
        },
        "question": {
            "type": "string",
            "description": "Specific visual question to answer about the image.",
        },
    }
    output_type = "string"

    def forward(self, path: str, question: str) -> str:
        if not COHERE_API_KEY:
            return "COHERE_API_KEY is not configured."

        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            return f"Image not found: {file_path}"

        mime_type, _ = mimetypes.guess_type(file_path.name)
        if mime_type not in {
            "image/png", "image/jpeg", "image/webp", "image/gif"
        }:
            return f"Unsupported image type: {mime_type}"

        import cohere

        data = base64.b64encode(file_path.read_bytes()).decode("utf-8")

        client = cohere.ClientV2(COHERE_API_KEY)
        response = client.chat(
            model=COHERE_MODEL,
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

        return response.message.content[0].text
