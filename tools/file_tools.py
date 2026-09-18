from pathlib import Path

from smolagents import Tool


class ReadFileTool(Tool):
    name = "read_local_file"
    description = (
        "Read a local GAIA attachment. Supports UTF-8 text, JSON, CSV, XML, "
        "HTML, Markdown and PDF files. Returns extracted text."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Absolute or repository-local path to the file.",
        },
    }
    output_type = "string"

    def forward(self, path: str) -> str:
        file_path = Path(path).expanduser().resolve()

        if not file_path.is_file():
            return f"File not found: {file_path}"

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(file_path))
                pages = [
                    page.extract_text() or ""
                    for page in reader.pages
                ]
                return "\n\n".join(pages)[:100000]
            except Exception as exc:
                return f"Could not read PDF: {exc}"

        text_suffixes = {
            ".txt", ".md", ".csv", ".json", ".xml", ".html",
            ".htm", ".py", ".js", ".ts", ".yaml", ".yml",
        }

        if suffix in text_suffixes:
            try:
                return file_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )[:100000]
            except Exception as exc:
                return f"Could not read text file: {exc}"

        return (
            f"Unsupported text format: {suffix}. "
            "Use Python for binary analysis or inspect the file metadata first."
        )


class InspectFileTool(Tool):
    name = "inspect_local_file"
    description = (
        "Inspect a local GAIA attachment and return its type, size, "
        "dimensions for images, and basic metadata."
    )
    inputs = {
        "path": {
            "type": "string",
            "description": "Absolute or repository-local path to the file.",
        },
    }
    output_type = "string"

    def forward(self, path: str) -> str:
        file_path = Path(path).expanduser().resolve()

        if not file_path.is_file():
            return f"File not found: {file_path}"

        result = [
            f"path: {file_path}",
            f"name: {file_path.name}",
            f"extension: {file_path.suffix.lower()}",
            f"size_bytes: {file_path.stat().st_size}",
        ]

        if file_path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
        }:
            try:
                from PIL import Image

                with Image.open(file_path) as image:
                    result.append(f"format: {image.format}")
                    result.append(f"mode: {image.mode}")
                    result.append(f"dimensions: {image.size[0]}x{image.size[1]}")
            except Exception as exc:
                result.append(f"image_error: {exc}")

        return "\n".join(result)
