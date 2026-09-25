import hashlib
import os
from pathlib import Path

from smolagents import (
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    PythonInterpreterTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)

from config import (
    MAX_AGENT_STEPS,
    MAX_WEBPAGE_TEXT,
    PLANNING_INTERVAL,
    WEB_MAX_RESULTS,
    WEB_RATE_LIMIT,
)
from models import FailoverModel
from tools import (
    AnalyzeImageTool,
    AnalyzeYouTubeVideoTool,
    ExtractYouTubeIdTool,
    InspectFileTool,
    WikipediaPageAsOfTool,
    ReadFileTool,
    TranscribeAudioTool,
    YouTubeTranscriptTool,
)

ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (ROOT / "prompts" / "gaia_system.txt").read_text(
    encoding="utf-8"
)

ANSWER_CACHE_VERSION = "1"
ANSWER_CACHE_ENABLED = os.getenv(
    "GAIA_CACHE_ENABLED", "true"
).strip().lower() not in {"0", "false", "no", "off"}
ANSWER_CACHE_DIR = Path(
    os.getenv("GAIA_CACHE_DIR", str(ROOT / ".gaia_cache"))
)


def _attachment_digest(attachment_path: str | None) -> str:
    if not attachment_path:
        return "none"

    path = Path(attachment_path)
    if not path.is_file():
        return "missing"

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _answer_cache_key(question: str, attachment_path: str | None) -> str:
    normalized_question = " ".join(question.split())
    prompt_digest = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    material = "\n".join(
        [
            ANSWER_CACHE_VERSION,
            normalized_question,
            _attachment_digest(attachment_path),
            prompt_digest,
            str(MAX_AGENT_STEPS),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _load_cached_answer(question: str, attachment_path: str | None):
    if not ANSWER_CACHE_ENABLED:
        return None

    path = ANSWER_CACHE_DIR / f"{_answer_cache_key(question, attachment_path)}.txt"
    try:
        answer = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None

    return answer or None


def _is_cacheable_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return False

    lowered = answer.lower()
    error_markers = (
        "agentgenerationerror",
        "all cohere model routes failed",
        "error in generating final llm output",
        "no_valid_response_generated",
        "no_tool_call_or_response_generated",
        "unprocessableentityerror",
        "i was unable to",
        "i am unable to",
        "i'm unable to",
        "i could not",
        "i couldn't",
        "cannot determine",
        "unable to determine",
    )
    return not any(marker in lowered for marker in error_markers)


def _save_cached_answer(
    question: str,
    attachment_path: str | None,
    answer: str,
) -> None:
    if not ANSWER_CACHE_ENABLED or not _is_cacheable_answer(answer):
        return

    try:
        ANSWER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = ANSWER_CACHE_DIR / f"{_answer_cache_key(question, attachment_path)}.txt"
        temp_path = path.with_name(path.name + ".tmp")
        temp_path.write_text(answer.strip(), encoding="utf-8")
        temp_path.replace(path)
    except OSError:
        pass


def create_model():
    return FailoverModel()


def _is_historical_wikipedia_task(question: str) -> bool:
    text = question.lower()
    return (
        "wikipedia" in text
        and (
            "latest 20" in text
            or "version of english wikipedia" in text
            or "version of wikipedia" in text
            or "as of 20" in text
        )
    )


def create_agent(question: str | None = None):
    historical_wikipedia = bool(question and _is_historical_wikipedia_task(question))

    if historical_wikipedia:
        tools = [WikipediaPageAsOfTool()]
        max_steps = min(MAX_AGENT_STEPS, 4)
    else:
        tools = [
            DuckDuckGoSearchTool(
                max_results=WEB_MAX_RESULTS,
                rate_limit=WEB_RATE_LIMIT,
            ),
            WikipediaPageAsOfTool(),
            WikipediaSearchTool(
                user_agent="gaia-agent/1.0",
                language="en",
                content_type="text",
                extract_format="WIKI",
            ),
            VisitWebpageTool(max_output_length=MAX_WEBPAGE_TEXT),
            PythonInterpreterTool(
                authorized_imports=[
                    "math",
                    "statistics",
                    "datetime",
                    "json",
                    "re",
                    "csv",
                    "collections",
                    "itertools",
                    "pathlib",
                ],
                timeout_seconds=30,
            ),
            ReadFileTool(),
            InspectFileTool(),
            AnalyzeImageTool(),
            AnalyzeYouTubeVideoTool(original_question=question),
            TranscribeAudioTool(),
            ExtractYouTubeIdTool(),
            YouTubeTranscriptTool(),
        ]
        max_steps = min(MAX_AGENT_STEPS, 10)

    return ToolCallingAgent(
        tools=tools,
        model=create_model(),
        max_steps=max_steps,
        verbosity_level=2,
        planning_interval=None,
        instructions=SYSTEM_PROMPT,
        return_full_result=True,
        max_tool_threads=3,
    )


def clean_answer(value) -> str:
    text = str(value).strip()
    if not text:
        return text

    for marker in (
        "FINAL ANSWER:",
        "FINAL ANSWER",
        "Final answer:",
        "Final Answer:",
    ):
        if text.startswith(marker):
            text = text[len(marker):].strip()
            break

    return text


def solve(question: str, attachment_path: str | None = None, debug: bool = False):
    if not debug:
        cached_answer = _load_cached_answer(question, attachment_path)
        if cached_answer is not None:
            return cached_answer

    agent = create_agent(question)

    if attachment_path:
        question = (
            f"{question}\n\n"
            f"A local GAIA attachment is available at:\n{attachment_path}\n"
            "Inspect the attachment before deciding it is irrelevant. "
            "Choose the analysis tool from the actual file type."
        )

    result = agent.run(question)
    answer = clean_answer(result.output if hasattr(result, "output") else result)

    if not debug:
        _save_cached_answer(
            question=question.split("\n\nA local GAIA attachment is available at:", 1)[0],
            attachment_path=attachment_path,
            answer=answer,
        )

    if debug:
        return answer, result, agent.model
    return answer
