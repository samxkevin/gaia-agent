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


def create_model():
    return FailoverModel()


def create_agent():
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
        TranscribeAudioTool(),
        ExtractYouTubeIdTool(),
        YouTubeTranscriptTool(),
    ]

    return ToolCallingAgent(
        tools=tools,
        model=create_model(),
        max_steps=MAX_AGENT_STEPS,
        verbosity_level=2,
        planning_interval=None,
        max_steps=min(MAX_AGENT_STEPS, 10),
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
    agent = create_agent()

    if attachment_path:
        question = (
            f"{question}\n\n"
            f"A local GAIA attachment is available at:\n{attachment_path}\n"
            "Inspect the attachment before deciding it is irrelevant. "
            "Choose the analysis tool from the actual file type."
        )

    result = agent.run(question)
    answer = clean_answer(result.output if hasattr(result, "output") else result)

    if debug:
        return answer, result, agent.model
    return answer
