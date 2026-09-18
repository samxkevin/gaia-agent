from pathlib import Path

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    OpenAIModel,
    PythonInterpreterTool,
    VisitWebpageTool,
)

from config import (
    COHERE_API_KEY,
    COHERE_BASE_URL,
    COHERE_MODEL,
    MAX_AGENT_STEPS,
)
from tools import AnalyzeImageTool, InspectFileTool, ReadFileTool


ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (ROOT / "prompts" / "gaia_system.txt").read_text(
    encoding="utf-8"
)


def create_model():
    if not COHERE_API_KEY:
        raise RuntimeError(
            "COHERE_API_KEY is required. Copy .env.example to .env and configure it."
        )

    return OpenAIModel(
        model_id=COHERE_MODEL,
        api_base=COHERE_BASE_URL,
        api_key=COHERE_API_KEY,
        temperature=0,
        reasoning_effort="high",
        max_tokens=4096,
    )


def create_agent():
    tools = [
        DuckDuckGoSearchTool(max_results=8, rate_limit=1.0),
        VisitWebpageTool(max_output_length=30000),
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
    ]

    return CodeAgent(
        tools=tools,
        model=create_model(),
        max_steps=MAX_AGENT_STEPS,
        verbosity_level=2,
        planning_interval=4,
        instructions=SYSTEM_PROMPT,
    )


def solve(question: str, attachment_path: str | None = None) -> str:
    agent = create_agent()

    if attachment_path:
        question = (
            f"{question}\n\n"
            f"IMPORTANT: The task includes a local attachment at:\n"
            f"{attachment_path}\n"
            "Inspect and use this attachment when relevant."
        )

    result = agent.run(question)
    return str(result).strip()
