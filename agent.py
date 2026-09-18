from pathlib import Path

from smolagents import CodeAgent, InferenceClientModel


ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (ROOT / "prompts" / "gaia_system.txt").read_text(encoding="utf-8")


def create_agent():
    """
    Create the GAIA solving agent.

    Model selection is kept isolated here so it can be swapped after
    current model/API benchmarking without changing the evaluation layer.
    """
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    )

    return CodeAgent(
        tools=[],
        model=model,
        max_steps=15,
        verbosity_level=2,
    )


def solve(question: str) -> str:
    agent = create_agent()
    result = agent.run(f"{SYSTEM_PROMPT}\n\nTask:\n{question}")
    return str(result).strip()
