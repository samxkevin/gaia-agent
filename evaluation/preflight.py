import importlib.util
import os

from evaluation.client import fetch_questions
from config import COHERE_FALLBACK_MODEL, COHERE_BASE_URL, MODEL_TIMEOUT_SECONDS
from models import get_chat_routes


def main():
    missing = []

    if not (os.getenv("COHERE_PRIMARY_API_KEY") or os.getenv("COHERE_API_KEY")):
        missing.append("COHERE_PRIMARY_API_KEY")
    if not os.getenv("COHERE_FALLBACK_API_KEY"):
        missing.append("COHERE_FALLBACK_API_KEY")
    if COHERE_FALLBACK_MODEL == "command-a-03-2025":
        missing.append("COHERE_FALLBACK_MODEL is still the legacy command-a-03-2025")

    for module in (
        "smolagents",
        "openai",
        "cohere",
        "requests",
        "ddgs",
        "wikipediaapi",
        "markdownify",
    ):
        if importlib.util.find_spec(module) is None:
            missing.append(f"python package: {module}")

    questions = fetch_questions()
    if not questions:
        raise RuntimeError("GAIA API returned no questions.")

    first = questions[0]
    required = {"task_id", "question", "Level", "file_name"}
    missing_fields = sorted(required.difference(first))

    if missing_fields:
        raise RuntimeError(
            f"GAIA schema is missing fields: {missing_fields}"
        )

    routes = get_chat_routes()

    from smolagents import DuckDuckGoSearchTool, VisitWebpageTool, WikipediaSearchTool

    DuckDuckGoSearchTool(max_results=1, rate_limit=1.0)
    WikipediaSearchTool(
        user_agent="gaia-agent/1.0",
        language="en",
        content_type="text",
        extract_format="WIKI",
    )
    VisitWebpageTool(max_output_length=1000)
    print("Tool dependencies: OK")

    print(f"GAIA questions available: {len(questions)}")
    print(f"First task: {first['task_id']}")
    print(f"Level: {first['Level']}")
    print(f"Attachment: {first['file_name'] or '<none>'}")
    print(f"Configured chat routes: {len(routes)}")
    print(f"Fallback model: {COHERE_FALLBACK_MODEL}")

    route_errors = []
    if not missing:
        from openai import OpenAI

        print("Probing all configured model/key routes...")
        for route in routes:
            try:
                client = OpenAI(
                    api_key=route.api_key,
                    base_url=COHERE_BASE_URL,
                    timeout=MODEL_TIMEOUT_SECONDS,
                    max_retries=0,
                )
                response = client.chat.completions.create(
                    model=route.model_id,
                    messages=[{"role": "user", "content": "Reply with OK."}],
                    max_tokens=4,
                    reasoning_effort="none",
                )
                if not response.choices:
                    raise RuntimeError("No completion choices returned.")
                print(f"  OK: {route.key_slot} -> {route.model_id}")
            except Exception as exc:
                message = (
                    f"{route.key_slot} -> {route.model_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                route_errors.append(message)
                print(f"  FAIL: {message}")

    for route in routes:
        print(f"  {route.key_slot} -> {route.model_id}")

    if missing or route_errors:
        problems = missing + [f"route probe failed: {error}" for error in route_errors]
        raise RuntimeError("Preflight failed: " + " | ".join(problems))

    print("Preflight: PASS")


if __name__ == "__main__":
    main()
