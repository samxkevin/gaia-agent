import importlib.util
import os

from evaluation.client import fetch_questions
from models import get_chat_routes


def main():
    missing = []

    if not (os.getenv("COHERE_PRIMARY_API_KEY") or os.getenv("COHERE_API_KEY")):
        missing.append("COHERE_PRIMARY_API_KEY")

    for module in ("smolagents", "openai", "cohere", "requests", "ddgs", "wikipediaapi"):
        if importlib.util.find_spec(module) is None:
            missing.append(f"python package: {module}")

    questions = fetch_questions()
    if not questions:
        raise RuntimeError("GAIA API returned no questions.")

    first = questions[0]
    required = {"task_id", "question", "Level", "file_name"}
    missing_fields = sorted(required.difference(first))

    if missing_fields:
        raise RuntimeError(f"GAIA schema is missing fields: {missing_fields}")

    routes = get_chat_routes()

    from smolagents import DuckDuckGoSearchTool, WikipediaSearchTool

    DuckDuckGoSearchTool(max_results=1, rate_limit=1.0)
    WikipediaSearchTool(
        user_agent="gaia-agent/1.0",
        language="en",
        content_type="text",
        extract_format="WIKI",
    )
    print("Tool dependencies: OK")

    print(f"GAIA questions available: {len(questions)}")
    print(f"First task: {first['task_id']}")
    print(f"Level: {first['Level']}")
    print(f"Attachment: {first['file_name'] or '<none>'}")
    print(f"Configured chat routes: {len(routes)}")

    for route in routes:
        print(f"  {route.key_slot} -> {route.model_id}")

    if missing:
        raise RuntimeError("Preflight failed: " + ", ".join(missing))

    print("Preflight: PASS")


if __name__ == "__main__":
    main()
