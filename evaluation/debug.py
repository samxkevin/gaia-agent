import argparse
import json
import time
from pathlib import Path

from agent import solve
from evaluation.client import download_file, fetch_question


def main():
    parser = argparse.ArgumentParser(
        description="Run one GAIA question without submitting anything."
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--task-id")
    parser.add_argument("--expected")
    parser.add_argument("--keep-attachment", action="store_true")
    args = parser.parse_args()

    task = fetch_question(index=args.index, task_id=args.task_id)
    task_id = task["task_id"]
    question = task["question"]
    file_name = task.get("file_name", "")

    print("=" * 72)
    print("GAIA SINGLE QUESTION DEBUG")
    print("=" * 72)
    print(f"task_id: {task_id}")
    print(f"level: {task.get('Level', '')}")
    print(f"file_name: {file_name or '<none>'}")
    print(f"question: {question}")
    print("=" * 72)

    attachment_path = None
    if file_name:
        attachment_path = download_file(
            task_id,
            file_name,
            ".gaia_debug_attachments",
        )
        print(f"attachment: {attachment_path}")

    started = time.perf_counter()
    answer, result = solve(
        question,
        attachment_path=attachment_path,
        debug=True,
    )
    elapsed = time.perf_counter() - started

    exact_match = None
    if args.expected is not None:
        exact_match = answer == args.expected
        print(f"expected: {args.expected}")
        print(f"exact_match: {exact_match}")

    print()
    print("=" * 72)
    print("AGENT RESULT")
    print("=" * 72)
    print(f"answer: {answer}")
    print(f"elapsed_seconds: {elapsed:.2f}")

    if hasattr(result, "steps"):
        print(f"steps: {len(result.steps)}")
    if hasattr(result, "token_usage") and result.token_usage:
        print(f"token_usage: {result.token_usage}")

    report = {
        "task_id": task_id,
        "question": question,
        "file_name": file_name,
        "attachment_path": attachment_path,
        "answer": answer,
        "expected": args.expected,
        "exact_match": exact_match,
        "elapsed_seconds": round(elapsed, 3),
        "steps": len(result.steps) if hasattr(result, "steps") else None,
        "token_usage": str(getattr(result, "token_usage", None)),
    }

    report_dir = Path(".gaia_debug_runs")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"{task_id}.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"report: {report_path}")

    if attachment_path and not args.keep_attachment:
        Path(attachment_path).unlink(missing_ok=True)
        print("attachment_cleanup: removed")

    print("submission: SKIPPED")


if __name__ == "__main__":
    main()
