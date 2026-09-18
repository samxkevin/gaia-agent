from pathlib import Path

from agent import solve
from config import AGENT_CODE_URL, HF_USERNAME
from evaluation.client import download_file, fetch_questions, submit_answers


def run():
    questions = fetch_questions()
    answers = []
    Path(".gaia_attachments").mkdir(exist_ok=True)

    for i, task in enumerate(questions, start=1):
        task_id = task["id"]
        instruction = task["instruction"]

        print(f"[{i}/{len(questions)}] {task_id}")

        try:
            attachment_path = None
            if task.get("file"):
                attachment_path = f".gaia_attachments/{task_id}"
                download_file(task_id, attachment_path)
            answer = solve(instruction, attachment_path=attachment_path)
            print(f"Answer: {answer}")
            answers.append(
                {
                    "task_id": task_id,
                    "submitted_answer": answer,
                }
            )
        except Exception as exc:
            print(f"Error: {exc}")

    if not answers:
        raise RuntimeError("No answers were produced.")

    result = submit_answers(
        username=HF_USERNAME,
        agent_code=AGENT_CODE_URL,
        answers=answers,
    )

    print(result)


if __name__ == "__main__":
    run()
