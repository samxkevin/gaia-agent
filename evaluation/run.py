from pathlib import Path

from agent import solve
from config import AGENT_CODE_URL, HF_USERNAME
from evaluation.client import download_file, fetch_questions, submit_answers


def run():
    questions = fetch_questions()
    answers = []
    attachment_dir = ".gaia_attachments"
    Path(attachment_dir).mkdir(exist_ok=True)

    for i, task in enumerate(questions, start=1):
        task_id = task["task_id"]
        instruction = task["question"]
        file_name = task.get("file_name", "")

        print(f"[{i}/{len(questions)}] {task_id}")

        try:
            attachment_path = None
            if file_name:
                attachment_path = download_file(
                    task_id,
                    file_name,
                    attachment_dir,
                )

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

    if len(answers) != len(questions):
        raise RuntimeError(
            f"Only produced {len(answers)} of {len(questions)} answers. "
            "Refusing to submit a partial run."
        )

    if not HF_USERNAME or not AGENT_CODE_URL:
        raise RuntimeError(
            "HF_USERNAME and AGENT_CODE_URL are required for submission."
        )

    result = submit_answers(
        username=HF_USERNAME,
        agent_code=AGENT_CODE_URL,
        answers=answers,
    )

    print(result)


if __name__ == "__main__":
    run()
