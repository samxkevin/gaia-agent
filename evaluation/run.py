from agent import solve
from config import AGENT_CODE_URL, HF_USERNAME
from evaluation.client import fetch_questions, submit_answers


def run():
    questions = fetch_questions()
    answers = []

    for i, task in enumerate(questions, start=1):
        task_id = task["id"]
        instruction = task["instruction"]

        print(f"[{i}/{len(questions)}] {task_id}")

        try:
            answer = solve(instruction)
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
