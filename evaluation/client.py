from pathlib import Path

import requests

from config import GAIA_API_URL


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "gaia-agent/1.0"})


def fetch_questions():
    response = SESSION.get(f"{GAIA_API_URL}/questions", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_question(index: int | None = None, task_id: str | None = None):
    questions = fetch_questions()

    if task_id:
        for question in questions:
            if question.get("task_id") == task_id:
                return question
        raise KeyError(f"Task not found: {task_id}")

    if index is None:
        index = 0

    try:
        return questions[index]
    except IndexError as exc:
        raise IndexError(
            f"Question index {index} is out of range for {len(questions)} questions."
        ) from exc


def download_file(task_id: str, file_name: str, output_dir: str):
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / file_name

    response = SESSION.get(
        f"{GAIA_API_URL}/files/{task_id}",
        timeout=60,
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return str(output_path)


def submit_answers(username: str, agent_code: str, answers: list[dict]):
    payload = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers,
    }

    response = SESSION.post(
        f"{GAIA_API_URL}/submit",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()
