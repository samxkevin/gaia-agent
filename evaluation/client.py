import requests

from config import GAIA_API_URL


def fetch_questions():
    response = requests.get(f"{GAIA_API_URL}/questions", timeout=30)
    response.raise_for_status()
    return response.json()


def download_file(task_id: str, output_path: str):
    response = requests.get(
        f"{GAIA_API_URL}/files/{task_id}",
        timeout=60,
    )
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def submit_answers(username: str, agent_code: str, answers: list[dict]):
    payload = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers,
    }

    response = requests.post(
        f"{GAIA_API_URL}/submit",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()
