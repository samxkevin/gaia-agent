import shutil
from pathlib import Path

import requests

from config import GAIA_API_URL, HF_TOKEN


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


def _official_gaia_file(task_id: str) -> Path:
    """Resolve an attachment through official GAIA metadata, never by guessing."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Attachment endpoint returned 404 and huggingface_hub is unavailable."
        ) from exc

    options = {"repo_id": "gaia-benchmark/GAIA", "repo_type": "dataset"}
    if HF_TOKEN:
        options["token"] = HF_TOKEN
    try:
        from pyarrow.parquet import read_table

        metadata = Path(
            hf_hub_download(filename="2023/validation/metadata.parquet", **options)
        )
        table = read_table(metadata, columns=["task_id", "file_path"])
        record = next(
            (
                row
                for row in table.to_pylist()
                if str(row.get("task_id")) == task_id
            ),
            None,
        )
        if record is None:
            raise RuntimeError(f"Task {task_id} is absent from official GAIA metadata.")
        dataset_path = record.get("file_path")
        if not dataset_path:
            raise RuntimeError(f"Task {task_id} has no attachment path in official GAIA metadata.")
        if Path(dataset_path).is_absolute() or ".." in Path(dataset_path).parts:
            raise RuntimeError(f"Task {task_id} has an unsafe attachment path in GAIA metadata.")
        return Path(hf_hub_download(filename=str(dataset_path), **options))
    except Exception as exc:
        raise RuntimeError(
            "Scoring attachment endpoint returned 404 and the official gated GAIA "
            "dataset fallback was unavailable. Request GAIA dataset access and set HF_TOKEN. "
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc


def download_file(task_id: str, file_name: str, output_dir: str):
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / Path(file_name).name
    if output_path.is_file() and output_path.stat().st_size:
        return str(output_path)

    response = SESSION.get(f"{GAIA_API_URL}/files/{task_id}", timeout=60)
    if response.status_code == 404:
        source = _official_gaia_file(task_id)
        shutil.copyfile(source, output_path)
    else:
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
