from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evaluation import client


class Response:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_parquet_metadata_resolves_authoritative_file_path(monkeypatch, tmp_path):
    metadata = tmp_path / "metadata.parquet"
    attachment = tmp_path / "resolved.png"
    attachment.write_bytes(b"attachment")
    pq.write_table(
        pa.table({
            "task_id": ["other", "wanted-task"],
            "file_path": ["2023/validation/other.txt", "2023/validation/authoritative.png"],
        }),
        metadata,
    )
    requested = []

    def download(*, filename, **kwargs):
        requested.append(filename)
        if filename == "2023/validation/metadata.parquet":
            return str(metadata)
        if filename == "2023/validation/authoritative.png":
            return str(attachment)
        pytest.fail(f"unexpected guessed path: {filename}")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert client._official_gaia_file("wanted-task") == attachment
    assert requested == [
        "2023/validation/metadata.parquet",
        "2023/validation/authoritative.png",
    ]


def test_gated_dataset_403_requires_authorized_hf_token(monkeypatch):
    def forbidden(**kwargs):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)
    with pytest.raises(RuntimeError, match="HF_TOKEN.*accepted GAIA dataset access"):
        client._official_gaia_file("task-id")


def test_scoring_endpoint_is_used_when_available(monkeypatch, tmp_path):
    monkeypatch.setattr(client.SESSION, "get", lambda *args, **kwargs: Response(200, b"official bytes"))
    monkeypatch.setattr(client, "_official_gaia_file", lambda task_id: pytest.fail("fallback called"))
    path = Path(client.download_file("task", "display.png", str(tmp_path)))
    assert path.read_bytes() == b"official bytes"


def test_404_uses_resolved_official_dataset_file_and_caches(monkeypatch, tmp_path):
    source = tmp_path / "dataset-source.png"
    source.write_bytes(b"real dataset bytes")
    calls = []
    monkeypatch.setattr(client.SESSION, "get", lambda *args, **kwargs: Response(404))
    monkeypatch.setattr(client, "_official_gaia_file", lambda task_id: calls.append(task_id) or source)
    output_dir = tmp_path / "output"

    first = Path(client.download_file("task-id", "display-name.png", str(output_dir)))
    second = Path(client.download_file("task-id", "display-name.png", str(output_dir)))

    assert first == second
    assert first.read_bytes() == b"real dataset bytes"
    assert calls == ["task-id"]


def test_non_404_does_not_use_dataset_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(client.SESSION, "get", lambda *args, **kwargs: Response(500))
    monkeypatch.setattr(client, "_official_gaia_file", lambda task_id: pytest.fail("fallback called"))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        client.download_file("task", "file.png", str(tmp_path))
