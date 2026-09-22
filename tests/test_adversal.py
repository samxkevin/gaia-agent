import json
from pathlib import Path

from tools.adversal import (
    AdversalAugmentedYouTubeVideoTool,
    _load_frames_json,
    classify_response,
    parse_remaining_minutes,
    parse_request_id,
    parse_time_value,
    resolve_adversal_command,
)
from tools.video import FrameObservation


def test_adversal_response_parser_reads_structured_content():
    class Result:
        structured_content = {"remaining_minutes": 96.0}
        content = []

    text = _result_text(Result())
    assert '"remaining_minutes": 96.0' in text
    assert parse_remaining_minutes(text) == 96.0



def test_time_parser_accepts_provider_clock_strings():
    assert parse_time_value("00:00:04") == 4.0
    assert parse_time_value("00:01:02.5") == 62.5
    assert parse_time_value(12.25) == 12.25


def test_load_frames_json_supports_provider_schema(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    image = frames_dir / "frame-001.jpg"
    image.write_bytes(b"image")
    (tmp_path / "frames.json").write_text(
        json.dumps([
            {
                "frame": "frames/frame-001.jpg",
                "timestamp": "00:01:22",
                "text": "penguin",
            }
        ]),
        encoding="utf-8",
    )
    frames = _load_frames_json(tmp_path)
    assert len(frames) == 1
    assert frames[0].timestamp == 82.0
    assert frames[0].path == image
    assert frames[0].ocr_text == "penguin"


def test_resolve_adversal_prefers_explicit_executable(monkeypatch):
    monkeypatch.setenv("ADVERSAL_CLI", r"C:\adversal\adversal-cli.exe")
    monkeypatch.setattr("tools.adversal.shutil.which", lambda _: None)
    assert resolve_adversal_command() == (r"C:\adversal\adversal-cli.exe", [])


def test_augmented_tool_falls_back_when_adversal_is_unavailable(monkeypatch):
    class FakeNative:
        def __init__(self, original_question=None):
            self.original_question = original_question

        def forward(self, url, question, max_frames=None):
            return "native-result"

    monkeypatch.setattr(
        "tools.adversal.NativeAnalyzeYouTubeVideoTool",
        FakeNative,
    )
    monkeypatch.setattr(
        "tools.adversal.run_adversal_video",
        lambda url, output_dir: type(
            "Evidence",
            (),
            {
                "status": "unavailable",
                "message": "not installed",
                "request_id": None,
                "remaining_minutes": None,
                "estimated_required_minutes": 3,
                "frames": (),
            },
        )(),
    )

    tool = AdversalAugmentedYouTubeVideoTool(
        original_question="highest number of bird species simultaneously on camera"
    )
    result = tool.forward("https://www.youtube.com/watch?v=abc12345678", "ignored")
    assert "native-result" in result


def test_adversal_maximum_uses_only_verified_frames_for_auxiliary_max(monkeypatch, tmp_path):
    class FakeNative:
        def __init__(self, original_question=None):
            self.original_question = original_question
            self.verification_diagnostics = []

        def _observe_frames(self, frames, plan, pass_name):
            return [
                FrameObservation(
                    timestamp=82.0,
                    visible_subjects=("birds",),
                    species=("a", "b", "c"),
                    count=3,
                    confidence=0.95,
                    uncertain=(),
                    pass_name=pass_name,
                    frame_path=str(frames[0][1]),
                )
            ]

        def _select_candidates(self, observations, duration=None):
            return observations[:1]

        def _verify_candidates(self, candidates, plan):
            return candidates

        def forward(self, url, question, max_frames=None):
            return json.dumps(
                {
                    "programmatic_maximum_observed": 2,
                    "same_frame_candidates": [
                        {
                            "timestamp": 40,
                            "species": ["x", "y"],
                            "count": 2,
                            "confidence": 0.9,
                            "pass_name": "refined",
                        }
                    ],
                }
            )

    fake_frame = tmp_path / "fake.jpg"
    fake_frame.write_bytes(b"x")
    evidence = type(
        "Evidence",
        (),
        {
            "status": "completed",
            "message": "ok",
            "request_id": "abc",
            "remaining_minutes": 50,
            "estimated_required_minutes": 3,
            "frames": (type("Frame", (), {"timestamp": 82.0, "path": fake_frame})(),),
        },
    )()

    monkeypatch.setattr("tools.adversal.NativeAnalyzeYouTubeVideoTool", FakeNative)
    monkeypatch.setattr("tools.adversal.run_adversal_video", lambda url, output_dir: evidence)

    tool = AdversalAugmentedYouTubeVideoTool(
        original_question="highest number of bird species simultaneously on camera"
    )
    result = tool.forward("https://www.youtube.com/watch?v=abc12345678", "ignored")
    assert '"combined_programmatic_maximum_observed": 3' in result
    assert '"source": "adversal_verified"' in result
