from types import SimpleNamespace

from PIL import Image

from tools.video import (
    AnalyzeYouTubeVideoTool,
    find_javascript_runtime,
    normalize_youtube_url,
)
from tools.vision import ExtractYouTubeIdTool


def response(text):
    return SimpleNamespace(
        message=SimpleNamespace(content=[SimpleNamespace(text=text)])
    )


class FakeClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return (
            response("timestamped visual evidence")
            if len(self.calls) == 1
            else response("aggregated answer")
        )


def test_markdown_youtube_url_is_normalized_and_id_extracted():
    markdown = "[watch this](https://www.youtube.com/watch?v=L1vXCYZAYYM)"
    assert (
        normalize_youtube_url(markdown)
        == "https://www.youtube.com/watch?v=L1vXCYZAYYM"
    )
    assert ExtractYouTubeIdTool().forward(markdown) == "L1vXCYZAYYM"


def test_frame_analysis_preserves_timestamps_and_uses_same_frame_instruction(
    tmp_path,
):
    frames = []
    for index in range(2):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (8, 8), color=(index * 10, 0, 0)).save(path)
        frames.append((index * 2.5, path))
    client = FakeClient()
    tool = AnalyzeYouTubeVideoTool(visual_client_factory=lambda: client)

    result = tool._analyze_frames(
        frames,
        "How many kinds are simultaneous?",
        5.0,
    )

    assert result == "aggregated answer"
    assert len(client.calls) == 2
    frame_content = client.calls[0]["messages"][0]["content"]
    texts = [part["text"] for part in frame_content if part["type"] == "text"]
    assert any("0.00 seconds" in text for text in texts)
    assert any("2.50 seconds" in text for text in texts)
    assert any("do not combine frames" in text for text in texts)
    synthesis = client.calls[1]["messages"][0]["content"]
    assert "never merge objects seen at different timestamps" in synthesis


def test_download_passes_bundled_ffmpeg_and_available_js_runtime(
    monkeypatch,
    tmp_path,
):
    tool = AnalyzeYouTubeVideoTool()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        (tmp_path / "video.mp4").write_bytes(b"video")

    monkeypatch.setattr(
        "tools.video.find_javascript_runtime",
        lambda: ("node", "/usr/bin/node"),
    )
    monkeypatch.setattr(
        "imageio_ffmpeg.get_ffmpeg_exe",
        lambda: "/bundled/ffmpeg",
    )
    monkeypatch.setattr("tools.video.subprocess.run", run)

    assert (
        tool._download("https://youtu.be/abcdefghi", tmp_path).name
        == "video.mp4"
    )
    command = commands[0]
    assert command[command.index("--ffmpeg-location") + 1] == "/bundled/ffmpeg"
    assert command[command.index("--js-runtimes") + 1] == "node:/usr/bin/node"


def test_dense_temporal_sampling_uses_full_budget(monkeypatch, tmp_path):
    tool = AnalyzeYouTubeVideoTool()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    commands = []

    def run(command, **kwargs):
        commands.append(command)

    monkeypatch.setattr(
        "imageio_ffmpeg.get_ffmpeg_exe",
        lambda: "/bundled/ffmpeg",
    )
    monkeypatch.setattr("tools.video.subprocess.run", run)
    tool._extract_frames(
        video,
        tmp_path / "frames",
        duration=30.0,
        max_frames=60,
    )

    command = commands[0]
    assert "fps=1/0.5" in command[command.index("-vf") + 1]
    assert command[command.index("-frames:v") + 1] == "60"


def test_javascript_runtime_detection_accepts_node_environment():
    paths = {"node": "/usr/local/bin/node"}
    assert find_javascript_runtime(paths.get) == (
        "node",
        "/usr/local/bin/node",
    )


def test_forward_composes_download_sampling_and_visual_analysis(
    monkeypatch,
    tmp_path,
):
    tool = AnalyzeYouTubeVideoTool()
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    seen = {}

    def fake_download(url, workdir):
        seen["url"] = url
        return video

    monkeypatch.setattr(tool, "_download", fake_download)
    monkeypatch.setattr(tool, "_duration", lambda path: 10.0)
    monkeypatch.setattr(tool, "_extract_frames", lambda *args: [(1.0, frame)])
    monkeypatch.setattr(
        tool,
        "_analyze_frames",
        lambda frames, question, duration: (
            f"{question}:{duration}:{len(frames)}"
        ),
    )

    result = tool.forward(
        "[video](https://youtu.be/abcdefghi)",
        "visible question",
        12,
    )

    assert seen["url"] == "https://youtu.be/abcdefghi"
    assert result == "visible question:10.0:1"
def test_temporal_count_overrides_sparse_model_request():
    tool = AnalyzeYouTubeVideoTool()

    assert tool._needs_dense_temporal_sampling(
        "What is the highest number of bird species on camera simultaneously?"
    )
    assert not tool._needs_dense_temporal_sampling(
        "What color is the bird?"
    )

def test_video_tool_uses_configured_vision_factory_by_default(monkeypatch):
    import tools.video as video_module

    seen = {}

    class FakeClient:
        pass

    def factory(*, model_id=None):
        seen["model_id"] = model_id
        return FakeClient()

    monkeypatch.setattr(video_module, "CohereFailoverClient", factory)
    tool = video_module.AnalyzeYouTubeVideoTool()
    client = tool.visual_client_factory()

    assert isinstance(client, FakeClient)
    assert seen["model_id"] == video_module.COHERE_VIDEO_MODEL
