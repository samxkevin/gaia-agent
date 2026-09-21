import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from tools.video import (
    AnalyzeYouTubeVideoTool,
    FrameObservation,
    VISION_BATCH_SIZE,
    build_video_counting_plan,
    find_javascript_runtime,
    normalize_youtube_url,
)
from tools.vision import ExtractYouTubeIdTool


def response(text):
    return SimpleNamespace(message=SimpleNamespace(content=[SimpleNamespace(text=text)]))


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return response(reply)


def observation_json(timestamps, counts=None):
    counts = counts or [1] * len(timestamps)
    return json.dumps([{"timestamp": t, "visible_subjects": ["birds"], "species": [f"species-{i}" for i in range(count)], "count": count, "confidence": .9, "uncertain": [], "note": "same frame"} for t, count in zip(timestamps, counts)])


def make_frames(tmp_path, timestamps):
    result = []
    for index, timestamp in enumerate(timestamps):
        path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (16, 16)).save(path)
        result.append((timestamp, path))
    return result


def test_semantics_preserve_species_vs_individuals():
    for question in ("highest number of bird species to be on camera simultaneously", "maximum number of distinct species visible together", "how many bird species appear at once"):
        plan = build_video_counting_plan(question)
        assert plan.distinct_categories and plan.simultaneous
        assert question in plan.instruction
    assert not build_video_counting_plan("maximum number of birds visible at once").distinct_categories


def test_markdown_url_normalization():
    url = "[watch](https://www.youtube.com/watch?v=L1vXCYZAYYM)"
    assert normalize_youtube_url(url).endswith("L1vXCYZAYYM")
    assert ExtractYouTubeIdTool().forward(url) == "L1vXCYZAYYM"


def test_structured_observations_preserve_timestamp_and_same_frame_prompt(tmp_path):
    frames = make_frames(tmp_path, [1.25, 2.25])
    # Model-returned timestamps are deliberately wrong; extracted timestamps win.
    visual = FakeClient([observation_json([91.5, 92.5], [2, 3])])
    tool = AnalyzeYouTubeVideoTool(lambda: visual)
    observations = tool._observe_frames(frames, build_video_counting_plan("maximum bird species visible at once"), "coarse")
    assert [o.timestamp for o in observations] == [1.25, 2.25]
    assert [o.count for o in observations] == [2, 3]
    prompt = visual.calls[0]["messages"][0]["content"][0]["text"]
    assert "Never merge evidence across images" in prompt
    assert len(frames) <= VISION_BATCH_SIZE


def test_provider_schema_drift_is_normalized_per_frame(tmp_path):
    batch = make_frames(tmp_path, [1, 2, 3, 4, 5])
    payload = {"observations": [
        {"species": ["a"], "visible_subjects": ["bird"], "uncertain": [], "confidence": .9},
        {"species": "b", "visible_subjects": "bird", "uncertain": False, "confidence": "bad"},
        {"species": ["c"], "visible_subjects": None, "uncertain": True, "count": "bad"},
        {"species": [], "visible_subjects": 7, "uncertain": None, "count": "2"},
        {"species": [], "visible_subjects": [], "uncertain": "none", "count": "not-a-number"},
    ]}
    observations = AnalyzeYouTubeVideoTool._parse_observations(
        json.dumps(payload), batch, "coarse"
    )
    assert [item.timestamp for item in observations] == [1, 2, 3, 4, 5]
    assert observations[0].uncertain == ()
    assert observations[1].species == ("b",)
    assert observations[1].uncertain == ()
    assert observations[2].uncertain == ("unspecified uncertainty",)
    assert observations[3].visible_subjects == ("7",)
    assert observations[3].count == 2
    assert observations[4].count == 0


def test_parser_accepts_wrapper_legacy_array_and_fenced_json(tmp_path):
    batch = make_frames(tmp_path, [1])
    item = {"species": ["a"], "visible_subjects": [], "count": 1, "confidence": 1, "uncertain": [], "note": ""}
    variants = [
        json.dumps({"observations": [item]}),
        json.dumps([item]),
        "```json\n" + json.dumps({"observations": [item]}) + "\n```",
    ]
    for text in variants:
        assert AnalyzeYouTubeVideoTool._parse_observations(text, batch, "coarse")[0].count == 1
    assert AnalyzeYouTubeVideoTool._parse_observations("plain prose", batch, "coarse") == []


def test_one_malformed_observation_does_not_discard_valid_siblings(tmp_path):
    batch = make_frames(tmp_path, [1, 2, 3])
    payload = {"observations": [
        {"species": ["a"], "confidence": .8},
        "malformed",
        {"species": ["b", "c"], "uncertain": False, "confidence": .9},
    ]}
    observations = AnalyzeYouTubeVideoTool._parse_observations(
        json.dumps(payload), batch, "coarse"
    )
    assert [(item.timestamp, item.count) for item in observations] == [(1, 1), (3, 2)]


def test_missing_provider_observations_get_temporal_placeholders(tmp_path):
    frames = make_frames(tmp_path, [1, 2, 3, 4])
    visual = FakeClient([json.dumps({"observations": [
        {"species": ["a"], "confidence": .9}
    ]})])
    tool = AnalyzeYouTubeVideoTool(lambda: visual)
    observations = tool._observe_frames(
        frames, build_video_counting_plan("maximum species visible at once"), "coarse"
    )
    assert [item.timestamp for item in observations] == [1, 2, 3, 4]
    assert [item.count for item in observations] == [1, 0, 0, 0]
    assert tool.observation_parse_diagnostics["coarse"] == {
        "requested_frame_count": 4,
        "parsed_observation_count": 1,
        "missing_observation_count": 3,
        "complete_batches": 0,
        "failed_or_incomplete_batches": 1,
    }
    assert tool.observation_batch_diagnostics["coarse"][0]["status"] == "wrong_observation_count"
    candidates = tool._select_candidates(observations, duration=4)
    assert candidates


def test_batch_diagnostics_distinguish_empty_parse_and_request_failures(tmp_path):
    frames = make_frames(tmp_path, [1])
    plan = build_video_counting_plan("maximum species visible at once")

    empty_tool = AnalyzeYouTubeVideoTool(lambda: FakeClient([""]))
    empty_tool._observe_frames(frames, plan, "coarse")
    assert empty_tool.observation_batch_diagnostics["coarse"][0]["status"] == "empty_response"

    prose_tool = AnalyzeYouTubeVideoTool(lambda: FakeClient(["not json"]))
    prose_tool._observe_frames(frames, plan, "coarse")
    assert prose_tool.observation_batch_diagnostics["coarse"][0]["status"] == "response_parsing_failure"

    failed_tool = AnalyzeYouTubeVideoTool(lambda: FakeClient([RuntimeError("provider down")]))
    failed_tool._observe_frames(frames, plan, "coarse")
    diagnostic = failed_tool.observation_batch_diagnostics["coarse"][0]
    assert diagnostic["status"] == "request_failure"
    assert diagnostic["exception_type"] == "RuntimeError"
    assert diagnostic["exception_message"] == "provider down"


def test_observation_calls_prefer_structured_response_format(tmp_path):
    frames = make_frames(tmp_path, [1])
    visual = FakeClient([json.dumps({"observations": [{"species": ["a"]}]})])
    AnalyzeYouTubeVideoTool(lambda: visual)._observe_frames(
        frames, build_video_counting_plan("maximum species visible at once"), "coarse"
    )
    assert visual.calls[0]["response_format"]["schema"]["required"] == ["observations"]


def test_independent_verifier_receives_actual_frame_without_prior_claims(tmp_path):
    frame = make_frames(tmp_path, [82.125])[0][1]
    # Independent verifier hallucinates a different timestamp; frame identity wins.
    verifier = FakeClient([observation_json([91.500], [2])])
    tool = AnalyzeYouTubeVideoTool(
        lambda: FakeClient([]),
        verification_client_factory=lambda: verifier,
    )
    candidate = FrameObservation(
        82.125, (), ("prior-secret-species",), 1, .7, (),
        pass_name="refined", frame_path=str(frame),
    )
    result = tool._verify_candidates(
        [candidate],
        build_video_counting_plan("maximum bird species visible at once"),
    )
    assert result[0].pass_name == "verification"
    assert result[0].timestamp == 82.125
    content = verifier.calls[0]["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert "Original question: maximum bird species visible at once" in content[0]["text"]
    assert "prior-secret-species" not in content[0]["text"]


def test_verification_is_bounded_to_candidate_limit(tmp_path):
    frames = make_frames(tmp_path, list(range(6)))
    verifier = FakeClient([observation_json([float(i)], [1]) for i in range(4)])
    tool = AnalyzeYouTubeVideoTool(
        lambda: FakeClient([]), verification_client_factory=lambda: verifier
    )
    candidates = [
        FrameObservation(float(i), (), ("a",), 1, .8, (), frame_path=str(path))
        for i, (_, path) in enumerate(frames)
    ]
    assert len(tool._verify_candidates(candidates, build_video_counting_plan("maximum species visible at once"))) == 4
    assert len(verifier.calls) == 4


def test_one_verifier_failure_does_not_cancel_later_candidate(tmp_path):
    frames = make_frames(tmp_path, [1.0, 8.0])
    verifier = FakeClient([
        RuntimeError("first candidate unavailable"),
        observation_json([8.0], [2]),
    ])
    tool = AnalyzeYouTubeVideoTool(
        lambda: FakeClient([]), verification_client_factory=lambda: verifier
    )
    candidates = [
        FrameObservation(timestamp, (), ("a",), 1, .8, (), frame_path=str(path))
        for timestamp, path in frames
    ]
    verified = tool._verify_candidates(
        candidates, build_video_counting_plan("maximum species visible at once")
    )
    assert [item.timestamp for item in verified] == [8.0]
    assert [item["status"] for item in tool.verification_diagnostics] == [
        "failed", "verified"
    ]


def test_malformed_verifier_json_is_reported_as_failed_attempt(tmp_path):
    frame = make_frames(tmp_path, [4.0])[0]
    verifier = FakeClient(["not structured json"])
    tool = AnalyzeYouTubeVideoTool(
        lambda: FakeClient([]), verification_client_factory=lambda: verifier
    )
    candidate = FrameObservation(
        4.0, (), ("a",), 1, .8, (), frame_path=str(frame[1])
    )
    assert tool._verify_candidates(
        [candidate], build_video_counting_plan("maximum species visible at once")
    ) == []
    assert tool.verification_diagnostics == [{
        "timestamp": 4.0,
        "status": "failed",
        "reason": "malformed structured response",
    }]


def test_candidate_selection_finds_high_counts_and_separates_regions():
    observations = [FrameObservation(t, (), ("a",) * c, c, .9, ()) for t, c in [(1, 1), (2, 3), (2.5, 3), (8, 2)]]
    selected = AnalyzeYouTubeVideoTool._select_candidates(observations)
    assert selected[0].timestamp == 2
    assert 2.5 not in [x.timestamp for x in selected]


def test_candidate_selection_is_temporally_diverse():
    observations = [
        FrameObservation(t, (), ("a",) * count, count, .9, ())
        for t, count in [(76, 3), (78, 3), (82, 2), (90, 2)]
    ]
    timestamps = [
        item.timestamp for item in AnalyzeYouTubeVideoTool._select_candidates(observations)
    ]
    assert 76 in timestamps
    assert 78 not in timestamps
    assert 82 in timestamps
    assert 90 in timestamps


def test_zero_count_frames_can_nominate_each_full_video_region():
    observations = [
        FrameObservation(float(timestamp), (), (), 0, .2, ())
        for timestamp in range(0, 120)
    ]
    candidates = AnalyzeYouTubeVideoTool._select_candidates(observations, duration=120)
    assert len(candidates) == 4
    assert [int(item.timestamp // 30) for item in candidates] == [0, 1, 2, 3]


def test_early_positive_cluster_cannot_eliminate_later_regions():
    observations = [
        FrameObservation(float(timestamp), (), (), 3 if timestamp < 30 else 0, .9, ())
        for timestamp in range(0, 120)
    ]
    candidates = AnalyzeYouTubeVideoTool._select_candidates(observations, duration=120)
    assert len(candidates) == 4
    assert candidates[0].count == 3
    assert any(item.timestamp >= 90 for item in candidates)


def test_candidate_selection_across_full_duration_is_deterministic():
    observations = [
        FrameObservation(float(timestamp), (), (), timestamp % 3, .5, ())
        for timestamp in range(0, 120)
    ]
    first = AnalyzeYouTubeVideoTool._select_candidates(observations, duration=120)
    second = AnalyzeYouTubeVideoTool._select_candidates(observations, duration=120)
    assert first == second
    assert len(first) == 4
    assert [int(item.timestamp // 30) for item in first] == [0, 1, 2, 3]


def test_coarse_diagnostics_distinguish_zero_scores_from_missing_observations():
    observations = [
        FrameObservation(0, (), (), 0, .5, ()),
        FrameObservation(1, (), ("a",), 1, .8, ()),
        FrameObservation(2, (), (), 0, .2, ("ambiguous",)),
    ]
    diagnostics = AnalyzeYouTubeVideoTool._coverage_diagnostics(
        observations, requested_frames=120
    )
    assert diagnostics == {
        "requested_frame_count": 120,
        "coarse_observation_count": 3,
        "minimum_timestamp_observed": 0,
        "maximum_timestamp_observed": 2,
        "zero_count_observations": 2,
        "positive_count_observations": 1,
        "uncertain_observations": 1,
    }


def test_refinement_region_reaches_peak_near_82_seconds(monkeypatch, tmp_path):
    calls = []
    tool = AnalyzeYouTubeVideoTool()
    candidate = FrameObservation(75.35, (), ("a",), 1, .9, ())

    def extract(video, output, duration, count, interval, offset):
        calls.append((count, interval, offset))
        return []

    monkeypatch.setattr(tool, "_extract_frames", extract)
    assert tool._extract_candidate_frames(
        tmp_path / "video", tmp_path / "frames", 120, [candidate]
    ) == []
    assert calls == [(31, 1.0, 60.0)]
    diagnostic = tool.refinement_diagnostics[0]
    assert diagnostic["start"] == 60
    assert diagnostic["end"] == 90
    assert diagnostic["start"] <= 82 <= diagnostic["end"]


def test_single_injected_factory_controls_visual_and_synthesis(tmp_path):
    frames = make_frames(tmp_path, [1.0])
    injected = FakeClient([observation_json([1.0], [1]), "summary"])
    tool = AnalyzeYouTubeVideoTool(lambda: injected)
    plan = build_video_counting_plan("maximum species visible at once")
    observations = tool._observe_frames(frames, plan, "coarse")
    assert "summary" in tool._synthesize(observations, plan, 2, 1)
    assert len(injected.calls) == 2


def test_synthesis_uses_injected_client_and_never_merges_timestamps():
    synthesis = FakeClient(["aggregated answer"])
    tool = AnalyzeYouTubeVideoTool(lambda: FakeClient([]), lambda: synthesis)
    observations = [FrameObservation(1, (), ("a", "b"), 2, .9, ()), FrameObservation(2, (), ("b", "c"), 2, .9, ())]
    result = tool._synthesize(observations, build_video_counting_plan("maximum species visible at once"), 3, 3)
    assert "aggregated answer" in result
    payload = synthesis.calls[0]["messages"][0]["content"]
    assert "Do not add species from different timestamps" in payload
    assert '"programmatic_maximum_observed": 2' in payload


def test_merge_keeps_stronger_coarse_when_refinement_is_lower():
    coarse = FrameObservation(10.0, (), ("a", "b", "c"), 3, .9, (), pass_name="coarse")
    refined = FrameObservation(10.05, (), ("a", "b"), 2, .95, (), pass_name="refined")
    assert AnalyzeYouTubeVideoTool._merge_observations([coarse, refined]) == [coarse]


def test_merge_prefers_stronger_refinement_and_deduplicates_overlap():
    coarse = FrameObservation(10.0, (), ("a",), 1, .7, (), pass_name="coarse")
    refined = FrameObservation(10.05, (), ("a", "b"), 2, .95, (), pass_name="refined")
    other = FrameObservation(12.0, (), ("c",), 1, .8, (), pass_name="coarse")
    merged = AnalyzeYouTubeVideoTool._merge_observations([coarse, refined, other])
    assert merged == [refined, other]


def test_matching_verification_preserves_high_confidence_candidate():
    coarse = FrameObservation(10, (), ("a", "b", "c"), 3, .9, (), pass_name="coarse")
    refined = FrameObservation(10.05, (), ("a", "b"), 2, .6, (), pass_name="refined")
    verified = FrameObservation(10.02, (), ("a", "b", "c"), 3, .95, (), pass_name="verification")
    assert AnalyzeYouTubeVideoTool._reconcile_observations([coarse, refined, verified]) == [verified]


def test_weak_contradictory_verification_does_not_erase_stronger_evidence():
    coarse = FrameObservation(10, (), ("a", "b", "c"), 3, .95, (), pass_name="coarse")
    verified = FrameObservation(10.02, (), ("a", "b"), 2, .4, (), pass_name="verification")
    assert AnalyzeYouTubeVideoTool._reconcile_observations([coarse, verified]) == [coarse]


def test_strong_consensus_can_reduce_low_confidence_candidate():
    coarse = FrameObservation(10, (), ("a", "b", "c"), 3, .35, (), pass_name="coarse")
    refined = FrameObservation(10.04, (), ("a", "b"), 2, .9, (), pass_name="refined")
    verified = FrameObservation(10.02, (), ("a", "b"), 2, .95, (), pass_name="verification")
    assert AnalyzeYouTubeVideoTool._reconcile_observations([coarse, refined, verified]) == [verified]


def test_reconciliation_timestamp_grouping_is_nontransitive():
    first = FrameObservation(10.000, (), ("a",), 1, .9, (), pass_name="coarse")
    bridge = FrameObservation(10.100, (), ("a",), 1, .8, (), pass_name="refined")
    third = FrameObservation(10.200, (), ("b",), 1, .9, (), pass_name="verification")
    reconciled = AnalyzeYouTubeVideoTool._reconcile_observations(
        [first, bridge, third], timestamp_tolerance=.125
    )
    assert reconciled == [first, third]


def test_reconciliation_never_merges_different_timestamps():
    first = FrameObservation(10, (), ("a", "b"), 2, .9, (), pass_name="coarse")
    second = FrameObservation(10.25, (), ("c",), 1, .9, (), pass_name="verification")
    assert AnalyzeYouTubeVideoTool._reconcile_observations([first, second]) == [first, second]


def test_programmatic_maximum_is_recomputed_from_merged_evidence():
    synthesis = FakeClient(["normalized"])
    tool = AnalyzeYouTubeVideoTool(lambda: FakeClient([]), lambda: synthesis)
    observations = [
        FrameObservation(1, (), ("a", "b", "c"), 3, .95, (), pass_name="coarse"),
        FrameObservation(1.05, (), ("a", "b"), 2, .9, (), pass_name="refined"),
        FrameObservation(4, (), ("d", "e"), 2, .8, (), pass_name="refined"),
    ]
    result = tool._synthesize(observations, build_video_counting_plan("maximum species visible at once"), 5, 5)
    assert '"programmatic_maximum_observed": 3' in result
    assert '"timestamp": 1' in result
    assert '"species": [\n        "a",\n        "b",\n        "c"' in result


def test_secondary_synthesis_failure_preserves_primary_result():
    synthesis = FakeClient([RuntimeError("down")])
    tool = AnalyzeYouTubeVideoTool(lambda: FakeClient([]), lambda: synthesis)
    obs = [FrameObservation(1, (), ("a", "b"), 2, .9, ())]
    assert '"programmatic_maximum_observed": 2' in tool._synthesize(obs, build_video_counting_plan("maximum species visible at once"), 2, 2)


def test_download_uses_720p_bundled_ffmpeg_and_js(monkeypatch, tmp_path):
    commands = []
    def run(command, **kwargs):
        commands.append(command); (tmp_path / "video.mp4").write_bytes(b"x")
    monkeypatch.setattr("tools.video.find_javascript_runtime", lambda: ("node", "/node"))
    monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: "/ffmpeg")
    monkeypatch.setattr("tools.video.subprocess.run", run)
    AnalyzeYouTubeVideoTool()._download("https://youtu.be/abcdef", tmp_path)
    command = commands[0]
    assert command[command.index("--ffmpeg-location") + 1] == "/ffmpeg"
    assert command[command.index("--js-runtimes") + 1] == "node:/node"
    assert "height<=720" in command[command.index("-f") + 1]


def test_temporal_questions_are_denser_and_refinement_is_conditional(monkeypatch, tmp_path):
    video = tmp_path / "v.mp4"; video.write_bytes(b"x")
    frame = tmp_path / "f.jpg"; frame.write_bytes(b"x")
    calls = []
    tool = AnalyzeYouTubeVideoTool(original_question="maximum number of species visible at once")
    monkeypatch.setattr(tool, "_download", lambda *a: video)
    monkeypatch.setattr(tool, "_duration", lambda p: 100.0)
    monkeypatch.setattr(tool, "_extract_frames", lambda *a, **kw: calls.append(kw.get("interval")) or [(5, frame)])
    monkeypatch.setattr(tool, "_observe_frames", lambda frames, plan, name: [FrameObservation(5, (), ("a",), 1, .9, (), pass_name=name)])
    monkeypatch.setattr(tool, "_extract_candidate_frames", lambda *a: [(5.1, frame)])
    monkeypatch.setattr(tool, "_synthesize", lambda *a: "ok")
    assert tool.forward("https://youtu.be/abcdef", "narrow paraphrase") == "ok"
    assert calls[0] == 1.0

    ordinary = AnalyzeYouTubeVideoTool(original_question="what color is the bird?")
    monkeypatch.setattr(ordinary, "_download", lambda *a: video)
    monkeypatch.setattr(ordinary, "_duration", lambda p: 100.0)
    monkeypatch.setattr(ordinary, "_extract_frames", lambda *a, **kw: [(5, frame)])
    monkeypatch.setattr(ordinary, "_observe_frames", lambda frames, plan, name: [])
    monkeypatch.setattr(ordinary, "_extract_candidate_frames", lambda *a: (_ for _ in ()).throw(AssertionError("unexpected refinement")))
    monkeypatch.setattr(ordinary, "_synthesize", lambda *a: "ok")
    assert ordinary.forward("https://youtu.be/abcdef", "paraphrase") == "ok"


def test_candidate_extraction_failure_is_diagnostic_and_other_candidates_continue(monkeypatch, tmp_path):
    tool = AnalyzeYouTubeVideoTool()
    calls = 0
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")

    def extract(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("decoder failed")
        return [(20.0, frame)]

    monkeypatch.setattr(tool, "_extract_frames", extract)
    candidates = [
        FrameObservation(5, (), ("a",), 1, .9, ()),
        FrameObservation(20, (), ("b",), 1, .9, ()),
    ]
    frames = tool._extract_candidate_frames(
        tmp_path / "video", tmp_path / "out", 30, candidates
    )
    assert frames == [(20.0, frame)]
    assert tool.refinement_diagnostics[0]["exception_type"] == "RuntimeError"
    assert tool.refinement_diagnostics[0]["exception_message"] == "decoder failed"
    assert tool.refinement_diagnostics[1]["status"] == "extracted"


def test_refinement_failure_preserves_complete_coarse_result(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"; video.write_bytes(b"video")
    frame = tmp_path / "frame.jpg"; frame.write_bytes(b"frame")
    coarse = FrameObservation(5, (), ("a", "b"), 2, .9, (), pass_name="coarse")
    captured = {}
    tool = AnalyzeYouTubeVideoTool(original_question="maximum species visible at once")
    monkeypatch.setattr(tool, "_download", lambda *args: video)
    monkeypatch.setattr(tool, "_duration", lambda path: 10.0)
    monkeypatch.setattr(tool, "_extract_frames", lambda *args, **kwargs: [(5, frame)])
    monkeypatch.setattr(tool, "_observe_frames", lambda frames, plan, name: [coarse])
    monkeypatch.setattr(tool, "_extract_candidate_frames", lambda *args: (_ for _ in ()).throw(RuntimeError("refinement failed")))
    monkeypatch.setattr(tool, "_synthesize", lambda observations, *args: captured.update(observations=observations) or "ok")
    assert tool.forward("https://youtu.be/abcdef", "ignored") == "ok"
    assert captured["observations"] == [coarse]
    diagnostic = tool.refinement_diagnostics[0]
    assert diagnostic["status"] == "failed"
    assert diagnostic["exception_type"] == "RuntimeError"
    assert diagnostic["exception_message"] == "refinement failed"
    assert diagnostic["fallback_coarse_retained"] is True


def test_synthesis_reports_refinement_failure():
    synthesis = FakeClient(["summary"])
    tool = AnalyzeYouTubeVideoTool(lambda: FakeClient([]), lambda: synthesis)
    tool.refinement_diagnostics = [{
        "attempted": True,
        "candidate_timestamp": 5,
        "start": 0,
        "end": 11,
        "extracted_frame_count": 0,
        "status": "failed",
        "exception_type": "RuntimeError",
        "exception_message": "ffmpeg failed",
        "fallback_coarse_retained": True,
    }]
    result = tool._synthesize(
        [FrameObservation(5, (), ("a",), 1, .9, ())],
        build_video_counting_plan("maximum species visible at once"),
        20,
        20,
    )
    assert '"refinement_failures": 1' in result
    assert '"exception_message": "ffmpeg failed"' in result


def test_verification_failure_does_not_fail_task_or_erase_evidence(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"; video.write_bytes(b"video")
    frame = tmp_path / "frame.jpg"; frame.write_bytes(b"frame")
    coarse = FrameObservation(
        5, (), ("a", "b"), 2, .9, (),
        pass_name="coarse", frame_path=str(frame),
    )
    verifier = FakeClient([RuntimeError("verification unavailable")])
    captured = {}
    tool = AnalyzeYouTubeVideoTool(
        lambda: FakeClient([]),
        verification_client_factory=lambda: verifier,
        original_question="maximum species visible at once",
    )
    monkeypatch.setattr(tool, "_download", lambda *args: video)
    monkeypatch.setattr(tool, "_duration", lambda path: 10.0)
    monkeypatch.setattr(tool, "_extract_frames", lambda *args, **kwargs: [(5, frame)])
    monkeypatch.setattr(
        tool, "_observe_frames", lambda frames, plan, name: [coarse] if name == "coarse" else []
    )
    monkeypatch.setattr(tool, "_extract_candidate_frames", lambda *args: [])
    monkeypatch.setattr(tool, "_synthesize", lambda observations, *args: captured.update(observations=observations) or "ok")
    assert tool.forward("https://youtu.be/abcdef", "ignored") == "ok"
    assert captured["observations"] == [coarse]
    assert len(verifier.calls) == 1


def test_frame_extraction_prefers_ffmpeg_reported_timestamps(monkeypatch, tmp_path):
    output = tmp_path / "frames"
    output.mkdir()
    for name in ("0001.jpg", "0002.jpg"):
        (output / name).write_bytes(b"frame")
    monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: "/ffmpeg")
    completed = SimpleNamespace(stderr=b"showinfo pts_time:0.125 x\nshowinfo pts_time:1.125 x")
    monkeypatch.setattr("tools.video.subprocess.run", lambda *args, **kwargs: completed)
    frames = AnalyzeYouTubeVideoTool()._extract_frames(
        tmp_path / "video", output, 10, 2, interval=1, offset=2
    )
    assert [timestamp for timestamp, _ in frames] == [2.125, 3.125]


def test_frame_extraction_is_resource_bounded(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: "/ffmpeg")
    monkeypatch.setattr("tools.video.subprocess.run", lambda command, **kwargs: commands.append(command))
    AnalyzeYouTubeVideoTool()._extract_frames(tmp_path / "v", tmp_path / "frames", 30, 30, interval=1)
    command = commands[0]
    assert "min(1280,iw)" in command[command.index("-vf") + 1]
    assert command[command.index("-frames:v") + 1] == "30"


def test_javascript_runtime_accepts_node():
    assert find_javascript_runtime({"node": "/node"}.get) == ("node", "/node")
