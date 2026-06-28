"""Tests for synchronization strategies on synthetic alignment data."""

from __future__ import annotations

from pathlib import Path

from bormosync.engine.strategies import get_strategy
from bormosync.models import AlignmentMap, Anchor, MediaClip


def _make_alignment(offset: float = 2.0, k: float = 1.001, n_anchors: int = 10) -> AlignmentMap:
    anchors = []
    for i in range(n_anchors):
        rec_t = 5.0 + i * 10.0
        cam_t = offset + k * rec_t
        anchors.append(Anchor(cam_time=cam_t, rec_time=rec_t, token=f"word{i}", confidence=0.9))
    return AlignmentMap(anchors=anchors, offset=offset, k=k, residual_ms=5.0)


def _make_video_clips() -> list[MediaClip]:
    return [
        MediaClip(
            path=Path("/videos/clip1.mp4"),
            kind="video",
            offset=0.0,
            in_point=0.0,
            duration=120.0,
            lane=1,
        ),
    ]


def test_strategy1_global_linear() -> None:
    alignment = _make_alignment(offset=2.0, k=1.001)
    strategy = get_strategy(1)
    rec_path = Path("/audio/recorder.wav")

    plan = strategy.plan(alignment, rec_path, rec_duration=100.0, video_clips=_make_video_clips())

    assert plan.strategy_id == 1
    audio_clips = [c for c in plan.clips if c.kind == "audio"]
    assert len(audio_clips) == 1

    clip = audio_clips[0]
    assert abs(clip.offset - 2.0) < 0.1
    assert abs(clip.duration - 100.0 * 1.001) < 0.5

    assert len(plan.audio_ops) == 1
    assert plan.audio_ops[0]["type"] == "atempo"


def test_strategy2_local_timestretch() -> None:
    alignment = _make_alignment(offset=2.0, k=1.001, n_anchors=10)
    strategy = get_strategy(2)
    rec_path = Path("/audio/recorder.wav")

    plan = strategy.plan(alignment, rec_path, rec_duration=100.0, video_clips=[])

    assert plan.strategy_id == 2
    assert len(plan.audio_ops) >= 2

    for op in plan.audio_ops:
        assert op["type"] == "atempo_segment"
        assert "factor" in op


def test_strategy3_silence_padding() -> None:
    alignment = _make_alignment(offset=2.0, k=1.001, n_anchors=10)
    strategy = get_strategy(3)
    rec_path = Path("/audio/recorder.wav")

    plan = strategy.plan(alignment, rec_path, rec_duration=100.0, video_clips=[])

    assert plan.strategy_id == 3
    audio_clips = [c for c in plan.clips if c.kind == "audio"]
    assert len(audio_clips) >= 1

    for clip in audio_clips:
        assert clip.lane == -1


def test_all_strategies_preserve_video_clips() -> None:
    """Every strategy must keep video clips (lane 1) in the plan, otherwise
    the exported FCPXML would contain no video at all."""
    alignment = _make_alignment(offset=2.0, k=1.001, n_anchors=12)
    video_clips = _make_video_clips()

    for sid in (1, 2, 3):
        strategy = get_strategy(sid)
        plan = strategy.plan(
            alignment,
            Path("/audio/rec.wav"),
            rec_duration=100.0,
            video_clips=video_clips,
        )
        video = [c for c in plan.clips if c.kind == "video"]
        audio = [c for c in plan.clips if c.kind == "audio"]
        assert len(video) == 1, f"Strategy {sid} dropped video clips"
        assert video[0].lane == 1
        assert len(audio) >= 1, f"Strategy {sid} produced no audio clips"
        # timeline must span at least the video extent
        assert plan.total_duration >= video[0].offset + video[0].duration - 1e-6


def test_strategy_offsets_within_tolerance() -> None:
    true_offset = 3.5
    true_k = 1.0005
    alignment = _make_alignment(offset=true_offset, k=true_k, n_anchors=20)

    for sid in [1, 2, 3]:
        strategy = get_strategy(sid)
        plan = strategy.plan(
            alignment,
            Path("/audio/rec.wav"),
            rec_duration=200.0,
            video_clips=[],
        )
        audio_clips = [c for c in plan.clips if c.kind == "audio"]
        for clip in audio_clips:
            expected_cam = true_offset + true_k * clip.in_point
            error_ms = abs(clip.offset - expected_cam) * 1000
            assert error_ms < 100, (
                f"Strategy {sid}: clip at rec={clip.in_point:.2f}s "
                f"offset error {error_ms:.1f}ms > 100ms"
            )
