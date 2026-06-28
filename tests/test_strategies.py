"""Tests for per-clip synchronization strategies."""

from __future__ import annotations

from pathlib import Path

from bormosync.config import BormoSyncConfig
from bormosync.engine.strategies import get_strategy
from bormosync.models import AlignmentMap, Anchor

REC = Path("/audio/recorder.wav")


def _alignment(offset: float, k: float, rec_times: list[float]) -> AlignmentMap:
    """Build alignment t_local = offset + k * t_rec with anchors at the given
    recorder times."""
    anchors = [
        Anchor(cam_time=offset + k * r, rec_time=r, token=f"w{i}", confidence=0.9)
        for i, r in enumerate(rec_times)
    ]
    return AlignmentMap(anchors=anchors, offset=offset, k=k, residual_ms=2.0)


def _two_dense_blocks(offset: float, k: float, pause: float = 4.0) -> AlignmentMap:
    """Two phrases of densely-spaced anchors separated by a long pause."""
    block1 = [1.0 + 0.25 * j for j in range(5)]  # 1.00 .. 2.00
    block2 = [block1[-1] + pause + 0.25 * j for j in range(5)]
    return _alignment(offset, k, block1 + block2)


def _rec_to_local(am: AlignmentMap, r: float) -> float:
    return am.offset + am.k * r


CONFIG = BormoSyncConfig()


def test_strategy1_single_clip() -> None:
    k = 1.001
    am = _alignment(offset=-5.0 * k, k=k, rec_times=[5.0, 9.0, 13.0])
    clips, ops = get_strategy(1).plan_clip(am, REC, 100.0, 10.0, 200.0, CONFIG)

    assert len(clips) == 1 and len(ops) == 1
    assert clips[0].lane == -1
    assert abs(clips[0].offset - 100.0) < 0.05  # audio starts at clip offset
    assert abs(clips[0].duration - 10.0) < 0.1
    assert ops[0]["type"] == "atempo_segment"
    assert abs(float(ops[0]["factor"]) - 1.0 / k) < 1e-4


def test_strategy3_makes_speech_blocks() -> None:
    am = _two_dense_blocks(offset=0.0, k=1.0)
    clips, ops = get_strategy(3).plan_clip(am, REC, 50.0, 20.0, 200.0, CONFIG)

    assert len(clips) == len(ops) == 2  # exactly two phrases
    for op in ops:
        assert op["type"] == "extract"  # pitch-safe: no tempo change


def test_strategy4_hybrid_stretches_blocks() -> None:
    k = 1.002
    am = _two_dense_blocks(offset=0.0, k=k)
    clips, ops = get_strategy(4).plan_clip(am, REC, 0.0, 20.0, 200.0, CONFIG)

    assert len(clips) == len(ops) == 2
    for op in ops:
        assert op["type"] == "atempo_segment"
        assert abs(float(op["factor"]) - 1.0 / k) < 1e-4


def test_all_strategies_place_audio_at_matched_timecode() -> None:
    """Core invariant: every produced audio clip sits at
    clip_offset + rec_to_local(op.start) — i.e. under the matching camera time.
    This is what guarantees timecode-accurate sync for any clip layout."""
    k = 1.0015
    clip_offset = 250.0
    am = _two_dense_blocks(offset=-3.0 * k, k=k)

    for sid in (1, 2, 3, 4):
        clips, ops = get_strategy(sid).plan_clip(am, REC, clip_offset, 27.0, 300.0, CONFIG)
        assert len(clips) == len(ops) >= 1
        for clip, op in zip(clips, ops, strict=True):
            expected = clip_offset + _rec_to_local(am, float(op["start"]))
            assert abs(clip.offset - expected) < 1e-6, f"strategy {sid} mis-placed audio"
            assert clip.lane == -1
            assert clip.in_point == 0.0


def test_strategy_falls_back_when_no_anchors() -> None:
    am = AlignmentMap(anchors=[], offset=-2.0, k=1.0, residual_ms=0.0)
    for sid in (2, 3, 4):
        clips, ops = get_strategy(sid).plan_clip(am, REC, 0.0, 10.0, 100.0, CONFIG)
        # falls back to global-linear: a single tempo segment
        assert len(clips) == 1 and ops[0]["type"] == "atempo_segment"
