"""Tests for timecode-based clip placement (no GPU/ffmpeg needed)."""

from __future__ import annotations

from bormosync.engine.pipeline import compute_master_offsets
from bormosync.models import AlignmentMap


def _align(rec_start: float, k: float = 1.0) -> AlignmentMap:
    # rec_start = -offset / k  ->  offset = -rec_start * k
    return AlignmentMap(anchors=[], offset=-rec_start * k, k=k, residual_ms=0.0)


def test_contiguous_clips_lie_end_to_end() -> None:
    # clip0 starts at recorder t=0 (dur 10), clip1 at recorder t=10 -> back to back
    offsets, unaligned = compute_master_offsets([_align(0.0), _align(10.0)], durations=[10.0, 5.0])
    assert offsets == [0.0, 10.0]
    assert unaligned == []


def test_gap_between_clips_is_preserved() -> None:
    # clip1 actually starts at recorder t=15 -> 5s real gap after clip0 (ends at 10)
    offsets, unaligned = compute_master_offsets([_align(0.0), _align(15.0)], durations=[10.0, 5.0])
    assert offsets[0] == 0.0
    assert abs(offsets[1] - 15.0) < 1e-9  # gap kept, NOT forced to 10
    assert unaligned == []


def test_timeline_anchored_at_earliest_clip() -> None:
    # first clip starts later on the recorder than the second
    offsets, _ = compute_master_offsets([_align(20.0), _align(5.0)], durations=[4.0, 4.0])
    # earliest recorder start (5s) becomes timeline origin 0
    assert abs(offsets[1] - 0.0) < 1e-9
    assert abs(offsets[0] - 15.0) < 1e-9


def test_unaligned_clip_falls_back_to_previous_end() -> None:
    offsets, unaligned = compute_master_offsets([_align(0.0), None], durations=[10.0, 5.0])
    assert offsets[0] == 0.0
    assert offsets[1] == 10.0  # placed right after clip0
    assert unaligned == [1]
