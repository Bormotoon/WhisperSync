"""Tests for timecode-based clip placement (no GPU/ffmpeg needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bormosync.config import BormoSyncConfig
from bormosync.engine.pipeline import compute_master_offsets, scan_cameras
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


def _fake_probe(path: Path):  # noqa: ANN202
    from fractions import Fraction

    from bormosync.engine.media import MediaInfo

    return MediaInfo(
        path=path,
        duration=10.0,
        fps=Fraction(25, 1),
        width=1920,
        height=1080,
        video_codec="h264",
        audio_codec="aac",
        audio_channels=2,
        audio_sample_rate=48000,
    )


def test_scan_cameras_flat_dir_is_single_camera(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("bormosync.engine.pipeline.probe", _fake_probe)
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mp4").touch()

    cams = scan_cameras(tmp_path, BormoSyncConfig())
    assert len(cams) == 1
    assert cams[0].lane == 1
    assert len(cams[0].clips) == 2
    assert all(c.lane == 1 for c in cams[0].clips)


def test_scan_cameras_subfolders_become_separate_lanes(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    monkeypatch.setattr("bormosync.engine.pipeline.probe", _fake_probe)
    (tmp_path / "camA").mkdir()
    (tmp_path / "camB").mkdir()
    (tmp_path / "camA" / "a1.mp4").touch()
    (tmp_path / "camA" / "a2.mp4").touch()
    (tmp_path / "camB" / "b1.mp4").touch()

    cams = scan_cameras(tmp_path, BormoSyncConfig())
    assert [c.name for c in cams] == ["camA", "camB"]
    assert cams[0].lane == 1 and cams[1].lane == 2
    assert len(cams[0].clips) == 2 and len(cams[1].clips) == 1
    assert all(c.lane == 1 for c in cams[0].clips)
    assert all(c.lane == 2 for c in cams[1].clips)


def test_scan_cameras_empty_dir_raises(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("bormosync.engine.pipeline.probe", _fake_probe)
    with pytest.raises(RuntimeError):
        scan_cameras(tmp_path, BormoSyncConfig())
