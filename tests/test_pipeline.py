"""Tests for timecode-based clip placement (no GPU/ffmpeg needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from bormosync.config import BormoSyncConfig
from bormosync.engine.pipeline import (
    CameraGroup,
    compute_master_offsets,
    make_timeline_entries,
    scan_cameras,
)
from bormosync.models import AlignmentMap, MediaClip


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


def _vclip(name: str, offset: float, dur: float, lane: int) -> MediaClip:
    return MediaClip(Path(f"{name}.mp4"), "video", offset, 0.0, dur, lane)


def _aclip(name: str, offset: float, dur: float, lane: int) -> MediaClip:
    return MediaClip(Path(f"{name}.wav"), "audio", offset, 0.0, dur, lane)


def test_timeline_entries_rows_speed_status() -> None:
    cameras = [
        CameraGroup("camA", 1, [], []),
        CameraGroup("camB", 2, [], []),
    ]
    video_clips = [_vclip("a1", 0.0, 10.0, 1), _vclip("b1", 0.0, 8.0, 2)]
    clip_camera = [0, 1]
    video_status = ["done", "done"]
    audio_clips = [_aclip("seg0", 0.0, 5.0, -1), _aclip("seg1", 5.0, 5.0, -1)]
    audio_speed = [0.999, 1.0]
    audio_track = ["Audio", "Audio"]
    audio_status = ["done", "working"]

    entries = make_timeline_entries(
        cameras,
        clip_camera,
        video_clips,
        video_status,
        audio_clips,
        audio_speed,
        audio_track,
        audio_status,
    )

    assert len(entries) == 4
    by_track = {e["track"]: e for e in entries if e["kind"] == "video"}
    assert by_track["camA"]["row"] == 0
    assert by_track["camB"]["row"] == 1
    assert all(e["speed"] == 1.0 for e in entries if e["kind"] == "video")

    audio = [e for e in entries if e["kind"] == "audio"]
    # audio rows stacked below the two cameras
    assert all(e["row"] == 2 for e in audio)
    assert audio[0]["speed"] == 0.999
    assert audio[1]["status"] == "working"


def test_timeline_entries_separate_audio_lanes() -> None:
    cameras = [CameraGroup("camera", 1, [], [])]
    video_clips = [_vclip("v", 0.0, 10.0, 1)]
    audio_clips = [_aclip("m1", 0.0, 10.0, -1), _aclip("m2", 0.0, 10.0, -2)]
    entries = make_timeline_entries(
        cameras,
        [0],
        video_clips,
        ["done"],
        audio_clips,
        [1.0, 1.0],
        ["Audio: micA", "Audio: micB"],
        ["done", "done"],
    )
    audio = [e for e in entries if e["kind"] == "audio"]
    rows = sorted(e["row"] for e in audio)
    assert rows == [1, 2]  # two distinct audio rows below the single camera
