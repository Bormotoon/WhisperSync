"""Tests for timecode-based clip placement (no GPU/ffmpeg needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from whispersync.config import WhisperSyncConfig
from whispersync.engine.pipeline import (
    CameraGroup,
    _snap_to_word_gap,
    clip_pieces,
    compute_master_offsets,
    make_timeline_entries,
    recorder_word_gaps,
    scan_cameras,
)
from whispersync.models import AlignmentMap, Anchor, MediaClip


def test_clip_pieces_tracks_nonlinear_drift() -> None:
    """Local Time-Stretch (strategy 2) must vary its stretch per segment to follow
    the matched word times — not collapse to one global factor (the old bug)."""
    anchors = [
        Anchor(cam_time=t + 0.1 * (t % 3), rec_time=float(t), token=f"w{t}", confidence=0.9)
        for t in range(1, 11)
    ]
    am = AlignmentMap(anchors=anchors, offset=0.0, k=1.0, residual_ms=5.0)
    cfg = WhisperSyncConfig()

    _, pieces = clip_pieces(am, clip_duration=12.0, rec_duration=20.0, strategy_id=2, config=cfg)
    factors = {round(f, 4) for _, _, f in pieces}
    assert len(pieces) >= 3
    assert len(factors) > 1, "piecewise warp should differ per segment, not be global"

    # strategy 1 is a single global stretch
    _, global_pieces = clip_pieces(am, 12.0, 20.0, strategy_id=1, config=cfg)
    assert len(global_pieces) == 1


def test_recorder_word_gaps_finds_silence_midpoints() -> None:
    # word A: [1.0, 1.5], word B: [2.0, 2.5] -> silence [1.5, 2.0], midpoint 1.75
    gaps = recorder_word_gaps([(1.0, 1.5), (2.0, 2.5)])
    assert gaps == [1.75]


def test_recorder_word_gaps_ignores_touching_or_overlapping_words() -> None:
    gaps = recorder_word_gaps([(1.0, 2.0), (2.0, 3.0), (2.9, 3.5)])
    assert gaps == []


def test_snap_to_word_gap_within_range() -> None:
    assert _snap_to_word_gap(10.3, [10.5], max_snap_s=0.4) == 10.5


def test_snap_to_word_gap_too_far_is_unchanged() -> None:
    assert _snap_to_word_gap(10.0, [10.5], max_snap_s=0.4) == 10.0


def test_snap_to_word_gap_no_gaps_is_unchanged() -> None:
    assert _snap_to_word_gap(10.0, [], max_snap_s=0.4) == 10.0


def test_clip_pieces_seam_snap_moves_breakpoint_to_nearest_gap() -> None:
    # A single interior anchor at rec_time=5.0, with a recorder word-gap
    # midpoint at 5.3 (within the default snap window) — the piece boundary
    # should land on the gap, not exactly on the anchor (avoids a mid-word cut).
    anchors = [
        Anchor(cam_time=5.0, rec_time=5.0, token="w1", confidence=0.9),
    ]
    am = AlignmentMap(anchors=anchors, offset=0.0, k=1.0, residual_ms=5.0)
    cfg = WhisperSyncConfig()

    _, pieces_unsnapped = clip_pieces(
        am, clip_duration=10.0, rec_duration=20.0, strategy_id=2, config=cfg
    )
    _, pieces_snapped = clip_pieces(
        am,
        clip_duration=10.0,
        rec_duration=20.0,
        strategy_id=2,
        config=cfg,
        rec_word_gaps=[5.3],
    )
    # unsnapped: the second piece starts exactly at the anchor (5.0)
    assert abs(pieces_unsnapped[1][0] - 5.0) < 1e-9
    # snapped: the second piece starts at the nearby word gap instead
    assert abs(pieces_snapped[1][0] - 5.3) < 1e-9


def test_clip_pieces_seam_snap_respects_max_distance() -> None:
    # A gap far outside seam_snap_max_s must not move the breakpoint.
    anchors = [Anchor(cam_time=5.0, rec_time=5.0, token="w1", confidence=0.9)]
    am = AlignmentMap(anchors=anchors, offset=0.0, k=1.0, residual_ms=5.0)
    cfg = WhisperSyncConfig(seam_snap_max_s=0.4)
    _, pieces = clip_pieces(
        am, clip_duration=10.0, rec_duration=20.0, strategy_id=2, config=cfg, rec_word_gaps=[7.0]
    )
    assert abs(pieces[1][0] - 5.0) < 1e-9  # far gap ignored, boundary stays put


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


def _fake_probe(path: Path, timeout: float = 30.0):  # noqa: ANN202
    from fractions import Fraction

    from whispersync.engine.media import MediaInfo

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
    monkeypatch.setattr("whispersync.engine.pipeline.probe", _fake_probe)
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mp4").touch()

    cams, warns = scan_cameras(tmp_path, WhisperSyncConfig())
    assert len(cams) == 1
    assert cams[0].lane == 1
    assert len(cams[0].clips) == 2
    assert all(c.lane == 1 for c in cams[0].clips)
    assert warns == []


def test_scan_cameras_subfolders_become_separate_lanes(
    tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    monkeypatch.setattr("whispersync.engine.pipeline.probe", _fake_probe)
    (tmp_path / "camA").mkdir()
    (tmp_path / "camB").mkdir()
    (tmp_path / "camA" / "a1.mp4").touch()
    (tmp_path / "camA" / "a2.mp4").touch()
    (tmp_path / "camB" / "b1.mp4").touch()

    cams, warns = scan_cameras(tmp_path, WhisperSyncConfig())
    assert [c.name for c in cams] == ["camA", "camB"]
    assert cams[0].lane == 1 and cams[1].lane == 2
    assert len(cams[0].clips) == 2 and len(cams[1].clips) == 1
    assert all(c.lane == 1 for c in cams[0].clips)
    assert all(c.lane == 2 for c in cams[1].clips)
    assert warns == []


def test_scan_cameras_warns_about_ignored_root_files(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # Camera sub-folders exist AND there's a stray video file in the root —
    # that file is silently excluded from the run unless we warn about it.
    monkeypatch.setattr("whispersync.engine.pipeline.probe", _fake_probe)
    (tmp_path / "camA").mkdir()
    (tmp_path / "camA" / "a1.mp4").touch()
    (tmp_path / "stray.mp4").touch()

    cams, warns = scan_cameras(tmp_path, WhisperSyncConfig())
    assert [c.name for c in cams] == ["camA"]
    assert len(warns) == 1
    assert "stray.mp4" in warns[0]


def test_scan_cameras_empty_dir_raises(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("whispersync.engine.pipeline.probe", _fake_probe)
    with pytest.raises(RuntimeError):
        scan_cameras(tmp_path, WhisperSyncConfig())


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


def test_preliminary_offsets_end_to_end_per_camera() -> None:
    from whispersync.engine.pipeline import preliminary_offsets

    clips = [
        _vclip("a1", 0.0, 10.0, 1),
        _vclip("a2", 0.0, 8.0, 1),
        _vclip("b1", 0.0, 5.0, 2),
    ]
    clip_camera = [0, 0, 1]
    offs = preliminary_offsets(clips, clip_camera)
    # camera 0: 0, 10; camera 1 restarts at 0
    assert offs == [0.0, 10.0, 0.0]


def test_sequence_order_warning_on_disordered_offsets() -> None:
    from whispersync.engine.pipeline import sequence_order_warnings

    # natural order a1,a2 but a2 placed BEFORE a1 -> warn
    clips = [_vclip("a1", 50.0, 10.0, 1), _vclip("a2", 5.0, 10.0, 1)]
    warns = sequence_order_warnings(clips, [0, 0], [True, True])
    assert len(warns) == 1 and "a2" in warns[0]


def test_sequence_order_no_warning_when_consistent() -> None:
    from whispersync.engine.pipeline import sequence_order_warnings

    clips = [_vclip("a1", 0.0, 10.0, 1), _vclip("a2", 12.0, 10.0, 1)]
    assert sequence_order_warnings(clips, [0, 0], [True, True]) == []


def test_camera_av_offset_resolution_defaults_and_overrides() -> None:
    # config.camera_av_offset_ms_by_camera.get(camera_name, default) is the
    # exact resolution pipeline.run_pipeline uses — locks the per-camera
    # override / global-default contract without needing a full pipeline run.
    cfg = WhisperSyncConfig(
        camera_av_offset_ms=10.0,
        camera_av_offset_ms_by_camera={"camB": -5.0},
    )
    assert cfg.camera_av_offset_ms_by_camera.get("camA", cfg.camera_av_offset_ms) == 10.0
    assert cfg.camera_av_offset_ms_by_camera.get("camB", cfg.camera_av_offset_ms) == -5.0


# --- sentence-wise strategy 3 -------------------------------------------------


def _jittered_alignment(k: float, jitter: list[float], step: float = 2.0) -> AlignmentMap:
    """Anchors on the line cam = k*rec with per-anchor cam-time jitter."""
    anchors = [
        Anchor(
            cam_time=k * (i * step) + jitter[i % len(jitter)],
            rec_time=i * step,
            token=f"w{i}",
            confidence=0.9,
        )
        for i in range(1, 40)
    ]
    return AlignmentMap(anchors=anchors, offset=0.0, k=k, residual_ms=30.0)


def _speech_words(sentences: list[tuple[float, float]], word_len: float = 0.3):
    """Dense word spans filling each sentence block (gap < pause inside)."""
    words: list[tuple[float, float]] = []
    for s, e in sentences:
        t = s
        while t < e - 1e-9:
            words.append((t, min(t + word_len, e)))
            t += word_len + 0.1  # 0.1s intra-sentence gaps (below the pause gate)
    return words


def test_sentence_blocks_groups_by_pause() -> None:
    from whispersync.engine.pipeline import _sentence_blocks

    words = [(0.0, 0.3), (0.5, 0.9), (2.0, 2.4), (2.5, 3.0)]  # 1.1s pause after 0.9
    blocks = _sentence_blocks(words, min_pause_s=0.6)
    assert blocks == [(0.0, 0.9), (2.0, 3.0)]


def test_sentence_mode_cuts_only_in_pauses() -> None:
    # Two sentences [5..15] and [17..27] (2s pause), anchors every 2s with
    # ±80ms jitter. Every interior piece boundary must land OUTSIDE speech.
    sentences = [(5.0, 15.0), (17.0, 27.0)]
    words = _speech_words(sentences)
    am = _jittered_alignment(1.0, [0.08, -0.06, 0.05, -0.08])
    cfg = WhisperSyncConfig()

    lead, pieces = clip_pieces(
        am, clip_duration=40.0, rec_duration=60.0, strategy_id=3, config=cfg, rec_words=words
    )
    assert pieces, "sentence mode must produce pieces"
    interior = [s for s, _d, _f in pieces[1:]]  # every piece start except the first
    for b in interior:
        for ws, we in words:
            assert not (ws + 1e-6 < b < we - 1e-6), f"boundary {b} lands inside word {ws}-{we}"


def test_sentence_mode_speech_factors_are_smooth_despite_jitter() -> None:
    # Anchor jitter of ±80ms used to swing per-piece factors by ±5-10%; with
    # the smoothed map, SPEECH pieces must stay within a fraction of a percent
    # of the true rate (K=1.0) — pause pieces may stretch freely.
    sentences = [(5.0, 15.0), (17.0, 27.0), (29.0, 39.0)]
    words = _speech_words(sentences)
    am = _jittered_alignment(1.0, [0.08, -0.06, 0.05, -0.08])
    cfg = WhisperSyncConfig()

    _lead, pieces = clip_pieces(
        am, clip_duration=45.0, rec_duration=60.0, strategy_id=3, config=cfg, rec_words=words
    )

    def is_speech_piece(start: float, dur: float) -> bool:
        mid = start + dur / 2.0
        return any(s <= mid <= e for s, e in sentences)

    speech_factors = [f for s, d, f in pieces if is_speech_piece(s, d) and d > 1.0]
    assert speech_factors, "expected speech pieces"
    for f in speech_factors:
        assert abs(f - 1.0) < 0.005, f"speech factor {f} polluted by anchor jitter"


def test_sentence_mode_pieces_tile_contiguously() -> None:
    # No recorder content may be skipped or repeated between pieces.
    sentences = [(5.0, 15.0), (17.0, 27.0)]
    words = _speech_words(sentences)
    am = _jittered_alignment(1.0005, [0.03, -0.03])
    cfg = WhisperSyncConfig()

    _lead, pieces = clip_pieces(
        am, clip_duration=40.0, rec_duration=60.0, strategy_id=3, config=cfg, rec_words=words
    )
    for (s0, d0, _f0), (s1, _d1, _f1) in zip(pieces, pieces[1:], strict=False):
        assert abs((s0 + d0) - s1) < 1e-6


def test_sentence_mode_places_sentences_on_target() -> None:
    # True mapping cam = rec + 1.0 (offset 1s, K=1): sentence onsets must land
    # within a few ms of their true camera positions despite anchor jitter.
    sentences = [(5.0, 15.0), (17.0, 27.0)]
    words = _speech_words(sentences)
    anchors = [
        Anchor(cam_time=1.0 + i * 2.0 + j, rec_time=i * 2.0, token=f"w{i}", confidence=0.9)
        for i, j in ((i, [0.05, -0.05][i % 2]) for i in range(1, 20))
    ]
    am = AlignmentMap(anchors=anchors, offset=1.0, k=1.0, residual_ms=30.0)
    cfg = WhisperSyncConfig()

    lead, pieces = clip_pieces(
        am, clip_duration=40.0, rec_duration=60.0, strategy_id=3, config=cfg, rec_words=words
    )
    # walk the plan and find the output position of each sentence onset
    local = lead
    positions: dict[float, float] = {}
    for s, d, f in pieces:
        positions[round(s, 3)] = local
        local += d / f
    for s, _e in sentences:
        # sentence piece starts at s - pad (0.08); its true cam position is s + 1 - pad
        key = round(s - 0.08, 3)
        assert key in positions
        assert abs(positions[key] - (s + 1.0 - 0.08)) < 0.03


def test_sentence_mode_without_words_falls_back_to_thinning() -> None:
    am = _jittered_alignment(1.0, [0.0])
    cfg = WhisperSyncConfig()
    _lead, pieces = clip_pieces(
        am, clip_duration=40.0, rec_duration=60.0, strategy_id=3, config=cfg
    )
    assert pieces  # old path still works when rec_words aren't available


def test_strategy2_snap_moves_both_sides_keeping_factors_stable() -> None:
    # Snapping a boundary 0.3s into a pause used to change one neighbour's
    # INPUT length while both OUTPUT lengths stayed put — factors jumped to
    # ~1.3/0.7 around every snapped seam. Moving the camera side along with
    # the recorder side keeps factors at the true rate.
    anchors = [
        Anchor(cam_time=float(t), rec_time=float(t), token=f"w{t}", confidence=0.9)
        for t in (2.0, 5.0, 8.0)
    ]
    am = AlignmentMap(anchors=anchors, offset=0.0, k=1.0, residual_ms=5.0)
    cfg = WhisperSyncConfig()
    _lead, pieces = clip_pieces(
        am,
        clip_duration=10.0,
        rec_duration=20.0,
        strategy_id=2,
        config=cfg,
        rec_word_gaps=[5.3],  # snap the middle boundary 0.3s to the right
    )
    assert any(abs(s - 5.3) < 1e-9 for s, _d, _f in pieces)  # snap happened
    for _s, _d, f in pieces:
        assert abs(f - 1.0) < 0.01, f"factor {f} destabilized by one-sided snap"
