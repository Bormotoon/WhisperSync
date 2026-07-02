"""Tests for Boundary Flex and pause ducking (pure logic, no ffmpeg)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from whispersync.config import WhisperSyncConfig
from whispersync.engine import acoustic
from whispersync.engine.pipeline import pause_spans_local
from whispersync.engine.timestretch import _duck_pause_expr
from whispersync.models import AlignmentMap, Anchor


def _am(cam_times: list[float]) -> AlignmentMap:
    anchors = [Anchor(cam_time=t, rec_time=t + 100.0, token="x", confidence=0.9) for t in cam_times]
    return AlignmentMap(anchors=anchors, offset=100.0, k=1.0, residual_ms=5.0)


# --- pause_spans_local -----------------------------------------------------


def test_pause_spans_detects_interior_gap() -> None:
    # speech at 1,2,3 then a 5s gap, then 8,9
    am = _am([1.0, 2.0, 3.0, 8.0, 9.0])
    spans = pause_spans_local(am, clip_duration=10.0, gap_threshold=0.6, min_pause=0.6)
    # interior gap 3->8 is a pause; head 0->1 and tail 9->10 are < min? head=1.0>=0.6 yes
    assert (3.0, 8.0) in spans
    assert (0.0, 1.0) in spans  # head pause
    # tail 9->10 == 1.0s >= min
    assert (9.0, 10.0) in spans


def test_pause_spans_ignores_short_gaps_and_clamps() -> None:
    am = _am([1.0, 1.3, 1.6, 5.0])  # 0.3s gaps are not pauses; 1.6->5.0 is
    spans = pause_spans_local(am, clip_duration=6.0, gap_threshold=0.6, min_pause=0.6)
    assert (1.6, 5.0) in spans
    # no sub-threshold gaps leaked in
    assert all(b - a >= 0.6 for a, b in spans)
    # all spans within [0, clip_duration]
    assert all(0.0 <= a < b <= 6.0 for a, b in spans)


def test_pause_spans_empty_without_anchors() -> None:
    am = AlignmentMap(anchors=[], offset=0.0, k=1.0, residual_ms=0.0)
    assert pause_spans_local(am, 10.0, 0.6, 0.6) == []


# --- pause_spans_local (word-based: duck only where BOTH tracks are silent) --


def test_pause_spans_word_based_requires_both_silent() -> None:
    # k=1, offset=0 -> recorder time == local time, so this is easy to reason about.
    am = AlignmentMap(anchors=[], offset=0.0, k=1.0, residual_ms=0.0)
    # camera has a real gap [3, 8) with no words (a real pause)
    cam_words = [(0.0, 1.0), (2.0, 3.0), (8.0, 9.0)]
    # recorder has a LOW-CONFIDENCE word at [4, 5) inside that window that never
    # became an anchor — it must NOT be ducked, because the recorder is not
    # actually silent there (PROJECT_ANALYSIS.md §2.5).
    rec_words = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (8.0, 9.0)]
    spans = pause_spans_local(
        am,
        clip_duration=10.0,
        gap_threshold=0.6,
        min_pause=0.6,
        cam_words=cam_words,
        rec_words=rec_words,
        rec_duration=10.0,
    )
    # the [3,8) camera gap is split by the recorder's word at [4,5) into two
    # sub-pauses where BOTH tracks are silent: [3,4) and [5,8).
    assert (3.0, 4.0) in spans
    assert (5.0, 8.0) in spans
    # no span covers the recorder's actual word
    assert not any(a < 4.5 < b for a, b in spans)


def test_pause_spans_word_based_no_overlap_means_no_pause() -> None:
    am = AlignmentMap(anchors=[], offset=0.0, k=1.0, residual_ms=0.0)
    # camera silent [3, 8); recorder has continuous speech through that window
    # AND through the tail, so there is no time where both tracks are silent.
    cam_words = [(0.0, 1.0), (2.0, 3.0), (8.0, 10.0)]
    rec_words = [(0.0, 10.0)]  # one long word/segment spanning the whole clip
    spans = pause_spans_local(
        am,
        clip_duration=10.0,
        gap_threshold=0.6,
        min_pause=0.6,
        cam_words=cam_words,
        rec_words=rec_words,
        rec_duration=10.0,
    )
    assert spans == []


# --- _duck_pause_expr ------------------------------------------------------


def test_duck_expr_references_edges_and_level() -> None:
    # one pause [2,4], duck to 0.1 linear, 0.1s fades
    expr = _duck_pause_expr(2.0, 4.0, duck_lin=0.1, fade=0.1)
    assert "0.100000" in expr  # the duck level
    assert expr.count("if(") == 4  # nested 4-way if for one pause
    # edges: a-fade=1.9, a=2.0, b=4.0, b+fade=4.1 appear in the expression
    assert "1.9000" in expr and "4.1000" in expr


# --- refine_piece_boundaries (monkeypatched, no ffmpeg) --------------------


def test_refine_shifts_only_confident_beyond_deadband(monkeypatch) -> None:
    cfg = WhisperSyncConfig(
        boundary_flex=True,
        flex_window_s=4.0,
        flex_min_sharpness=80.0,
        flex_deadband_s=0.025,
        flex_max_shift_s=0.15,
        acoustic_max_lag_s=1.0,
    )
    # three contiguous pieces, factor 1.0, 5s each
    pieces = [(100.0, 5.0, 1.0), (105.0, 5.0, 1.0), (110.0, 5.0, 1.0)]

    # stub out track decoding (in-memory Boundary Flex loads each full track
    # once — see acoustic.load_mono16k_track) so no ffmpeg/files are needed
    fake_track = np.zeros(700 * 16000)
    monkeypatch.setattr(acoustic, "load_mono16k_track", lambda p: fake_track)

    # piece 0: confident, big lag → shift; piece 1: confident but tiny lag → deadband, no shift;
    # piece 2: low sharpness → no shift.
    calls = {"i": 0}

    def fake_gcc(cam, rec, sr, max_lag, eps):
        i = calls["i"]
        calls["i"] += 1
        return [(-0.08, 200.0), (-0.005, 200.0), (-0.08, 10.0)][i]

    monkeypatch.setattr(acoustic, "gcc_phat", fake_gcc)

    refined = acoustic.refine_piece_boundaries(
        pieces,
        lead=0.0,
        cam_audio_wav=Path("c"),
        rec_audio_path=Path("r"),
        clip_duration=15.0,
        rec_duration=600.0,
        config=cfg,
    )
    # piece 0 start moved by +0.08 (=-lag); 1 and 2 unchanged
    assert abs(refined[0][0] - 100.08) < 1e-6
    assert abs(refined[1][0] - 105.0) < 1e-6
    assert abs(refined[2][0] - 110.0) < 1e-6
    # durations/factors preserved
    assert all(p[1] == 5.0 and p[2] == 1.0 for p in refined)


def test_refine_clamps_to_max_shift(monkeypatch) -> None:
    cfg = WhisperSyncConfig(boundary_flex=True, flex_max_shift_s=0.15, flex_min_sharpness=80.0)
    pieces = [(100.0, 5.0, 1.0)]
    fake_track = np.zeros(700 * 16000)
    monkeypatch.setattr(acoustic, "load_mono16k_track", lambda p: fake_track)
    monkeypatch.setattr(acoustic, "gcc_phat", lambda *a, **k: (-1.0, 300.0))  # huge lag
    refined = acoustic.refine_piece_boundaries(pieces, 0.0, Path("c"), Path("r"), 15.0, 600.0, cfg)
    # clamped to +0.15, not +1.0
    assert abs(refined[0][0] - 100.15) < 1e-6
