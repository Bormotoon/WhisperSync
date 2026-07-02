"""Tests for the acoustic fallback ("Strategy 0") — pure logic with
monkeypatched track loading/correlation, no ffmpeg. Verified separately
against real ffmpeg-generated synthetic audio during development (see
PROJECT_ANALYSIS.md §10.2); these tests lock the geometry/sign conventions
and the pipeline integration without needing real media files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from whispersync.config import WhisperSyncConfig
from whispersync.engine import acoustic
from whispersync.engine.pipeline import _try_acoustic_fallback
from whispersync.models import AlignmentMap


def test_acoustic_coarse_align_recovers_offset_and_k(monkeypatch) -> None:
    # Fake tracks (contents don't matter, gcc_phat and _window_slice are both
    # monkeypatched); simulate a clip whose content matches the recorder at
    # rec_time = cam_time + 5 (offset=-5, k=1). _window_slice is faked to
    # return the requested center time itself (as a 1-element array) so the
    # fake gcc_phat below can read back "which cam/rec time is being probed"
    # and score exactly the correct (cam_time, rec_time) pair as the sharp
    # match — everything else scores low, so the picked point is unambiguous.
    monkeypatch.setattr(acoustic, "load_mono16k_track", lambda p: np.zeros(1))

    def fake_window_slice(track, center_s, win_s, sr):
        return np.array([center_s])

    monkeypatch.setattr(acoustic, "_window_slice", fake_window_slice)

    def fake_gcc(cam_win, rec_win, sr, max_lag, eps):
        cam_t, rec_t = float(cam_win[0]), float(rec_win[0])
        # true match is rec_t == cam_t + 5 (offset=-5, k=1); score it sharp,
        # everything else low, with lag=0 so best_rec_time = the probed t_rec.
        if abs(rec_t - (cam_t + 5.0)) < 1e-6:
            return (0.0, 300.0)
        return (0.0, 5.0)

    monkeypatch.setattr(acoustic, "gcc_phat", fake_gcc)

    result = acoustic.acoustic_coarse_align(
        Path("cam.wav"),
        Path("rec.wav"),
        clip_duration=20.0,
        rec_duration=40.0,
        grid_s=5.0,
        window_s=4.0,
        min_sharpness=50.0,
    )
    assert result is not None
    offset, k = result
    assert abs(offset - (-5.0)) < 1e-6
    assert abs(k - 1.0) < 1e-6


def test_acoustic_coarse_align_no_confident_points_returns_none(monkeypatch) -> None:
    fake_track = np.zeros(50 * 16000)
    monkeypatch.setattr(acoustic, "load_mono16k_track", lambda p: fake_track)
    monkeypatch.setattr(acoustic, "gcc_phat", lambda *a, **k: (0.0, 5.0))  # always below gate

    result = acoustic.acoustic_coarse_align(
        Path("cam.wav"),
        Path("rec.wav"),
        clip_duration=20.0,
        rec_duration=30.0,
        grid_s=5.0,
        window_s=4.0,
        min_sharpness=50.0,
    )
    assert result is None


def test_try_acoustic_fallback_wraps_result_in_alignment_map(monkeypatch) -> None:
    monkeypatch.setattr(
        "whispersync.engine.pipeline.acoustic_coarse_align", lambda *a, **k: (-5.0, 1.0)
    )
    cfg = WhisperSyncConfig()
    am = _try_acoustic_fallback(Path("clip.wav"), 20.0, Path("rec.wav"), 40.0, cfg)
    assert isinstance(am, AlignmentMap)
    assert am.anchors == []  # no text breakpoints -> clip_pieces uses one global stretch
    assert am.offset == -5.0
    assert am.k == 1.0


def test_try_acoustic_fallback_returns_none_when_no_match(monkeypatch) -> None:
    monkeypatch.setattr("whispersync.engine.pipeline.acoustic_coarse_align", lambda *a, **k: None)
    cfg = WhisperSyncConfig()
    am = _try_acoustic_fallback(Path("clip.wav"), 20.0, Path("rec.wav"), 40.0, cfg)
    assert am is None


def test_try_acoustic_fallback_swallows_ffmpeg_errors(monkeypatch) -> None:
    def raise_runtime(*a, **k):
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr("whispersync.engine.pipeline.acoustic_coarse_align", raise_runtime)
    cfg = WhisperSyncConfig()
    am = _try_acoustic_fallback(Path("clip.wav"), 20.0, Path("rec.wav"), 40.0, cfg)
    assert am is None


def test_acoustic_fallback_config_defaults() -> None:
    cfg = WhisperSyncConfig()
    assert cfg.acoustic_fallback is True
    assert cfg.acoustic_fallback_grid_s > 0
    assert cfg.acoustic_fallback_window_s > 0


@pytest.mark.parametrize("min_sharpness", [10.0, 200.0])
def test_acoustic_fallback_min_sharpness_is_configurable(min_sharpness: float) -> None:
    cfg = WhisperSyncConfig(acoustic_fallback_min_sharpness=min_sharpness)
    assert cfg.acoustic_fallback_min_sharpness == min_sharpness
