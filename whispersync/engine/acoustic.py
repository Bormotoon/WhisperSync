"""Acoustic cross-correlation for sub-frame lip-sync ("Boundary Flex").

Whisper word timings are only ±50–100 ms accurate. This module measures the true
recorder<->camera lag directly from the audio waveform — independent of the
transcript — using PHAT-weighted cross-correlation (GCC-PHAT, ``gcc_phat``), which
is robust to the different mics and reverb of camera vs recorder.

``refine_piece_boundaries`` (Boundary Flex) uses this to acoustically nudge each
rendered piece's recorder start so speech lands under the picture to sub-frame
accuracy, independent of Whisper's word timings. A measurement is only trusted when
its cross-correlation peak is confidently sharp (silence/wind/music windows are
rejected) and the residual exceeds a deadband, so it only removes real drift rather
than injecting GCC measurement noise.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import tempfile
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from whispersync.config import WhisperSyncConfig
from whispersync.engine.media import extract_audio_window

logger = logging.getLogger(__name__)

_REFINE_SR = 16000  # all acoustic windows are cut to mono 16 kHz


def _pool_context() -> Any:
    """'fork' context for the boundary-measurement pool (fast, no __main__ re-import);
    falls back to the default context where fork is unavailable."""
    try:
        return mp.get_context("fork")
    except (ValueError, RuntimeError):  # pragma: no cover - platform dependent
        return mp.get_context()


def read_wav_mono16k(path: Path) -> tuple[np.ndarray, int]:
    """Read a small PCM WAV (as produced by ``extract_audio_window``) into a float
    array in [-1, 1]. Uses the stdlib ``wave`` module — no scipy/soundfile dep."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        sampwidth = w.getsampwidth()
        nchannels = w.getnchannels()
        raw = w.readframes(n)
    if sampwidth != 2:
        raise ValueError(f"expected pcm_s16le (2-byte) wav, got sampwidth={sampwidth}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    # If a window slipped through as stereo, fold to mono.
    if nchannels == 2 and data.size:
        data = data.reshape(-1, 2).mean(axis=1)
    return data / 32768.0, sr


def gcc_phat(
    sig_ref: np.ndarray, sig_query: np.ndarray, sr: int, max_lag_s: float, eps: float
) -> tuple[float, float]:
    """PHAT-weighted cross-correlation lag between two signals.

    Returns ``(lag_seconds, sharpness)`` where ``lag`` is the shift to ADD to the
    query's time so it aligns with the reference (positive ⇒ query currently lags;
    its event happens later in the query than in the reference), and ``sharpness``
    = peak / median(|cc|) is a confidence score (≈240–335 for clear speech windows,
    ≈12 for silence/uncorrelated — gate around 50). Sub-sample accurate via
    parabolic interpolation around the integer peak.
    """
    n = min(len(sig_ref), len(sig_query))
    if n < 8:
        return 0.0, 0.0
    a = sig_ref[:n] - sig_ref[:n].mean()
    b = sig_query[:n] - sig_query[:n].mean()
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    A = np.fft.rfft(a, nfft)  # noqa: N806 — conventional FFT notation
    B = np.fft.rfft(b, nfft)  # noqa: N806
    R = A * np.conj(B)  # noqa: N806
    R /= np.abs(R) + eps  # PHAT whitening  # noqa: N806
    cc = np.fft.irfft(R, nfft)
    # Reorder so index n-1 is zero lag, spanning lags [-(n-1) .. n-1].
    cc = np.concatenate((cc[-(n - 1) :], cc[:n]))
    abs_cc = np.abs(cc)

    # Restrict the peak search to ±max_lag.
    max_lag = int(min(n - 1, round(max_lag_s * sr)))
    center = n - 1
    lo = center - max_lag
    hi = center + max_lag
    window = abs_cc[lo : hi + 1]
    if window.size == 0:
        return 0.0, 0.0
    rel_peak = int(np.argmax(window))
    peak = lo + rel_peak

    median = float(np.median(abs_cc)) or eps
    sharpness = float(abs_cc[peak] / median)

    # Parabolic sub-sample interpolation around the integer peak.
    delta = 0.0
    if 0 < peak < len(abs_cc) - 1:
        y0, y1, y2 = abs_cc[peak - 1], abs_cc[peak], abs_cc[peak + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom

    lag_samples = (peak - center) + delta
    return lag_samples / sr, sharpness


# (rec_start, rec_in_duration, atempo_factor) — the piece tuple produced by clip_pieces.
Piece = tuple[float, float, float]


def _measure_boundary(
    idx: int,
    cam_mid: float,
    rec_mid: float,
    cam_audio_wav: Path,
    rec_audio_path: Path,
    tmp_dir: Path,
    win: float,
    max_lag_s: float,
    eps: float,
    min_sharpness: float,
    deadband_s: float,
    max_shift_s: float,
) -> tuple[int, float]:
    """Measure one boundary's acoustic correction (seconds to add to rec_start).

    Module-level and self-contained (writes only its own indexed scratch files) so it
    runs in a process pool. Returns ``(idx, correction)``; correction is 0.0 when the
    peak is not confident or the residual is within the deadband.
    """
    cam_win = tmp_dir / f"flex_cam_{idx:05d}.wav"
    rec_win = tmp_dir / f"flex_rec_{idx:05d}.wav"
    try:
        extract_audio_window(cam_audio_wav, cam_win, cam_mid - win / 2.0, win, _REFINE_SR)
        extract_audio_window(rec_audio_path, rec_win, rec_mid - win / 2.0, win, _REFINE_SR)
        cam_sig, _ = read_wav_mono16k(cam_win)
        rec_sig, _ = read_wav_mono16k(rec_win)
        lag_s, sharp = gcc_phat(cam_sig, rec_sig, _REFINE_SR, max_lag_s, eps)
        # gcc_phat's lag is the shift to add to the query's (recorder's) time to
        # align it with the reference (camera); -lag_s is therefore the seconds
        # to add to the recorder read time so the cut lands under the picture.
        if sharp >= min_sharpness and abs(lag_s) > deadband_s:
            return idx, max(-max_shift_s, min(max_shift_s, -lag_s))
        return idx, 0.0
    except (RuntimeError, ValueError) as e:
        logger.debug("flex boundary %d failed: %s", idx, e)
        return idx, 0.0
    finally:
        cam_win.unlink(missing_ok=True)
        rec_win.unlink(missing_ok=True)


def refine_piece_boundaries(
    pieces: list[Piece],
    lead: float,
    cam_audio_wav: Path,
    rec_audio_path: Path,
    clip_duration: float,
    rec_duration: float,
    config: WhisperSyncConfig,
    tmp_dir: Path | None = None,
    workers: int = 1,
) -> list[Piece]:
    """Acoustically nudge each piece's recorder start so its speech lands under the
    picture, independent of Whisper's word timings ("Boundary Flex").

    Pieces are contiguous in OUTPUT (camera) time, beginning at ``lead``: piece i's
    local start is ``lead + sum(out_dur[<i])`` where ``out_dur = rec_dur / factor``.
    For each piece we cross-correlate a short camera window at that local time against
    the recorder window at the piece's current ``rec_start``; if the peak is sharp and
    the measured residual exceeds the deadband, we shift ``rec_start`` by it (clamped).
    The piece's duration/factor are left unchanged, so the exact-length contract and
    the overall warp are preserved — only WHERE in the recorder we cut from moves.

    The independent per-boundary measurements are spread across ``workers`` processes;
    the geometry (which window each boundary probes) is deterministic, so the result is
    identical regardless of worker count.
    """
    if not pieces:
        return pieces

    win = config.flex_window_s
    own_tmp = tmp_dir is None
    if own_tmp:
        tmp_dir = Path(tempfile.mkdtemp(prefix="whispersync_flex_"))
    assert tmp_dir is not None

    # First pass (cheap, sequential): the window geometry for every in-bounds boundary.
    jobs: list[tuple[int, float, float]] = []  # (idx, cam_mid, rec_mid)
    local = lead
    for idx, (rec_start, rec_dur, factor) in enumerate(pieces):
        out_dur = rec_dur / factor if factor else rec_dur
        probe = min(out_dur, win)
        cam_mid = local + probe / 2.0
        rec_mid = rec_start + min(rec_dur, win) / 2.0
        if (
            win / 2.0 <= cam_mid <= clip_duration - win / 2.0
            and win / 2.0 <= rec_mid <= rec_duration - win / 2.0
        ):
            jobs.append((idx, cam_mid, rec_mid))
        local += out_dur

    # Second pass (slow ffmpeg+GCC): measure each boundary, in parallel when asked.
    corrections: dict[int, float] = {}
    args = (
        cam_audio_wav,
        rec_audio_path,
        tmp_dir,
        win,
        config.acoustic_max_lag_s,
        config.gcc_eps,
        config.flex_min_sharpness,
        config.flex_deadband_s,
        config.flex_max_shift_s,
    )
    try:
        if workers <= 1 or len(jobs) <= 1:
            for idx, cam_mid, rec_mid in jobs:
                i, corr = _measure_boundary(idx, cam_mid, rec_mid, *args)
                corrections[i] = corr
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=_pool_context()) as pool:
                futs = [
                    pool.submit(_measure_boundary, idx, cam_mid, rec_mid, *args)
                    for idx, cam_mid, rec_mid in jobs
                ]
                for fut in futs:
                    i, corr = fut.result()
                    corrections[i] = corr
    finally:
        if own_tmp:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Apply corrections, clamping each shifted start within the recorder bounds.
    refined: list[Piece] = []
    n_shifted = 0
    for idx, (rec_start, rec_dur, factor) in enumerate(pieces):
        corr = corrections.get(idx, 0.0)
        if corr != 0.0:
            n_shifted += 1
        new_start = max(0.0, min(rec_start + corr, rec_duration - rec_dur))
        refined.append((new_start, rec_dur, factor))

    logger.info("Boundary Flex: nudged %d/%d piece starts", n_shifted, len(pieces))
    return refined
