"""Tier-2 acoustic micro-alignment ("Flex Time").

Whisper word timings are only ±50–100 ms accurate, and the piecewise warp can only
be as good as the anchors feeding it. This pass re-measures the true recorder↔camera
offset directly from the audio waveform — independent of the transcript — using
PHAT-weighted cross-correlation (GCC-PHAT), which is robust to the different mics
and reverb of camera vs recorder.

It samples a grid of points across the clip, cross-correlates a short window of the
camera's own audio against the recorder audio at each point, and builds a piecewise
correction curve from the confident measurements. The curve corrects the anchor
recorder-times sub-sample and injects synthetic anchors at the grid points, so the
downstream warp tracks the real (non-linear) drift even between sparse text anchors.
Windows whose correlation peak is not sharp (silence, wind, music) are rejected and
that region falls back to the text map (zero correction).
"""

from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path

import numpy as np

from whispersync.config import WhisperSyncConfig
from whispersync.engine.media import extract_audio_window
from whispersync.models import Anchor

logger = logging.getLogger(__name__)

_REFINE_SR = 16000  # all acoustic windows are cut to mono 16 kHz


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


def _measure_grid(
    cam_audio_wav: Path,
    rec_audio_path: Path,
    am_offset: float,
    am_k: float,
    clip_duration: float,
    rec_duration: float,
    config: WhisperSyncConfig,
    tmp_dir: Path,
) -> list[tuple[float, float, float]]:
    """Cross-correlate camera vs recorder on a time grid across the clip.

    Returns accepted ``(cam_time, correction_s, sharpness)`` points, where
    ``correction_s`` is how much later (positive) the recorder audio actually lands
    versus the Tier-1 line prediction — to be added to predicted recorder times.
    """
    win = config.acoustic_window_s
    grid = config.acoustic_grid_s
    half = win / 2.0

    def cam_to_rec(t_cam: float) -> float:
        return (t_cam - am_offset) / am_k if am_k else t_cam

    points: list[tuple[float, float, float]] = []
    t = half
    idx = 0
    while t <= clip_duration - half:
        t_rec = cam_to_rec(t)
        if half <= t_rec <= rec_duration - half:
            cam_win = tmp_dir / f"cam_{idx:04d}.wav"
            rec_win = tmp_dir / f"rec_{idx:04d}.wav"
            try:
                extract_audio_window(cam_audio_wav, cam_win, t - half, win, _REFINE_SR)
                extract_audio_window(rec_audio_path, rec_win, t_rec - half, win, _REFINE_SR)
                cam_sig, _ = read_wav_mono16k(cam_win)
                rec_sig, _ = read_wav_mono16k(rec_win)
                # ref=camera, query=recorder window centered on the PREDICTED rec
                # time. gcc_phat returns lag such that a query whose content sits
                # later than ref gives a NEGATIVE lag (verified). So the recorder
                # content actually sits at t_rec + (-lag): the correction to add to
                # the recorder read time is -lag_s.
                lag_s, sharp = gcc_phat(
                    cam_sig, rec_sig, _REFINE_SR, config.acoustic_max_lag_s, config.gcc_eps
                )
                correction_s = -lag_s
            except (RuntimeError, ValueError) as e:
                logger.debug("acoustic grid point %d failed: %s", idx, e)
                correction_s, sharp = 0.0, 0.0
            finally:
                cam_win.unlink(missing_ok=True)
                rec_win.unlink(missing_ok=True)
            if sharp >= config.acoustic_min_sharpness:
                points.append((t, correction_s, sharp))
        t += grid
        idx += 1
    logger.info("Acoustic refine: %d confident grid point(s) (of %d sampled)", len(points), idx)
    return points


def _correction_at(points: list[tuple[float, float, float]], t_cam: float) -> float:
    """Piecewise-linear correction (seconds) at a camera time, interpolating between
    confident grid points and flat-extrapolating at the ends. Empty ⇒ 0 (fall back
    to the text map)."""
    if not points:
        return 0.0
    if t_cam <= points[0][0]:
        return points[0][1]
    if t_cam >= points[-1][0]:
        return points[-1][1]
    # points are sorted by cam_time; find the bracketing pair.
    for (t0, c0, _), (t1, c1, _) in zip(points, points[1:], strict=False):
        if t0 <= t_cam <= t1:
            frac = (t_cam - t0) / (t1 - t0) if t1 > t0 else 0.0
            return c0 + frac * (c1 - c0)
    return points[-1][1]


def refine_anchors(
    anchors: list[Anchor],
    cam_audio_wav: Path,
    rec_audio_path: Path,
    am_offset: float,
    am_k: float,
    clip_duration: float,
    rec_duration: float,
    config: WhisperSyncConfig,
    tmp_dir: Path | None = None,
) -> list[Anchor]:
    """Correct anchor recorder-times by acoustic cross-correlation and inject
    synthetic grid anchors. ``am_offset``/``am_k`` are left to the caller — only
    ``anchors`` is enriched, so timeline placement is unaffected.
    """
    if not anchors:
        return anchors

    own_tmp = tmp_dir is None
    if own_tmp:
        tmp_dir = Path(tempfile.mkdtemp(prefix="whispersync_acoustic_"))
    assert tmp_dir is not None
    try:
        points = _measure_grid(
            cam_audio_wav,
            rec_audio_path,
            am_offset,
            am_k,
            clip_duration,
            rec_duration,
            config,
            tmp_dir,
        )
    finally:
        if own_tmp:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    if not points:
        return anchors  # nothing confident: keep the text anchors untouched

    inv_k = 1.0 / am_k if am_k else 1.0

    # Correct existing text anchors. A camera-time correction c(t) shifts the
    # recorder time by c/k (since cam = offset + k*rec).
    corrected: list[Anchor] = []
    for a in anchors:
        c = _correction_at(points, a.cam_time)
        corrected.append(
            Anchor(
                cam_time=a.cam_time,
                rec_time=a.rec_time + c * inv_k,
                token=a.token,
                confidence=a.confidence,
            )
        )

    # Inject synthetic anchors at the grid points so the warp has dense breakpoints
    # even where the text tier had none. Their recorder time is the predicted line
    # plus the measured correction; confidence scales with peak sharpness.
    for t_cam, correction_s, sharp in points:
        t_rec_pred = (t_cam - am_offset) / am_k if am_k else t_cam
        corrected.append(
            Anchor(
                cam_time=t_cam,
                rec_time=t_rec_pred + correction_s * inv_k,
                token="~gcc",
                confidence=min(1.0, sharp / 300.0),
            )
        )

    # Re-sort by recorder time and drop any anchor that breaks monotonicity (same
    # guard the difflib matcher applies).
    corrected.sort(key=lambda a: a.rec_time)
    monotonic: list[Anchor] = []
    last_cam = last_rec = -float("inf")
    for a in corrected:
        if a.cam_time > last_cam and a.rec_time > last_rec:
            monotonic.append(a)
            last_cam, last_rec = a.cam_time, a.rec_time
    return monotonic


# (rec_start, rec_in_duration, atempo_factor) — the piece tuple produced by clip_pieces.
Piece = tuple[float, float, float]


def refine_piece_boundaries(
    pieces: list[Piece],
    lead: float,
    cam_audio_wav: Path,
    rec_audio_path: Path,
    clip_duration: float,
    rec_duration: float,
    config: WhisperSyncConfig,
    tmp_dir: Path | None = None,
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
    """
    if not pieces:
        return pieces

    win = config.flex_window_s
    own_tmp = tmp_dir is None
    if own_tmp:
        tmp_dir = Path(tempfile.mkdtemp(prefix="whispersync_flex_"))
    assert tmp_dir is not None

    refined: list[Piece] = []
    n_shifted = 0
    try:
        local = lead
        for idx, (rec_start, rec_dur, factor) in enumerate(pieces):
            out_dur = rec_dur / factor if factor else rec_dur
            # Probe a window centered a little into the piece (avoid the seam itself).
            probe = min(out_dur, win)
            cam_mid = local + probe / 2.0
            rec_mid = rec_start + min(rec_dur, win) / 2.0
            corr = 0.0
            if (
                win / 2.0 <= cam_mid <= clip_duration - win / 2.0
                and win / 2.0 <= rec_mid <= rec_duration - win / 2.0
            ):
                cam_win = tmp_dir / f"flex_cam_{idx:05d}.wav"
                rec_win = tmp_dir / f"flex_rec_{idx:05d}.wav"
                try:
                    extract_audio_window(
                        cam_audio_wav, cam_win, cam_mid - win / 2.0, win, _REFINE_SR
                    )
                    extract_audio_window(
                        rec_audio_path, rec_win, rec_mid - win / 2.0, win, _REFINE_SR
                    )
                    cam_sig, _ = read_wav_mono16k(cam_win)
                    rec_sig, _ = read_wav_mono16k(rec_win)
                    lag_s, sharp = gcc_phat(
                        cam_sig, rec_sig, _REFINE_SR, config.acoustic_max_lag_s, config.gcc_eps
                    )
                    # -lag_s = seconds to add to the recorder read time (see _measure_grid).
                    if sharp >= config.flex_min_sharpness and abs(lag_s) > config.flex_deadband_s:
                        corr = max(-config.flex_max_shift_s, min(config.flex_max_shift_s, -lag_s))
                        n_shifted += 1
                except (RuntimeError, ValueError) as e:
                    logger.debug("flex boundary %d failed: %s", idx, e)
                finally:
                    cam_win.unlink(missing_ok=True)
                    rec_win.unlink(missing_ok=True)
            # Keep the shifted start within the recorder bounds.
            new_start = max(0.0, min(rec_start + corr, rec_duration - rec_dur))
            refined.append((new_start, rec_dur, factor))
            local += out_dur
    finally:
        if own_tmp:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Boundary Flex: nudged %d/%d piece starts", n_shifted, len(pieces))
    return refined
