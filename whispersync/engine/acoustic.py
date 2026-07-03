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
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from whispersync.config import WhisperSyncConfig
from whispersync.engine.media import extract_audio_to_wav

logger = logging.getLogger(__name__)

_REFINE_SR = 16000  # all acoustic analysis happens on mono 16 kHz


def read_wav_mono16k(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV into a float array in [-1, 1]. Uses the stdlib ``wave``
    module — no scipy/soundfile dep."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        sampwidth = w.getsampwidth()
        nchannels = w.getnchannels()
        raw = w.readframes(n)
    if sampwidth != 2:
        raise ValueError(f"expected pcm_s16le (2-byte) wav, got sampwidth={sampwidth}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    # If the file is stereo, fold to mono.
    if nchannels == 2 and data.size:
        data = data.reshape(-1, 2).mean(axis=1)
    return data / 32768.0, sr


def load_mono16k_track(path: Path) -> np.ndarray:
    """Decode an entire audio file (any format/channel layout ffmpeg reads) to a
    mono 16 kHz float array, once. Boundary Flex used to re-run ffmpeg (via
    ``extract_audio_window``) for every single boundary it measured — for a
    clip with hundreds of pieces that's hundreds of short-lived ffmpeg
    processes just to cut small windows. Decoding each full track exactly once
    and slicing the resulting numpy array for every window instead removes
    that spawn overhead entirely (the FFT-based ``gcc_phat`` cost dominates
    once ffmpeg is out of the loop). See PROJECT_ANALYSIS.md §6.2.
    """
    import tempfile

    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    import os

    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        extract_audio_to_wav(path, tmp_path, sample_rate=_REFINE_SR, mono=True)
        sig, _ = read_wav_mono16k(tmp_path)
        return sig
    finally:
        tmp_path.unlink(missing_ok=True)


def _window_slice(track: np.ndarray, center_s: float, win_s: float, sr: int) -> np.ndarray:
    """A ``win_s``-second slice of ``track`` centered on ``center_s``, clamped to
    the track's bounds (shorter at the edges rather than raising)."""
    half = int(round(win_s / 2.0 * sr))
    c = int(round(center_s * sr))
    lo = max(0, c - half)
    hi = min(len(track), c + half)
    return track[lo:hi]


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


def acoustic_coarse_align(
    cam_audio_wav: Path,
    rec_audio_path: Path,
    clip_duration: float,
    rec_duration: float,
    grid_s: float = 30.0,
    window_s: float = 8.0,
    max_lag_s: float = 1.0,
    min_sharpness: float = 50.0,
    gcc_eps: float = 1e-8,
) -> tuple[float, float] | None:
    """Acoustic fallback offset/K estimate when there's no usable transcript
    match (too little speech, music, a foreign language Whisper garbles, or
    near-silence) — the alignment paths in ``matcher.py`` all fail without at
    least a couple of matched words. This works directly on the waveforms,
    exactly like Boundary Flex, but coarsely: cross-correlate a window of the
    camera's own audio against the recorder at each point on a grid across
    the WHOLE recorder span (assuming the clip could start anywhere in it),
    then fit an ``offset, K`` line through the confident (sharp-peak) points
    the same way ``ransac_linear_fit`` does for text anchors.

    Returns ``(offset, k)`` such that ``t_cam = offset + k * t_rec``, or
    ``None`` if too few grid points were confident to fit a line. This is
    the "Strategy 0" acoustic fallback (see PROJECT_ANALYSIS.md §10.2): it
    turns WhisperSync from "works when there's transcribable speech" into
    "works on anything with correlated audio between the two tracks" —
    music, ambient noise, or a language Whisper can't transcribe well, as
    long as the same physical event reaches both mics.
    """
    cam_track = load_mono16k_track(cam_audio_wav)
    rec_track = load_mono16k_track(rec_audio_path)

    half = window_s / 2.0
    points: list[tuple[float, float]] = []  # (cam_time, rec_time) of confident matches
    t_cam = half
    while t_cam <= clip_duration - half:
        t_rec = half
        best_sharp = 0.0
        best_rec_time: float | None = None
        while t_rec <= rec_duration - half:
            cam_win = _window_slice(cam_track, t_cam, window_s, _REFINE_SR)
            rec_win = _window_slice(rec_track, t_rec, window_s, _REFINE_SR)
            # ref=camera window (fixed content), query=recorder window: a
            # sharp peak means these two windows' content actually matches.
            # Same convention as acoustic._measure_boundary (Boundary Flex):
            # the true recorder time for this camera moment is the probed
            # window center corrected by -lag_s.
            lag_s, sharp = gcc_phat(cam_win, rec_win, _REFINE_SR, max_lag_s, gcc_eps)
            if sharp > best_sharp:
                best_sharp, best_rec_time = sharp, t_rec - lag_s
            t_rec += grid_s
        if best_rec_time is not None and best_sharp >= min_sharpness:
            points.append((t_cam, best_rec_time))
        t_cam += grid_s

    if len(points) < 2:
        logger.info("Acoustic coarse align: only %d confident point(s), giving up", len(points))
        return None

    cam_times = np.array([p[0] for p in points])
    rec_times = np.array([p[1] for p in points])
    coeffs = np.polyfit(rec_times, cam_times, 1)
    k, offset = float(coeffs[0]), float(coeffs[1])
    logger.info(
        "Acoustic coarse align: offset=%.3fs k=%.6f from %d confident point(s)",
        offset,
        k,
        len(points),
    )
    return offset, k


# (rec_start, rec_in_duration, atempo_factor) — the piece tuple produced by clip_pieces.
Piece = tuple[float, float, float]


def _measure_boundary(
    cam_track: np.ndarray,
    rec_track: np.ndarray,
    cam_mid: float,
    rec_mid: float,
    win: float,
    max_lag_s: float,
    eps: float,
    min_sharpness: float,
    deadband_s: float,
    max_shift_s: float,
) -> float:
    """Measure one boundary's acoustic correction (seconds to add to rec_start),
    by slicing the two pre-decoded tracks in memory — no ffmpeg call, no scratch
    files. Returns 0.0 when the peak is not confident or the residual is within
    the deadband.
    """
    cam_sig = _window_slice(cam_track, cam_mid, win, _REFINE_SR)
    rec_sig = _window_slice(rec_track, rec_mid, win, _REFINE_SR)
    lag_s, sharp = gcc_phat(cam_sig, rec_sig, _REFINE_SR, max_lag_s, eps)
    # gcc_phat's lag is the shift to add to the query's (recorder's) time to
    # align it with the reference (camera); -lag_s is therefore the seconds
    # to add to the recorder read time so the cut lands under the picture.
    if sharp >= min_sharpness and abs(lag_s) > deadband_s:
        return max(-max_shift_s, min(max_shift_s, -lag_s))
    return 0.0


_FLEX_MIN_PIECE_S = 0.05


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
) -> tuple[float, list[Piece]]:
    """Acoustically nudge each piece's onset so its speech lands under the
    picture, independent of Whisper's word timings ("Boundary Flex").
    Returns ``(lead, pieces)`` — the lead can change when piece 0's onset moves.

    Pieces are contiguous in OUTPUT (camera) time, beginning at ``lead``: piece i's
    local start is ``lead + sum(out_dur[<i])`` where ``out_dur = rec_dur / factor``.
    For each piece we cross-correlate a short camera window at that local time against
    the recorder window at the piece's current ``rec_start``; if the peak is sharp and
    the measured residual exceeds the deadband, the BOUNDARY between this piece and
    its predecessor moves by it (clamped).

    Moving the boundary — not just this piece's start — is what keeps the plan free
    of content gaps and overlaps: the previous piece's duration absorbs the shift
    (its factor is recomputed so its OUTPUT length grows/shrinks by exactly the
    output this piece loses/gains), the shifted piece keeps its own tempo factor, and
    every later piece's output position is untouched. The old behaviour slid the
    whole piece window without touching the neighbour, so a −80 ms nudge made the
    last 80 ms of piece N and the first 80 ms of piece N+1 the SAME recorder
    content played twice — the mid-word micro-repeat («подга-га-товил») users
    heard with Boundary Flex enabled. With sentence-wise plans the boundary sits
    in room tone and the neighbour is a pause piece, so the absorbed shift is
    inaudible by construction.

    Both tracks are decoded to mono 16 kHz numpy arrays exactly once (regardless
    of piece count), and every boundary window is a slice of those arrays — no
    per-boundary ffmpeg subprocess. The independent measurements are spread
    across a thread pool (``gcc_phat``'s FFT calls release the GIL, so threads
    parallelize this fine and avoid the fork-safety concerns of a process pool);
    the geometry is deterministic, so the result is identical regardless of
    worker count. ``tmp_dir`` is accepted for backward compatibility but is
    unused now that no scratch files are written.
    """
    if not pieces:
        return lead, pieces

    win = config.flex_window_s

    # First pass (cheap): the window geometry for every in-bounds boundary.
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

    if not jobs:
        return lead, pieces

    # Decode both full tracks to mono 16k once; every boundary below just
    # slices these arrays.
    cam_track = load_mono16k_track(cam_audio_wav)
    rec_track = load_mono16k_track(rec_audio_path)

    args = (
        win,
        config.acoustic_max_lag_s,
        config.gcc_eps,
        config.flex_min_sharpness,
        config.flex_deadband_s,
        config.flex_max_shift_s,
    )
    corrections: dict[int, float] = {}
    if workers <= 1 or len(jobs) <= 1:
        for idx, cam_mid, rec_mid in jobs:
            corrections[idx] = _measure_boundary(cam_track, rec_track, cam_mid, rec_mid, *args)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_measure_boundary, cam_track, rec_track, cam_mid, rec_mid, *args): idx
                for idx, cam_mid, rec_mid in jobs
            }
            for fut, idx in futs.items():
                corrections[idx] = fut.result()

    # Apply each correction as a BOUNDARY move (content stays contiguous):
    #   piece i:   onset += δ, duration −= δ, factor kept → its speech rate is
    #              untouched, its output shortens/lengthens by δ/factor;
    #   piece i−1: duration += δ (covers the ceded/vacated recorder content),
    #              factor recomputed so its output length changes by exactly
    #              +δ/factor_i — every later piece's output position is
    #              preserved. For piece 0 the lead absorbs the output change.
    refined: list[list[float]] = [list(p) for p in pieces]
    out_durs = [rec_dur / factor if factor else rec_dur for (_s, rec_dur, factor) in pieces]
    new_lead = lead
    n_shifted = 0
    for idx in sorted(corrections):
        delta = corrections[idx]
        if delta == 0.0:
            continue
        start, dur, factor = refined[idx]
        # Clamps: this piece keeps a sliver of duration; the neighbour (or the
        # lead) must be able to absorb; the shifted onset stays in the recorder.
        delta = min(delta, dur - _FLEX_MIN_PIECE_S)
        delta = max(delta, -start)
        if idx > 0:
            delta = max(delta, -(refined[idx - 1][1] - _FLEX_MIN_PIECE_S))
        else:
            f = factor or 1.0
            if new_lead + delta / f < 0.0:
                delta = -new_lead * f
        if abs(delta) < 1e-6:
            continue
        n_shifted += 1
        f = factor or 1.0
        refined[idx] = [start + delta, dur - delta, factor]
        out_durs[idx] -= delta / f
        if idx > 0:
            p_start, p_dur, _p_factor = refined[idx - 1]
            new_p_dur = p_dur + delta
            new_p_out = out_durs[idx - 1] + delta / f
            refined[idx - 1] = [
                p_start,
                new_p_dur,
                max(0.25, min(4.0, new_p_dur / new_p_out)) if new_p_out > 1e-6 else 1.0,
            ]
            out_durs[idx - 1] = new_p_out
        else:
            new_lead += delta / f

    logger.info("Boundary Flex: nudged %d/%d piece onsets", n_shifted, len(pieces))
    return new_lead, [(s, d, f) for s, d, f in refined]
