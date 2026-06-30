"""Tier-1 text alignment: banded DTW over word indices.

The legacy matcher finds anchors with a single ``difflib.SequenceMatcher`` LCS.
That is fragile when *both* the camera clip and the recorder contain repeated
takes/drafts of the same speech: an LCS can stitch camera-take-A to a recorder
copy of take-B — monotonic in time, but latched to the wrong repeat, which makes
the synced audio float.

This module replaces the LCS with a banded Dynamic Time Warping pass in word-index
space. The coarse pass (``estimate_coarse_delta``) already assumes K ≈ 1 and locates
the clip inside the (windowed) recorder, so DTW only needs a narrow Sakoe-Chiba band
around the resulting diagonal. The cell cost mixes a token-mismatch term with a
*slope-deviation* penalty: a camera word whose token also occurs far off the
diagonal (the wrong take) pays a large slope penalty, so DTW prefers the on-diagonal
(correct take) occurrence even though both score a perfect token match. Every DTW
step is non-decreasing in both indices, so emitted anchors are monotonic and dense
(one per matched word), preserving intra-phrase drift for the piecewise warp.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from whispersync.config import WhisperSyncConfig
from whispersync.models import Anchor, Word

logger = logging.getLogger(__name__)

# Sentinel "infinite" cost for cells outside the band / unreachable.
_INF = float("inf")


def _mid(w: Word) -> float:
    return (w.start + w.end) / 2


def _coarse_j0(cam_words: list[Word], rec_words: list[Word], coarse_delta: float | None) -> int:
    """Recorder word index whose mid-time best matches the first camera word's
    expected recorder time (cam_mid[0] + coarse_delta). Falls back to 0 when the
    coarse delta is unknown."""
    if coarse_delta is None:
        return 0
    target = _mid(cam_words[0]) + coarse_delta
    rec_mids = [_mid(w) for w in rec_words]
    # rec_words are sorted by time; binary-search the closest mid-time.
    j = int(np.searchsorted(rec_mids, target))
    if j <= 0:
        return 0
    if j >= len(rec_mids):
        return len(rec_mids) - 1
    return j if abs(rec_mids[j] - target) < abs(rec_mids[j - 1] - target) else j - 1


def _band_radius(cam_words: list[Word], rec_words: list[Word], config: WhisperSyncConfig) -> int:
    """Sakoe-Chiba half-width in recorder-word indices: convert the matcher's
    time-domain slack (match_window_margin) into a word count via the recorder's
    word rate, then clamp."""
    rec_span = max(_mid(rec_words[-1]) - _mid(rec_words[0]), 1e-3)
    rec_word_rate = len(rec_words) / rec_span  # words per second
    band = math.ceil(config.match_window_margin * rec_word_rate * config.dtw_band_margin_frac)
    return int(max(config.dtw_band_min, min(config.dtw_band_max, band)))


def dtw_anchors(
    cam_words: list[Word],
    rec_words: list[Word],
    config: WhisperSyncConfig,
    coarse_delta: float | None,
) -> list[Anchor]:
    """Dense, monotonic anchors from a banded DTW over normalized words.

    ``cam_words``/``rec_words`` must be normalized (``w.norm`` populated) and sorted
    by time, exactly as ``matcher.normalize_words`` produces. Returns a list of
    ``Anchor`` in the same shape ``_anchors_from_words`` produces, so it is a drop-in
    feed for ``reject_gross_outliers`` → ``ransac_linear_fit``.
    """
    n = len(cam_words)
    m = len(rec_words)
    if n == 0 or m == 0:
        return []

    band = _band_radius(cam_words, rec_words, config)
    j0 = _coarse_j0(cam_words, rec_words, coarse_delta)

    # Expected recorder index for camera index i: a diagonal of slope (m-1)/(n-1)
    # anchored so cam index 0 maps to j0 (the coarse match). K ≈ 1, but using the
    # index-count ratio keeps the band centered even when word counts differ.
    slope = (m - 1) / (n - 1) if n > 1 else 0.0

    def j_expected(i: int) -> float:
        return j0 + slope * i

    w_tok = config.dtw_token_weight
    w_slope = config.dtw_slope_weight
    subst = config.dtw_subst_cost
    gap = config.dtw_gap_cost

    cam_norm = [w.norm for w in cam_words]
    rec_norm = [w.norm for w in rec_words]

    # Rolling DTW over the band only. For row i we store, for each in-band j, the
    # cumulative cost and a back-pointer (0=diag, 1=up/cam-gap, 2=left/rec-gap).
    # Bands for consecutive rows overlap, so we keep the previous row's window keyed
    # by absolute j via a dict for O(band) memory.
    prev_cost: dict[int, float] = {}
    # back[i] maps absolute j -> backpointer, kept for the whole grid to backtrack.
    back_rows: list[dict[int, int]] = []

    for i in range(n):
        center = j_expected(i)
        lo = max(0, int(math.floor(center - band)))
        hi = min(m - 1, int(math.ceil(center + band)))
        cur_cost: dict[int, float] = {}
        cur_back: dict[int, int] = {}
        ci = cam_norm[i]
        for j in range(lo, hi + 1):
            tok = 0.0 if ci == rec_norm[j] else subst
            # Quadratic slope penalty, scaled so the band edge costs w_slope * a
            # full band-worth of token substitutions — i.e. straying to a wrong,
            # off-diagonal take is far more expensive than tolerating local
            # substitutions on the correct diagonal. This is what defeats the
            # two-sided repeated-takes latch.
            frac = (j - center) / band
            local = w_tok * tok + w_slope * subst * band * frac * frac

            # Three predecessors: diagonal (i-1,j-1), cam-gap (i-1,j), rec-gap (i,j-1).
            # bp: 0=diag/start, 1=cam-gap(up), 2=rec-gap(left). The path must be
            # connected — no free restarts — so DTW cannot teleport across the
            # recorder to a cheaper wrong-take region.
            if i == 0:
                # First row seeds the path: a start (cost 0) anywhere in the band,
                # or extend a rec-gap from the left within the row.
                best, bp = 0.0, 0
                left = cur_cost.get(j - 1, _INF)
                if left + gap < best:
                    best, bp = left + gap, 2
            else:
                best, bp = _INF, 0
                diag = prev_cost.get(j - 1, _INF)
                if diag < best:
                    best, bp = diag, 0
                up = prev_cost.get(j, _INF)
                if up + gap < best:
                    best, bp = up + gap, 1
                left = cur_cost.get(j - 1, _INF)
                if left + gap < best:
                    best, bp = left + gap, 2
            if best == _INF:
                continue  # unreachable cell (band gap): leave it out, no teleport
            cur_cost[j] = best + local
            cur_back[j] = bp
        if not cur_cost:
            # Band produced no reachable cells; bail to the caller's difflib fallback.
            logger.debug("DTW row %d produced no reachable in-band cells", i)
            return []
        prev_cost = cur_cost
        back_rows.append(cur_back)

    # Backtrack from the min-cost cell in the last row's band.
    j = min(prev_cost, key=lambda jj: prev_cost[jj])
    path: list[tuple[int, int]] = []
    i = n - 1
    while i >= 0:
        path.append((i, j))
        bp = back_rows[i].get(j, 0)
        if i == 0:
            break
        if bp == 0:  # diagonal
            i, j = i - 1, j - 1
        elif bp == 1:  # cam-gap (up)
            i = i - 1
        else:  # rec-gap (left): stay on same row
            j = j - 1
            if j < 0:
                break
    path.reverse()

    # Emit an anchor only for true token matches above the confidence floor.
    anchors: list[Anchor] = []
    last_i = last_j = -1
    for i, j in path:
        if i == last_i or j == last_j:
            continue  # skip the stationary side of a gap step
        last_i, last_j = i, j
        if cam_norm[i] != rec_norm[j]:
            continue
        cw, rw = cam_words[i], rec_words[j]
        conf = min(cw.probability, rw.probability)
        if conf < config.dtw_min_anchor_conf:
            continue
        anchors.append(
            Anchor(
                cam_time=_mid(cw),
                rec_time=_mid(rw),
                token=cw.norm,
                confidence=conf,
            )
        )

    logger.info(
        "DTW: %d anchors from %d cam / %d rec words (band=%d, j0=%d)",
        len(anchors),
        n,
        m,
        band,
        j0,
    )
    return anchors
