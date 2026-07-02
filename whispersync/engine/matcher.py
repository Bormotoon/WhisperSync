"""Anchor matching and time alignment between transcripts."""

from __future__ import annotations

import difflib
import logging
import random
import re
from collections import Counter, defaultdict

import numpy as np

from whispersync.config import WhisperSyncConfig
from whispersync.models import AlignmentMap, Anchor, Transcript, Word

logger = logging.getLogger(__name__)

# Strip anything that isn't a Unicode word character. string.punctuation (the
# old approach) only covers ASCII punctuation, so Russian/typographic marks
# like «», —, … pass through untouched and words end up with stray leading/
# trailing punctuation that never matches its "clean" counterpart on the other
# track — a silent source of lost anchors on non-English (esp. Russian) audio.
_NON_WORD_RE = re.compile(r"[^\w]", re.UNICODE)


def normalize_token(text: str) -> str:
    return _NON_WORD_RE.sub("", text.lower())


def normalize_words(words: list[Word], min_confidence: float) -> list[Word]:
    result: list[Word] = []
    for w in words:
        if w.probability < min_confidence:
            continue
        norm = normalize_token(w.text)
        if not norm:
            continue
        w.norm = norm
        result.append(w)
    return result


def _mid(w: Word) -> float:
    return (w.start + w.end) / 2


def estimate_coarse_delta(
    cam_words: list[Word], rec_words: list[Word], config: WhisperSyncConfig
) -> float | None:
    """Roughly locate the clip inside a (possibly very long) reference by voting
    on the time delta ``rec_time - cam_time`` of shared rare words. Returns the
    estimated delta (recorder time of the clip's start ≈ cam time + delta), or
    None if there is no confident peak.

    Assumes K ≈ 1 for the coarse pass, which is accurate enough over a single
    clip to pick the right window; the fine pass recovers the exact K.
    """
    rec_count = Counter(w.norm for w in rec_words)
    rec_positions: dict[str, list[float]] = defaultdict(list)
    for w in rec_words:
        if rec_count[w.norm] <= config.seed_max_occurrences:
            rec_positions[w.norm].append(_mid(w))

    bin_width = config.seed_bin_width
    votes: Counter[int] = Counter()
    delta_sum: dict[int, float] = defaultdict(float)
    for cw in cam_words:
        ct = _mid(cw)
        for rt in rec_positions.get(cw.norm, ()):
            d = rt - ct
            b = round(d / bin_width)
            votes[b] += 1
            delta_sum[b] += d

    if not votes:
        return None

    best_bin, best_votes = votes.most_common(1)[0]
    if best_votes < max(3, config.min_anchors // 2):
        return None

    # weighted mean over the winning bin and its immediate neighbours
    total_n = 0
    total_d = 0.0
    for b in (best_bin - 1, best_bin, best_bin + 1):
        if b in votes:
            total_n += votes[b]
            total_d += delta_sum[b]
    return total_d / total_n


def _anchors_from_words(cam_words: list[Word], rec_words: list[Word]) -> list[Anchor]:
    cam_norms = [w.norm for w in cam_words]
    rec_norms = [w.norm for w in rec_words]

    all_tokens = cam_norms + rec_norms
    token_counts = Counter(all_tokens)

    matcher = difflib.SequenceMatcher(None, cam_norms, rec_norms, autojunk=False)
    raw_anchors: list[Anchor] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for ci, ri in zip(range(i1, i2), range(j1, j2), strict=True):
            cw = cam_words[ci]
            rw = rec_words[ri]
            raw_anchors.append(
                Anchor(
                    cam_time=(cw.start + cw.end) / 2,
                    rec_time=(rw.start + rw.end) / 2,
                    token=cw.norm,
                    confidence=min(cw.probability, rw.probability),
                )
            )

    seen_tokens: set[str] = set()
    unique_anchors: list[Anchor] = []
    for a in raw_anchors:
        if token_counts[a.token] <= 4:
            if a.token in seen_tokens:
                continue
            seen_tokens.add(a.token)
        unique_anchors.append(a)

    monotonic: list[Anchor] = []
    last_cam = -float("inf")
    last_rec = -float("inf")
    for a in unique_anchors:
        if a.cam_time > last_cam and a.rec_time > last_rec:
            monotonic.append(a)
            last_cam = a.cam_time
            last_rec = a.rec_time

    logger.info(
        "Found %d anchors (%d raw, %d after uniqueness)",
        len(monotonic),
        len(raw_anchors),
        len(unique_anchors),
    )
    return monotonic


def find_anchors(
    cam_transcript: Transcript,
    rec_transcript: Transcript,
    min_confidence: float = 0.6,
) -> list[Anchor]:
    cam_words = normalize_words(list(cam_transcript.words), min_confidence)
    rec_words = normalize_words(list(rec_transcript.words), min_confidence)
    if not cam_words or not rec_words:
        return []
    return _anchors_from_words(cam_words, rec_words)


def reject_gross_outliers(
    anchors: list[Anchor], window: int = 10, tol_s: float = 0.30
) -> list[Anchor]:
    """Drop anchors whose ``cam_time - rec_time`` delta disagrees with their local
    neighbourhood — i.e. a word matched to the wrong (far-away) occurrence.

    Unlike a global-linear inlier test, this keeps anchors that follow a *smooth*
    (possibly non-linear) drift, because each is only compared to its neighbours.
    ``anchors`` must be sorted by rec_time (as produced by ``_anchors_from_words``).

    The window shrinks (down to a minimum of 2) for short anchor lists instead of
    disabling the filter outright — a clip with, say, 12 anchors used to skip this
    check entirely (it needed >= 2*window=20), so a single 5-second-off outlier
    would ride straight through to the piecewise warp. See PROJECT_ANALYSIS.md §2.6.
    """
    n = len(anchors)
    if n < 4:
        return anchors
    eff_window = min(window, max(2, n // 4))
    deltas = [a.cam_time - a.rec_time for a in anchors]
    kept: list[Anchor] = []
    for i, a in enumerate(anchors):
        lo = max(0, i - eff_window)
        hi = min(n, i + eff_window + 1)
        local = sorted(deltas[lo:hi])
        median = local[len(local) // 2]
        if abs(deltas[i] - median) <= tol_s:
            kept.append(a)
    return kept


def reject_residual_outliers(
    anchors: list[Anchor],
    offset: float,
    k: float,
    min_factor: float = 3.0,
    min_residual_s: float = 0.25,
) -> list[Anchor]:
    """Drop anchors whose residual against the fitted line (offset, k) is both
    far from the pack (> ``min_factor`` times the median residual) AND
    absolutely large (> ``min_residual_s``), so a lone isolated mismatch that
    slipped past ``reject_gross_outliers`` (e.g. because its local neighbourhood
    happened to also be sparse/noisy) doesn't feed the piecewise warp — a single
    such anchor can force a 0.5-5s stretch on its piece. Requires both
    conditions so a genuinely noisy-but-honest alignment (all residuals modestly
    elevated) isn't gutted. See PROJECT_ANALYSIS.md §2.6.
    """
    if len(anchors) < 4:
        return anchors
    residuals = [abs((offset + k * a.rec_time) - a.cam_time) for a in anchors]
    sorted_res = sorted(residuals)
    median = sorted_res[len(sorted_res) // 2]
    threshold = max(min_factor * median, min_residual_s)
    kept = [a for a, r in zip(anchors, residuals, strict=True) if r <= threshold]
    return kept if len(kept) >= 2 else anchors


def ransac_linear_fit(
    anchors: list[Anchor],
    n_iterations: int = 200,
    inlier_threshold_ms: float = 100.0,
) -> tuple[float, float, list[Anchor]]:
    rng = random.Random(42)
    best_inliers: list[Anchor] = []

    for _ in range(n_iterations):
        sample = rng.sample(anchors, 2)
        dr = sample[1].rec_time - sample[0].rec_time
        if abs(dr) < 1e-9:
            continue
        k = (sample[1].cam_time - sample[0].cam_time) / dr
        offset = sample[0].cam_time - k * sample[0].rec_time

        inliers: list[Anchor] = []
        for a in anchors:
            predicted = offset + k * a.rec_time
            residual_ms = abs(predicted - a.cam_time) * 1000
            if residual_ms < inlier_threshold_ms:
                inliers.append(a)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < 2:
        best_inliers = anchors

    rec_times = np.array([a.rec_time for a in best_inliers])
    cam_times = np.array([a.cam_time for a in best_inliers])
    coeffs = np.polyfit(rec_times, cam_times, 1)
    k_final = float(coeffs[0])
    offset_final = float(coeffs[1])

    return offset_final, k_final, best_inliers


def _window_recorder(
    cam_words: list[Word], rec_words: list[Word], config: WhisperSyncConfig
) -> list[Word]:
    """If the recorder is much longer than the clip, restrict matching to a
    window around the coarse estimate; otherwise return all recorder words."""
    rec_span = _mid(rec_words[-1]) - _mid(rec_words[0])
    cam_lo = min(w.start for w in cam_words)
    cam_hi = max(w.end for w in cam_words)
    margin = config.match_window_margin

    # Only worth windowing when the reference dwarfs the needed window.
    if rec_span <= (cam_hi - cam_lo) + 4 * margin:
        return rec_words

    delta = estimate_coarse_delta(cam_words, rec_words, config)
    if delta is None:
        return rec_words

    lo = cam_lo + delta - margin
    hi = cam_hi + delta + margin
    windowed = [w for w in rec_words if lo <= _mid(w) <= hi]
    if len(windowed) < 2:
        return rec_words
    logger.info(
        "Windowed match: delta=%.1fs window=[%.0f,%.0f]s, %d -> %d recorder words",
        delta,
        lo,
        hi,
        len(rec_words),
        len(windowed),
    )
    return windowed


def _match_words(
    cam_words: list[Word], rec_words: list[Word], config: WhisperSyncConfig
) -> list[Anchor]:
    """Produce anchors from normalized words, dispatching on ``config.align_mode``.

    "dtw" uses the repeat-robust banded DTW; if it yields fewer than ``min_anchors``
    it falls back to the legacy difflib matcher on the same words (mirrors the
    windowed-vs-full retry policy in ``align``). "legacy" uses difflib directly.
    """
    if config.align_mode == "dtw":
        # Local import avoids a hard dependency for the legacy path.
        from whispersync.engine.dtw import dtw_anchors

        coarse = estimate_coarse_delta(cam_words, rec_words, config)
        anchors = dtw_anchors(cam_words, rec_words, config, coarse)
        if len(anchors) >= config.min_anchors:
            return anchors
        logger.info(
            "DTW produced %d anchors (< min %d); falling back to difflib",
            len(anchors),
            config.min_anchors,
        )
    return _anchors_from_words(cam_words, rec_words)


def align(
    cam_transcript: Transcript,
    rec_transcript: Transcript,
    config: WhisperSyncConfig,
) -> AlignmentMap:
    cam_words = normalize_words(list(cam_transcript.words), config.anchor_min_confidence)
    rec_words = normalize_words(list(rec_transcript.words), config.anchor_min_confidence)
    if not cam_words or not rec_words:
        raise ValueError("No usable words to align (check confidence threshold / speech content).")

    rec_used = _window_recorder(cam_words, rec_words, config)
    anchors = _match_words(cam_words, rec_used, config)

    # If the coarse window was misleading, retry once against the full reference.
    if len(anchors) < config.min_anchors and rec_used is not rec_words:
        logger.info("Windowed match weak (%d anchors); retrying full reference", len(anchors))
        anchors = _match_words(cam_words, rec_words, config)

    if len(anchors) < 2:
        raise ValueError(
            f"Only {len(anchors)} anchor(s) found — need at least 2. "
            "The transcripts may not contain enough matching speech."
        )

    # Drop gross mismatches (a word matched to a wrong far-away occurrence) while
    # KEEPING anchors that follow the smooth local drift. These feed the per-clip
    # piecewise warp, so they must track the real (non-linear) drift — not be
    # flattened onto a single line.
    kept = reject_gross_outliers(anchors)
    if len(kept) < 2:
        kept = anchors

    # A robust global line (offset, K) still drives timeline placement and the
    # Global-Linear strategy; it is fit on the cleaned anchors.
    offset, k, line_inliers = ransac_linear_fit(kept)

    # A second pass against the fitted line catches an ISOLATED outlier that
    # reject_gross_outliers missed (its own local neighbourhood was sparse/noisy
    # enough to not flag it). Applied to `kept` (not just the RANSAC inlier set)
    # so the piecewise warp — which uses every kept anchor, not only inliers —
    # never sees a single-anchor 0.5-5s mismatch. See PROJECT_ANALYSIS.md §2.6.
    kept = reject_residual_outliers(kept, offset, k)

    residuals_ms = [abs((offset + k * a.rec_time) - a.cam_time) * 1000 for a in line_inliers]
    residual_ms = float(np.median(residuals_ms)) if residuals_ms else 0.0

    if len(line_inliers) < config.min_anchors:
        logger.warning(
            "Only %d inlier anchors (minimum recommended: %d). Alignment may be inaccurate.",
            len(line_inliers),
            config.min_anchors,
        )

    logger.info(
        "Alignment: offset=%.4fs, K=%.6f, anchors=%d (kept %d of %d raw), residual=%.1fms",
        offset,
        k,
        len(line_inliers),
        len(kept),
        len(anchors),
        residual_ms,
    )

    return AlignmentMap(
        anchors=kept,
        offset=offset,
        k=k,
        residual_ms=residual_ms,
    )
