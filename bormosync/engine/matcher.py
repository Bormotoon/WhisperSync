"""Anchor matching and time alignment between transcripts."""

from __future__ import annotations

import difflib
import logging
import random
import string
from collections import Counter, defaultdict

import numpy as np

from bormosync.config import BormoSyncConfig
from bormosync.models import AlignmentMap, Anchor, Transcript, Word

logger = logging.getLogger(__name__)


def normalize_token(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


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
    cam_words: list[Word], rec_words: list[Word], config: BormoSyncConfig
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
    cam_words: list[Word], rec_words: list[Word], config: BormoSyncConfig
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


def align(
    cam_transcript: Transcript,
    rec_transcript: Transcript,
    config: BormoSyncConfig,
) -> AlignmentMap:
    cam_words = normalize_words(list(cam_transcript.words), config.anchor_min_confidence)
    rec_words = normalize_words(list(rec_transcript.words), config.anchor_min_confidence)
    if not cam_words or not rec_words:
        raise ValueError("No usable words to align (check confidence threshold / speech content).")

    rec_used = _window_recorder(cam_words, rec_words, config)
    anchors = _anchors_from_words(cam_words, rec_used)

    # If the coarse window was misleading, retry once against the full reference.
    if len(anchors) < config.min_anchors and rec_used is not rec_words:
        logger.info("Windowed match weak (%d anchors); retrying full reference", len(anchors))
        anchors = _anchors_from_words(cam_words, rec_words)

    if len(anchors) < 2:
        raise ValueError(
            f"Only {len(anchors)} anchor(s) found — need at least 2. "
            "The transcripts may not contain enough matching speech."
        )

    offset, k, inliers = ransac_linear_fit(anchors)

    residuals_ms = [abs((offset + k * a.rec_time) - a.cam_time) * 1000 for a in inliers]
    residual_ms = float(np.median(residuals_ms))

    if len(inliers) < config.min_anchors:
        logger.warning(
            "Only %d inlier anchors (minimum recommended: %d). Alignment may be inaccurate.",
            len(inliers),
            config.min_anchors,
        )

    logger.info(
        "Alignment: offset=%.4fs, K=%.6f, anchors=%d, residual=%.1fms",
        offset,
        k,
        len(inliers),
        residual_ms,
    )

    return AlignmentMap(
        anchors=inliers,
        offset=offset,
        k=k,
        residual_ms=residual_ms,
    )
