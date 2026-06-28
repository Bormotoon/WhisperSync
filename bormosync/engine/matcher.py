"""Anchor matching and time alignment between transcripts."""

from __future__ import annotations

import difflib
import logging
import random
import string
from collections import Counter

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


def find_anchors(
    cam_transcript: Transcript,
    rec_transcript: Transcript,
    min_confidence: float = 0.6,
) -> list[Anchor]:
    cam_words = normalize_words(list(cam_transcript.words), min_confidence)
    rec_words = normalize_words(list(rec_transcript.words), min_confidence)

    if not cam_words or not rec_words:
        return []

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


def align(
    cam_transcript: Transcript,
    rec_transcript: Transcript,
    config: BormoSyncConfig,
) -> AlignmentMap:
    anchors = find_anchors(cam_transcript, rec_transcript, config.anchor_min_confidence)

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
