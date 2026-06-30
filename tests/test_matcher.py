"""Tests for anchor matching and alignment on synthetic data."""

from __future__ import annotations

from pathlib import Path

from whispersync.config import WhisperSyncConfig
from whispersync.engine.matcher import (
    align,
    estimate_coarse_delta,
    find_anchors,
    normalize_token,
    normalize_words,
    reject_gross_outliers,
)
from whispersync.models import Anchor, Segment, Transcript, Word


def test_reject_gross_outliers_keeps_smooth_drift() -> None:
    """Anchors following a smooth (even non-linear) drift are kept; a single word
    matched to a far-away occurrence is dropped."""
    anchors = [
        Anchor(cam_time=t + 0.01 * t, rec_time=float(t), token=f"w{t}", confidence=0.9)
        for t in range(40)
    ]
    bad = Anchor(cam_time=anchors[20].cam_time + 5.0, rec_time=20.0, token="x", confidence=0.9)
    anchors[20] = bad

    kept = reject_gross_outliers(anchors, window=8, tol_s=0.3)

    assert bad not in kept
    assert len(kept) >= 38  # every smooth-drift anchor survives


def _make_transcript(
    words_data: list[tuple[str, float, float]],
    source: str = "test.wav",
) -> Transcript:
    words = [
        Word(text=text, start=start, end=end, probability=0.95) for text, start, end in words_data
    ]
    seg = Segment(start=words[0].start, end=words[-1].end, words=words)
    return Transcript(
        source_path=Path(source),
        language="en",
        duration=words[-1].end,
        segments=[seg],
    )


def test_normalize_token() -> None:
    assert normalize_token("Hello!") == "hello"
    assert normalize_token("  World,  ") == "world"
    assert normalize_token("...") == ""


def test_find_anchors_basic() -> None:
    cam_words = [
        ("the", 1.0, 1.2),
        ("quick", 1.5, 1.8),
        ("brown", 2.0, 2.3),
        ("fox", 2.5, 2.8),
        ("jumps", 3.0, 3.3),
    ]
    rec_words = [
        ("the", 5.0, 5.2),
        ("quick", 5.5, 5.8),
        ("brown", 6.0, 6.3),
        ("fox", 6.5, 6.8),
        ("jumps", 7.0, 7.3),
    ]
    cam_t = _make_transcript(cam_words, "cam.wav")
    rec_t = _make_transcript(rec_words, "rec.wav")

    anchors = find_anchors(cam_t, rec_t, min_confidence=0.5)
    assert len(anchors) >= 3


def test_align_known_offset_and_k() -> None:
    true_offset = 2.0
    true_k = 1.001

    cam_words: list[tuple[str, float, float]] = []
    rec_words: list[tuple[str, float, float]] = []
    tokens = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
        "quebec",
        "romeo",
        "sierra",
        "tango",
    ]

    for i, token in enumerate(tokens):
        rec_start = 10.0 + i * 5.0
        rec_end = rec_start + 0.3
        cam_start = true_offset + true_k * rec_start
        cam_end = true_offset + true_k * rec_end
        rec_words.append((token, rec_start, rec_end))
        cam_words.append((token, cam_start, cam_end))

    cam_t = _make_transcript(cam_words, "cam.wav")
    rec_t = _make_transcript(rec_words, "rec.wav")
    config = WhisperSyncConfig(min_anchors=5, anchor_min_confidence=0.5)

    result = align(cam_t, rec_t, config)

    assert abs(result.offset - true_offset) < 0.05
    assert abs(result.k - true_k) < 0.001
    assert len(result.anchors) >= 10
    assert result.residual_ms < 50.0


# vocabulary big enough to fill a long recorder without trivial repetition
_VOCAB = [f"word{i:04d}" for i in range(4000)]


def _long_recorder_with_clip(
    clip_tokens: list[str],
    clip_offset_in_rec: float,
    total_rec_minutes: float,
    true_k: float,
) -> tuple[Transcript, Transcript]:
    """Build a multi-minute recorder whose content at ``clip_offset_in_rec``
    matches a clip whose LOCAL time starts at 0. A few clip words are also
    sprinkled in a far region as distractors. The true mapping is
    cam_local = true_k * (t_rec - clip_offset_in_rec), i.e. offset = -true_k * clip_offset.
    """
    rec: list[tuple[str, float, float]] = []
    total_sec = total_rec_minutes * 60
    t = 0.0
    vi = 0
    while t < total_sec:
        rec.append((_VOCAB[vi % len(_VOCAB)], t, t + 0.4))
        vi += 1
        t += 0.5

    cam: list[tuple[str, float, float]] = []
    for j, tok in enumerate(clip_tokens):
        r = clip_offset_in_rec + j * 0.5
        rec.append((tok, r, r + 0.4))
        c = true_k * (r - clip_offset_in_rec)  # local clip time, starts at 0
        cam.append((tok, c, c + 0.4))

    # a few distractor copies elsewhere (fewer than the true run, so it loses)
    for j, tok in enumerate(clip_tokens[:5]):
        r = total_sec * 0.8 + j * 0.5
        rec.append((tok, r, r + 0.4))

    rec.sort(key=lambda x: x[1])
    return _make_transcript(cam, "cam.wav"), _make_transcript(rec, "rec.wav")


def test_estimate_coarse_delta_finds_region_despite_distractors() -> None:
    clip_tokens = [f"anchorword{i:03d}" for i in range(15)]
    cam_t, rec_t = _long_recorder_with_clip(
        clip_tokens, clip_offset_in_rec=1800.0, total_rec_minutes=60.0, true_k=1.0
    )
    cfg = WhisperSyncConfig(anchor_min_confidence=0.5)
    cam_w = normalize_words(list(cam_t.words), 0.5)
    rec_w = normalize_words(list(rec_t.words), 0.5)
    delta = estimate_coarse_delta(cam_w, rec_w, cfg)
    assert delta is not None
    # clip local starts at 0, content sits at recorder t≈1800 -> delta ≈ 1800
    assert abs(delta - 1800.0) < 5.0


def test_align_windowed_on_long_recorder() -> None:
    clip_tokens = [f"anchorword{i:03d}" for i in range(20)]
    clip_offset, true_k = 2400.0, 1.0008
    cam_t, rec_t = _long_recorder_with_clip(
        clip_tokens, clip_offset_in_rec=clip_offset, total_rec_minutes=80.0, true_k=true_k
    )
    cfg = WhisperSyncConfig(min_anchors=5, anchor_min_confidence=0.5)
    result = align(cam_t, rec_t, cfg)
    assert abs(result.k - true_k) < 0.002
    # the clip's local 0 must map back to recorder t≈2400
    assert abs(result.rec_to_cam(clip_offset)) < 0.5
    assert len(result.anchors) >= 10
