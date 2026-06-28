"""Tests for anchor matching and alignment on synthetic data."""

from __future__ import annotations

from pathlib import Path

from bormosync.config import BormoSyncConfig
from bormosync.engine.matcher import align, find_anchors, normalize_token
from bormosync.models import Segment, Transcript, Word


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
    config = BormoSyncConfig(min_anchors=5, anchor_min_confidence=0.5)

    result = align(cam_t, rec_t, config)

    assert abs(result.offset - true_offset) < 0.05
    assert abs(result.k - true_k) < 0.001
    assert len(result.anchors) >= 10
    assert result.residual_ms < 50.0
