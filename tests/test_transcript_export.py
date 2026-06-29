"""Tests for transcript JSON/SRT export."""

from __future__ import annotations

import json
from pathlib import Path

from bormosync.engine.transcript_export import (
    build_sentence_groups,
    format_srt_timestamp,
    save_transcript,
)
from bormosync.models import Segment, Transcript, Word


def _transcript() -> Transcript:
    seg1 = Segment(
        start=0.0,
        end=1.5,
        words=[
            Word("Hello", 0.0, 0.5, 0.99),
            Word("world.", 0.6, 1.5, 0.97),
        ],
    )
    seg2 = Segment(
        start=2.0,
        end=3.0,
        words=[Word("Next", 2.0, 2.4, 0.95), Word("phrase", 2.5, 3.0, 0.9)],
    )
    return Transcript(
        source_path=Path("rec.wav"), language="en", duration=3.0, segments=[seg1, seg2]
    )


def test_format_srt_timestamp() -> None:
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(3661.5) == "01:01:01,500"


def test_build_sentence_groups_splits_on_punctuation() -> None:
    groups = build_sentence_groups(_transcript().segments)
    assert len(groups) >= 1
    assert groups[0]["text"].startswith("Hello world.")


def test_save_transcript_writes_json_and_srt(tmp_path: Path) -> None:
    t = _transcript()
    json_path, srt_path = save_transcript(
        t,
        tmp_path,
        "rec",
        audio_path=Path("/audio/rec.wav"),
        model="large-v3",
        device="cpu",
        compute_type="float32",
        mode="fast",
    )
    assert json_path.exists() and srt_path.exists()

    data = json.loads(json_path.read_text())
    assert data["model"] == "large-v3"
    assert data["language"] == "en"
    assert data["device"] == "cpu"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["text"] == "Hello world."
    assert data["segments"][0]["words"][0]["word"] == "Hello"

    srt = srt_path.read_text()
    assert "00:00:00,000 --> 00:00:01,500" in srt
    assert "Hello world." in srt
