"""Save full transcripts to JSON + SRT (Podcast Reels Forge-compatible format).

The transcription is computed anyway for alignment, so we persist it next to the
output as a reusable artifact (subtitles, archive, re-runs).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from whispersync.models import Segment, Transcript

_SENTENCE_END = re.compile(r"[.!?…！？]$")
_SENTENCE_MAX_CHARS = 140


def _segment_text(segment: Segment) -> str:
    return " ".join(w.text for w in segment.words).strip()


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp (HH:MM:SS,mmm)."""
    total_ms = max(0, round(seconds * 1000.0))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem %= 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def build_sentence_groups(segments: list[Segment]) -> list[dict[str, Any]]:
    """Group segments into sentence-like blocks (split on sentence-ending
    punctuation or after ~140 chars)."""
    sentences: list[dict[str, Any]] = []
    current: list[Segment] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(_segment_text(s) for s in current).strip()
        if text:
            sentences.append(
                {
                    "start": round(current[0].start, 3),
                    "end": round(current[-1].end, 3),
                    "text": text,
                    "segment_count": len(current),
                }
            )
        current.clear()

    for seg in segments:
        text = _segment_text(seg)
        if not text:
            continue
        current.append(seg)
        joined = " ".join(_segment_text(s) for s in current)
        if _SENTENCE_END.search(text) or len(joined) >= _SENTENCE_MAX_CHARS:
            flush()
    flush()
    return sentences


def transcript_to_dict(
    transcript: Transcript,
    *,
    audio_path: Path,
    model: str,
    device: str,
    compute_type: str,
    mode: str,
) -> dict[str, Any]:
    """Serialize a Transcript to the Reels Forge JSON shape."""
    segments = [
        {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": _segment_text(seg),
            "words": [
                {
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "word": w.text,
                    "probability": round(w.probability, 3),
                }
                for w in seg.words
            ],
        }
        for seg in transcript.segments
    ]
    return {
        "audio": str(audio_path),
        "source_audio": str(audio_path),
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "mode": mode,
        "language": transcript.language,
        "duration": transcript.duration,
        "timing_version": 2,
        "segments": segments,
        "sentences": build_sentence_groups(transcript.segments),
    }


def dump_srt(path: Path, transcript: Transcript) -> None:
    lines: list[str] = []
    idx = 1
    for seg in transcript.segments:
        text = _segment_text(seg)
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}")
        lines.append(text)
        lines.append("")
        idx += 1
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def save_transcript(
    transcript: Transcript,
    out_dir: Path,
    stem: str,
    *,
    audio_path: Path,
    model: str,
    device: str,
    compute_type: str,
    mode: str,
) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.srt`` into ``out_dir``; return paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    srt_path = out_dir / f"{stem}.srt"
    data = transcript_to_dict(
        transcript,
        audio_path=audio_path,
        model=model,
        device=device,
        compute_type=compute_type,
        mode=mode,
    )
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_srt(srt_path, transcript)
    return json_path, srt_path
