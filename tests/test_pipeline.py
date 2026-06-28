"""Tests for pipeline helpers that don't require GPU/ffmpeg (mocked)."""

from __future__ import annotations

from pathlib import Path

import bormosync.engine.pipeline as pipeline_mod
from bormosync.models import MediaClip, Segment, Transcript, Word


class _FakeEngine:
    """Returns canned transcripts in order, one per transcribe() call."""

    def __init__(self, transcripts: list[Transcript]) -> None:
        self._transcripts = transcripts
        self._i = 0

    def transcribe(self, path: Path, progress_callback=None) -> Transcript:  # noqa: ANN001
        t = self._transcripts[self._i]
        self._i += 1
        return t


def _single_word_transcript(word: str) -> Transcript:
    seg = Segment(start=1.0, end=1.5, words=[Word(text=word, start=1.0, end=1.5, probability=0.9)])
    return Transcript(source_path=Path("x"), language="en", duration=10.0, segments=[seg])


def test_build_camera_transcript_shifts_and_merges(monkeypatch) -> None:  # noqa: ANN001
    # avoid invoking ffmpeg: extraction becomes identity
    monkeypatch.setattr(pipeline_mod, "extract_audio_to_wav", lambda p: p)

    clips = [
        MediaClip(Path("a.mp4"), "video", offset=0.0, in_point=0.0, duration=10.0, lane=1),
        MediaClip(Path("b.mp4"), "video", offset=10.0, in_point=0.0, duration=10.0, lane=1),
    ]
    engine = _FakeEngine([_single_word_transcript("hello"), _single_word_transcript("world")])
    cleanup: list[Path] = []

    transcript = pipeline_mod.build_camera_transcript(engine, clips, cleanup)
    words = transcript.words

    assert len(words) == 2
    # first clip word stays at local time (offset 0)
    assert words[0].text == "hello"
    assert abs(words[0].start - 1.0) < 1e-9
    # second clip word is shifted onto the camera timeline by clip offset 10s
    assert words[1].text == "world"
    assert abs(words[1].start - 11.0) < 1e-9
    assert abs(words[1].end - 11.5) < 1e-9
    # merged duration spans both clips
    assert abs(transcript.duration - 20.0) < 1e-9
    # both extracted audio paths were registered for cleanup
    assert len(cleanup) == 2
