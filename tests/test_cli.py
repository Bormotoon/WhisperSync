"""Tests for CLI argument parsing."""

from __future__ import annotations

from pathlib import Path

from whispersync.cli import _build_parser


def test_single_audio_file() -> None:
    args = _build_parser().parse_args(["--video-dir", "vids", "--audio-file", "rec.wav"])
    assert args.audio_files == [Path("rec.wav")]
    assert args.strategy == 1
    assert args.recorder_mode is None  # default applied later via config


def test_multiple_audio_files_accumulate() -> None:
    args = _build_parser().parse_args(
        [
            "--video-dir",
            "vids",
            "--audio-file",
            "lavA.wav",
            "--audio-file",
            "lavB.wav",
            "--recorder-mode",
            "all",
            "--strategy",
            "4",
        ]
    )
    assert args.audio_files == [Path("lavA.wav"), Path("lavB.wav")]
    assert args.recorder_mode == "all"
    assert args.strategy == 4


def test_strategy_choices_include_hybrid() -> None:
    args = _build_parser().parse_args(
        ["--video-dir", "v", "--audio-file", "r.wav", "--strategy", "4"]
    )
    assert args.strategy == 4


def test_crossfade_toggle() -> None:
    parser = _build_parser()
    assert parser.parse_args(["--video-dir", "v", "--audio-file", "r.wav"]).crossfade is None
    assert (
        parser.parse_args(["--video-dir", "v", "--audio-file", "r.wav", "--no-crossfade"]).crossfade
        is False
    )
    args = parser.parse_args(
        ["--video-dir", "v", "--audio-file", "r.wav", "--crossfade", "--crossfade-ms", "20"]
    )
    assert args.crossfade is True
    assert args.crossfade_ms == 20


def test_timebase_and_camera_flags() -> None:
    args = _build_parser().parse_args(
        [
            "--video-dir",
            "v",
            "--audio-file",
            "r.wav",
            "--timebase-source",
            "recorder",
            "--audio-source-camera",
            "camB",
        ]
    )
    assert args.timebase_source == "recorder"
    assert args.audio_source_camera == "camB"
