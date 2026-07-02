"""Tests for the main.py entry-point dispatcher (GUI vs --cli)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_MAIN = Path(__file__).resolve().parents[1] / "main.py"


def test_cli_flag_dispatches_to_cli_and_strips_itself() -> None:
    # --cli must be consumed by the dispatcher, not passed through to
    # argparse (which doesn't know that flag and would reject it as
    # "unrecognized arguments"). --version is a cheap way to exercise the
    # full argparse path without needing real media files.
    result = subprocess.run(
        [sys.executable, str(_MAIN), "--cli", "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "whispersync" in result.stdout.lower()


def test_cli_flag_usage_error_exits_2() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_MAIN),
            "--cli",
            "--video-dir",
            "/does/not/exist",
            "--audio-file",
            "x.wav",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
