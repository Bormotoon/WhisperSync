"""Tests for config loading (missing file, unknown keys, CLI overrides)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from whispersync.config import WhisperSyncConfig, load_config


def test_load_config_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_load_config_no_path_uses_defaults() -> None:
    cfg = load_config(None)
    assert cfg == WhisperSyncConfig()


def test_render_master_wav_defaults_off() -> None:
    assert WhisperSyncConfig().render_master_wav is False


def test_render_master_wav_cli_override() -> None:
    cfg = load_config(None, render_master_wav=True)
    assert cfg.render_master_wav is True


def test_load_config_reads_known_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": "medium", "default_strategy": 2}))
    cfg = load_config(path)
    assert cfg.model == "medium"
    assert cfg.default_strategy == 2


def test_load_config_unknown_key_warns_but_does_not_fail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.json"
    # Typo: "pause_duck_dB" instead of "pause_duck_db" — must not silently no-op.
    path.write_text(json.dumps({"model": "medium", "pause_duck_dB": -12.0}))
    with caplog.at_level(logging.WARNING):
        cfg = load_config(path)
    assert cfg.model == "medium"
    assert any("pause_duck_dB" in r.message for r in caplog.records)


def test_load_config_cli_overrides_win(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": "medium"}))
    cfg = load_config(path, model="large-v3")
    assert cfg.model == "large-v3"


def test_voice_segment_and_ambience_defaults() -> None:
    cfg = WhisperSyncConfig()
    assert cfg.voice_segment_minutes == 0  # monolith by default
    assert cfg.ambience_track is True  # ambience extraction on by default
    assert cfg.boundary_flex is True
    assert cfg.timebase_source == "camera"
