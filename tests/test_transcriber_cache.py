"""Tests for the transcript cache key (pure logic, no GPU/model load)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from whispersync.config import WhisperSyncConfig
from whispersync.engine.transcriber import WhisperEngine


def test_cache_key_differs_by_resolved_device(tmp_path: Path) -> None:
    audio = tmp_path / "rec.wav"
    audio.write_bytes(b"fake audio bytes")
    cfg = WhisperSyncConfig(compute_type="auto")

    # Two runs with the literal config compute_type="auto" but different
    # RESOLVED devices (e.g. one machine has a GPU, one doesn't, or a run fell
    # back CUDA->CPU mid-transcription) must not collide on the same cache key
    # despite producing different transcripts. See PROJECT_ANALYSIS.md §2.7.
    key_cuda = WhisperEngine._cache_key(audio, cfg, "cuda", "float16")
    key_cpu = WhisperEngine._cache_key(audio, cfg, "cpu", "float32")
    assert key_cuda != key_cpu


def test_cache_key_stable_for_same_resolved_device(tmp_path: Path) -> None:
    audio = tmp_path / "rec.wav"
    audio.write_bytes(b"fake audio bytes")
    cfg = WhisperSyncConfig(compute_type="auto")

    key1 = WhisperEngine._cache_key(audio, cfg, "cuda", "float16")
    key2 = WhisperEngine._cache_key(audio, cfg, "cuda", "float16")
    assert key1 == key2


def test_cache_key_changes_with_decoding_params(tmp_path: Path) -> None:
    audio = tmp_path / "rec.wav"
    audio.write_bytes(b"fake audio bytes")
    cfg_a = WhisperSyncConfig(beam_size=5)
    cfg_b = WhisperSyncConfig(beam_size=10)

    key_a = WhisperEngine._cache_key(audio, cfg_a, "cpu", "float32")
    key_b = WhisperEngine._cache_key(audio, cfg_b, "cpu", "float32")
    assert key_a != key_b


def test_on_model_loading_fires_once_before_first_load() -> None:
    calls: list[str] = []
    cfg = WhisperSyncConfig()
    engine = WhisperEngine(cfg, on_model_loading=lambda: calls.append("loading"))

    with patch.object(WhisperEngine, "_load", return_value=object()):
        engine._ensure_model()
        engine._ensure_model()  # already loaded -> must not fire again

    assert calls == ["loading"]


def test_on_model_loading_not_required() -> None:
    cfg = WhisperSyncConfig()
    engine = WhisperEngine(cfg)  # no callback passed
    with patch.object(WhisperEngine, "_load", return_value=object()):
        engine._ensure_model()  # must not raise
