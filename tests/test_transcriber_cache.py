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
    engine = WhisperEngine(cfg, on_model_loading=calls.append)

    with (
        patch.object(WhisperEngine, "_load", return_value=object()),
        patch("whispersync.engine.transcriber._local_model_path", return_value=None),
    ):
        engine._ensure_model()
        engine._ensure_model()  # already loaded -> must not fire again

    assert len(calls) == 1


def test_on_model_loading_reports_cached_model_and_uses_local_path() -> None:
    calls: list[str] = []
    cfg = WhisperSyncConfig()
    engine = WhisperEngine(cfg, on_model_loading=calls.append)

    with (
        patch.object(WhisperEngine, "_load", return_value=object()),
        patch(
            "whispersync.engine.transcriber._local_model_path",
            return_value="/fake/hub/snapshots/abc",
        ),
    ):
        engine._ensure_model()

    assert len(calls) == 1
    assert "found on disk" in calls[0]
    assert "download" not in calls[0].lower()
    # The actual load must go through the LOCAL path (fully offline), not the
    # model name (which would re-check the hub online on every start).
    assert engine._model_source == "/fake/hub/snapshots/abc"


def test_on_model_loading_reports_download_when_not_cached() -> None:
    calls: list[str] = []
    cfg = WhisperSyncConfig()
    engine = WhisperEngine(cfg, on_model_loading=calls.append)

    with (
        patch.object(WhisperEngine, "_load", return_value=object()),
        patch("whispersync.engine.transcriber._local_model_path", return_value=None),
    ):
        engine._ensure_model()

    assert len(calls) == 1
    assert "downloading" in calls[0].lower()
    assert engine._model_source == cfg.model  # name -> faster-whisper downloads


def test_on_model_loading_not_required() -> None:
    cfg = WhisperSyncConfig()
    engine = WhisperEngine(cfg)  # no callback passed
    with (
        patch.object(WhisperEngine, "_load", return_value=object()),
        patch("whispersync.engine.transcriber._local_model_path", return_value=None),
    ):
        engine._ensure_model()  # must not raise


def test_local_model_path_accepts_ct2_directory(tmp_path: Path) -> None:
    from whispersync.engine.transcriber import _local_model_path

    model_dir = tmp_path / "my-ct2-model"
    model_dir.mkdir()
    assert _local_model_path(str(model_dir)) == str(model_dir)


def test_local_model_path_none_for_unknown_model() -> None:
    from whispersync.engine.transcriber import _local_model_path

    # A model that certainly isn't in the local HF cache -> needs a download.
    assert _local_model_path("definitely-not-a-real-model-xyz") is None


def test_prune_cache_removes_only_stale_entries(tmp_path: Path) -> None:
    import os
    import time

    from whispersync.engine.transcriber import _prune_cache

    old = tmp_path / "old.json"
    fresh = tmp_path / "fresh.json"
    other = tmp_path / "not_cache.txt"  # non-.json files are never touched
    for f in (old, fresh, other):
        f.write_text("{}")
    stale_mtime = time.time() - 10 * 86400
    os.utime(old, (stale_mtime, stale_mtime))
    os.utime(other, (stale_mtime, stale_mtime))

    removed = _prune_cache(tmp_path, max_age_days=7)
    assert removed == 1
    assert not old.exists()
    assert fresh.exists()
    assert other.exists()


def test_prune_cache_missing_dir_is_noop(tmp_path: Path) -> None:
    from whispersync.engine.transcriber import _prune_cache

    assert _prune_cache(tmp_path / "does_not_exist", max_age_days=7) == 0


def test_engine_init_prunes_when_configured(tmp_path: Path) -> None:
    import os
    import time

    old = tmp_path / "stale.json"
    old.write_text("{}")
    stale_mtime = time.time() - 30 * 86400
    os.utime(old, (stale_mtime, stale_mtime))

    cfg = WhisperSyncConfig(cache_dir=str(tmp_path), cache_max_age_days=7)
    WhisperEngine(cfg)
    assert not old.exists()


def test_engine_init_keeps_cache_forever_by_default(tmp_path: Path) -> None:
    import os
    import time

    old = tmp_path / "ancient.json"
    old.write_text("{}")
    ancient = time.time() - 365 * 86400
    os.utime(old, (ancient, ancient))

    cfg = WhisperSyncConfig(cache_dir=str(tmp_path))  # cache_max_age_days=0
    WhisperEngine(cfg)
    assert old.exists()
