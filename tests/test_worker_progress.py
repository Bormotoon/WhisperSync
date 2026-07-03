"""Tests for the GUI worker's stage-weighted overall progress (pure logic,
no Qt/PyQt6 import needed for this specific function)."""

from __future__ import annotations

from whispersync.gui.worker import _STAGE_WEIGHTS, overall_progress


def test_stage_weights_sum_to_one() -> None:
    assert abs(sum(w for _, w in _STAGE_WEIGHTS) - 1.0) < 1e-9


def test_overall_progress_starts_at_zero() -> None:
    assert overall_progress("scanning", 0.0) == 0


def test_overall_progress_ends_at_100() -> None:
    assert overall_progress("done", 1.0) == 100


def test_overall_progress_is_monotonic_across_stage_transitions() -> None:
    # Finishing one stage at 100% must not read as LOWER than starting the
    # next stage at 0% — this is exactly the "bar jumps backwards" bug that
    # motivated weighting stages in the first place.
    stages = [name for name, _ in _STAGE_WEIGHTS]
    prev = -1
    for stage in stages:
        start_val = overall_progress(stage, 0.0)
        end_val = overall_progress(stage, 1.0)
        assert start_val >= prev
        assert end_val >= start_val
        prev = end_val


def test_overall_progress_unknown_stage_falls_back_to_raw() -> None:
    assert overall_progress("some_future_stage", 0.5) == 50


def test_worker_logs_only_changed_messages() -> None:
    # A stage that reuses the same message on every progress tick (dozens of
    # ticks per transcribed clip) must produce ONE log line, not a flood —
    # regression for the "[INFO] DJI_0829.MOV" x18 spam.
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pathlib import Path

    from PyQt6.QtWidgets import QApplication

    from whispersync.config import WhisperSyncConfig
    from whispersync.engine.pipeline import PipelineProgress
    from whispersync.gui.worker import SyncWorker

    app = QApplication.instance() or QApplication([])  # noqa: F841
    worker = SyncWorker(
        config=WhisperSyncConfig(),
        video_dir=Path("/x"),
        audio_files=[Path("/x/a.wav")],
        strategy_id=3,
        output_path=Path("/x/out.fcpxml"),
    )
    logged: list[str] = []
    worker.log.connect(logged.append)

    for prog in (0.1, 0.2, 0.3):
        worker._on_progress(
            PipelineProgress(stage="transcribing_camera", progress=prog, message="DJI_0829.MOV")
        )
    worker._on_progress(
        PipelineProgress(stage="transcribing_camera", progress=0.4, message="DJI_0830.MOV")
    )
    worker._on_progress(PipelineProgress(stage="transcribing_camera", progress=0.5, message=""))

    assert logged == ["DJI_0829.MOV", "DJI_0830.MOV"]
