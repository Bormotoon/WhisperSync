"""QObject-based worker for background pipeline execution."""

import threading
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from whispersync.config import WhisperSyncConfig
from whispersync.engine.pipeline import PipelineProgress, run_pipeline

# Rough share of total wall-clock time each pipeline stage takes on a typical
# run, used to blend per-stage progress (0..1) into one overall percentage.
# Transcription dominates (GPU-bound, scales with media length); rendering is
# the other major cost. Without this, the bar used to reset to 0% and race to
# 100% on every single stage transition — technically correct per-stage but
# visually looked like the run kept restarting. See PROJECT_ANALYSIS.md §7.5.
# Must sum to 1.0 across all stages the pipeline can emit, in the order they
# occur (see pipeline.py's `_notify` call sites for the stage names).
_STAGE_WEIGHTS: list[tuple[str, float]] = [
    ("scanning", 0.02),
    ("transcribing_recorder", 0.20),
    ("transcribing_camera", 0.30),
    ("aligning", 0.03),
    ("planning", 0.03),
    ("processing", 0.35),
    ("exporting", 0.05),
    ("done", 0.02),
]


def _stage_starts(weights: list[tuple[str, float]]) -> dict[str, float]:
    starts: dict[str, float] = {}
    cumulative = 0.0
    for name, weight in weights:
        starts[name] = cumulative
        cumulative += weight
    return starts


_STAGE_START = _stage_starts(_STAGE_WEIGHTS)


def overall_progress(stage: str, stage_progress: float) -> int:
    """Blend a pipeline stage name + its own 0..1 progress into an overall
    0..100 percentage, using ``_STAGE_WEIGHTS``. Unknown stage names (should
    not happen, but defensive) fall back to the stage's own raw progress."""
    start = _STAGE_START.get(stage)
    weight = dict(_STAGE_WEIGHTS).get(stage)
    if start is None or weight is None:
        return int(max(0.0, min(1.0, stage_progress)) * 100)
    return int(max(0.0, min(100.0, (start + weight * stage_progress) * 100)))


class SyncWorker(QObject):
    progress = pyqtSignal(int)
    stage = pyqtSignal(str)
    log = pyqtSignal(str)
    timeline = pyqtSignal(object)  # list[dict] timeline snapshot
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        config: WhisperSyncConfig,
        video_dir: Path,
        audio_files: list[Path],
        strategy_id: int,
        output_path: Path,
    ) -> None:
        super().__init__()
        self.config = config
        self.video_dir = video_dir
        self.audio_files = audio_files
        self.strategy_id = strategy_id
        self.output_path = output_path
        # A threading.Event (not a plain bool) so run_pipeline can poll it
        # directly from inside a long render job — between individual ffmpeg
        # pieces, not just between whole clips/stages. See
        # PROJECT_ANALYSIS.md §3.5.
        self._cancel_event = threading.Event()

    def _on_progress(self, p: PipelineProgress) -> None:
        if self._cancel_event.is_set():
            raise InterruptedError("Cancelled by user")
        self.stage.emit(p.stage)
        self.progress.emit(overall_progress(p.stage, p.progress))
        if p.message:
            self.log.emit(p.message)
        if p.clips is not None:
            self.timeline.emit(p.clips)

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = run_pipeline(
                config=self.config,
                video_dir=self.video_dir,
                audio_files=self.audio_files,
                strategy_id=self.strategy_id,
                output_path=self.output_path,
                progress_callback=self._on_progress,
                cancel_event=self._cancel_event,
            )
            self.finished.emit(result)
        except InterruptedError:
            self.log.emit("Pipeline cancelled by user")
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self) -> None:
        self._cancel_event.set()
