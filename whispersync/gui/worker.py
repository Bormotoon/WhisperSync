"""QObject-based worker for background pipeline execution."""

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from whispersync.config import WhisperSyncConfig
from whispersync.engine.pipeline import PipelineProgress, run_pipeline


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
        self._cancelled = False

    def _on_progress(self, p: PipelineProgress) -> None:
        if self._cancelled:
            raise InterruptedError("Cancelled by user")
        self.stage.emit(p.stage)
        self.progress.emit(int(p.progress * 100))
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
            )
            self.finished.emit(result)
        except InterruptedError:
            self.log.emit("Pipeline cancelled by user")
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self) -> None:
        self._cancelled = True
