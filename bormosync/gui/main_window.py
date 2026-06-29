"""Main application window for BormoSync GUI."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QThread
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from bormosync.config import BormoSyncConfig
from bormosync.gui.widgets.drop_zone import DropZone
from bormosync.gui.widgets.help_page import HelpPage
from bormosync.gui.widgets.log_view import LogView
from bormosync.gui.widgets.timeline_preview import TimelinePreview
from bormosync.gui.worker import SyncWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BormoSync — Audio/Video Synchronization")
        # Floor below which the layout would get cramped; the left column scrolls
        # rather than crushing its groups. Open larger so everything fits at once.
        self.setMinimumSize(1040, 640)
        self.resize(1280, 940)

        self.config = BormoSyncConfig()
        self.settings = QSettings("BormoSync", "BormoSync")
        self._worker: SyncWorker | None = None
        self._thread: QThread | None = None

        self._setup_ui()
        self._restore_state()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 8, 16)
        left_layout.setSpacing(10)

        title = QLabel("BormoSync")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #E53935; margin-bottom: 0px;")
        left_layout.addWidget(title)

        subtitle = QLabel("Advanced Audio/Video Synchronization")
        subtitle.setStyleSheet("color: #9CA0A6; font-size: 12px; margin-bottom: 8px;")
        left_layout.addWidget(subtitle)

        video_group = QGroupBox("Video Folder")
        video_layout = QVBoxLayout(video_group)
        self.video_drop = DropZone(
            placeholder="Drop video folder here",
            accept_dirs=True,
            accepted_extensions=[],
        )
        self.video_drop.path_dropped.connect(self._on_video_dropped)
        video_layout.addWidget(self.video_drop)
        self.btn_browse_video = QPushButton("Browse...")
        self.btn_browse_video.clicked.connect(self._browse_video)
        video_layout.addWidget(self.btn_browse_video)
        left_layout.addWidget(video_group)

        audio_group = QGroupBox("Recorder Audio")
        audio_layout = QVBoxLayout(audio_group)
        self.audio_drop = DropZone(
            placeholder="Drop audio file here",
            accept_dirs=False,
            accepted_extensions=self.config.audio_exts,
        )
        self.audio_drop.path_dropped.connect(self._on_audio_dropped)
        audio_layout.addWidget(self.audio_drop)
        self.btn_browse_audio = QPushButton("Browse...")
        self.btn_browse_audio.clicked.connect(self._browse_audio)
        audio_layout.addWidget(self.btn_browse_audio)
        left_layout.addWidget(audio_group)

        strategy_group = QGroupBox("Sync Strategy")
        strategy_layout = QVBoxLayout(strategy_group)
        self.radio1 = QRadioButton("1 — Global Linear Calibration")
        self.radio2 = QRadioButton("2 — Local Time-Stretch")
        self.radio3 = QRadioButton("3 — Silence Padding (pitch-safe)")
        self.radio4 = QRadioButton("4 — Hybrid (Global + Silence)")
        self.radio1.setChecked(True)
        for r in (self.radio1, self.radio2, self.radio3, self.radio4):
            r.setMinimumHeight(26)  # never let the label clip vertically
            r.toggled.connect(self._on_strategy_changed)
            strategy_layout.addWidget(r)
        left_layout.addWidget(strategy_group)

        options_group = QGroupBox("Options")
        options_layout = QFormLayout(options_group)
        self.timebase_combo = QComboBox()
        self.timebase_combo.addItems(["camera", "recorder"])
        options_layout.addRow("Timebase source:", self.timebase_combo)
        self.crossfade_check = QCheckBox("Crossfade segment seams (declick)")
        self.crossfade_check.setChecked(self.config.crossfade_enabled)
        options_layout.addRow(self.crossfade_check)
        left_layout.addWidget(options_group)

        self.btn_sync = QPushButton("SYNC")
        self.btn_sync.setObjectName("primaryButton")
        self.btn_sync.setMinimumHeight(48)
        self.btn_sync.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sync.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.btn_sync.clicked.connect(self._start_sync)
        left_layout.addWidget(self.btn_sync)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_sync)
        left_layout.addWidget(self.btn_cancel)

        left_layout.addStretch()

        right_tabs = QTabWidget()

        run_tab = QWidget()
        right_layout = QVBoxLayout(run_tab)
        right_layout.setContentsMargins(8, 12, 8, 8)
        right_layout.setSpacing(12)

        timeline_group = QGroupBox("Timeline")
        timeline_layout = QVBoxLayout(timeline_group)
        self.timeline_preview = TimelinePreview()
        self.timeline_preview.setMinimumHeight(180)
        timeline_layout.addWidget(self.timeline_preview)
        right_layout.addWidget(timeline_group, stretch=1)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.stage_label = QLabel("Ready")
        self.stage_label.setStyleSheet("color: #9CA0A6; font-size: 13px;")
        progress_layout.addWidget(self.stage_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        right_layout.addWidget(progress_group)

        result_group = QGroupBox("Results")
        result_layout = QVBoxLayout(result_group)
        self.result_label = QLabel("No results yet")
        self.result_label.setStyleSheet("color: #9CA0A6;")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # A word-wrapped label reports a collapsed height hint; reserve room for
        # the four metric lines (the output path may wrap onto a fifth) so the
        # readout never clips.
        self.result_label.setMinimumHeight(96)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # Monospace metrics so the K / offset / residual figures line up.
        result_font = QFont()
        result_font.setFamilies(["JetBrains Mono", "DejaVu Sans Mono", "Consolas", "monospace"])
        result_font.setStyleHint(QFont.StyleHint.Monospace)
        result_font.setPointSize(10)
        self.result_label.setFont(result_font)
        result_layout.addWidget(self.result_label)
        self.btn_open_folder = QPushButton("Open Output Folder")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self._open_output_folder)
        result_layout.addWidget(self.btn_open_folder)
        right_layout.addWidget(result_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_view = LogView()
        log_layout.addWidget(self.log_view)
        right_layout.addWidget(log_group, stretch=1)

        right_tabs.addTab(run_tab, "Run")

        self.help_page = HelpPage()
        right_tabs.addTab(self.help_page, "Help")
        self.right_tabs = right_tabs

        # Wrap the controls column in a scroll area so a short window scrolls it
        # instead of crushing the groups (the radios used to clip). The panel keeps
        # its natural width and never shrinks below what the content needs.
        left_panel.setMinimumWidth(320)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(340)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 0)  # controls column stays compact
        splitter.setStretchFactor(1, 1)  # timeline / simulator side absorbs resize
        splitter.setSizes([400, 760])

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Affordance: every clickable control gets the hand cursor.
        for btn in self.findChildren(QPushButton):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        for rb in self.findChildren(QRadioButton):
            rb.setCursor(Qt.CursorShape.PointingHandCursor)
        self.crossfade_check.setCursor(Qt.CursorShape.PointingHandCursor)

        self._on_strategy_changed()

    def _get_strategy_id(self) -> int:
        if self.radio2.isChecked():
            return 2
        if self.radio3.isChecked():
            return 3
        if self.radio4.isChecked():
            return 4
        return 1

    def _on_strategy_changed(self) -> None:
        self.help_page.set_strategy(self._get_strategy_id())

    def _on_video_dropped(self, path: str) -> None:
        self.settings.setValue("last_video_dir", path)
        self.log_view.append_log(f"Video folder: {path}")

    def _on_audio_dropped(self, path: str) -> None:
        self.settings.setValue("last_audio_file", path)
        self.log_view.append_log(f"Audio file: {path}")

    def _browse_video(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if path:
            self.video_drop.set_path(path)

    def _browse_audio(self) -> None:
        exts = " ".join(f"*{e}" for e in self.config.audio_exts)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", f"Audio Files ({exts})"
        )
        if path:
            self.audio_drop.set_path(path)

    def _start_sync(self) -> None:
        video_path = self.video_drop.current_path
        audio_path = self.audio_drop.current_path

        if not video_path or not Path(video_path).is_dir():
            QMessageBox.warning(self, "Error", "Please select a video folder.")
            return
        if not audio_path or not Path(audio_path).is_file():
            QMessageBox.warning(self, "Error", "Please select an audio file.")
            return

        output_dir = self.config.resolved_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "sync_output.fcpxml"

        strategy_id = self._get_strategy_id()
        self.config.timebase_source = self.timebase_combo.currentText()
        self.config.crossfade_enabled = self.crossfade_check.isChecked()

        self.right_tabs.setCurrentIndex(0)  # show the Run tab during processing
        self.btn_sync.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_view.clear_log()
        self.log_view.append_log(f"Starting sync with Strategy {strategy_id}...")

        self._worker = SyncWorker(
            config=self.config,
            video_dir=Path(video_path),
            audio_files=[Path(audio_path)],
            strategy_id=strategy_id,
            output_path=output_path,
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.stage.connect(self._on_stage)
        self._worker.log.connect(lambda msg: self.log_view.append_log(msg))
        self._worker.timeline.connect(self.timeline_preview.set_tracks)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._thread.start()

    def _cancel_sync(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.log_view.append_log("Cancellation requested...", "WARNING")

    def _on_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    def _on_stage(self, stage: str) -> None:
        self.stage_label.setText(stage)
        self.status_bar.showMessage(stage)

    def _on_finished(self, result: object) -> None:
        self.btn_sync.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)
        self.stage_label.setText("Done!")

        from bormosync.models import SyncResult

        if isinstance(result, SyncResult):
            self.result_label.setStyleSheet("color: #F0F0F1;")
            self.result_label.setText(
                f"{'Anchors':<9}{result.anchors_used}\n"
                f"{'K':<9}{result.alignment.k:.6f}\n"
                f"{'Residual':<9}{result.alignment.residual_ms:.1f} ms\n"
                f"{'Output':<9}{result.fcpxml_path}"
            )
            self.btn_open_folder.setEnabled(True)
            self._output_path = result.fcpxml_path.parent
            # The timeline is kept live via the worker's `timeline` signal.

        self.log_view.append_log("Sync complete!", "INFO")
        self.status_bar.showMessage("Sync complete!")

    def _on_error(self, msg: str) -> None:
        self.btn_sync.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.stage_label.setText("Error!")
        self.log_view.append_log(f"ERROR: {msg}", "ERROR")
        self.status_bar.showMessage("Error!")
        QMessageBox.critical(self, "Sync Error", msg)

    def _open_output_folder(self) -> None:
        if hasattr(self, "_output_path"):
            import subprocess

            subprocess.Popen(["xdg-open", str(self._output_path)])

    def _restore_state(self) -> None:
        last_video = self.settings.value("last_video_dir", "")
        last_audio = self.settings.value("last_audio_file", "")
        if last_video and Path(str(last_video)).exists():
            self.video_drop.set_path(str(last_video))
        if last_audio and Path(str(last_audio)).exists():
            self.audio_drop.set_path(str(last_audio))

    def closeEvent(self, event: object) -> None:
        if self._thread and self._thread.isRunning():
            if self._worker:
                self._worker.cancel()
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)  # type: ignore[arg-type]


def main() -> None:
    app = QApplication(sys.argv)

    # Modern UI font stack with explicit anti-aliasing; falls back gracefully
    # to whatever the platform provides.
    app_font = QFont()
    app_font.setFamilies(
        ["Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Arial", "sans-serif"]
    )
    app_font.setPointSize(10)
    app_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(app_font)

    qss_path = Path(__file__).parent / "theme.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
