from __future__ import annotations

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent, QPainter, QPen
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class DropZone(QFrame):
    path_dropped = pyqtSignal(str)

    def __init__(
        self,
        parent=None,
        accepted_extensions: list[str] | None = None,
        accept_dirs: bool = True,
        placeholder: str = "Drop video folder here",
    ) -> None:
        super().__init__(parent)
        self.accepted_extensions = accepted_extensions or [
            ".mp4",
            ".mkv",
            ".avi",
            ".mov",
            ".webm",
            ".mp3",
            ".wav",
            ".flac",
            ".aac",
            ".ogg",
        ]
        self.accept_dirs = accept_dirs
        self._hover = False
        self._dropped_name: str | None = None
        self._path: str | None = None

        self.setAcceptDrops(True)
        self.setMinimumHeight(96)
        self._apply_surface(hover=False)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel(placeholder)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("color: #9CA0A6; font-size: 13px; background: transparent;")
        layout.addWidget(self._label)

    def _apply_surface(self, hover: bool) -> None:
        bg = "#20140F" if hover else "#1A1A1D"
        self.setStyleSheet(f"background: {bg}; border-radius: 8px;")

    @property
    def current_path(self) -> str | None:
        return self._path

    def set_path(self, path: str) -> None:
        self._path = path
        self._dropped_name = os.path.basename(path)
        self._label.setText(self._dropped_name)
        self._label.setStyleSheet(
            "color: #F0F0F1; font-size: 13px; font-weight: 600; background: transparent;"
        )
        self.update()

    def _is_accepted(self, path: str) -> bool:
        if os.path.isdir(path):
            return self.accept_dirs
        _, ext = os.path.splitext(path)
        return ext.lower() in self.accepted_extensions

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if self._is_accepted(url.toLocalFile()):
                    event.acceptProposedAction()
                    self._hover = True
                    self._apply_surface(hover=True)
                    self.update()
                    return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self._hover = False
        self._apply_surface(hover=False)
        self.update()

    def dropEvent(self, event: QDropEvent) -> None:
        self._hover = False
        self._apply_surface(hover=False)
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if self._is_accepted(path):
                self.set_path(path)
                self.path_dropped.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen_color = QColor("#FF8A65") if self._hover else QColor("#FF6E40")
        pen = QPen(pen_color, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 0)))

        r = self.rect().adjusted(4, 4, -4, -4)
        painter.drawRoundedRect(r, 8, 8)
        painter.end()
