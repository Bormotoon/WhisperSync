from __future__ import annotations

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent, QPainter, QPen
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class DropZone(QFrame):
    path_dropped = pyqtSignal(str)
    # Emitted (in addition to path_dropped, with the first path) when
    # accept_multiple is set and one or more accepted files are dropped.
    paths_dropped = pyqtSignal(list)

    def __init__(
        self,
        parent=None,
        accepted_extensions: list[str] | None = None,
        accept_dirs: bool = True,
        accept_multiple: bool = False,
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
        self.accept_multiple = accept_multiple
        self._hover = False
        self._dropped_name: str | None = None
        self._path: str | None = None
        self._paths: list[str] = []

        self.setAcceptDrops(True)
        self.setMinimumHeight(84)
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

    @property
    def current_paths(self) -> list[str]:
        """All selected paths when accept_multiple is set (falls back to a
        single-element list from current_path otherwise)."""
        if self._paths:
            return list(self._paths)
        return [self._path] if self._path else []

    def set_path(self, path: str) -> None:
        self._path = path
        self._paths = [path]
        self._dropped_name = os.path.basename(path)
        self._label.setText(self._dropped_name)
        self._label.setStyleSheet(
            "color: #F0F0F1; font-size: 13px; font-weight: 600; background: transparent;"
        )
        self.update()

    def set_paths(self, paths: list[str]) -> None:
        if not paths:
            return
        if len(paths) == 1:
            self.set_path(paths[0])
            return
        self._paths = list(paths)
        self._path = paths[0]
        self._label.setText(f"{len(paths)} files: " + ", ".join(os.path.basename(p) for p in paths))
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
        if self.accept_multiple:
            accepted = [
                url.toLocalFile()
                for url in event.mimeData().urls()
                if self._is_accepted(url.toLocalFile())
            ]
            if accepted:
                self.set_paths(accepted)
                self.path_dropped.emit(accepted[0])
                self.paths_dropped.emit(accepted)
                event.acceptProposedAction()
                return
            event.ignore()
            return
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
