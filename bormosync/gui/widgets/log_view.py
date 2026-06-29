from __future__ import annotations

from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QTextEdit


class LogView(QTextEdit):
    _COLORS: dict[str, str] = {
        "DEBUG": "#9CA0A6",
        "INFO": "#F0F0F1",
        "WARNING": "#FF6E40",
        "ERROR": "#F05550",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        # Visual styling (background, rounded border) comes from theme.qss.
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.setFont(mono)

    def append_log(self, message: str, level: str = "INFO") -> None:
        color = self._COLORS.get(level.upper(), "#F0F0F1")
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(f"[{level.upper():>7}] {message}\n", fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def clear_log(self) -> None:
        self.clear()
