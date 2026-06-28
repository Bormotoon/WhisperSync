from __future__ import annotations

from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QTextEdit


class LogView(QTextEdit):
    _COLORS: dict[str, str] = {
        "DEBUG": "#888888",
        "INFO": "#EEEEEE",
        "WARNING": "#FF5722",
        "ERROR": "#D32F2F",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("QTextEdit { background: #0A0A0A; color: #EEEEEE; border: none; }")
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.setFont(mono)

    def append_log(self, message: str, level: str = "INFO") -> None:
        color = self._COLORS.get(level.upper(), "#EEEEEE")
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(f"[{level.upper():>7}] {message}\n", fmt)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def clear_log(self) -> None:
        self.clear()
