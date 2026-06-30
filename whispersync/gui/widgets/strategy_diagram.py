from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class StrategyDiagram(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._strategy_id = 1
        self.setMinimumHeight(80)
        self.setMaximumHeight(80)
        self.setStyleSheet("background: #0A0A0A;")

    def set_strategy(self, strategy_id: int) -> None:
        self._strategy_id = strategy_id
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._strategy_id == 1:
            self._paint_global(painter)
        elif self._strategy_id == 2:
            self._paint_segments(painter)
        elif self._strategy_id == 3:
            self._paint_speech_silence(painter)
        elif self._strategy_id == 4:
            self._paint_hybrid(painter)

        painter.end()

    def _block_rect(self, x: int, w: int, h: int = 28) -> QRect:
        cy = (self.height() - h) // 2
        return QRect(x, cy, w, h)

    def _draw_block(self, painter: QPainter, rect: QRect, color: QColor, label: str) -> None:
        painter.setPen(QPen(QColor("#2A2A30"), 1))
        painter.setBrush(QBrush(color))
        painter.drawRoundedRect(rect, 4, 4)

        painter.setPen(QColor("#F0F0F1"))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_global(self, painter: QPainter) -> None:
        margin = 16
        rect = self._block_rect(margin, self.width() - 2 * margin)
        self._draw_block(painter, rect, QColor("#E53935"), "Global atempo")

    def _paint_segments(self, painter: QPainter) -> None:
        colors = [
            QColor("#E53935"),
            QColor("#E53935"),
            QColor("#FF5722"),
            QColor("#FF7043"),
            QColor("#D84315"),
        ]
        widths = [0.25, 0.18, 0.22, 0.15, 0.20]
        margin = 16
        total_w = self.width() - 2 * margin
        x = margin

        for i, w_ratio in enumerate(widths):
            w = int(total_w * w_ratio)
            rect = self._block_rect(x, w)
            self._draw_block(painter, rect, colors[i], f"Seg {i + 1}")
            x += w + 4

    def _paint_speech_silence(self, painter: QPainter) -> None:
        margin = 16
        segments = [
            (0.18, "#E53935", "Speech"),
            (0.06, "#333333", None),
            (0.22, "#E53935", "Speech"),
            (0.08, "#333333", None),
            (0.15, "#E53935", "Speech"),
            (0.05, "#333333", None),
            (0.20, "#E53935", "Speech"),
        ]
        total_w = self.width() - 2 * margin
        x = margin

        for w_ratio, color, label in segments:
            w = int(total_w * w_ratio)
            rect = self._block_rect(x, w)

            if label is None:
                painter.setPen(QPen(QColor("#555555"), 1, Qt.PenStyle.DashLine))
                painter.setBrush(QBrush(QColor(color)))
                painter.drawRoundedRect(rect, 4, 4)
            else:
                self._draw_block(painter, rect, QColor(color), label)

            x += w + 4

    def _paint_hybrid(self, painter: QPainter) -> None:
        # phrases tempo-corrected (orange = stretched) with silence gaps between
        margin = 16
        segments = [
            (0.20, "#FF6E40", "K·Phrase"),
            (0.06, "#333333", None),
            (0.18, "#FF6E40", "K·Phrase"),
            (0.08, "#333333", None),
            (0.22, "#FF6E40", "K·Phrase"),
            (0.05, "#333333", None),
            (0.16, "#FF6E40", "K·Phrase"),
        ]
        total_w = self.width() - 2 * margin
        x = margin

        for w_ratio, color, label in segments:
            w = int(total_w * w_ratio)
            rect = self._block_rect(x, w)
            if label is None:
                painter.setPen(QPen(QColor("#555555"), 1, Qt.PenStyle.DashLine))
                painter.setBrush(QBrush(QColor(color)))
                painter.drawRoundedRect(rect, 4, 4)
            else:
                self._draw_block(painter, rect, QColor(color), label)
            x += w + 4
