from __future__ import annotations

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class TimelinePreview(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._clips: list[dict] = []
        self.setMinimumHeight(120)
        self.setMaximumHeight(120)
        self.setStyleSheet("background: #0A0A0A;")

    def set_clips(self, clips: list[dict]) -> None:
        self._clips = clips
        self.update()

    def _time_to_x(self, t: float, total: float, area_w: int) -> int:
        return int((t / total) * area_w)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin_l = 60
        margin_r = 12
        area_w = self.width() - margin_l - margin_r
        timeline_top = 28
        row_h = 32
        gap = 6

        if not self._clips:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No clips loaded")
            painter.end()
            return

        total_duration = max(c["offset"] + c["duration"] for c in self._clips)

        painter.setPen(QPen(QColor("#333333"), 1))
        painter.drawLine(margin_l, timeline_top, margin_l + area_w, timeline_top)

        step = max(1, int(total_duration / 10))
        for t in range(0, int(total_duration) + 1, step):
            x = margin_l + self._time_to_x(t, total_duration, area_w)
            painter.setPen(QColor("#888888"))
            painter.drawText(x - 15, 18, f"{t}s")
            painter.setPen(QPen(QColor("#333333"), 1))
            painter.drawLine(x, timeline_top, x, timeline_top + 6)

        for clip in self._clips:
            lane = clip.get("lane", 0)
            y = timeline_top + 12 if lane >= 1 else timeline_top + 12 + row_h + gap

            x = margin_l + self._time_to_x(clip["offset"], total_duration, area_w)
            w = max(4, self._time_to_x(clip["duration"], total_duration, area_w))

            color = QColor("#1E88E5") if lane >= 1 else QColor("#D32F2F")
            rect = QRect(x, y, w, row_h)

            painter.setPen(QPen(color.lighter(130), 1))
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(rect, 3, 3)

            painter.setPen(QColor("#EEEEEE"))
            font = painter.font()
            font.setPointSize(7)
            painter.setFont(font)
            name = clip.get("name", "")
            painter.drawText(rect.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter, name)

        painter.setPen(QColor("#888888"))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(4, timeline_top + 12 + row_h // 2 + 4, "V")
        painter.drawText(4, timeline_top + 12 + row_h + gap + row_h // 2 + 4, "A")

        painter.end()
