"""Multi-track timeline view.

Renders one row per camera (video) and per audio lane, showing each clip's
position (how far it moved), applied speed change, and live sync status
(pending / working / done). Hover a clip for full details.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

# status -> (fill, border) colours
_VIDEO_BASE = "#1E88E5"
_AUDIO_BASE = "#D32F2F"
_STATUS_ALPHA = {"pending": 55, "working": 130, "done": 255}
_WORKING_BORDER = "#FF5722"

_MARGIN_L = 96
_MARGIN_R = 16
_RULER_H = 22
_ROW_H = 34
_ROW_GAP = 8
_TOP = 28


class TimelinePreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._clips: list[dict[str, Any]] = []
        self._rows: list[tuple[int, str]] = []  # (row_index, track_label)
        self._hit: list[tuple[QRect, dict[str, Any]]] = []
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #0A0A0A;")

    # ------------------------------------------------------------------ API
    def set_tracks(self, clips: list[dict[str, Any]]) -> None:
        self._clips = list(clips or [])
        rows: dict[int, str] = {}
        for c in self._clips:
            rows.setdefault(int(c.get("row", 0)), str(c.get("track", "")))
        self._rows = sorted(rows.items())
        # grow to fit all rows
        needed = _TOP + _RULER_H + max(1, len(self._rows)) * (_ROW_H + _ROW_GAP) + 12
        self.setMinimumHeight(needed)
        self.update()

    # keep the old name working for any existing callers
    def set_clips(self, clips: list[dict[str, Any]]) -> None:
        self.set_tracks(clips)

    # -------------------------------------------------------------- helpers
    def _area_w(self) -> int:
        return max(1, self.width() - _MARGIN_L - _MARGIN_R)

    def _x(self, t: float, total: float) -> int:
        return _MARGIN_L + int((t / total) * self._area_w()) if total > 0 else _MARGIN_L

    @staticmethod
    def _speed_label(speed: float) -> str:
        pct = (speed - 1.0) * 100.0
        if abs(pct) < 0.005:
            return ""
        return f"{pct:+.2f}%"

    # --------------------------------------------------------------- paint
    def paintEvent(self, event: Any) -> None:  # noqa: ANN401
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._hit = []

        if not self._clips:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No timeline yet")
            painter.end()
            return

        total = max(c["offset"] + c["duration"] for c in self._clips) or 1.0

        self._paint_ruler(painter, total)
        self._paint_rows(painter)
        for clip in self._clips:
            self._paint_clip(painter, clip, total)

        painter.end()

    def _paint_ruler(self, painter: QPainter, total: float) -> None:
        y = _TOP
        painter.setPen(QPen(QColor("#333333"), 1))
        painter.drawLine(_MARGIN_L, y, _MARGIN_L + self._area_w(), y)
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        ticks = 10
        step = total / ticks
        for i in range(ticks + 1):
            t = step * i
            x = self._x(t, total)
            painter.setPen(QPen(QColor("#333333"), 1))
            painter.drawLine(x, y, x, y + 5)
            painter.setPen(QColor("#888888"))
            painter.drawText(x - 18, y - 6, 40, 12, Qt.AlignmentFlag.AlignCenter, _fmt_time(t))

    def _row_y(self, row_pos: int) -> int:
        return _TOP + _RULER_H + row_pos * (_ROW_H + _ROW_GAP)

    def _paint_rows(self, painter: QPainter) -> None:
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        for pos, (_row, label) in enumerate(self._rows):
            y = self._row_y(pos)
            painter.setBrush(QBrush(QColor("#141414")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(_MARGIN_L, y, self._area_w(), _ROW_H)
            painter.setPen(QColor("#AAAAAA"))
            painter.drawText(4, y, _MARGIN_L - 10, _ROW_H, Qt.AlignmentFlag.AlignVCenter, label)

    def _row_pos(self, row: int) -> int:
        for pos, (r, _label) in enumerate(self._rows):
            if r == row:
                return pos
        return 0

    def _paint_clip(self, painter: QPainter, clip: dict[str, Any], total: float) -> None:
        kind = clip.get("kind", "video")
        status = clip.get("status", "done")
        base = QColor(_VIDEO_BASE if kind == "video" else _AUDIO_BASE)
        base.setAlpha(_STATUS_ALPHA.get(status, 255))

        y = self._row_y(self._row_pos(int(clip.get("row", 0)))) + 3
        h = _ROW_H - 6
        x = self._x(clip["offset"], total)
        w = max(3, self._x(clip["offset"] + clip["duration"], total) - x)
        rect = QRect(x, y, w, h)

        if status == "working":
            painter.setPen(QPen(QColor(_WORKING_BORDER), 2))
        elif status == "pending":
            painter.setPen(QPen(base.lighter(160), 1, Qt.PenStyle.DashLine))
        else:
            painter.setPen(QPen(base.lighter(140), 1))
        painter.setBrush(QBrush(base))
        painter.drawRoundedRect(rect, 3, 3)

        self._hit.append((rect, clip))

        # labels: name + speed badge (if any)
        painter.setPen(QColor("#FFFFFF"))
        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)
        name = str(clip.get("name", ""))
        speed_txt = self._speed_label(float(clip.get("speed", 1.0)))
        text = f"{name}  {speed_txt}" if speed_txt else name
        if w > 28:
            painter.drawText(rect.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter, text)

    # ------------------------------------------------------------- tooltips
    def mouseMoveEvent(self, event: Any) -> None:  # noqa: ANN401
        pos = event.position().toPoint()
        for rect, clip in reversed(self._hit):
            if rect.contains(pos):
                speed = float(clip.get("speed", 1.0))
                speed_txt = self._speed_label(speed) or "0% (unchanged)"
                self.setToolTip(
                    f"{clip.get('name', '')}\n"
                    f"track: {clip.get('track', '')} ({clip.get('kind', '')})\n"
                    f"offset: {clip.get('offset', 0.0):.3f}s\n"
                    f"duration: {clip.get('duration', 0.0):.3f}s\n"
                    f"source in-point: {clip.get('in_point', 0.0):.3f}s\n"
                    f"speed: {speed_txt}\n"
                    f"status: {clip.get('status', '')}"
                )
                return
        self.setToolTip("")


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"
