"""Multi-track timeline view.

Renders one row per camera (video) and per audio lane, showing each clip's
position (how far it moved), applied speed change, and live sync status
(pending / working / done). Hover a clip for full details.
"""

from __future__ import annotations

import math
import time
from typing import Any

from PyQt6.QtCore import QRect, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

# status -> (fill, border) colours
_VIDEO_BASE = "#1E88E5"
_AUDIO_BASE = "#E53935"
_STATUS_ALPHA = {"pending": 55, "working": 130, "done": 255}
_WORKING_BORDER = "#FF6E40"

# A synced audio clip is tinted by how much its tempo had to change to lock to the
# picture (|speed-1|): a barely-touched clip is green, a heavily corrected one trends
# through yellow toward red. ~1.5% tempo change ≈ full red — beyond gentle drift.
_DRIFT_FULL_SCALE = 0.015


def _drift_color(speed: float) -> QColor:
    """Green→yellow→red by tempo-correction magnitude. speed==1.0 → green."""
    t = min(abs(speed - 1.0) / _DRIFT_FULL_SCALE, 1.0)
    # green (62,179,80) → yellow (224,196,64) → red (240,85,80)
    if t < 0.5:
        u = t / 0.5
        r = int(62 + (224 - 62) * u)
        g = int(179 + (196 - 179) * u)
        b = int(80 + (64 - 80) * u)
    else:
        u = (t - 0.5) / 0.5
        r = int(224 + (240 - 224) * u)
        g = int(196 + (85 - 196) * u)
        b = int(64 + (80 - 64) * u)
    return QColor(r, g, b)


_MARGIN_L = 96
_MARGIN_R = 16
_RULER_H = 22
_ROW_H = 34
_ROW_GAP = 8
_TOP = 28


_APPEAR_MS = 280.0  # fade/grow-in duration for a clip reaching "done"


def _ease_out(t: float) -> float:
    """Cubic ease-out for a snappy-but-soft appearance."""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


class TimelinePreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._clips: list[dict[str, Any]] = []
        self._rows: list[tuple[int, str]] = []  # (row_index, track_label)
        self._hit: list[tuple[QRect, dict[str, Any]]] = []
        # Animation state: per-clip appearance start time (keyed by identity) and a
        # free-running clock that drives the working-clip pulse. A single ~60fps
        # timer ticks only while something is animating, so an idle timeline is quiet.
        self._appear_start: dict[str, float] = {}
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #0A0A0A;")

    # ------------------------------------------------------------------ API
    def set_tracks(self, clips: list[dict[str, Any]]) -> None:
        prev = {self._clip_key(c): c.get("status") for c in self._clips}
        self._clips = list(clips or [])
        now = time.monotonic()
        # Start an appearance animation the moment a clip first becomes "done".
        for c in self._clips:
            key = self._clip_key(c)
            if c.get("status") == "done" and prev.get(key) != "done":
                self._appear_start[key] = now
        rows: dict[int, str] = {}
        for c in self._clips:
            rows.setdefault(int(c.get("row", 0)), str(c.get("track", "")))
        self._rows = sorted(rows.items())
        # grow to fit all rows
        needed = _TOP + _RULER_H + max(1, len(self._rows)) * (_ROW_H + _ROW_GAP) + 12
        self.setMinimumHeight(needed)
        self._sync_anim_timer()
        self.update()

    @staticmethod
    def _clip_key(clip: dict[str, Any]) -> str:
        return f"{clip.get('row', 0)}:{clip.get('kind', '')}:{clip.get('name', '')}"

    def _animating(self) -> bool:
        """True while any clip is mid-appearance or any clip is 'working' (pulses)."""
        now = time.monotonic()
        for c in self._clips:
            if c.get("status") == "working":
                return True
            start = self._appear_start.get(self._clip_key(c))
            if start is not None and (now - start) * 1000.0 < _APPEAR_MS:
                return True
        return False

    def _sync_anim_timer(self) -> None:
        if self._animating():
            if not self._anim_timer.isActive():
                self._anim_timer.start()
        elif self._anim_timer.isActive():
            self._anim_timer.stop()

    def _on_anim_tick(self) -> None:
        self.update()
        self._sync_anim_timer()  # stop once everything has settled

    def _appear_progress(self, clip: dict[str, Any]) -> float:
        """0→1 eased appearance progress for a clip; 1.0 if it has no pending anim."""
        start = self._appear_start.get(self._clip_key(clip))
        if start is None:
            return 1.0
        return _ease_out((time.monotonic() - start) * 1000.0 / _APPEAR_MS)

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
            painter.setPen(QColor("#9CA0A6"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No timeline yet — drop media and press SYNC to preview clip placement",
            )
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
        painter.setPen(QPen(QColor("#2A2A30"), 1))
        painter.drawLine(_MARGIN_L, y, _MARGIN_L + self._area_w(), y)
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        ticks = 10
        step = total / ticks
        for i in range(ticks + 1):
            t = step * i
            x = self._x(t, total)
            painter.setPen(QPen(QColor("#2A2A30"), 1))
            painter.drawLine(x, y, x, y + 5)
            painter.setPen(QColor("#9CA0A6"))
            painter.drawText(x - 18, y - 6, 40, 12, Qt.AlignmentFlag.AlignCenter, _fmt_time(t))

    def _row_y(self, row_pos: int) -> int:
        return _TOP + _RULER_H + row_pos * (_ROW_H + _ROW_GAP)

    def _paint_rows(self, painter: QPainter) -> None:
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        for pos, (_row, label) in enumerate(self._rows):
            y = self._row_y(pos)
            painter.setBrush(QBrush(QColor("#16161A")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(_MARGIN_L, y, self._area_w(), _ROW_H)
            painter.setPen(QColor("#9CA0A6"))
            painter.drawText(4, y, _MARGIN_L - 10, _ROW_H, Qt.AlignmentFlag.AlignVCenter, label)

    def _row_pos(self, row: int) -> int:
        for pos, (r, _label) in enumerate(self._rows):
            if r == row:
                return pos
        return 0

    def _paint_clip(self, painter: QPainter, clip: dict[str, Any], total: float) -> None:
        kind = clip.get("kind", "video")
        status = clip.get("status", "done")
        if kind == "video":
            base = QColor(_VIDEO_BASE)
        elif status == "done":
            # Finished audio: tint by how much tempo correction was applied (drift).
            base = _drift_color(float(clip.get("speed", 1.0)))
        else:
            # Still pending/working: keep the neutral recorder red until measured.
            base = QColor(_AUDIO_BASE)
        base.setAlpha(_STATUS_ALPHA.get(status, 255))

        full_y = self._row_y(self._row_pos(int(clip.get("row", 0)))) + 3
        full_h = _ROW_H - 6
        x = self._x(clip["offset"], total)
        w = max(3, self._x(clip["offset"] + clip["duration"], total) - x)

        # Appearance micro-animation: a freshly-"done" clip grows up from its centre
        # and fades in over _APPEAR_MS. The hit-rect uses the final geometry so hover
        # works immediately.
        appear = self._appear_progress(clip)
        h = max(2, int(full_h * (0.4 + 0.6 * appear)))
        y = full_y + (full_h - h) // 2
        rect = QRect(x, y, w, h)
        if appear < 1.0:
            painter.setOpacity(0.25 + 0.75 * appear)

        if status == "working":
            # Pulsing border to show this clip is being rendered right now.
            pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 6.0)
            bw = QColor(_WORKING_BORDER)
            bw.setAlpha(int(140 + 115 * pulse))
            painter.setPen(QPen(bw, 2))
        elif status == "pending":
            painter.setPen(QPen(base.lighter(160), 1, Qt.PenStyle.DashLine))
        else:
            painter.setPen(QPen(base.lighter(140), 1))
        painter.setBrush(QBrush(base))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setOpacity(1.0)

        # Hit-test against the final (full-height) geometry, not the animated one.
        self._hit.append((QRect(x, full_y, w, full_h), clip))

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
                lines = [
                    str(clip.get("name", "")),
                    f"track: {clip.get('track', '')} ({clip.get('kind', '')})",
                    f"offset: {clip.get('offset', 0.0):.3f}s",
                    f"duration: {clip.get('duration', 0.0):.3f}s",
                    f"source in-point: {clip.get('in_point', 0.0):.3f}s",
                    f"speed: {speed_txt}",
                    f"status: {clip.get('status', '')}",
                ]
                if clip.get("kind") == "audio" and clip.get("status") == "done":
                    lines.append(f"sync correction: {_drift_label(speed)}")
                self.setToolTip("\n".join(lines))
                return
        self.setToolTip("")


def _drift_label(speed: float) -> str:
    """Human description of the tempo correction magnitude (matches the clip tint)."""
    t = min(abs(speed - 1.0) / _DRIFT_FULL_SCALE, 1.0)
    if t < 0.15:
        return "minimal (in sync)"
    if t < 0.5:
        return "moderate"
    if t < 0.85:
        return "significant"
    return "heavy"


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"
