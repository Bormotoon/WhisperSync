"""Interactive micro-synchronization simulator.

A teaching playground that mirrors the look of the real WhisperSync timeline so a
user recognises it instantly. Two tracks — a blue **reference** (camera/video)
and a red **recorder** (clean audio) — are laid out exactly like the run-time
timeline, with the same ruler, row strips and per-clip speed badges.

Drag the two sliders (per-phrase clock **drift** and **phrase length**) and the
panel shows, for the currently selected strategy, how the recorder phrases are
reshaped to line up with the picture, plus the resulting **accuracy** vs
**distortion** trade-off and a sweeping playhead.

The method is *not* chosen here — it follows the strategy radio buttons in the
main window (``set_strategy``), so exploring strategies and learning are one and
the same action.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# -- palette: identical tokens to TimelinePreview --------------------------
_REF = "#1E88E5"  # reference / camera (video, blue)
_REC = "#E53935"  # recorder (clean audio, red)
_PAD = "#3A2A2E"  # inserted padding (faint)
_PLAY = "#64B5F6"  # playhead
_ACCENT = "#FF6E40"  # distortion / stretch highlight (= timeline working border)
_GOOD = "#3FB950"
_BAD = "#F05550"
_TEXT = "#F0F0F1"
_MUTED = "#9CA0A6"
_BORDER = "#2A2A30"
_ROWBG = "#16161A"

# layout constants borrowed from the timeline so the two read the same
_MARGIN_L = 96
_MARGIN_R = 16
_RULER_H = 22
_ROW_H = 34
_ROW_GAP = 14
_TOP = 26

_METHOD = {1: "Global linear", 2: "Local time-stretch", 3: "Hybrid"}
_NOTE = {
    1: "One global atempo over the whole clip — no seams. Fully cancels drift "
    "only when the drift is linear; speech is gently sped up/slowed everywhere.",
    2: "Every phrase is time-stretched to fit its slot. Alignment is perfect, but "
    "the speech itself is stretched — audible on large drift.",
    3: "Stretch per phrase + padded gaps — near-perfect alignment with roughly "
    "half the distortion of pure stretching. The usual best choice.",
}

_N_PHRASES = 4
_GAP_MS = 700
_SWEEP_MS = 4200  # time for the playhead to cross once


@dataclass
class _SimModel:
    strategy: int = 1
    drift_ms: int = 140
    phrase_ms: int = 3000
    play_t: float = 0.0  # 0..1 playhead position

    def total_ms(self) -> int:
        return _N_PHRASES * (self.phrase_ms + _GAP_MS)

    def speed_pct(self) -> float:
        """Tempo change applied to speech by the current strategy (0 == none)."""
        if self.drift_ms == 0:
            return 0.0
        # hybrid shares the correction with padding -> stretches roughly half
        eff = self.drift_ms / 2.0 if self.strategy == 3 else float(self.drift_ms)
        speed = self.phrase_ms / (self.phrase_ms + eff)
        return (speed - 1.0) * 100.0

    def metrics(self) -> tuple[float, float]:
        """Return (accuracy %, distortion index 0..1) for the current settings."""
        d = abs(self.drift_ms)
        s = self.strategy
        if s in (1, 2):
            return 100.0, min(d / 500.0, 1.0)
        return 100.0, min(d / 1000.0, 1.0)  # hybrid: ~half the distortion


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


# Same green→yellow→red drift language as the real timeline, driven here by the
# tempo change the strategy applies (in %). ~1.5% ≈ full red.
_DRIFT_FULL_PCT = 1.5


def _drift_color(speed_pct: float) -> QColor:
    t = min(abs(speed_pct) / _DRIFT_FULL_PCT, 1.0)
    if t < 0.5:
        u = t / 0.5
        return QColor(int(62 + 162 * u), int(179 + 17 * u), int(80 - 16 * u))
    u = (t - 0.5) / 0.5
    return QColor(int(224 + 16 * u), int(196 - 111 * u), int(64 + 16 * u))


class _SimCanvas(QWidget):
    """The painted part: two tracks, blocks, playhead, ruler and legend."""

    def __init__(self, model: _SimModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.m = model
        self.setMinimumHeight(190)
        self.setMinimumWidth(440)

    def _x(self, ms: float, total: float) -> float:
        area = max(1, self.width() - _MARGIN_L - _MARGIN_R)
        return _MARGIN_L + (ms / total) * area if total > 0 else _MARGIN_L

    def _row_y(self, row: int) -> int:
        return _TOP + _RULER_H + row * (_ROW_H + _ROW_GAP)

    def paintEvent(self, event: object) -> None:  # noqa: ANN001
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = self.m
        total = float(m.total_ms())

        self._paint_ruler(p, total)
        self._paint_rows(p)

        ref_y = self._row_y(0) + 3
        rec_y = self._row_y(1) + 3
        bh = _ROW_H - 6
        pct = m.speed_pct()
        badge = "" if abs(pct) < 0.005 else f"{pct:+.2f}%"

        for i in range(_N_PHRASES):
            rs = i * (m.phrase_ms + _GAP_MS)
            ref_x = self._x(rs, total)
            ref_w = max(3.0, self._x(rs + m.phrase_ms, total) - ref_x)
            self._block(p, ref_x, ref_y, ref_w, bh, QColor(_REF), "video")

            g_x = self._x(rs, total)
            g_end = self._x(rs + m.phrase_ms, total)
            g_w = max(3.0, g_end - g_x)
            # Tint the synced phrase by the tempo correction, exactly like the real
            # timeline: green when barely touched, red when heavily stretched.
            rec_color = _drift_color(pct)
            self._block(p, g_x, rec_y, g_w, bh, rec_color, "audio", badge=badge)

            # padding bars: strategy 3 (Hybrid) also adjusts the silence between
            # phrases to absorb the residual after each phrase's own stretch.
            if m.strategy == 3 and i < _N_PHRASES - 1:
                next_x = self._x((i + 1) * (m.phrase_ms + _GAP_MS), total)
                if next_x - g_end > 5:
                    self._padding(p, g_end, rec_y, next_x - g_end, bh)

        if m.strategy == 1 and m.drift_ms != 0:
            p.setPen(QColor(_ACCENT))
            f = QFont()
            f.setPointSize(8)
            p.setFont(f)
            p.drawText(
                _MARGIN_L,
                self._row_y(1) + _ROW_H + 2,
                self.width() - _MARGIN_L - _MARGIN_R,
                14,
                Qt.AlignmentFlag.AlignHCenter,
                "one global atempo across the whole clip",
            )

        self._paint_playhead(p, total)
        self._paint_legend(p)
        p.end()

    # primitives ------------------------------------------------------------
    def _block(
        self,
        p: QPainter,
        x: float,
        y: float,
        w: float,
        h: float,
        color: QColor,
        kind: str,
        badge: str = "",
    ) -> None:
        p.setPen(QPen(color.lighter(140), 1))
        p.setBrush(QBrush(color))
        p.drawRoundedRect(int(x), int(y), int(w), int(h), 3, 3)
        if badge:  # this speech is time-stretched -> flag it like a real seam
            p.setPen(QPen(QColor(_ACCENT), 2))
            p.drawLine(int(x) + 2, int(y) + int(h) - 1, int(x + w) - 2, int(y) + int(h) - 1)
        f = QFont()
        f.setPointSize(7)
        p.setFont(f)
        p.setPen(QColor("#FFFFFF"))
        text = f"{kind}  {badge}" if badge else kind
        if w > 30:
            p.drawText(int(x) + 4, int(y), int(w) - 8, int(h), Qt.AlignmentFlag.AlignVCenter, text)

    def _padding(self, p: QPainter, x: float, y: float, w: float, h: float) -> None:
        p.setPen(QPen(QColor(_ACCENT), 1, Qt.PenStyle.DashLine))
        p.setBrush(QBrush(QColor(_PAD)))
        ph = h * 0.66
        p.drawRoundedRect(int(x), int(y + (h - ph) / 2), int(w), int(ph), 3, 3)
        if w > 40:
            p.setPen(QColor(_MUTED))
            f = QFont()
            f.setPointSize(7)
            p.setFont(f)
            p.drawText(int(x), int(y), int(w), int(h), Qt.AlignmentFlag.AlignCenter, "pad")

    def _paint_ruler(self, p: QPainter, total: float) -> None:
        y = _TOP
        area_r = self.width() - _MARGIN_R
        p.setPen(QPen(QColor(_BORDER), 1))
        p.drawLine(_MARGIN_L, y, area_r, y)
        f = QFont()
        f.setPointSize(8)
        p.setFont(f)
        ticks = 8
        for i in range(ticks + 1):
            t = total * i / ticks
            x = self._x(t, total)
            p.setPen(QPen(QColor(_BORDER), 1))
            p.drawLine(int(x), y, int(x), y + 5)
            p.setPen(QColor(_MUTED))
            p.drawText(
                int(x) - 18, y - 6, 40, 12, Qt.AlignmentFlag.AlignCenter, _fmt_time(t / 1000.0)
            )

    def _paint_rows(self, p: QPainter) -> None:
        f = QFont()
        f.setPointSize(8)
        p.setFont(f)
        for row, label in ((0, "Camera"), (1, "Recorder")):
            y = self._row_y(row)
            p.setBrush(QBrush(QColor(_ROWBG)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(_MARGIN_L, y, self.width() - _MARGIN_L - _MARGIN_R, _ROW_H)
            p.setPen(QColor(_MUTED))
            p.drawText(4, y, _MARGIN_L - 10, _ROW_H, Qt.AlignmentFlag.AlignVCenter, label)

    def _paint_playhead(self, p: QPainter, total: float) -> None:
        x = self._x(self.m.play_t * total, total)
        y_top = _TOP
        y_bot = self._row_y(1) + _ROW_H
        p.setPen(QPen(QColor(_PLAY), 2))
        p.drawLine(int(x), y_top, int(x), y_bot)
        p.setBrush(QBrush(QColor(_PLAY)))
        p.setPen(Qt.PenStyle.NoPen)
        head = QPolygonF(
            [
                QPointF(x - 4, y_top - 6),
                QPointF(x + 4, y_top - 6),
                QPointF(x, y_top),
            ]
        )
        p.drawPolygon(head)

    def _paint_legend(self, p: QPainter) -> None:
        f = QFont()
        f.setPointSize(7)
        p.setFont(f)
        y = self.height() - 12
        x = _MARGIN_L
        for color, label in (
            (_REF, "Reference"),
            (_GOOD, "In sync"),
            (_BAD, "Heavy correction"),
            (_PAD, "Padding"),
            (_ACCENT, "Stretched / residual"),
        ):
            p.setBrush(QBrush(QColor(color)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(x, y - 8, 8, 8)
            p.setPen(QColor(_MUTED))
            w = 18 + len(label) * 6
            p.drawText(x + 12, y - 10, w, 12, Qt.AlignmentFlag.AlignVCenter, label)
            x += w + 14


class SyncSimulator(QWidget):
    """Full simulator panel: header, canvas, readouts and sliders."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = _SimModel()
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)
        self._build()
        self._refresh()

    # construction ----------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Micro-sync simulator")
        title.setStyleSheet("color: #F0F0F1; font-size: 14px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_play.clicked.connect(self.toggle)
        self.btn_restart = QPushButton("⟲ Restart")
        self.btn_restart.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_restart.clicked.connect(self.restart)
        header.addWidget(self.btn_play)
        header.addWidget(self.btn_restart)
        root.addLayout(header)

        self.canvas = _SimCanvas(self.model)
        root.addWidget(self.canvas, stretch=1)

        readouts = QHBoxLayout()
        readouts.setSpacing(22)
        self.lbl_acc = self._stat("Accuracy", "100%")
        self.lbl_dist = self._stat("Distortion index", "0.00")
        readouts.addLayout(self._acc_box)
        readouts.addLayout(self._dist_box)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet("color: #9CA0A6; font-size: 11px;")
        self.note.setMinimumHeight(44)
        readouts.addWidget(self.note, stretch=1)
        root.addLayout(readouts)

        self.drift = QSlider(Qt.Orientation.Horizontal)
        self.drift.setRange(-500, 500)
        self.drift.setSingleStep(10)
        self.drift.setPageStep(50)
        self.drift.setValue(self.model.drift_ms)
        self.drift.valueChanged.connect(self._on_drift)
        self.lbl_drift = self._mono_value(f"{self.model.drift_ms:+d} ms")
        root.addLayout(self._slider_row("Clock drift / phrase", self.drift, self.lbl_drift))

        self.phrase = QSlider(Qt.Orientation.Horizontal)
        self.phrase.setRange(1000, 7000)
        self.phrase.setSingleStep(100)
        self.phrase.setPageStep(500)
        self.phrase.setValue(self.model.phrase_ms)
        self.phrase.valueChanged.connect(self._on_phrase)
        self.lbl_phrase = self._mono_value(f"{self.model.phrase_ms} ms")
        root.addLayout(self._slider_row("Phrase length", self.phrase, self.lbl_phrase))

        self.method_lbl = QLabel("")
        self.method_lbl.setStyleSheet("color: #9CA0A6; font-size: 11px;")
        self.method_lbl.setWordWrap(True)
        root.addWidget(self.method_lbl)

    def _stat(self, caption: str, value: str) -> QLabel:
        box = QVBoxLayout()
        box.setSpacing(0)
        cap = QLabel(caption)
        cap.setStyleSheet("color: #9CA0A6; font-size: 11px;")
        val = QLabel(value)
        val.setStyleSheet("color: #F0F0F1; font-size: 22px; font-weight: bold;")
        box.addWidget(cap)
        box.addWidget(val)
        if caption == "Accuracy":
            self._acc_box = box
        else:
            self._dist_box = box
        return val

    def _mono_value(self, text: str) -> QLabel:
        lbl = QLabel(text)
        f = QFont()
        f.setFamilies(["JetBrains Mono", "DejaVu Sans Mono", "Consolas", "monospace"])
        f.setStyleHint(QFont.StyleHint.Monospace)
        lbl.setFont(f)
        lbl.setStyleSheet("color: #F0F0F1;")
        lbl.setMinimumWidth(70)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return lbl

    def _slider_row(self, label: str, slider: QSlider, value: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        cap = QLabel(label)
        cap.setStyleSheet("color: #9CA0A6; font-size: 12px;")
        cap.setMinimumWidth(140)
        row.addWidget(cap)
        row.addWidget(slider, stretch=1)
        row.addWidget(value)
        return row

    # public API ------------------------------------------------------------
    def set_strategy(self, strategy_id: int) -> None:
        self.model.strategy = strategy_id
        self._refresh()

    def play(self) -> None:
        self._timer.start()
        self.btn_play.setText("⏸ Pause")

    def pause(self) -> None:
        self._timer.stop()
        self.btn_play.setText("▶ Play")

    def toggle(self) -> None:
        self.pause() if self._timer.isActive() else self.play()

    def restart(self) -> None:
        self.model.play_t = 0.0
        self.canvas.update()
        self.play()

    # internals -------------------------------------------------------------
    def _tick(self) -> None:
        self.model.play_t = (self.model.play_t + 30.0 / _SWEEP_MS) % 1.0
        self.canvas.update()

    def _on_drift(self, value: int) -> None:
        self.model.drift_ms = value
        self.lbl_drift.setText(f"{value:+d} ms")
        self._refresh()

    def _on_phrase(self, value: int) -> None:
        self.model.phrase_ms = value
        self.lbl_phrase.setText(f"{value} ms")
        self._refresh()

    def _refresh(self) -> None:
        acc, dist = self.model.metrics()
        self.lbl_acc.setText(f"{acc:.0f}%")
        self.lbl_dist.setText(f"{dist:.2f}")
        acc_color = _GOOD if acc >= 99 else (_ACCENT if acc >= 90 else _BAD)
        dist_color = _GOOD if dist < 0.15 else (_ACCENT if dist < 0.6 else _BAD)
        self.lbl_acc.setStyleSheet(f"color: {acc_color}; font-size: 22px; font-weight: bold;")
        self.lbl_dist.setStyleSheet(f"color: {dist_color}; font-size: 22px; font-weight: bold;")
        s = self.model.strategy
        self.note.setText(_NOTE.get(s, ""))
        self.method_lbl.setText(
            f"Method follows your selection: <b>Strategy {s} — {_METHOD.get(s, '?')}</b>. "
            f"Change the strategy radios on the left to compare."
        )
        self.method_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.canvas.update()

    # pause the animation when the panel is not visible (save CPU) ----------
    def hideEvent(self, event: object) -> None:  # noqa: ANN001
        self._timer.stop()
        self.btn_play.setText("▶ Play")
        super().hideEvent(event)  # type: ignore[arg-type]

    def showEvent(self, event: object) -> None:  # noqa: ANN001
        self.play()
        super().showEvent(event)  # type: ignore[arg-type]
