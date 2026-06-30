"""Help & tutorial page.

A scrollable, self-contained explainer of how WhisperSync works, with the
interactive micro-sync simulator embedded so the concepts are tangible before
the user picks a strategy and runs anything.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from whispersync.gui.widgets.sync_simulator import SyncSimulator

_H = "color:#FF6E40; font-size:15px; font-weight:bold;"
_BODY = "color:#C7C9CE; font-size:12.5px;"


class HelpPage(QWidget):
    """Tabbed 'Help' page: documentation + embedded simulator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        self._lay = QVBoxLayout(content)
        self._lay.setContentsMargins(18, 16, 18, 18)
        self._lay.setSpacing(12)

        self._title("Help & Tutorial")

        self._heading("What problem does this solve?")
        self._para(
            "WhisperSync is built for <b>dual-system sound</b>: you film on one device "
            "(say a DJI Pocket) whose built-in microphone is weak, and capture clean "
            "audio separately (a phone with a radio mic). The two devices run on "
            "independent quartz clocks, so over minutes-to-hours their timing slowly "
            "diverges — non-linear <b>clock drift</b> of up to about a second. Waveform "
            "matchers such as PluralEyes lock the <i>start</i> but cannot track that "
            "creeping drift, so the dialogue slides out of sync later on."
        )

        self._heading("How WhisperSync works")
        self._para(
            "<ol>"
            "<li><b>Probe</b> — read the duration and format of every clip.</li>"
            "<li><b>Transcribe</b> — Whisper turns both the camera scratch audio and the "
            "recorder audio into word-level transcripts (a timestamp per word).</li>"
            "<li><b>Match anchors</b> — words shared by both transcripts become time "
            "anchors. For long recordings a coarse rare-word vote first locates each "
            "clip, then a precise match runs in a narrow window.</li>"
            "<li><b>Fit (RANSAC)</b> — a robust linear fit estimates the clock ratio "
            "<b>K</b> and the <b>offset</b>, discarding mismatched words as outliers.</li>"
            "<li><b>Re-align</b> — the chosen strategy warps the recorder audio so every "
            "word lands under the picture.</li>"
            "<li><b>Export</b> — an <b>FCPXML</b> is written for Final Cut Pro / DaVinci "
            "Resolve, referencing your original media by path.</li>"
            "</ol>"
        )

        self._heading("Reading the timeline")
        self._para(
            "The simulator below — and the real <i>Run</i> timeline — share one visual "
            "language:"
            "<ul>"
            "<li><span style='color:#1E88E5'>&#9632;</span> <b>Blue</b> blocks are video / "
            "camera lanes (one row per camera).</li>"
            "<li>Finished <b>synced-audio</b> blocks are tinted by how much their tempo had "
            "to be corrected to lock to the picture: "
            "<span style='color:#3FB950'>&#9632; green</span> = barely touched (already in "
            "sync), through <span style='color:#E0C440'>&#9632; yellow</span>, to "
            "<span style='color:#F05550'>&#9632; red</span> = heavy correction. Most clips "
            "should sit green&ndash;yellow; a red clip is worth a listen.</li>"
            "<li>A <span style='color:#FF6E40'>speed badge</span> like "
            "<span style='color:#FF6E40'>-0.34%</span> shows the exact tempo change; while a "
            "clip is being rendered its border <b>pulses</b>, and each clip <b>fades in</b> "
            "as it finishes.</li>"
            "<li>Dashed <b>pad</b> blocks are inserted silence.</li>"
            "</ul>"
        )

        self._heading("Try it — the micro-sync simulator")
        self._para(
            "Drag the two sliders and switch the <b>strategy radios on the left</b> to feel "
            "the trade-off. <i>Drift</i> is how far each phrase slips; <i>phrase length</i> "
            "is how long people talk between pauses."
        )
        self.simulator = SyncSimulator()
        sim_frame = QFrame()
        sim_frame.setObjectName("simFrame")
        sim_lay = QVBoxLayout(sim_frame)
        sim_lay.setContentsMargins(0, 0, 0, 0)
        sim_lay.addWidget(self.simulator)
        self._lay.addWidget(sim_frame)

        self._heading("The four strategies")
        self._para(
            "<ul>"
            "<li><b>1 · Global linear</b> — one atempo for the whole clip. No seams; best "
            "when the drift is steady (linear).</li>"
            "<li><b>2 · Local time-stretch</b> — each phrase stretched to fit. Highest "
            "alignment, but the speech itself is stretched (audible on large drift).</li>"
            "<li><b>3 · Silence padding</b> — speech is never touched (zero distortion); "
            "silence is padded or trimmed instead. Residual drift remains inside long "
            "phrases.</li>"
            "<li><b>4 · Hybrid</b> — a gentle per-phrase stretch plus padded gaps. "
            "Near-perfect alignment at roughly half the distortion — <b>the default</b>, "
            "selected for you.</li>"
            "</ul>"
            "In short: stretching buys alignment by touching the speech; padding protects "
            "the speech but leaves residual drift; hybrid balances both. The "
            "<b>distortion index</b> and <b>accuracy</b> readouts above make the trade-off "
            "concrete."
        )

        self._heading("Polishing the sync (Options)")
        self._para(
            "<ul>"
            "<li><b>Boundary Flex</b> (on by default) — after the strategy places each "
            "phrase, this pass fine-tunes its position by directly cross-correlating the "
            "camera and recorder <i>waveforms</i>, independent of the transcript. It pulls "
            "lip-sync down to about a frame even where the word timings were fuzzy. It adds "
            "a little processing; turn it off for the fastest run.</li>"
            "<li><b>Duck pauses</b> (off by default) — lowers the volume <i>between</i> "
            "phrases by the amount set on the <b>Pause level</b> slider (0&nbsp;dB&nbsp;=&nbsp;"
            "off … &minus;&infin; = silence). Speech is left untouched; only the gaps are "
            "attenuated, so a faintly mis-synced room tone / ambience in the pauses becomes "
            "inaudible. Handy when the recorder and camera rooms don't quite match.</li>"
            "</ul>"
        )

        self._heading("Tips")
        self._para(
            "<ul>"
            "<li>Put each camera in its own subfolder &rarr; separate lanes.</li>"
            "<li>Multiple recorders: <b>best</b> picks the strongest per clip; <b>all</b> "
            "gives each recorder its own lane.</li>"
            "<li>Very long audio (6&ndash;8 h) is handled by windowed matching — just drop "
            "it in.</li>"
            "<li>The defaults (<b>Hybrid + Boundary Flex + crossfade</b>) are tuned for the "
            "best result out of the box — just drop media and press SYNC.</li>"
            "<li><b>Duck pauses</b> on the Pause-level slider hides ambience desync in the "
            "gaps; leave it off if you want the raw room tone.</li>"
            "<li><b>Timebase source</b> chooses whose frame rate the FCPXML inherits.</li>"
            "<li>Full transcripts are saved next to the output for reuse.</li>"
            "</ul>"
        )

        self._lay.addStretch()

    # public ----------------------------------------------------------------
    def set_strategy(self, strategy_id: int) -> None:
        self.simulator.set_strategy(strategy_id)

    # builders --------------------------------------------------------------
    def _title(self, text: str) -> None:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#E53935; font-size:22px; font-weight:bold;")
        self._lay.addWidget(lbl)

    def _heading(self, text: str) -> None:
        lbl = QLabel(text)
        lbl.setStyleSheet(_H)
        lbl.setContentsMargins(0, 6, 0, 0)
        self._lay.addWidget(lbl)

    def _para(self, html: str) -> None:
        lbl = QLabel(html)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(_BODY)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        lbl.setOpenExternalLinks(True)
        self._lay.addWidget(lbl)
