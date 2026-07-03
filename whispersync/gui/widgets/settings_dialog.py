"""Modal dialog exposing transcription settings not covered by the main
window's quick options (model/language/device/compute-type/initial-prompt/
transcribe-mode) — previously only reachable via the CLI or a JSON config
file. See PROJECT_ANALYSIS.md §Stage 7.5."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
)

from whispersync.config import WhisperSyncConfig

_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
_DEVICES = ["auto", "cuda", "cpu"]
_COMPUTE_TYPES = ["auto", "float16", "int8_float16", "int8", "float32"]
_MODES = ["fast", "quality"]


class SettingsDialog(QDialog):
    """Edits a copy of the config's transcription fields; the caller applies
    the result via ``apply_to``/reads the individual properties on accept."""

    def __init__(self, config: WhisperSyncConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Transcription Settings")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)  # custom/local model names are valid too
        self.model_combo.addItems(_MODELS)
        self.model_combo.setCurrentText(config.model)
        form.addRow("Model:", self.model_combo)

        self.language_edit = QLineEdit(config.language or "")
        self.language_edit.setPlaceholderText("auto-detect (leave empty)")
        form.addRow("Language:", self.language_edit)

        self.device_combo = QComboBox()
        self.device_combo.addItems(_DEVICES)
        self.device_combo.setCurrentText(config.device)
        form.addRow("Device:", self.device_combo)

        self.compute_type_combo = QComboBox()
        self.compute_type_combo.setEditable(True)
        self.compute_type_combo.addItems(_COMPUTE_TYPES)
        self.compute_type_combo.setCurrentText(config.compute_type)
        form.addRow("Compute type:", self.compute_type_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(_MODES)
        self.mode_combo.setCurrentText(config.transcribe_mode)
        self.mode_combo.setToolTip(
            "fast = batched pipeline (default, ~real-time on GPU). "
            "quality = sequential with context + hallucination guard, ~10x slower."
        )
        form.addRow("Transcribe mode:", self.mode_combo)

        self.prompt_edit = QPlainTextEdit(config.initial_prompt)
        self.prompt_edit.setPlaceholderText("Optional domain context to bias vocabulary")
        self.prompt_edit.setMaximumHeight(80)
        form.addRow("Initial prompt:", self.prompt_edit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply_to(self, config: WhisperSyncConfig) -> WhisperSyncConfig:
        """Return a copy of ``config`` with this dialog's fields applied."""
        import dataclasses

        language = self.language_edit.text().strip() or None
        return dataclasses.replace(
            config,
            model=self.model_combo.currentText().strip() or config.model,
            language=language,
            device=self.device_combo.currentText(),
            compute_type=self.compute_type_combo.currentText(),
            transcribe_mode=self.mode_combo.currentText(),
            initial_prompt=self.prompt_edit.toPlainText(),
        )
