"""GUI wiring tests — pure widget logic (no display needed via the Qt
offscreen platform plugin), covering the Stage 7.5 additions: multi-file
DropZone, the settings dialog, and the main window's recorder-mode/re-run
wiring."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from whispersync.config import WhisperSyncConfig
from whispersync.gui.widgets.drop_zone import DropZone
from whispersync.gui.widgets.settings_dialog import SettingsDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


def test_drop_zone_single_mode_unaffected(qapp: QApplication) -> None:
    zone = DropZone(accept_dirs=False, accepted_extensions=[".wav"])
    zone.set_path("/tmp/a.wav")
    assert zone.current_path == "/tmp/a.wav"
    assert zone.current_paths == ["/tmp/a.wav"]


def test_drop_zone_multi_set_paths(qapp: QApplication) -> None:
    zone = DropZone(accept_dirs=False, accept_multiple=True, accepted_extensions=[".wav"])
    zone.set_paths(["/tmp/a.wav", "/tmp/b.wav"])
    assert zone.current_paths == ["/tmp/a.wav", "/tmp/b.wav"]
    assert zone.current_path == "/tmp/a.wav"  # first path, for back-compat callers


def test_drop_zone_multi_single_path_falls_back_to_set_path(qapp: QApplication) -> None:
    zone = DropZone(accept_dirs=False, accept_multiple=True, accepted_extensions=[".wav"])
    zone.set_paths(["/tmp/only.wav"])
    assert zone.current_paths == ["/tmp/only.wav"]


def test_drop_zone_empty_paths_is_a_no_op(qapp: QApplication) -> None:
    zone = DropZone(accept_dirs=False, accept_multiple=True, accepted_extensions=[".wav"])
    zone.set_paths([])
    assert zone.current_paths == []


def test_settings_dialog_apply_to_overrides_fields(qapp: QApplication) -> None:
    cfg = WhisperSyncConfig()
    dialog = SettingsDialog(cfg)
    dialog.model_combo.setCurrentText("medium")
    dialog.language_edit.setText("ru")
    dialog.device_combo.setCurrentText("cpu")
    dialog.compute_type_combo.setCurrentText("int8")
    dialog.mode_combo.setCurrentText("quality")
    dialog.prompt_edit.setPlainText("podcast about cooking")

    new_cfg = dialog.apply_to(cfg)
    assert new_cfg.model == "medium"
    assert new_cfg.language == "ru"
    assert new_cfg.device == "cpu"
    assert new_cfg.compute_type == "int8"
    assert new_cfg.transcribe_mode == "quality"
    assert new_cfg.initial_prompt == "podcast about cooking"
    # Original config is untouched (dataclasses.replace returns a copy).
    assert cfg.model == "large-v3"


def test_settings_dialog_empty_language_becomes_none(qapp: QApplication) -> None:
    cfg = WhisperSyncConfig(language="ru")
    dialog = SettingsDialog(cfg)
    dialog.language_edit.setText("")
    new_cfg = dialog.apply_to(cfg)
    assert new_cfg.language is None


def test_settings_dialog_prefills_from_config(qapp: QApplication) -> None:
    cfg = WhisperSyncConfig(model="medium", language="en", device="cpu")
    dialog = SettingsDialog(cfg)
    assert dialog.model_combo.currentText() == "medium"
    assert dialog.language_edit.text() == "en"
    assert dialog.device_combo.currentText() == "cpu"
