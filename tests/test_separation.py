"""Tests for the ambience-separation helper (path logic, output naming)."""

from __future__ import annotations

from pathlib import Path

from whispersync.engine import separation


def test_separator_absent(tmp_path: Path) -> None:
    assert separation.separator_cli(tmp_path) is None
    assert separation.is_available(tmp_path) is False


def test_separator_present(tmp_path: Path) -> None:
    cli = tmp_path / ".sep-venv" / "bin" / "audio-separator"
    cli.parent.mkdir(parents=True)
    cli.write_text("")
    assert separation.separator_cli(tmp_path) == cli
    assert separation.is_available(tmp_path) is True


def test_expected_output_name(tmp_path: Path) -> None:
    out = separation._expected_output(
        tmp_path, Path("/x/DJI_0829.wav"), "melband_roformer_inst_v2.ckpt"
    )
    assert out.name == "DJI_0829_(Instrumental)_melband_roformer_inst_v2.wav"


def test_extract_requires_venv(tmp_path: Path) -> None:
    # No .sep-venv under repo_root → clear error, no subprocess attempted.
    import pytest

    with pytest.raises(RuntimeError, match="sep-venv"):
        separation.extract_ambience(tmp_path / "cam.wav", tmp_path / "out", tmp_path, "model.ckpt")
