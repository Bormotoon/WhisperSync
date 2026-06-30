"""Camera-ambience extraction (remove the camera's own voice, keep the room tone).

When the recorder's clean voice and the camera's built-in-mic voice both play on the
final timeline, their ~tens-of-ms offset is heard as a doubled/echoed voice — even
though lip-sync is fine, two near-identical voices comb-filter. Plainly ducking the
camera would also kill the ambience the editor wants. Instead we run a source-
separation model over the camera audio to strip the *vocal* and keep the
*instrumental* (= ambience/room tone), and lay that on its own lane next to the
synced voice. The voice then comes only from the clean recorder track, with no echo.

The separator (audio-separator + a RoFormer model) lives in a SEPARATE Python venv
(``.sep-venv``) because it needs an older Python than the main app; we invoke it as a
subprocess. The model runs on the GPU.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# audio-separator names its output "<input-stem>_(<Stem>)_<model-stem>.wav".
_STEM = "Instrumental"


def separator_python(repo_root: Path) -> Path | None:
    """Path to the isolated separation venv's python, or None if it isn't set up."""
    candidate = repo_root / ".sep-venv" / "bin" / "python"
    return candidate if candidate.exists() else None


def separator_cli(repo_root: Path) -> Path | None:
    """Path to the isolated venv's ``audio-separator`` console script, or None.

    We invoke this entry point (not ``python -m audio_separator.utils.cli``, which
    has no ``__main__`` guard and would silently no-op)."""
    candidate = repo_root / ".sep-venv" / "bin" / "audio-separator"
    return candidate if candidate.exists() else None


def is_available(repo_root: Path) -> bool:
    return separator_cli(repo_root) is not None


def _expected_output(out_dir: Path, input_path: Path, model_filename: str) -> Path:
    model_stem = Path(model_filename).stem
    return out_dir / f"{input_path.stem}_({_STEM})_{model_stem}.wav"


def extract_ambience(
    camera_audio: Path,
    out_dir: Path,
    repo_root: Path,
    model_filename: str,
    model_dir: Path | None = None,
    timeout: int = 1800,
) -> Path:
    """Run the separator on ``camera_audio`` and return the ambience-only WAV.

    Strips the vocal (the camera's own voice) and keeps the instrumental stem
    (room tone / ambience). Raises ``RuntimeError`` if the separator venv is missing
    or the subprocess fails.
    """
    cli = separator_cli(repo_root)
    if cli is None:
        raise RuntimeError(
            "Ambience separation needs the '.sep-venv' environment "
            "(audio-separator). It is not set up — run setup_sep_venv.sh."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cli),
        str(camera_audio),
        "--model_filename",
        model_filename,
        "--output_dir",
        str(out_dir),
        "--output_format",
        "WAV",
        "--single_stem",
        _STEM,
        "--log_level",
        "warning",
    ]
    if model_dir is not None:
        cmd += ["--model_file_dir", str(model_dir)]

    logger.info("Extracting ambience (%s) from %s", model_filename, camera_audio.name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Ambience separation failed (exit {result.returncode}): "
            f"{result.stderr[-600:] or result.stdout[-600:]}"
        )

    produced = _expected_output(out_dir, camera_audio, model_filename)
    if not produced.exists():
        # Some model-name variants get truncated in the filename; fall back to the
        # newest matching instrumental WAV in the output dir.
        candidates = sorted(
            out_dir.glob(f"{camera_audio.stem}_({_STEM})_*.wav"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise RuntimeError(
                f"Separator reported success but no instrumental output was found "
                f"in {out_dir} for {camera_audio.name}."
            )
        produced = candidates[-1]
    return produced
