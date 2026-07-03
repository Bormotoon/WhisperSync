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


def _normalized_base(name: str) -> str:
    """A filename base suitable for matching separator outputs to inputs.

    audio-separator builds output names as "<base>_(<Stem>)_<model>.wav" but
    normalizes the input's base first — observed in the field: a temp file
    named ``tmpsj40fum_.wav`` produced ``tmpsj40fum_(Instrumental)_...`` (the
    trailing underscore swallowed), which the old exact/glob prediction could
    never match. Strip trailing separator characters from both sides of the
    comparison instead of trying to predict the separator's exact escaping.
    """
    return name.rstrip("_-. ").lower()


def _find_output(out_dir: Path, input_path: Path, model_filename: str) -> Path | None:
    """The separator's instrumental output for ``input_path``, or None.

    Tries the exact predicted name first, then falls back to scanning the
    output dir for a WAV whose part before "(<Stem>)" normalizes to the same
    base as the input — exact normalized equality, so one input's stem being
    a prefix of another's can't cross-match. Multiple survivors (e.g. stale
    files from a previous run) resolve to the newest by mtime.
    """
    exact = _expected_output(out_dir, input_path, model_filename)
    if exact.exists():
        return exact
    marker = f"({_STEM})"
    want = _normalized_base(input_path.stem)
    candidates = [
        f
        for f in out_dir.iterdir()
        if f.suffix.lower() == ".wav"
        and marker in f.name
        and _normalized_base(f.name.split(marker, 1)[0]) == want
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_ambience(
    camera_audio: Path,
    out_dir: Path,
    repo_root: Path,
    model_filename: str,
    model_dir: Path | None = None,
    timeout: int = 1800,
) -> Path:
    """Run the separator on a single ``camera_audio`` file and return the
    ambience-only WAV. For more than one clip, prefer ``extract_ambience_batch``
    — it loads the (1.5+ GB) model once instead of once per call.
    """
    results = extract_ambience_batch(
        [camera_audio], out_dir, repo_root, model_filename, model_dir, timeout
    )
    return results[camera_audio]


def extract_ambience_batch(
    camera_audios: list[Path],
    out_dir: Path,
    repo_root: Path,
    model_filename: str,
    model_dir: Path | None = None,
    timeout: int = 3600,
) -> dict[Path, Path]:
    """Run the separator once over every file in ``camera_audios`` and return a
    ``{input_path: ambience_wav_path}`` map.

    ``audio-separator``'s CLI accepts multiple positional inputs and loads the
    (1.5+ GB) model a single time for the whole batch — calling it once per
    clip (the previous behaviour) reloaded the model from scratch for every
    camera clip in a multi-clip shoot, each load costing tens of seconds. A
    partial failure for one input still raises (the CLI doesn't report
    per-file exit codes), so callers that want per-clip fault isolation should
    catch the RuntimeError and fall back to the single-file
    ``extract_ambience`` for the remaining files. See PROJECT_ANALYSIS.md §6.3.
    """
    if not camera_audios:
        return {}
    cli = separator_cli(repo_root)
    if cli is None:
        raise RuntimeError(
            "Ambience separation needs the '.sep-venv' environment "
            "(audio-separator). It is not set up — run setup_sep_venv.sh."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cli),
        *(str(p) for p in camera_audios),
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

    logger.info(
        "Extracting ambience (%s) from %d clip(s) in one batch", model_filename, len(camera_audios)
    )
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Ambience separation failed (exit {result.returncode}): "
            f"{result.stderr[-600:] or result.stdout[-600:]}"
        )

    outputs: dict[Path, Path] = {}
    for camera_audio in camera_audios:
        produced = _find_output(out_dir, camera_audio, model_filename)
        if produced is None:
            raise RuntimeError(
                f"Separator reported success but no instrumental output was found "
                f"in {out_dir} for {camera_audio.name}."
            )
        outputs[camera_audio] = produced
    return outputs
