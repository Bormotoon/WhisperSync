from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def check_ffmpeg() -> dict:
    result: dict = {"ffmpeg": False, "ffprobe": False, "details": {}}
    for cmd in ("ffmpeg", "ffprobe"):
        try:
            r = subprocess.run([cmd, "-version"], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                line = r.stdout.split("\n")[0]
                _ok(f"{cmd} — {line}")
                result[cmd] = True
                result["details"][cmd] = line
            else:
                _fail(f"{cmd} not working")
                result["details"][cmd] = r.stderr.strip()
        except FileNotFoundError:
            _fail(f"{cmd} not found in PATH")
            result["details"][cmd] = "not found"
    return result


def check_cuda() -> dict:
    """Report whether faster-whisper (the actual transcription backend) can use
    CUDA — via ctranslate2's own device probe, the same check
    ``transcriber.resolve_device`` makes at runtime. The app does not depend on
    torch at all (transcription runs on ctranslate2); checking only
    ``torch.cuda.is_available()`` (the old implementation) reported "CUDA
    check skipped" on a perfectly working GPU machine whenever torch wasn't
    installed, which is the common case here. See PROJECT_ANALYSIS.md §3.6.
    Torch, if present, additionally supplies the GPU name/VRAM for the report.
    """
    result: dict = {"cuda_available": False, "gpu_name": None, "vram_gb": None}
    try:
        from whispersync.engine.transcriber import _ct2_cuda_available

        ct2_cuda = _ct2_cuda_available()
    except ImportError as e:
        _fail(f"CUDA check error (ctranslate2/faster_whisper not importable): {e}")
        result["details"] = str(e)
        return result

    result["cuda_available"] = ct2_cuda
    if not ct2_cuda:
        _warn("CUDA not available to ctranslate2 (faster-whisper will run on CPU)")
        result["details"] = "ctranslate2 reports no CUDA device"
        return result

    # torch is optional; use it only to enrich the report with GPU name/VRAM.
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            name = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)
            vram_gb = round(props.total_memory / (1024**3), 2)
            result["gpu_name"] = name
            result["vram_gb"] = vram_gb
            _ok(f"CUDA — {name}, {vram_gb} GB VRAM (ctranslate2 + torch)")
            return result
    except ImportError:
        pass
    _ok("CUDA — available to ctranslate2 (install torch for GPU name/VRAM detail)")
    return result


def check_python() -> dict:
    result: dict = {"python_version": sys.version}
    _ok(f"Python {sys.version.split()[0]}")
    return result


def check_dependencies() -> dict:
    deps = {
        "ctranslate2": "ctranslate2",
        "faster_whisper": "faster_whisper",
        "PyQt6": "PyQt6",
    }
    result: dict = {}
    for name, mod in deps.items():
        try:
            __import__(mod)
            result[name] = True
            _ok(f"{name} — installed")
        except ImportError:
            result[name] = False
            _warn(f"{name} — not installed")
    return result


def check_ambience_separator() -> dict:
    """Whether the optional ``.sep-venv`` environment (ambience-track feature,
    ``--ambience-track``) is set up. Not fatal — the feature is opt-in — but
    unlike the other checks this one previously wasn't reported at all, so a
    user enabling the GUI checkbox only learned it was missing after a run
    failed. See PROJECT_ANALYSIS.md §3.6."""
    from whispersync.engine import separation

    repo_root = Path(__file__).resolve().parents[2]
    available = separation.is_available(repo_root)
    if available:
        _ok("Ambience separator (.sep-venv) — available")
    else:
        _warn(
            "Ambience separator (.sep-venv) — not set up "
            "(optional; run setup_sep_venv.sh to enable --ambience-track)"
        )
    return {"available": available, "repo_root": str(repo_root)}


def check_disk_space(min_gb: int = 10) -> dict:
    usage = shutil.disk_usage("/")
    free_gb = usage.free / (1024**3)
    ok = free_gb >= min_gb
    if ok:
        _ok(f"Disk space: {free_gb:.1f} GB free")
    else:
        _fail(f"Disk space: {free_gb:.1f} GB free (min {min_gb} GB)")
    return {"free_gb": round(free_gb, 1), "ok": ok}


def run_all_checks() -> dict:
    print("WhisperSync — System Check")
    print("=" * 40)

    print("\n[FFmpeg]")
    ff = check_ffmpeg()

    print("\n[CUDA]")
    cu = check_cuda()

    print("\n[Disk]")
    ds = check_disk_space()

    print("\n[Python]")
    py = check_python()

    print("\n[Dependencies]")
    deps = check_dependencies()

    print("\n[Ambience separator]")
    sep = check_ambience_separator()

    report = {
        "ffmpeg": ff,
        "ffprobe": ff,
        "cuda": cu,
        "disk": ds,
        "python": py,
        "dependencies": deps,
        "ambience_separator": sep,
    }

    # In the current working directory (where the user runs the command from),
    # not next to the module inside the installed package — the README has
    # always documented "report.json" without a package-internal path, and a
    # user has no reason to go looking inside site-packages for it. See
    # PROJECT_ANALYSIS.md §3.6.
    path = Path.cwd() / "report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {path}")

    return report


if __name__ == "__main__":
    run_all_checks()
