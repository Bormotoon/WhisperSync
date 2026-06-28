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
    result: dict = {"cuda_available": False, "gpu_name": None, "vram_gb": None}
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            name = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)
            vram_gb = round(props.total_memory / (1024**3), 2)
            _ok(f"CUDA — {name}, {vram_gb} GB VRAM")
            result["cuda_available"] = True
            result["gpu_name"] = name
            result["vram_gb"] = vram_gb
        else:
            _warn("CUDA not available")
            result["details"] = "cuda not available"
    except ImportError:
        _warn("CUDA check skipped — torch not installed")
        result["details"] = "torch not installed"
    except Exception as e:
        _fail(f"CUDA check error: {e}")
        result["details"] = str(e)
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
    print("BormoSync — System Check")
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

    report = {
        "ffmpeg": ff,
        "ffprobe": ff,
        "cuda": cu,
        "disk": ds,
        "python": py,
        "dependencies": deps,
    }

    path = Path(__file__).parent / "report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {path}")

    return report


if __name__ == "__main__":
    run_all_checks()
