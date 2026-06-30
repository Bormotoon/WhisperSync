# WhisperSync — Advanced Audio/Video Synchronization Tool

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Code style](https://img.shields.io/badge/code%20style-black%20%7C%20ruff%20%7C%20mypy-000000)

[Русский](README.md) · **English**

---

**WhisperSync** synchronizes audio and video for dual-system recording: the camera
films with a scratch audio track while an external recorder (lav mic, Zoom,
Tascam) captures clean sound separately. WhisperSync automatically finds the exact
time mapping between the tracks and generates an FCPXML project for Final Cut Pro
or DaVinci Resolve.

How it works: **transcription** → **anchor matching** → **K/offset regression**
→ **sync strategy** → **FCPXML export**.

Using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2),
WhisperSync transcribes both audio streams with word-level timestamps, finds
matching words (anchors) via sequence alignment, applies RANSAC regression to
robustly estimate the linear clock drift (K = rate, offset = shift), then applies
the best sync strategy for the recording conditions. The result is a render-free
FCPXML that references the original media.

![WhisperSync GUI](docs/images/main_window.png)

> Main window: drag-and-drop sources, strategy selection, a multi-track timeline
> with live status, and a real-time log.

## Table of Contents

- [Features](#features)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage) — [GUI](#gui) · [CLI](#cli)
- [Strategies Guide](#strategies-guide)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## Features

- **4 sync strategies** — Global Linear (linear drift), Local Time-Stretch
  (non-linear drift), Silence Padding (pitch-safe), Hybrid (general purpose)
- **PyQt6 GUI** with a dark theme, drag-and-drop, strategy diagrams and a live
  multi-track timeline
- **CLI headless mode** with JSON output for automation
- **Per-clip timecode placement** — clips need not be contiguous; real gaps are
  preserved (works for arbitrary sources)
- **Multi-camera** (one lane per camera) and **multi-recorder** (best/all lane
  modes) support
- **Windowed matching** for multi-hour recordings (coarse rare-word localization,
  then precise match in a window)
- **Production Whisper engine** — auto device/compute-type, batched GPU inference
  with OOM fallback ladder, anti-hallucination decoding
- **Transcription cache** (SHA-256) and **full transcript export** (JSON + SRT)
- **FCPXML export** without rendering — for FCP / DaVinci Resolve
- **NVIDIA GPU** acceleration (CUDA/cuDNN) with CPU fallback
- **Cross-platform** — Windows, macOS, Linux

## Screenshots

### Multi-track timeline

![Timeline](docs/images/timeline.png)

One row per camera and per audio lane. You can see each clip's real position
(`DJI_0838` and `DJI_0839` with a gap between them; the second camera `GX010024`
with its own offset on a separate lane), the applied audio speed change
(`−0.10%`, `+0.11%`), and live status: **done** (filled), **working** (orange
outline), **pending** (dashed). Hover a clip for offset / duration / in-point /
speed / status.

### Strategy diagrams

| Strategy 1 — Global Linear | Strategy 2 — Local Time-Stretch |
|----------------------------|---------------------------------|
| ![S1](docs/images/strategy_1.png) | ![S2](docs/images/strategy_2.png) |
| One block, uniform rescale | Per-segment factor |

| Strategy 3 — Silence Padding | Strategy 4 — Hybrid |
|------------------------------|---------------------|
| ![S3](docs/images/strategy_3.png) | ![S4](docs/images/strategy_4.png) |
| Speech untouched, gaps move | Phrases corrected + gaps absorb the rest |

### Help tab & interactive simulator

![Simulator](docs/images/simulator.png)

The **Help** tab is a built-in tutorial that explains the whole pipeline and
embeds an interactive **micro-sync simulator**. Drag the **clock drift** and
**phrase length** sliders, then switch strategies on the left and watch the
recorder track (red) reshape against the picture (blue) — exactly like the real
timeline. The **accuracy** vs **distortion index** readouts make the trade-off
concrete: time-stretch reaches 100 % alignment but distorts speech, padding adds
zero distortion but leaves residual drift in long phrases, and Hybrid balances
both. Great for understanding which strategy to pick before you run anything.

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | >= 3.10 | 3.11+ |
| GPU | CPU fallback | NVIDIA GPU (CUDA/cuDNN) |
| ffmpeg/ffprobe | in PATH | latest |
| RAM | 4 GB | 8+ GB |
| Disk | 10 GB free | SSD |

- **NVIDIA GPU** — recommended for fast transcription (CPU fallback works, but is
  slower). **ffmpeg / ffprobe** must be in `PATH`. **CUDA / cuDNN** are required
  for GPU mode.

## Installation

```bash
git clone https://github.com/Bormotoon/WhisperSync.git
cd WhisperSync

python -m venv venv
source venv/bin/activate     # Linux/macOS
# venv\Scripts\activate      # Windows

pip install -r requirements.txt

# Verify the environment (ffmpeg, CUDA, Python, deps, disk)
python whispersync/engine/system_check.py
```

## Usage

### GUI

```bash
python main.py
```

Drag-and-drop a video folder and a recorder file, pick a strategy and options,
press **SYNC**, then import the generated `.fcpxml` into Final Cut Pro or
DaVinci Resolve. The timeline updates live as clips are processed.

### CLI

```bash
# Basic (Strategy 1 — Global Linear)
python main.py --cli --video-dir ./videos --audio-file rec.wav \
  --strategy 1 --output out.fcpxml

# Hybrid strategy, JSON output
python main.py --cli --video-dir ./videos --audio-file rec.wav \
  --strategy 4 --json

# Multi-camera (subfolders) + multiple recorders on separate lanes
python main.py --cli --video-dir ./shoot \
  --audio-file lavA.wav --audio-file lavB.wav --recorder-mode all

# CPU, smaller model, dry-run (alignment only)
python main.py --cli --video-dir ./videos --audio-file rec.wav \
  --device cpu --compute-type int8 --model medium --dry-run
```

Key flags: `--strategy {1,2,3,4}`, `--device {auto,cuda,cpu}`,
`--compute-type {auto,float16,int8,...}`, `--batch-size`, `--mode {fast,quality}`,
`--timebase-source {camera,recorder}`, `--audio-source-camera`,
`--recorder-mode {best,all}`, `--crossfade/--no-crossfade`,
`--save-transcripts/--no-save-transcripts`, `--config`, `--no-cache`,
`--dry-run`, `--json`, `--verbose`. Run `python main.py --cli --help` for the
full list. CLI flags override the JSON config, which overrides defaults.

## Strategies Guide

| Strategy | Name | When to use |
|----------|------|-------------|
| **1** | Global Linear | Linear clock drift (most common). One `atempo` factor for the whole file. |
| **2** | Local Time-Stretch | Non-linear drift. Per-segment `atempo` between anchors. |
| **3** | Silence Padding | Pitch matters. Speech is untouched; only inter-phrase silence is adjusted. |
| **4** | Hybrid (Global + Silence) | General purpose, recommended. Each phrase is corrected by the clip's global `K` and placed at its anchor; silence absorbs the rest. |

> **Clip placement.** Every clip is aligned to the recorder independently and
> positioned from matched timecodes — clips need not be contiguous, real gaps
> between recordings are preserved.

> **Multi-camera.** Put each camera's clips in its own sub-folder of
> `--video-dir`. Each camera gets its own lane; the clean audio is synced once
> from a reference camera (`--audio-source-camera`, auto by default).

> **Multiple recorders.** Pass `--audio-file` several times. `--recorder-mode
> best` keeps one audio lane (best recorder per clip); `all` puts each recorder
> on its own lane. (Chunks of one device share a clock — concatenate them first
> instead of passing them as separate recorders.)

## Configuration

WhisperSync reads a JSON config via `--config config.json`. Priority:
**CLI flags > JSON config > defaults**. Example:

```json
{
    "model": "large-v3",
    "device": "auto",
    "compute_type": "auto",
    "language": "auto",
    "batch_size": 16,
    "transcribe_mode": "fast",
    "fcpxml_version": "1.9",
    "default_strategy": 1,
    "use_cache": true,
    "save_transcripts": true,
    "timebase_source": "camera",
    "recorder_mode": "best",
    "crossfade_enabled": true,
    "min_anchors": 8,
    "anchor_min_confidence": 0.6
}
```

## Troubleshooting

- **`CUDA not available, falling back to CPU`** — install the NVIDIA CUDA Toolkit
  and cuDNN; check `python -c "import torch; print(torch.cuda.is_available())"`.
- **`ffmpeg not found in PATH`** — `sudo apt install ffmpeg` /
  `brew install ffmpeg` / download from ffmpeg.org (Windows).
- **Few anchors** — ensure both tracks contain audible speech; try
  `--language`, lower `anchor_min_confidence`, or a larger model.
- **High residual** — try Strategy 2 (non-linear drift) or Strategy 4; verify
  anchors are spread across the whole length.
- **Reset cache** — `rm -rf ~/.cache/whispersync/` or run with `--no-cache`.

## Architecture

```
WhisperSync/
├── main.py                  # Entry point (GUI / CLI dispatch)
├── whispersync/
│   ├── cli.py               # argparse CLI
│   ├── config.py            # WhisperSyncConfig + JSON loader
│   ├── models.py            # Word, Segment, Transcript, Anchor, SyncPlan...
│   ├── engine/
│   │   ├── pipeline.py      # End-to-end orchestration
│   │   ├── transcriber.py   # WhisperEngine + SHA-256 cache
│   │   ├── matcher.py       # Anchor finding + RANSAC + windowed match
│   │   ├── strategies.py    # 4 sync strategies (per clip)
│   │   ├── timestretch.py   # ffmpeg atempo / segment / fade wrappers
│   │   ├── media.py         # ffprobe, audio extraction, file:// URIs
│   │   ├── naming.py        # natural sort + sequence detection
│   │   ├── export.py        # FCPXML generation
│   │   ├── transcript_export.py  # JSON + SRT transcript export
│   │   └── system_check.py  # Environment validation
│   └── gui/                 # PyQt6 window, worker, widgets, theme
└── tests/                   # pytest suite
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md). Before
opening a PR, make sure these pass:

```bash
ruff check whispersync/ tests/
black --check whispersync/ tests/
mypy whispersync/ main.py
pytest
```

## License

MIT License — Copyright (c) 2024-2025 WhisperSync Contributors. See
[LICENSE](LICENSE).
