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
→ **sync strategy** → **synced-audio render** → **FCPXML export**.

Using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2),
WhisperSync transcribes both audio streams with word-level timestamps, finds
matching words (anchors) via sequence alignment, applies RANSAC regression to
robustly estimate the linear clock drift (K = rate, offset = shift), then uses a
sync strategy to reassemble the recorder audio under the picture. The render keeps
the recorder's native channel count and bit depth throughout (no forced mono/
16-bit downmix), and uses a transparent resample instead of time-stretch wherever
the real clock drift is small enough for it to be inaudible.

The result is an FCPXML that references the original video files and the
rendered synced audio, ready to import into Final Cut Pro or DaVinci Resolve.
Source media is never modified.

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

- **3 sync strategies** — Global Linear (linear drift), Local Time-Stretch
  (non-linear drift), Hybrid (general purpose, recommended)
- **Bit-perfect render** — preserves the recorder's channel count and bit depth
  end to end; transparent resample instead of atempo/WSOLA for small drift
  (< 0.5%); fades only on seams that are actually discontinuous
- **Seam-snap-to-silence** — piecewise-strategy piece boundaries snap to
  inter-word silence so a seam never lands mid-word
- **Boundary Flex** — acoustic cross-correlation (GCC-PHAT) refines each
  piece's position for sub-frame lip-sync, on by default
- **PyQt6 GUI** with a dark theme, drag-and-drop, strategy diagrams and a live
  multi-track timeline
- **CLI headless mode** with JSON output for automation (`--json` sends
  progress to stderr, the report to stdout)
- **Per-clip timecode placement** — clips need not be contiguous; real gaps are
  preserved (works for arbitrary sources); a thin timecode fit (too few
  anchors) falls back to filename order instead of trusting a guess
- **Multi-camera** (one lane per camera) and **multi-recorder** (best/all lane
  modes) support
- **Windowed matching** for multi-hour recordings (coarse rare-word localization,
  then precise match in a window)
- **Production Whisper engine** — auto device/compute-type via ctranslate2
  (torch not required), batched GPU inference with an OOM fallback ladder,
  anti-hallucination decoding
- **Transcription cache** (SHA-256, keyed by the actually-resolved device/
  compute-type) and **full transcript export** (JSON + SRT)
- **RANSAC regression** with a two-stage outlier filter for robust drift
  estimation even with mismatched anchors
- **Auto-strategy** — recommends a better-suited strategy from the drift
  characteristics when the one you picked isn't the best fit
- **Acoustic fallback** — a clip with no usable transcript match at all
  (music, heavy noise, a language Whisper garbles) falls back to a coarse
  waveform cross-correlation, no words required
- **Per-camera lip-sync calibration** — a constant mic-to-lips offset that no
  acoustic method can see, correctable per camera
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

| Strategy 1 — Global Linear | Strategy 2 — Local Time-Stretch | Strategy 3 — Hybrid |
|----------------------------|---------------------------------|-----------------------|
| ![S1](docs/images/strategy_1.png) | ![S2](docs/images/strategy_2.png) | ![S4](docs/images/strategy_4.png) |
| One block, uniform rescale | Per-segment factor | Phrases corrected + gaps absorb the rest |

### Help tab & interactive simulator

![Simulator](docs/images/simulator.png)

The **Help** tab is a built-in tutorial that explains the whole pipeline and
embeds an interactive **micro-sync simulator**. Drag the **clock drift** and
**phrase length** sliders, then switch strategies on the left and watch the
recorder track (red) reshape against the picture (blue) — exactly like the real
timeline. The **accuracy** vs **distortion index** readouts make the
stretch-vs-padding trade-off concrete. Great for understanding which strategy to
pick before you run anything.

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | >= 3.10 | 3.11+ |
| GPU | CPU fallback | NVIDIA GPU (CUDA/cuDNN) |
| ffmpeg/ffprobe | in PATH | latest, with libsoxr |
| RAM | 4 GB | 8+ GB |
| Disk | 10 GB free | SSD |

- **NVIDIA GPU** — recommended for fast transcription. Transcription runs on
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper)/ctranslate2, which
  does **not require torch** — CUDA is detected via ctranslate2's own probe.
  CPU fallback works, just slower.
- **ffmpeg / ffprobe** must be in `PATH`. A build with `libsoxr` gives
  higher-quality resampling; WhisperSync automatically falls back to the
  default resampler if `libsoxr` isn't available.
- **CUDA / cuDNN** are required for GPU mode.

## Installation

```bash
git clone https://github.com/Bormotoon/WhisperSync.git
cd WhisperSync

python -m venv venv
source venv/bin/activate     # Linux/macOS
# venv\Scripts\activate      # Windows

pip install -r requirements.txt

# Verify the environment (ffmpeg, CUDA via ctranslate2, Python, deps, disk,
# optional .sep-venv)
python -m whispersync.engine.system_check
```

## Usage

### GUI

```bash
python main.py
```

Drag-and-drop a video folder and one or more recorder files (2+ files unlocks
a `best`/`all` recorder-mode picker), pick a strategy and options (Boundary
Flex, pause ducking, ambience track), optionally open **Transcription
Settings...** for model/language/device/compute-type/initial-prompt/mode,
press **SYNC**, then import the generated `.fcpxml` into Final Cut Pro or
DaVinci Resolve. The timeline updates live as clips are processed, and a
**Re-run with Selected Strategy** button appears after a successful run —
transcripts are cached, so switching strategies re-runs alignment/render only.

### CLI

```bash
# Basic (default strategy — 3, Hybrid)
python main.py --cli --video-dir ./videos --audio-file rec.wav --output out.fcpxml

# Strategy 1 — Global Linear, JSON output (progress goes to stderr)
python main.py --cli --video-dir ./videos --audio-file rec.wav \
  --strategy 1 --json 2>/dev/null

# Multi-camera (subfolders) + multiple recorders on separate lanes
python main.py --cli --video-dir ./shoot \
  --audio-file lavA.wav --audio-file lavB.wav --recorder-mode all

# CPU, smaller model, dry-run (alignment only)
python main.py --cli --video-dir ./videos --audio-file rec.wav \
  --device cpu --compute-type int8 --model medium --dry-run

# Version
python main.py --cli --version
```

Key flags: `--strategy {1,2,3}` (default: `WhisperSyncConfig.default_strategy`,
`3`; `4` is accepted as a deprecated alias for `3`), `--device {auto,cuda,cpu}`,
`--compute-type {auto,float16,int8,...}`, `--batch-size`, `--mode {fast,quality}`,
`--timebase-source {camera,recorder}`, `--audio-source-camera`,
`--camera-av-offset-ms` (constant per-camera lip-sync calibration),
`--recorder-mode {best,all}`, `--crossfade/--no-crossfade`, `--render-workers`,
`--boundary-flex/--no-boundary-flex` (on by default), `--pause-duck/
--no-pause-duck`, `--pause-duck-db`, `--ambience-track`,
`--render-master-wav` (also render one WAV spanning the whole timeline, voice +
ambience mixed at their timeline offsets, for users without an NLE),
`--save-transcripts/--no-save-transcripts`, `--config`, `--no-cache`,
`--dry-run`, `--verify` (post-render lip-sync self-check, see
`tools/verify_sync.py`), `--json`, `--verbose`, `--version`. Run
`python main.py --cli --help` for the full list. CLI flags override the JSON
config, which overrides defaults.

Exit codes: `0` success, `1` run failure (no anchors, ffmpeg error, etc.),
`2` usage/argument error.

## Strategies Guide

| Strategy | Name | When to use |
|----------|------|-------------|
| **1** | Global Linear | Linear clock drift (most common). One tempo-conform factor for the whole file. |
| **2** | Local Time-Stretch | Non-linear drift. Per-segment factor between anchors, with the boundary snapped to the nearest inter-word silence. |
| **3** | Hybrid (Global + Silence) | General purpose, **recommended default**. Each phrase is corrected by the clip's global `K` and placed at its anchor; the gap absorbs the rest. |

> **About pitch.** For real clock drift (typically a fraction of a percent),
> WhisperSync automatically uses a transparent resample instead of time-stretch
> (`atempo`/WSOLA) — pitch shifts by that same tiny fraction (inaudible on
> speech) but without WSOLA's phase artifacts. `atempo` only kicks in once a
> piece's actual correction exceeds that threshold (see `stretch_method` /
> `RESAMPLE_CONFORM_MAX_DEVIATION` in the config).

> **Mid-word stutter.** In the piecewise strategies (2, 3), piece boundaries
> snap to the nearest inter-word silence in the recorder (seam-snap-to-silence)
> instead of landing exactly on the anchor's timecode — this removes the
> characteristic mid-word stutter ("подготовил" → "подга-га-товил") without
> affecting sync accuracy.

> **Clip placement.** Every clip is aligned to the recorder independently and
> positioned from matched timecodes — clips need not be contiguous, real gaps
> between recordings are preserved. A clip with too few anchors (`min_anchors`)
> doesn't trust its timecode fit and falls back to filename order with a
> warning instead of risking a wildly wrong position.

> **Multi-camera.** Put each camera's clips in its own sub-folder of
> `--video-dir`. Each camera gets its own lane; the clean audio is synced once
> from a reference camera (`--audio-source-camera`, auto by default). Video
> files left directly in `--video-dir`'s root when camera sub-folders exist are
> ignored with a warning.

> **Multiple recorders.** Pass `--audio-file` several times. `--recorder-mode
> best` keeps one audio lane (best recorder per clip); `all` puts each recorder
> on its own lane. (Chunks of one device share a clock — concatenate them first
> instead of passing them as separate recorders.)

## Configuration

WhisperSync reads a JSON config via `--config config.json`. Priority:
**CLI flags > JSON config > defaults**. An unknown key logs a warning instead of
silently doing nothing, and a missing `--config` path is a hard error. Example:

```json
{
    "model": "large-v3",
    "device": "auto",
    "compute_type": "auto",
    "language": "auto",
    "batch_size": 16,
    "transcribe_mode": "fast",
    "fcpxml_version": "1.9",
    "default_strategy": 3,
    "use_cache": true,
    "cache_max_age_days": 0,
    "save_transcripts": true,
    "timebase_source": "camera",
    "recorder_mode": "best",
    "crossfade_enabled": true,
    "stretch_method": "auto",
    "seam_snap_max_s": 0.4,
    "render_workers": 0,
    "min_anchors": 8,
    "anchor_min_confidence": 0.6,
    "boundary_flex": true,
    "acoustic_fallback": true,
    "pause_duck_enabled": false,
    "ambience_track": false,
    "render_master_wav": false
}
```

## Troubleshooting

- **`CUDA not available, falling back to CPU`** — install the NVIDIA CUDA
  Toolkit and cuDNN; run `python -m whispersync.engine.system_check` (it
  checks via ctranslate2, the same path the transcription engine uses at
  runtime — torch is not required).
- **`ffmpeg not found in PATH`** — `sudo apt install ffmpeg` /
  `brew install ffmpeg` / download from ffmpeg.org (Windows).
- **Few anchors** — ensure both tracks contain audible speech; try
  `--language`, lower `anchor_min_confidence`, or a larger model. A clip below
  `min_anchors` falls back to filename-order placement with a warning. A clip
  with NO usable transcript match at all (music, heavy noise, a language
  Whisper garbles) automatically falls back to `acoustic_fallback` — a coarse
  GCC-PHAT cross-correlation scan across the whole recorder, directly on the
  waveforms, no words required. Less precise than text anchors, but turns a
  hard failure into a still-working (if rougher) placement.
- **High residual** — try Strategy 2 (non-linear drift); verify anchors are
  spread across the whole length.
- **Reset cache** — `rm -rf ~/.cache/whispersync/` or run with `--no-cache`.
  The cache key includes the resolved device/compute-type, so a GPU run and a
  CPU-fallback run never collide.

## Architecture

```
WhisperSync/
├── main.py                  # Thin shim -> whispersync.app:main (for running from a checkout)
├── whispersync/
│   ├── app.py                # Entry point (GUI / CLI dispatch); also the whispersync-gui script
│   ├── cli.py                 # argparse CLI
│   ├── config.py              # WhisperSyncConfig + JSON loader
│   ├── models.py              # Word, Segment, Transcript, Anchor, AlignmentMap, MediaClip, SyncPlan...
│   ├── engine/
│   │   ├── pipeline.py        # End-to-end orchestration (incl. clip_pieces — the real per-strategy planning)
│   │   ├── transcriber.py     # WhisperEngine + SHA-256 cache (keyed by resolved device/compute-type)
│   │   ├── matcher.py         # Anchor finding + RANSAC + two-stage outlier rejection + windowed match
│   │   ├── strategies.py      # Strategy id -> name/description registry
│   │   ├── acoustic.py        # GCC-PHAT cross-correlation, Boundary Flex
│   │   ├── separation.py      # Ambience-track extraction via the isolated .sep-venv
│   │   ├── timestretch.py     # ffmpeg resample-conform/atempo/segment/assemble wrappers
│   │   ├── media.py           # ffprobe, audio extraction, lossless master WAV, atempo chain
│   │   ├── naming.py          # natural sort
│   │   ├── export.py          # FCPXML generation
│   │   ├── transcript_export.py  # JSON + SRT transcript export
│   │   └── system_check.py    # Environment validation
│   └── gui/                   # PyQt6 window, worker, widgets, theme
└── tests/                     # pytest suite
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md).
[PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md) has the full technical audit, known
issues and roadmap. Before opening a PR, make sure these pass:

```bash
ruff check whispersync/ tests/
black --check whispersync/ tests/
mypy whispersync/ main.py
pytest
```

## License

MIT License — Copyright (c) 2024-2026 WhisperSync Contributors. See
[LICENSE](LICENSE).
