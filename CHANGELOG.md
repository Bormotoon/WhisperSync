# Changelog

All notable changes to BormoSync will be documented in this file.

## [Unreleased]

### Added
- **Toggleable seam crossfades:** short equal-power fades at audio segment
  joints declick the seams produced by the Local Time-Stretch strategy. They
  are length-preserving (no extra drift) and can be turned off via the GUI
  checkbox or `--crossfade`/`--no-crossfade` (`crossfade_enabled`, `crossfade_ms`).
- **Production-grade Whisper engine** (settings ported from the Podcast Reels
  Forge pipeline, tuned on an RTX 5060 Ti 16GB):
  - `device`/`compute_type` now default to `auto` — CUDA when available with
    float16 (or int8_float16 on older GPUs), float32 on CPU.
  - Batched GPU inference (`batch_size`, default 16) with an OOM fallback ladder
    (GPU batch → batch/2 → … → CPU) for fast multi-hour transcription.
  - Anti-hallucination decoding: temperature ladder, `condition_on_previous_text`
    off by default, `repetition_penalty`, `no_repeat_ngram_size`, plus
    compression-ratio / log-prob / no-speech thresholds and tuned VAD params —
    kills the "endless Спасибо." loop, yielding cleaner anchors.
  - `fast` (batched) vs `quality` (sequential, context-aware) modes, optional
    `initial_prompt`, `best_of`, `patience`. New CLI: `--batch-size`, `--mode`,
    `--initial-prompt`. `HF_HUB_DISABLE_XET=1` set to avoid HF network hangs.
- **Multiple recorders (different devices):** pass several `--audio-file` flags.
  Each clip is aligned against every recorder; the timeline is placed from the
  best-covering ("primary") recorder. `recorder_mode` / `--recorder-mode`:
  `best` (default) syncs each clip from its strongest recorder on one audio lane;
  `all` places every recorder on its own audio lane (-1, -2, …) for multi-mic /
  multi-speaker setups. (Chunks of one device should just be concatenated first —
  same clock, lossless.)
- **Multi-camera support:** put each camera's clips in its own sub-folder of the
  video directory; each camera is placed on its own lane (1, 2, 3, …) and aligned
  to the recorder independently. The clean audio is synced once from a chosen
  reference camera (`audio_source_camera` / `--audio-source-camera`, default
  auto-picks the best-aligned camera) so it isn't duplicated across angles.
- **Windowed matching for long recordings:** each clip is first coarsely located
  in the (possibly multi-hour) reference by rare-word delta voting, then matched
  precisely only inside a window around that estimate. This avoids O(N²) difflib
  over the full stream and the false matches caused by phrases repeating across
  hours. Falls back to a full search if the window looks weak. Tunable via
  `match_window_margin`, `seed_max_occurrences`, `seed_bin_width`.
- **Strategy 4 — Hybrid (Global + Silence):** each phrase is tempo-corrected by
  the clip's global drift K and then placed at its anchor position with silence
  absorbing the rest. Robust against non-linear drift and near pitch-perfect.
- **Per-clip alignment & timecode-based placement:** every camera clip is now
  aligned to the recorder independently and positioned on the timeline from its
  matched recorder start time. Clips are no longer assumed contiguous — real
  gaps between recordings are preserved (works for arbitrary sources).
- **Timebase source selection** (`timebase_source`, `--timebase-source`, GUI
  dropdown): choose whether FCPXML audio time values snap to the camera (default)
  or recorder sample rate.

### Changed
- Strategies now plan per camera clip (`plan_clip`) instead of one global plan.

### Fixed
- **Critical:** only the first camera clip was transcribed, so on the real
  "one long recorder track + many video files" workflow anchors covered just
  the first clip and drift across the full session was never corrected. The
  pipeline now transcribes the scratch audio of *every* clip and merges it
  onto the concatenated camera timeline (word times shifted by clip offset).
- **Critical:** sync strategies dropped the camera video clips, so exported
  FCPXML contained no video — all three strategies now keep video on lane 1
- **Critical:** Local Time-Stretch / Silence Padding produced clips whose
  `start` pointed past the trimmed segment file length; pipeline now resets
  `in_point` to 0 after extraction/atempo
- Sequence/gap duration now spans the full video extent, not just audio
- FCPXML now emits standard `<asset-clip>` elements (was non-standard
  `<clip ref=...>`) with audio assets declared on the audio sample-rate
  timebase and proper `audioRate`/`audioChannels`
- Strategy and alignment-quality warnings are now propagated into `SyncResult`
- `--no-cache` now actually disables the transcription cache (`config.use_cache`)
- Whisper `unload()` now runs `gc.collect()` + `torch.cuda.empty_cache()` to
  truly free VRAM

### Changed
- Segment extraction uses fast input seeking (`-ss` before `-i`) to avoid
  re-decoding the whole recording for every segment

### Added
- Project scaffolding with full package structure (bormosync/engine/, gui/, widgets/)
- Core data models: Word, Segment, Transcript, Anchor, AlignmentMap, MediaClip, SyncPlan, SyncResult
- Configuration management with JSON config support and CLI overrides
- System check utility (ffmpeg, CUDA, Python, dependencies, disk space)
- Media probing via ffprobe (duration, fps, codecs, sample rate)
- Audio extraction to 16kHz mono WAV for Whisper
- WhisperEngine with lazy model loading and SHA256-based transcription cache
- Anchor matching with SequenceMatcher, uniqueness and monotonicity filters
- RANSAC-based linear regression for clock drift (K) and offset estimation
- Three synchronization strategies:
  - Strategy 1: Global Linear Calibration (single atempo pass)
  - Strategy 2: Local Time-Stretch (per-segment atempo between anchors)
  - Strategy 3: Silence Padding (pitch-safe, speech segments untouched)
- Time-stretch utilities: atempo chain decomposition, segment extraction, silence generation, concatenation, crossfade
- FCPXML generator with rational time values, DOCTYPE, spine/gap/clip structure
- End-to-end pipeline orchestrator with progress callbacks
- PyQt6 GUI with aggressive dark theme (#0A0A0A, #D32F2F, #FF5722)
  - Drag & drop zones for video folder and audio file
  - Strategy selection with visual diagrams
  - Timeline preview with dual-lane visualization
  - Colored log viewer with autoscroll
  - QObject worker with moveToThread pattern and cancellation
  - QSettings persistence for last paths
- CLI headless mode with argparse (--video-dir, --audio-file, --strategy, --json, --verbose)
- PyInstaller spec for --onedir packaging
- Comprehensive test suite: matcher, strategies, timestretch, export (16 tests)

## [0.1.0] - 2026-06-28

### Added
- Initial release
