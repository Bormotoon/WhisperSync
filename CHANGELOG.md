# Changelog

All notable changes to WhisperSync will be documented in this file.

## [Unreleased]

### Added — new features (Stage 7, 2026-07-03)

- **Auto-strategy recommendation**: after a run, the residual/local-drift
  characteristics of the best alignment are checked against the strategy
  actually used; if a different strategy would likely fit better, a warning
  suggests it (transcripts are cached, so re-running is cheap).
- **Acoustic fallback ("Strategy 0")**: a clip with no usable transcript match
  against any recorder (music, background noise, near-silence, a language
  Whisper garbles) now falls back to a coarse GCC-PHAT cross-correlation grid
  scan across the waveforms to estimate offset/K directly — turning a hard
  failure into a still-working (if less precise) placement, as long as the
  same physical audio event reaches both the camera and recorder mic. On by
  default (`acoustic_fallback`).
- **Per-camera AV/lip-sync calibration**: `camera_av_offset_ms` /
  `camera_av_offset_ms_by_camera` (and `--camera-av-offset-ms`) apply a
  constant correction for a camera's own mic-to-lips delay, which no
  acoustic method can see on its own.
- **`--render-master-wav`**: optionally render a single WAV spanning the
  whole timeline (every synced voice clip, and the ambience track if
  enabled, mixed at their timeline offsets over a silent bed) next to the
  FCPXML, for anyone without an NLE.
- **GUI parity with the CLI**: the Recorder Audio drop zone now accepts
  multiple files (drag-drop or Browse) with a `best`/`all` recorder-mode
  picker; a new "Transcription Settings..." dialog exposes
  model/language/device/compute-type/initial-prompt/transcribe-mode; a
  "Re-run with Selected Strategy" button appears after a successful run
  (transcripts are cached, so it skips straight to alignment/render); and
  the status/log now shows "Loading Whisper model..." during a first-time
  (possibly HuggingFace-downloading) model load instead of looking frozen.

### Changed — quality/reliability overhaul (2026-07-02)

A full project audit (`PROJECT_ANALYSIS.md`) found that the rendered voice
track was audibly worse than the recorder source for reasons unrelated to
synchronization, that strategies 3 and 4 had silently become identical, and a
long tail of reliability/cross-platform/dead-code issues. This release fixes
all of it:

- **Bit-perfect render path**: the render no longer forces mono/16-bit —
  every stage (extract, resample-conform/atempo, assemble, pause-duck)
  preserves the recorder's native channel count and a lossless PCM codec
  matching its bit depth. Recorders are normalized to a lossless master WAV
  once up front (fixes non-sample-accurate cutting from lossy sources like
  mp3/m4a, and format mismatches between pieces and lead-silence). A new
  transparent resample ("varispeed") conform replaces `atempo`/WSOLA for the
  small tempo changes real clock drift produces, avoiding WSOLA's phase/
  texture artifacts where they weren't needed. Fades apply only to seams that
  are acoustically discontinuous (e.g. nudged by Boundary Flex), not to every
  piece boundary — the old behaviour carved an audible volume dip into
  otherwise-continuous audio on nearly every seam of the phrase-wise
  strategies. Pause-ducking is folded into the assembly pass instead of a
  second full decode/encode.
- **Strategies 3 and 4 merged**: in the real render path they had become
  byte-identical (old strategy 3 "Silence Padding" promised zero pitch-shift
  but was actually time-stretching every phrase like Hybrid). Now one honest
  "Hybrid" strategy at id 3; `--strategy 4` is a deprecated alias.
- **Seam-snap-to-silence** replaces the old tempo-factor smoothing (which
  fixed the mid-word stutter by averaging atempo factors but let speech drift
  off the picture by up to 1.4s): piece boundaries now snap to the nearest
  recorder inter-word silence, so a seam never lands mid-word without
  touching any piece's tempo factor.
- **Pause ducking** now ducks only where BOTH the camera and recorder tracks
  are actually silent (from full word lists), not gaps between matched
  anchors — a quiet phrase or a word Whisper missed no longer gets
  attenuated as a false "pause".
- **Reliability**: Whisper's VRAM is freed right after alignment instead of
  being held through rendering/ambience separation (a common GPU-OOM cause);
  clips with no audio track no longer abort the whole run; a JSON config with
  a typo'd key now warns instead of silently no-opping; the transcript cache
  key is keyed by the actually-resolved device/compute-type, not the literal
  `"auto"`.
- **Cross-platform**: `file://` URIs use `Path.as_uri()` (the old hand-rolled
  version mis-encoded Windows paths); "Open Output Folder" uses
  `QDesktopServices` instead of Linux-only `xdg-open`.
- **CLI**: `--json` now sends all human-readable output to stderr so the
  report on stdout is clean for piping; added `--version`; exit codes
  distinguish usage errors (2) from run failures (1); fixed `main.py --cli`
  passing `--cli` through to argparse unstripped.
- **Dead code removed**: `engine/dtw.py` (banded-DTW anchor matcher — shelved
  after real-data measurements showed it performed worse than the legacy
  matcher), `acoustic.refine_anchors` (Tier-2 anchor correction, superseded
  once the real float bug was found in the render path, not the anchor
  layer), the entire `plan_clip`-based `SyncStrategy` class hierarchy in
  `strategies.py` (the pipeline only ever read `.name`; real planning lives
  in `pipeline.clip_pieces`), and several unused `timestretch`/`naming`
  helpers.
- **Packaging**: `pyproject.toml` now declares a build backend and its actual
  dependencies (`pip install .` previously installed nothing); dropped the
  unused `pydub`/`ffmpeg-python`; the GUI/CLI dispatcher moved from
  repo-root `main.py` into `whispersync/app.py` so the `whispersync-gui`
  entry point resolves after a wheel install.
- Unified defaults: `default_strategy` (1 → 3) and `boundary_flex`
  (off → on) are now read from `WhisperSyncConfig` by both the CLI and the
  GUI, instead of disagreeing with each other.

See `PROJECT_ANALYSIS.md` for the full technical audit this release addresses.

### Added
- **GitHub-ready project**: real GUI screenshots (rendered via Qt offscreen) in
  the README, bilingual README (RU + `README.en.md`), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates, a GitHub Actions CI
  workflow (ruff + black + mypy + pytest on 3.11/3.12), and richer `pyproject`
  metadata (urls, classifiers, keywords).
- **Full transcript export (JSON + SRT):** the transcription is computed for
  alignment anyway, so it is now also saved next to the output under
  `output/transcripts/` — one `.json` + `.srt` per recorder and per camera clip,
  in the Podcast Reels Forge format (segments with word-level timestamps +
  sentence groups). Toggle with `save_transcripts` / `--save-transcripts` /
  `--no-save-transcripts` (default on).
- **Filename-aware ordering & preliminary layout:** clips are sorted in natural
  order (DJI_9 < DJI_10 < DJI_100) and consecutive runs (DJI_0838, DJI_0839, …)
  are detected. The timeline is now populated from filenames right after scanning
  (clips laid end-to-end per camera) and the current clip is highlighted while it
  transcribes, so progress is visible before alignment finishes. Clips that fail
  alignment fall back to their filename order, and a warning is raised if matched
  timecodes contradict the filename order (likely misalignment).
- **Full multi-track timeline in the GUI:** one row per camera and per audio
  lane, showing each clip's real position (how far it moved), the applied speed
  change (e.g. `+0.10%`), and live sync status — pending (dashed/dim), working
  (orange outline), done (solid). Updates live as clips are processed; hover a
  clip for offset / duration / in-point / speed / status. Driven by per-clip
  timeline snapshots emitted from the pipeline (`PipelineProgress.clips`).
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
- Project scaffolding with full package structure (whispersync/engine/, gui/, widgets/)
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
