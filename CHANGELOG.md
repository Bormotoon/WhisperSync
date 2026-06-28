# Changelog

All notable changes to BormoSync will be documented in this file.

## [Unreleased]

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
