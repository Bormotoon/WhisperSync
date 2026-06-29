"""CLI interface for BormoSync headless mode."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from bormosync.config import BormoSyncConfig, load_config
from bormosync.engine.pipeline import PipelineProgress, run_pipeline
from bormosync.logging_setup import setup_logging

try:
    from rich.console import Console

    _console: Console | None = Console()
except ImportError:
    _console = None

logger = logging.getLogger("bormosync.cli")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print(msg: str) -> None:
    if _console is not None:
        _console.print(msg, highlight=False)
    else:
        print(msg, flush=True)


def _progress_printer(p: PipelineProgress) -> None:
    pct = int(p.progress * 100)
    msg = p.message or p.stage
    _print(f"  [{p.stage:<25s}] {pct:3d}%  {msg}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bormosync",
        description="BormoSync — Advanced Audio/Video Synchronization Tool",
        epilog=(
            "Example:\n"
            "  bormosync --video-dir ./videos --audio-file rec.wav "
            "--strategy 1 --output output/sync.fcpxml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video-dir", required=True, type=Path, help="Path to folder with video files"
    )
    parser.add_argument(
        "--audio-file",
        required=True,
        type=Path,
        action="append",
        dest="audio_files",
        help="Path to recorder audio file (repeat for multiple recorders)",
    )
    parser.add_argument(
        "--recorder-mode",
        choices=["best", "all"],
        default=None,
        help="Multiple recorders: 'best' = one lane (best per clip), "
        "'all' = each recorder on its own lane (default: best)",
    )
    parser.add_argument(
        "--crossfade",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Declick audio segment seams with short fades (default: on). "
        "Use --no-crossfade to disable.",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=int,
        default=None,
        help="Crossfade/declick fade length in ms (default: 10)",
    )
    parser.add_argument(
        "--save-transcripts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save full transcripts (JSON + SRT) to output/transcripts/ (default: on)",
    )
    parser.add_argument(
        "--strategy",
        choices=[1, 2, 3, 4],
        default=1,
        type=int,
        help="Sync strategy: 1=global linear, 2=local stretch, 3=silence padding, "
        "4=hybrid (default: 1)",
    )
    parser.add_argument(
        "--timebase-source",
        choices=["camera", "recorder"],
        default=None,
        help="Audio sample-rate reference for FCPXML time values (default: camera)",
    )
    parser.add_argument(
        "--audio-source-camera",
        default=None,
        help="Multicam: camera sub-folder name to sync audio from (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sync_output.fcpxml"),
        help="Output FCPXML path",
    )
    parser.add_argument("--model", default=None, help="Whisper model name")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu")
    parser.add_argument(
        "--compute-type", default=None, help="Compute type: auto, float16, int8, etc."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batched-inference batch size; main GPU speed lever (default: 16)",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "quality"],
        default=None,
        dest="transcribe_mode",
        help="fast=batched (default); quality=sequential, context-aware, slower but more accurate",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional domain context to bias Whisper vocabulary",
    )
    parser.add_argument("--language", default=None, help="Language code (e.g. ru, en)")
    parser.add_argument("--fcpxml-version", default=None, help="FCPXML version (default: 1.9)")
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON config file")
    parser.add_argument("--no-cache", action="store_true", help="Disable transcription cache")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only scan + transcribe + align, skip processing"
    )
    parser.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


# ---------------------------------------------------------------------------
# Dry-run: scan + transcribe + align only
# ---------------------------------------------------------------------------


def _run_dry_run(
    config: BormoSyncConfig,
    video_dir: Path,
    audio_files: list[Path],
    progress_callback: Any = None,
) -> Any:
    import os
    from contextlib import suppress

    from bormosync.engine.matcher import align as match_align
    from bormosync.engine.media import extract_audio_to_wav
    from bormosync.engine.pipeline import scan_video_clips
    from bormosync.engine.transcriber import WhisperEngine

    def _notify(stage: str, progress: float = 0.0, message: str = "") -> None:
        if progress_callback is not None:
            progress_callback(PipelineProgress(stage=stage, progress=progress, message=message))

    cleanup_paths: list[Path] = []
    engine: WhisperEngine | None = None

    try:
        _notify("scanning", 0.0, "Scanning video directory...")
        _, video_clips = scan_video_clips(config=config, video_dir=video_dir)
        _notify("scanning", 1.0, f"Found {len(video_clips)} video clip(s)")

        engine = WhisperEngine(config)

        rec_transcripts = []
        for ri, rp in enumerate(audio_files):
            _notify("transcribing_recorder", ri / len(audio_files), f"Recorder: {rp.name}")
            rec_transcripts.append(
                engine.transcribe(rp, lambda p: _notify("transcribing_recorder", p))
            )

        # Align each clip against each recorder and return the richest alignment.
        best: Any = None
        n = len(video_clips)
        for idx, clip in enumerate(video_clips):
            _notify("transcribing_camera", idx / max(n, 1), f"Clip {idx + 1}/{n}: {clip.path.name}")
            clip_audio = extract_audio_to_wav(clip.path)
            cleanup_paths.append(clip_audio)
            clip_transcript = engine.transcribe(clip_audio)
            for rec_transcript in rec_transcripts:
                try:
                    am = match_align(clip_transcript, rec_transcript, config)
                except ValueError:
                    continue
                if best is None or len(am.anchors) > len(best.anchors):
                    best = am

        if best is None:
            raise RuntimeError("No camera clip could be aligned to any recorder audio.")
        _notify("aligning", 1.0, f"Best alignment: {len(best.anchors)} anchors")
        return best
    finally:
        if engine is not None:
            engine.unload()
        for p in cleanup_paths:
            with suppress(OSError):
                os.unlink(p)


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def _print_alignment_summary(alignment: Any) -> None:
    _print("")
    _print("=== Dry-Run Alignment Result ===")
    _print(f"  Anchors:  {len(alignment.anchors)}")
    _print(f"  Offset:   {alignment.offset:.4f} s")
    _print(f"  K:        {alignment.k:.6f}")
    _print(f"  Residual: {alignment.residual_ms:.1f} ms")


def _print_sync_result(result: Any) -> None:
    _print("")
    _print("=== Sync Complete ===")
    _print(f"  Anchors:    {result.anchors_used}")
    _print(f"  K:          {result.alignment.k:.6f}")
    _print(f"  Offset:     {result.alignment.offset:.4f} s")
    _print(f"  Residual:   {result.alignment.residual_ms:.1f} ms")
    _print(f"  Output:     {result.fcpxml_path}")
    if result.warnings:
        _print(f"  Warnings:   {', '.join(result.warnings)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.video_dir.is_dir():
        _print(f"Error: {args.video_dir} is not a directory")
        sys.exit(1)
    for af in args.audio_files:
        if not af.is_file():
            _print(f"Error: {af} is not a file")
            sys.exit(1)

    overrides: dict[str, object] = {}
    if args.model:
        overrides["model"] = args.model
    if args.device:
        overrides["device"] = args.device
    if args.compute_type:
        overrides["compute_type"] = args.compute_type
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.transcribe_mode:
        overrides["transcribe_mode"] = args.transcribe_mode
    if args.initial_prompt is not None:
        overrides["initial_prompt"] = args.initial_prompt
    if args.language:
        overrides["language"] = args.language
    if args.fcpxml_version:
        overrides["fcpxml_version"] = args.fcpxml_version
    if args.timebase_source:
        overrides["timebase_source"] = args.timebase_source
    if args.audio_source_camera:
        overrides["audio_source_camera"] = args.audio_source_camera
    if args.recorder_mode:
        overrides["recorder_mode"] = args.recorder_mode
    if args.crossfade is not None:
        overrides["crossfade_enabled"] = args.crossfade
    if args.crossfade_ms is not None:
        overrides["crossfade_ms"] = args.crossfade_ms
    if args.save_transcripts is not None:
        overrides["save_transcripts"] = args.save_transcripts

    if args.no_cache:
        overrides["use_cache"] = False

    config = load_config(args.config, **overrides)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    _print("BormoSync — Starting synchronization")
    _print(f"  Video dir:   {args.video_dir}")
    _print(f"  Audio files: {', '.join(str(a) for a in args.audio_files)}")
    _print(f"  Strategy:    {args.strategy}")
    _print(f"  Output:      {args.output}")
    _print("")

    try:
        if args.dry_run:
            alignment = _run_dry_run(
                config=config,
                video_dir=args.video_dir,
                audio_files=args.audio_files,
                progress_callback=_progress_printer,
            )
            if args.json_output:
                report = {
                    "offset": alignment.offset,
                    "k": alignment.k,
                    "anchors": len(alignment.anchors),
                    "residual_ms": alignment.residual_ms,
                    "fcpxml_path": None,
                    "warnings": [],
                }
                print(json.dumps(report, indent=2))
            else:
                _print_alignment_summary(alignment)
        else:
            result = run_pipeline(
                config=config,
                video_dir=args.video_dir,
                audio_files=args.audio_files,
                strategy_id=args.strategy,
                output_path=args.output,
                progress_callback=_progress_printer,
            )
            if args.json_output:
                report = {
                    "offset": result.alignment.offset,
                    "k": result.alignment.k,
                    "anchors": result.anchors_used,
                    "residual_ms": result.alignment.residual_ms,
                    "fcpxml_path": str(result.fcpxml_path),
                    "warnings": result.warnings,
                }
                print(json.dumps(report, indent=2))
            else:
                _print_sync_result(result)

    except Exception as exc:
        _print(f"\nError: {exc}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
