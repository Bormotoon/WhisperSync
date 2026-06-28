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
        "--audio-file", required=True, type=Path, help="Path to recorder audio file"
    )
    parser.add_argument(
        "--strategy",
        choices=[1, 2, 3],
        default=1,
        type=int,
        help="Sync strategy (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sync_output.fcpxml"),
        help="Output FCPXML path",
    )
    parser.add_argument("--model", default=None, help="Whisper model name")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu")
    parser.add_argument("--compute-type", default=None, help="Compute type: float16, int8, etc.")
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
    audio_file: Path,
    progress_callback: Any = None,
) -> Any:
    import os
    from contextlib import suppress

    from bormosync.engine.matcher import align as match_align
    from bormosync.engine.media import extract_audio_to_wav
    from bormosync.engine.transcriber import WhisperEngine

    def _notify(stage: str, progress: float = 0.0, message: str = "") -> None:
        if progress_callback is not None:
            progress_callback(PipelineProgress(stage=stage, progress=progress, message=message))

    cleanup_paths: list[Path] = []
    engine: WhisperEngine | None = None

    try:
        _notify("scanning", 0.0, "Scanning video directory...")
        exts = tuple(config.video_exts)
        video_paths = sorted(
            [p for p in video_dir.iterdir() if p.suffix.lower() in exts],
            key=lambda p: p.name,
        )
        if not video_paths:
            raise RuntimeError(f"No video files found in {video_dir}")
        _notify("scanning", 1.0, f"Found {len(video_paths)} video(s)")

        _notify("extracting", 0.0, "Extracting camera audio...")
        cam_audio = extract_audio_to_wav(video_paths[0])
        cleanup_paths.append(cam_audio)
        _notify("extracting", 1.0, "Camera audio extracted")

        engine = WhisperEngine(config)

        _notify("transcribing_camera", 0.0, "Transcribing camera audio...")
        cam_transcript = engine.transcribe(cam_audio, lambda p: _notify("transcribing_camera", p))

        _notify("transcribing_recorder", 0.0, "Transcribing recorder audio...")
        rec_transcript = engine.transcribe(
            audio_file, lambda p: _notify("transcribing_recorder", p)
        )

        _notify("aligning", 0.0, "Aligning transcripts...")
        alignment = match_align(cam_transcript, rec_transcript, config)
        _notify("aligning", 1.0, f"Alignment complete: {len(alignment.anchors)} anchors")

        return alignment
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
    if not args.audio_file.is_file():
        _print(f"Error: {args.audio_file} is not a file")
        sys.exit(1)

    overrides: dict[str, object] = {}
    if args.model:
        overrides["model"] = args.model
    if args.device:
        overrides["device"] = args.device
    if args.compute_type:
        overrides["compute_type"] = args.compute_type
    if args.language:
        overrides["language"] = args.language
    if args.fcpxml_version:
        overrides["fcpxml_version"] = args.fcpxml_version

    if args.no_cache:
        overrides["use_cache"] = False

    config = load_config(args.config, **overrides)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    _print("BormoSync — Starting synchronization")
    _print(f"  Video dir:  {args.video_dir}")
    _print(f"  Audio file: {args.audio_file}")
    _print(f"  Strategy:   {args.strategy}")
    _print(f"  Output:     {args.output}")
    _print("")

    try:
        if args.dry_run:
            alignment = _run_dry_run(
                config=config,
                video_dir=args.video_dir,
                audio_file=args.audio_file,
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
                audio_file=args.audio_file,
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
