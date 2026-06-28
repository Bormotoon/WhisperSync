"""CLI interface for BormoSync headless mode."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bormosync.config import load_config
from bormosync.engine.pipeline import PipelineProgress, run_pipeline
from bormosync.logging_setup import setup_logging


def _progress_printer(p: PipelineProgress) -> None:
    pct = int(p.progress * 100)
    msg = p.message or p.stage
    print(f"  [{p.stage:<25s}] {pct:3d}%  {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BormoSync — Advanced Audio/Video Synchronization Tool",
        epilog=(
            "Example:\n"
            "  python main.py --cli --video-dir ./videos --audio-file rec.wav --strategy 1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video-dir", required=True, type=Path, help="Path to video folder")
    parser.add_argument("--audio-file", required=True, type=Path, help="Path to recorder audio")
    parser.add_argument(
        "--strategy", type=int, choices=[1, 2, 3], default=1, help="Sync strategy (default: 1)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/sync_output.fcpxml"), help="Output FCPXML path"
    )
    parser.add_argument("--model", default=None, help="Whisper model name")
    parser.add_argument("--device", default=None, help="Device: cuda or cpu")
    parser.add_argument("--compute-type", default=None, help="Compute type: float16, int8, etc.")
    parser.add_argument("--language", default=None, help="Language code (e.g. ru, en)")
    parser.add_argument("--fcpxml-version", default=None, help="FCPXML version (default: 1.9)")
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON config file")
    parser.add_argument("--no-cache", action="store_true", help="Disable transcription cache")
    parser.add_argument("--dry-run", action="store_true", help="Only align, don't process audio")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level)

    if not args.video_dir.is_dir():
        print(f"Error: {args.video_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    if not args.audio_file.is_file():
        print(f"Error: {args.audio_file} is not a file", file=sys.stderr)
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

    config = load_config(args.config, **overrides)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("BormoSync — Starting synchronization")
    print(f"  Video dir:  {args.video_dir}")
    print(f"  Audio file: {args.audio_file}")
    print(f"  Strategy:   {args.strategy}")
    print(f"  Output:     {args.output}")
    print()

    try:
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
            print()
            print("=== Sync Complete ===")
            print(f"  Anchors:    {result.anchors_used}")
            print(f"  K:          {result.alignment.k:.6f}")
            print(f"  Offset:     {result.alignment.offset:.4f} s")
            print(f"  Residual:   {result.alignment.residual_ms:.1f} ms")
            print(f"  Output:     {result.fcpxml_path}")
            if result.warnings:
                print(f"  Warnings:   {', '.join(result.warnings)}")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
