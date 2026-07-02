#!/usr/bin/env python3
"""Measure realized audio/video lip-sync lag between a rendered voice track and
its source video, independent of the pipeline that produced it.

This is the measurement harness the project's flex-sync debugging was done
with (see PROJECT_ANALYSIS.md and the project memory notes): sample a grid of
points across the clip, cross-correlate a short window of the video's own
audio against the rendered voice track at each point (GCC-PHAT, the same
function Boundary Flex uses), and report the realized lag distribution. A
sharp, confident correlation peak means "this is genuinely how far off the two
tracks are" — not GCC measurement noise — so points below the sharpness gate
are dropped rather than counted as zero lag.

Usage:
    python -m tools.verify_sync --video DJI_0830.MOV --voice DJI_0830_voice.wav
    python -m tools.verify_sync --video DJI_0830.MOV --voice DJI_0830_voice.wav --json

Exit code is 0 if the median |lag| is within --median-threshold-ms (default
20ms, roughly half a frame at 24-30fps); 1 otherwise — so this doubles as a
regression check in a CI/local verification step (see --json for a
machine-readable report instead of the human-readable table).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whispersync.engine.acoustic import (  # noqa: E402
    _REFINE_SR,
    _window_slice,
    gcc_phat,
    load_mono16k_track,
)

DEFAULT_GRID_S = 5.0
DEFAULT_WINDOW_S = 4.0
DEFAULT_MAX_LAG_S = 1.0
DEFAULT_MIN_SHARPNESS = 50.0
DEFAULT_GCC_EPS = 1e-8


@dataclass
class LagSample:
    t: float
    lag_ms: float
    sharpness: float


@dataclass
class VerifyReport:
    video: str
    voice: str
    samples: list[LagSample] = field(default_factory=list)

    @property
    def confident(self) -> list[LagSample]:
        return self.samples

    def summary(self) -> dict[str, float | int]:
        if not self.confident:
            return {"n_confident": 0, "n_total": len(self.samples)}
        lags = sorted(abs(s.lag_ms) for s in self.confident)
        n = len(lags)
        return {
            "n_confident": n,
            "n_total": len(self.samples),
            "median_abs_lag_ms": lags[n // 2],
            "p90_abs_lag_ms": lags[min(n - 1, int(n * 0.9))],
            "max_abs_lag_ms": lags[-1],
        }


def measure(
    video_audio_path: Path,
    voice_path: Path,
    grid_s: float = DEFAULT_GRID_S,
    window_s: float = DEFAULT_WINDOW_S,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    min_sharpness: float = DEFAULT_MIN_SHARPNESS,
) -> VerifyReport:
    """Cross-correlate ``video_audio_path`` (the source's own scratch audio)
    against ``voice_path`` (the rendered synced voice) on a time grid, and
    return every measured point (confident or not — see
    ``VerifyReport.confident`` to filter)."""
    video_track = load_mono16k_track(video_audio_path)
    voice_track = load_mono16k_track(voice_path)
    duration_s = min(len(video_track), len(voice_track)) / _REFINE_SR

    report = VerifyReport(video=str(video_audio_path), voice=str(voice_path))
    half = window_s / 2.0
    t = half
    while t <= duration_s - half:
        video_win = _window_slice(video_track, t, window_s, _REFINE_SR)
        voice_win = _window_slice(voice_track, t, window_s, _REFINE_SR)
        lag_s, sharp = gcc_phat(video_win, voice_win, _REFINE_SR, max_lag_s, DEFAULT_GCC_EPS)
        if sharp >= min_sharpness:
            report.samples.append(LagSample(t=t, lag_ms=lag_s * 1000.0, sharpness=sharp))
        t += grid_s
    return report


def _print_table(report: VerifyReport) -> None:
    print(f"video: {report.video}")
    print(f"voice: {report.voice}")
    print()
    if not report.confident:
        print("No confident correlation points found (audio too dissimilar/quiet).")
        return
    print(f"{'t (s)':>8}  {'lag (ms)':>10}  {'sharpness':>10}")
    for s in report.confident:
        print(f"{s.t:8.1f}  {s.lag_ms:10.1f}  {s.sharpness:10.1f}")
    print()
    summary = report.summary()
    print(
        f"confident points: {summary['n_confident']}/{summary['n_total']}  "
        f"median |lag|: {summary.get('median_abs_lag_ms', 0):.1f} ms  "
        f"p90: {summary.get('p90_abs_lag_ms', 0):.1f} ms  "
        f"max: {summary.get('max_abs_lag_ms', 0):.1f} ms"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_sync",
        description=(
            "Measure realized lip-sync lag between a video's own audio and a "
            "rendered synced voice track, via GCC-PHAT cross-correlation."
        ),
    )
    parser.add_argument("--video", required=True, type=Path, help="Source video/audio file")
    parser.add_argument("--voice", required=True, type=Path, help="Rendered synced voice WAV")
    parser.add_argument("--grid-s", type=float, default=DEFAULT_GRID_S, help="Grid spacing (s)")
    parser.add_argument(
        "--window-s", type=float, default=DEFAULT_WINDOW_S, help="Correlation window (s)"
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=DEFAULT_MIN_SHARPNESS,
        help="Reject points below this confidence",
    )
    parser.add_argument(
        "--median-threshold-ms",
        type=float,
        default=20.0,
        help="Exit 1 if median |lag| exceeds this",
    )
    parser.add_argument("--json", dest="json_output", action="store_true", help="JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    report = measure(
        args.video,
        args.voice,
        grid_s=args.grid_s,
        window_s=args.window_s,
        min_sharpness=args.min_sharpness,
    )

    if args.json_output:
        print(
            json.dumps(
                {
                    "video": report.video,
                    "voice": report.voice,
                    "samples": [
                        {"t": s.t, "lag_ms": s.lag_ms, "sharpness": s.sharpness}
                        for s in report.samples
                    ],
                    "summary": report.summary(),
                },
                indent=2,
            )
        )
    else:
        _print_table(report)

    median = report.summary().get("median_abs_lag_ms")
    if median is None or median > args.median_threshold_ms:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
