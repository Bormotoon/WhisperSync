"""Time-stretch utilities via ffmpeg atempo filter chains."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from bormosync.engine.media import build_atempo_chain

logger = logging.getLogger(__name__)


def edge_fade_filters(out_duration: float, fade_ms: int) -> list[str]:
    """Equal-power fade-in/out at the edges of a segment of length
    ``out_duration`` seconds. Declicks segment seams without changing the
    segment's length (so no drift is introduced). Empty if fades are disabled
    or the segment is too short."""
    if fade_ms <= 0 or out_duration <= 0:
        return []
    fade = min(fade_ms / 1000.0, out_duration / 2.0)
    if fade <= 0:
        return []
    out_start = max(0.0, out_duration - fade)
    return [
        f"afade=t=in:st=0:d={fade:.4f}",
        f"afade=t=out:st={out_start:.4f}:d={fade:.4f}",
    ]


def apply_atempo(input_path: Path, output_path: Path, factor: float) -> Path:
    chain = build_atempo_chain(factor)
    af = ",".join(chain)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        af,
        str(output_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg atempo failed: {result.stderr}")
    logger.info("atempo done → %s", output_path)
    return output_path


def apply_atempo_segment(
    input_path: Path,
    output_dir: Path,
    start: float,
    duration: float,
    factor: float,
    segment_index: int,
    fade_ms: int = 0,
) -> Path:
    output_path = output_dir / f"segment_{segment_index:04d}.wav"
    chain = build_atempo_chain(factor)
    # atempo=p changes length: out = in / p, so the output is duration/factor long.
    out_duration = duration / factor if factor else duration
    chain += edge_fade_filters(out_duration, fade_ms)
    af = ",".join(chain)
    # -ss before -i enables fast input seeking (sample-accurate for PCM/WAV),
    # so cutting many segments doesn't re-decode the whole file each time.
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(input_path),
        "-af",
        af,
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg atempo segment failed: {result.stderr}")
    return output_path


def extract_segment(
    input_path: Path,
    output_dir: Path,
    start: float,
    duration: float,
    segment_index: int,
    fade_ms: int = 0,
) -> Path:
    output_path = output_dir / f"segment_{segment_index:04d}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(input_path),
    ]
    fades = edge_fade_filters(duration, fade_ms)
    if fades:
        cmd += ["-af", ",".join(fades)]
    cmd += ["-acodec", "pcm_s16le", str(output_path)]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg extract segment failed: {result.stderr}")
    return output_path


def generate_silence(
    output_path: Path,
    duration: float,
    sample_rate: int = 48000,
) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",
        "-t",
        str(duration),
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg silence generation failed: {result.stderr}")
    return output_path


def concatenate_segments(segment_paths: list[Path], output_path: Path) -> Path:
    fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="filelist_")
    try:
        with os.fdopen(fd, "w") as f:
            for p in segment_paths:
                f.write(f"file '{p.resolve()}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            str(output_path),
        ]
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")
    finally:
        os.unlink(list_path)
    return output_path


def crossfade_segments(
    seg_a: Path,
    seg_b: Path,
    output_path: Path,
    fade_ms: int = 10,
) -> Path:
    fade_s = fade_ms / 1000.0
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(seg_a),
        "-i",
        str(seg_b),
        "-filter_complex",
        f"acrossfade=d={fade_s}:c1=tri:c2=tri",
        str(output_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg crossfade failed: {result.stderr}")
    return output_path
