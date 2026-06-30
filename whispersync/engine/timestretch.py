"""Time-stretch utilities via ffmpeg atempo filter chains."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from whispersync.engine.media import build_atempo_chain

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


def assemble_clip(
    placements: list[tuple[Path, float]],
    total_duration: float,
    sample_rate: int,
    output_path: Path,
) -> Path:
    """Mix pre-cut speech segments onto a silent bed of ``total_duration`` so the
    result is a single WAV exactly as long as the source video clip.

    ``placements`` is a list of ``(segment_wav, local_start_seconds)``; each
    segment is delayed to its position and summed (segments do not overlap, so a
    plain sum reproduces them with silence in the gaps). One ffmpeg call.
    """
    if not placements:
        return generate_silence(output_path, total_duration, sample_rate)

    inputs: list[str] = [
        "-f",
        "lavfi",
        "-t",
        f"{total_duration:.6f}",
        "-i",
        f"anullsrc=r={sample_rate}:cl=mono",  # input 0: silent bed
    ]
    for seg_path, _local in placements:
        inputs += ["-i", str(seg_path)]

    parts: list[str] = []
    labels = ["[0:a]"]  # the bed
    for i, (_seg_path, local) in enumerate(placements):
        delay_ms = max(0, round(local * 1000))
        parts.append(f"[{i + 1}:a]adelay={delay_ms}:all=1[s{i}]")
        labels.append(f"[s{i}]")
    n_mix = len(labels)
    parts.append(
        f"{''.join(labels)}amix=inputs={n_mix}:normalize=0:duration=first[mx];"
        f"[mx]atrim=0:{total_duration:.6f},asetpts=PTS-STARTPTS[out]"
    )
    filter_complex = ";".join(parts)

    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    logger.info("Assembling synced clip (%d segments) → %s", len(placements), output_path)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip assembly failed: {result.stderr[-800:]}")
    return output_path


def assemble_continuous(
    segment_paths: list[Path],
    lead_silence: float,
    total_duration: float,
    sample_rate: int,
    output_path: Path,
) -> Path:
    """Concatenate stretched speech pieces (in order, no gaps) into one WAV of
    exactly ``total_duration`` seconds.

    ``lead_silence`` seconds of silence precede the first piece (only when the
    recorder does not cover the clip's start); the tail is padded to length. The
    audio *between* sync points is preserved and time-stretched — nothing is cut
    out and no silence is inserted mid-clip, so phrases never get clipped.
    """
    if not segment_paths:
        return generate_silence(output_path, total_duration, sample_rate)

    inputs: list[str] = []
    n = 0
    if lead_silence > 1e-3:
        inputs += [
            "-f",
            "lavfi",
            "-t",
            f"{lead_silence:.6f}",
            "-i",
            f"anullsrc=r={sample_rate}:cl=mono",
        ]
        n += 1
    for p in segment_paths:
        inputs += ["-i", str(p)]
        n += 1

    # Normalise every input to the target rate/mono so concat accepts them, glue
    # in order, then pad+trim to the exact clip length.
    parts = [f"[{j}:a]aresample={sample_rate},aformat=channel_layouts=mono[n{j}]" for j in range(n)]
    norm = "".join(f"[n{j}]" for j in range(n))
    parts.append(
        f"{norm}concat=n={n}:v=0:a=1[c];"
        f"[c]apad,atrim=0:{total_duration:.6f},asetpts=PTS-STARTPTS[out]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        ";".join(parts),
        "-map",
        "[out]",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    logger.info("Assembling continuous clip (%d pieces) → %s", len(segment_paths), output_path)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg continuous assembly failed: {result.stderr[-800:]}")
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
