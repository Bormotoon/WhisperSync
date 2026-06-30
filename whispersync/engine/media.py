"""Media probing and audio extraction via ffmpeg/ffprobe."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote


@dataclass
class MediaInfo:
    path: Path
    duration: float
    fps: Fraction | None
    width: int | None
    height: int | None
    video_codec: str | None
    audio_codec: str | None
    audio_channels: int | None
    audio_sample_rate: int | None


def probe(path: Path) -> MediaInfo:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")

    data = json.loads(result.stdout)

    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    # Some professional containers (MXF, certain MOV) omit format.duration;
    # fall back to the video, then audio, stream duration.
    duration_str = data.get("format", {}).get("duration")
    if duration_str is None and video_stream is not None:
        duration_str = video_stream.get("duration")
    if duration_str is None and audio_stream is not None:
        duration_str = audio_stream.get("duration")
    if duration_str is None:
        raise RuntimeError(f"Could not determine duration for {path}")
    duration = float(duration_str)

    fps: Fraction | None = None
    width: int | None = None
    height: int | None = None
    video_codec: str | None = None

    if video_stream:
        r_frame_rate = video_stream.get("r_frame_rate", "")
        if "/" in r_frame_rate:
            num, den = r_frame_rate.split("/")
            fps = Fraction(int(num), int(den))
        elif r_frame_rate:
            fps = Fraction(r_frame_rate)
        width = int(video_stream.get("width", 0)) or None
        height = int(video_stream.get("height", 0)) or None
        video_codec = video_stream.get("codec_name")

        # Prefer a frame-accurate video duration: the container duration often runs
        # a few frames longer than the picture (longer audio track / muxing slack),
        # and that overshoot grows with length, so the FCPXML would claim more
        # frames than the file holds and Final Cut would refuse to relink.
        nb_frames = video_stream.get("nb_frames")
        stream_dur = video_stream.get("duration")
        if nb_frames and str(nb_frames).isdigit() and int(nb_frames) > 0 and fps:
            duration = int(nb_frames) / float(fps)
        elif stream_dur:
            with contextlib.suppress(ValueError):
                duration = float(stream_dur)

    audio_codec: str | None = None
    audio_channels: int | None = None
    audio_sample_rate: int | None = None

    if audio_stream:
        audio_codec = audio_stream.get("codec_name")
        audio_channels = int(audio_stream.get("channels", 0)) or None
        sr = audio_stream.get("sample_rate")
        audio_sample_rate = int(sr) if sr else None

    return MediaInfo(
        path=path,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        video_codec=video_codec,
        audio_codec=audio_codec,
        audio_channels=audio_channels,
        audio_sample_rate=audio_sample_rate,
    )


def extract_audio_to_wav(
    input_path: Path,
    output_path: Path | None = None,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    if output_path is None:
        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        output_path = Path(tmp_name)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
    ]
    if mono:
        cmd.extend(["-ac", "1"])
    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")

    return output_path


def extract_audio_window(
    input_path: Path,
    output_path: Path,
    start: float,
    duration: float,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """Cut a short audio window ``[start, start+duration]`` to WAV.

    Like ``extract_audio_to_wav`` but seeks to an arbitrary timestamp — ``-ss``
    and ``-t`` go BEFORE ``-i`` for fast, sample-accurate seeking on PCM/WAV (the
    same convention as ``timestretch.extract_segment``). Used by the acoustic
    refine pass to pull matching camera/recorder windows for cross-correlation.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, start):.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
    ]
    if mono:
        cmd.extend(["-ac", "1"])
    cmd.append(str(output_path))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg window extraction failed: {result.stderr}")

    return output_path


def path_to_file_uri(path: Path) -> str:
    absolute = path.resolve()
    encoded = quote(str(absolute), safe="/:")
    return f"file://{encoded}"


def build_atempo_chain(factor: float) -> list[str]:
    filters: list[str] = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return filters
