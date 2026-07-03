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
    # Sample format as reported by ffprobe (e.g. "s16", "s32", "fltp") and its bit
    # depth, when known. Used to pick a lossless PCM codec for rendered audio that
    # matches (or safely exceeds) the source, instead of hard-coding 16-bit.
    audio_sample_fmt: str | None = None
    audio_bits_per_sample: int | None = None


def probe(path: Path, timeout: float = 30.0) -> MediaInfo:
    """Read duration/fps/codecs/etc via ffprobe. ``timeout`` (seconds) is
    configurable — the default is generous for local files, but a clip on
    network/NAS storage can legitimately take longer to respond. See
    PROJECT_ANALYSIS.md §6.6.
    """
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
    audio_sample_fmt: str | None = None
    audio_bits_per_sample: int | None = None

    if audio_stream:
        audio_codec = audio_stream.get("codec_name")
        audio_channels = int(audio_stream.get("channels", 0)) or None
        sr = audio_stream.get("sample_rate")
        audio_sample_rate = int(sr) if sr else None
        audio_sample_fmt = audio_stream.get("sample_fmt") or None
        # bits_per_raw_sample is the true source depth (e.g. 24-bit in a 32-bit
        # container); bits_per_sample is the container's storage width. Prefer the
        # raw value when ffprobe reports it.
        bps = audio_stream.get("bits_per_raw_sample") or audio_stream.get("bits_per_sample")
        audio_bits_per_sample = int(bps) if bps and str(bps).isdigit() and int(bps) > 0 else None

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
        audio_sample_fmt=audio_sample_fmt,
        audio_bits_per_sample=audio_bits_per_sample,
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


def extract_audio_master(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    channels: int,
    codec: str,
) -> Path:
    """Transcode a recorder file to lossless PCM WAV, once, for the render path.

    Cutting pieces directly from a lossy source (mp3/m4a) with ``-ss`` before
    ``-i`` is not sample-accurate (seek lands on a frame boundary, ~26ms for
    mp3), and re-decoding a lossy file on every cut compounds artifacts. This
    produces a single PCM master at the render's target sample rate and native
    channel count so every downstream cut/concat operates on identical,
    sample-accurate, uncompressed audio. Uses the highest-quality resampler
    ffmpeg ships (soxr) when available, falling back to swr.
    """
    resamplers = ("soxr", "swr")
    last_stderr = ""
    for i, resampler in enumerate(resamplers):
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-af",
            f"aresample=resampler={resampler}",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-acodec",
            codec,
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            return output_path
        last_stderr = result.stderr
        if i < len(resamplers) - 1:
            # This ffmpeg build may lack libsoxr (or some other resampler-specific
            # issue); retry with the next resampler rather than parsing stderr text,
            # which varies across ffmpeg versions.
            continue
    raise RuntimeError(f"ffmpeg master extraction failed: {last_stderr}")


def pcm_codec_for_bit_depth(bits_per_sample: int | None) -> str:
    """Lossless PCM codec that matches (or safely covers) a source bit depth.

    Rendering everything through 16-bit PCM regardless of source depth throws
    away real resolution from 24-/32-bit recorders and adds a fresh quantization
    step at every intermediate render stage. ``None`` (unknown depth, or a lossy
    source codec ffprobe can't report PCM depth for) defaults to 24-bit, which
    covers the vast majority of professional recorders without truncation.
    """
    if bits_per_sample is not None and bits_per_sample <= 16:
        return "pcm_s16le"
    if bits_per_sample is not None and bits_per_sample >= 32:
        return "pcm_s32le"
    return "pcm_s24le"


def path_to_file_uri(path: Path) -> str:
    """A ``file://`` URI for ``path``, percent-encoded and platform-correct.

    ``Path.as_uri()`` handles this properly cross-platform — notably on
    Windows, where a hand-rolled ``f"file://{quote(str(path))}"`` would encode
    backslashes as ``%5C`` instead of converting them to the forward slashes a
    URI requires, producing a URI FCPXML/other tools can't resolve. See
    PROJECT_ANALYSIS.md §3.1.
    """
    return path.resolve().as_uri()


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
