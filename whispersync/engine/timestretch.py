"""Time-stretch utilities via ffmpeg atempo/resample filter chains.

Everything here preserves the recorder's native channel count and a caller-
chosen lossless PCM codec end to end (extract -> stretch -> assemble), instead
of collapsing to 16-bit mono at every stage. See PROJECT_ANALYSIS.md §2.0 for
why that mattered: a multi-pass render through 16-bit mono compounds
quantization noise and throws away stereo width the user recorded on purpose.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from whispersync.engine.media import build_atempo_chain

logger = logging.getLogger(__name__)

# Default channel count / codec for call sites that don't yet pass one explicitly
# (kept only so this module still has sane standalone defaults; the pipeline
# always passes the recorder's real values).
_DEFAULT_CHANNELS = 1
_DEFAULT_CODEC = "pcm_s24le"

# Above this fractional tempo change, ffmpeg's atempo (WSOLA) is used; at or
# below it, a plain resample ("varispeed") conform is audibly transparent and
# avoids WSOLA's phase/texture artifacts entirely. Real clock drift between a
# camera and a recorder is typically 0.01-0.5%, well under this threshold.
RESAMPLE_CONFORM_MAX_DEVIATION = 0.005


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


def edge_fade_filter_in(fade_s: float) -> str:
    return f"afade=t=in:st=0:d={fade_s:.4f}"


def edge_fade_filter_out(out_duration: float, fade_s: float) -> str:
    out_start = max(0.0, out_duration - fade_s)
    return f"afade=t=out:st={out_start:.4f}:d={fade_s:.4f}"


def seam_fade_filters(
    out_duration: float, fade_ms: int, fade_in: bool, fade_out: bool
) -> list[str]:
    """Like ``edge_fade_filters`` but each edge is independently toggled.

    A fade is only needed on an edge that is acoustically discontinuous with
    its neighbour (the boundary was acoustically nudged, or the pieces are not
    contiguous in the source). Fading BOTH edges of every piece — even ones
    that butt up against an untouched neighbour — creates an audible dip to
    silence at every seam (a fade-out into a fade-in), which on the phrase-wise
    strategies happens on nearly every seam. See PROJECT_ANALYSIS.md §2.0.
    """
    if fade_ms <= 0 or out_duration <= 0 or not (fade_in or fade_out):
        return []
    fade = min(fade_ms / 1000.0, out_duration / 2.0)
    if fade <= 0:
        return []
    filters: list[str] = []
    if fade_in:
        filters.append(edge_fade_filter_in(fade))
    if fade_out:
        filters.append(edge_fade_filter_out(out_duration, fade))
    return filters


def _duck_pause_expr(a: float, b: float, duck_lin: float, fade: float) -> str:
    """A single ``volume`` expression (eval=frame) for ONE pause [a, b]: 1.0 outside
    [a-fade, b+fade], a linear ramp 1→duck on [a-fade, a], flat ``duck_lin`` on
    [a, b], and a ramp duck→1 on [b, b+fade].

    One filter per pause keeps each expression tiny; the pipeline chains them, which
    ffmpeg handles far better than a single product expression over many pauses (a
    huge expression overflows the filtergraph parser)."""
    d = duck_lin
    ramp_in = f"(1-(1-{d:.6f})*(t-({a - fade:.4f}))/{fade:.4f})"
    ramp_out = f"({d:.6f}+(1-{d:.6f})*(t-({b:.4f}))/{fade:.4f})"
    return (
        f"if(lt(t,{a - fade:.4f}),1,"
        f"if(lt(t,{a:.4f}),{ramp_in},"
        f"if(lt(t,{b:.4f}),{d:.6f},"
        f"if(lt(t,{b + fade:.4f}),{ramp_out},1))))"
    )


def duck_filter_chain(
    pauses: list[tuple[float, float]], duck_db: float, fade_ms: int
) -> str | None:
    """Build the ``volume=...`` filter chain that ducks ``pauses``, or ``None`` if
    there is nothing to duck (no spans, or ``duck_db`` disables ducking).

    Split out from ``apply_pause_ducking`` so ``assemble_continuous`` can fold
    ducking into its own filter graph in a single ffmpeg pass instead of a
    second full decode/encode over an intermediate file.
    """
    fade = max(fade_ms, 1) / 1000.0
    spans = [(a, b) for a, b in pauses if b - a > 1e-3]
    if not spans or duck_db >= 0.0:
        return None
    duck_lin = 0.0 if duck_db <= -120.0 else 10.0 ** (duck_db / 20.0)
    # One small volume filter per pause, chained (commas). A single combined
    # expression over all pauses overflows ffmpeg's filtergraph parser.
    return ",".join(
        f"volume=volume='{_duck_pause_expr(a, b, duck_lin, fade)}':eval=frame" for a, b in spans
    )


def apply_pause_ducking(
    input_path: Path,
    output_path: Path,
    pauses: list[tuple[float, float]],
    duck_db: float,
    fade_ms: int,
    sample_rate: int,
    channels: int = _DEFAULT_CHANNELS,
    codec: str = _DEFAULT_CODEC,
) -> Path:
    """Attenuate the given ``pauses`` (local-time [start, end] spans, seconds) by
    ``duck_db`` decibels with an equal-time linear fade at each edge.

    ``duck_db`` 0 = no change; -inf (or very negative) = full silence. Speech regions
    are left at unity gain. A no-op copy is returned when there is nothing to duck.
    Kept as a standalone step for callers that don't go through
    ``assemble_continuous`` (e.g. re-ducking an already-assembled file); the
    pipeline itself asks ``assemble_continuous`` to duck inline (one pass).
    """
    chain = duck_filter_chain(pauses, duck_db, fade_ms)
    if chain is None:
        # Nothing to do — straight copy so callers always get an output file.
        cmd = ["ffmpeg", "-y", "-i", str(input_path), "-c", "copy", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg pause-duck copy failed: {result.stderr}")
        return output_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        chain,
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-acodec",
        codec,
        str(output_path),
    ]
    logger.info("Pause-ducking %d pause(s) by %.1f dB", len(pauses), duck_db)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg pause-duck failed: {result.stderr[-800:]}")
    return output_path


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
    fade_in: bool = True,
    fade_out: bool = True,
    channels: int = _DEFAULT_CHANNELS,
    codec: str = _DEFAULT_CODEC,
) -> Path:
    output_path = output_dir / f"segment_{segment_index:04d}.wav"
    chain = build_atempo_chain(factor)
    # atempo=p changes length: out = in / p, so the output is duration/factor long.
    out_duration = duration / factor if factor else duration
    chain += seam_fade_filters(out_duration, fade_ms, fade_in, fade_out)
    # atempo's actual output is a few ms SHORTER than duration/factor (filter priming
    # / fractional-sample rounding) — a consistent ~-3 ms/piece bias. With hundreds
    # of pieces concatenated by assemble_continuous, that accumulates into seconds of
    # progressive A/V drift. Force every piece to its exact intended length: pad any
    # shortfall with silence, then hard-trim to out_duration so concatenation is
    # sample-exact regardless of atempo rounding.
    chain += ["apad", f"atrim=0:{out_duration:.6f}", "asetpts=PTS-STARTPTS"]
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
        "-ac",
        str(channels),
        "-acodec",
        codec,
        str(output_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg atempo segment failed: {result.stderr}")
    return output_path


def resample_conform_segment(
    input_path: Path,
    output_dir: Path,
    start: float,
    duration: float,
    factor: float,
    segment_index: int,
    sample_rate: int,
    fade_ms: int = 0,
    fade_in: bool = True,
    fade_out: bool = True,
    channels: int = _DEFAULT_CHANNELS,
    codec: str = _DEFAULT_CODEC,
) -> Path:
    """Conform a piece's tempo by resampling ("varispeed") instead of atempo/WSOLA.

    For the small tempo changes real clock drift produces (well under 1%), a
    resample is audibly transparent — it shifts pitch by the same tiny fraction
    (inaudible on speech) but, unlike WSOLA, introduces no phase/texture
    artifacts, because it doesn't try to preserve pitch while changing duration.
    Reserved for ``|factor - 1| <= RESAMPLE_CONFORM_MAX_DEVIATION``; larger
    changes (an outlier anchor, or a phrase actually needing correction) still
    go through ``apply_atempo_segment``. See PROJECT_ANALYSIS.md §2.0.
    """
    output_path = output_dir / f"segment_{segment_index:04d}.wav"
    out_duration = duration / factor if factor else duration
    # Reinterpreting the sample rate by `factor` and resampling back to the
    # target rate is exactly a duration/pitch-shifting conform: reading the
    # source `factor`x "faster" makes it `factor`x shorter, then aresample
    # brings it back to the pipeline's common sample rate.
    new_rate = max(1, round(sample_rate * factor))
    chain = [f"asetrate={new_rate}", f"aresample={sample_rate}:resampler=soxr"]
    chain += seam_fade_filters(out_duration, fade_ms, fade_in, fade_out)
    chain += ["apad", f"atrim=0:{out_duration:.6f}", "asetpts=PTS-STARTPTS"]
    af = ",".join(chain)
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
        "-ac",
        str(channels),
        "-acodec",
        codec,
        str(output_path),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0 and ":resampler=soxr" in af:
        # This ffmpeg build may lack libsoxr (error text varies by version, so we
        # don't parse it); retry once with the default resampler.
        cmd[cmd.index("-af") + 1] = af.replace(":resampler=soxr", "")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg resample-conform segment failed: {result.stderr}")
    return output_path


def extract_segment(
    input_path: Path,
    output_dir: Path,
    start: float,
    duration: float,
    segment_index: int,
    fade_ms: int = 0,
    fade_in: bool = True,
    fade_out: bool = True,
    channels: int = _DEFAULT_CHANNELS,
    codec: str = _DEFAULT_CODEC,
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
    fades = seam_fade_filters(duration, fade_ms, fade_in, fade_out)
    if fades:
        cmd += ["-af", ",".join(fades)]
    cmd += ["-ac", str(channels), "-acodec", codec, str(output_path)]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg extract segment failed: {result.stderr}")
    return output_path


def render_piece(
    input_path: Path,
    output_dir: Path,
    rec_start: float,
    rec_dur: float,
    factor: float,
    index: int,
    fade_ms: int,
    fade_in: bool = True,
    fade_out: bool = True,
    sample_rate: int = 48000,
    channels: int = _DEFAULT_CHANNELS,
    codec: str = _DEFAULT_CODEC,
    stretch_method: str = "auto",
) -> Path:
    """Render one timeline piece to ``segment_{index:04d}.wav``: a plain cut when
    the tempo is unchanged, a transparent resample conform for a small tempo
    change, or an atempo (WSOLA) stretch for a larger one. A pure, side-effect-
    free wrapper (only writes its own indexed file) so it can run in a process
    pool — the output is identical and order-independent because each piece
    owns a distinct filename.

    ``stretch_method``: "auto" picks resample-conform for
    ``|factor-1| <= RESAMPLE_CONFORM_MAX_DEVIATION`` and atempo otherwise;
    "atempo"/"resample" force one method regardless of the factor (mainly for
    testing/comparison).
    """
    if abs(factor - 1.0) <= 1e-6:
        return extract_segment(
            input_path,
            output_dir,
            rec_start,
            rec_dur,
            index,
            fade_ms=fade_ms,
            fade_in=fade_in,
            fade_out=fade_out,
            channels=channels,
            codec=codec,
        )
    use_resample = stretch_method == "resample" or (
        stretch_method == "auto" and abs(factor - 1.0) <= RESAMPLE_CONFORM_MAX_DEVIATION
    )
    if use_resample:
        return resample_conform_segment(
            input_path,
            output_dir,
            rec_start,
            rec_dur,
            factor,
            index,
            sample_rate,
            fade_ms=fade_ms,
            fade_in=fade_in,
            fade_out=fade_out,
            channels=channels,
            codec=codec,
        )
    return apply_atempo_segment(
        input_path,
        output_dir,
        rec_start,
        rec_dur,
        factor,
        index,
        fade_ms=fade_ms,
        fade_in=fade_in,
        fade_out=fade_out,
        channels=channels,
        codec=codec,
    )


def generate_silence(
    output_path: Path,
    duration: float,
    sample_rate: int = 48000,
    channels: int = _DEFAULT_CHANNELS,
    codec: str = _DEFAULT_CODEC,
) -> Path:
    layout = "mono" if channels == 1 else "stereo" if channels == 2 else f"{channels}c"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={sample_rate}:cl={layout}",
        "-t",
        str(duration),
        "-acodec",
        codec,
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
    channels: int = _DEFAULT_CHANNELS,
    codec: str = _DEFAULT_CODEC,
    duck_pauses: list[tuple[float, float]] | None = None,
    duck_db: float = 0.0,
    duck_fade_ms: int = 80,
) -> Path:
    """Concatenate stretched speech pieces (in order, no gaps) into one WAV of
    exactly ``total_duration`` seconds.

    ``lead_silence`` seconds of silence precede the first piece (only when the
    recorder does not cover the clip's start); the tail is padded to length. The
    audio *between* sync points is preserved and time-stretched — nothing is cut
    out and no silence is inserted mid-clip, so phrases never get clipped.

    ``channels``/``codec`` are the recorder's native channel count and a
    lossless PCM codec matching its bit depth — every piece (from
    ``render_piece``) and the lead-silence bed already share this format, so
    concatenation and this final encode never touch the audio's resolution.

    Pass ``duck_pauses`` (with ``duck_db`` < 0) to fold pause-ducking into this
    same ffmpeg invocation instead of a separate ``apply_pause_ducking`` pass
    over an intermediate file — one encode instead of two.
    """
    if not segment_paths:
        return generate_silence(output_path, total_duration, sample_rate, channels, codec)

    # Concatenate via the concat *demuxer* (a single -i reading the list file),
    # not by opening every segment as its own -i input: a clip can contain
    # thousands of pieces and one fd per input blows past the open-files limit
    # ("Too many open files"). The demuxer streams the segments one at a time.
    #
    # An optional lead-silence WAV is generated up front and prepended to the
    # list. Output is resampled to the target rate (segments inherit the
    # recording's rate, which may differ), then padded+trimmed to the exact clip
    # length so phrases are never clipped. Channel count is preserved throughout
    # (no forced downmix), so this resample never touches channel layout.
    list_entries: list[Path] = list(segment_paths)
    lead_path: Path | None = None
    if lead_silence > 1e-3:
        lead_path = output_path.parent / f".lead_{output_path.stem}.wav"
        generate_silence(lead_path, lead_silence, sample_rate, channels, codec)
        list_entries.insert(0, lead_path)

    duck_chain = duck_filter_chain(duck_pauses or [], duck_db, duck_fade_ms)

    fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="whispersync_concat_")
    try:
        with os.fdopen(fd, "w") as f:
            for p in list_entries:
                # Single quotes in paths must be escaped for the concat demuxer.
                escaped = str(p.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        af = (
            f"aresample={sample_rate}:resampler=soxr,"
            f"apad,atrim=0:{total_duration:.6f},asetpts=PTS-STARTPTS"
        )
        if duck_chain:
            af += f",{duck_chain}"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-af",
            af,
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-acodec",
            codec,
            str(output_path),
        ]
        logger.info(
            "Assembling continuous clip (%d pieces%s) → %s",
            len(segment_paths),
            f", ducking {len(duck_pauses or [])} pause(s)" if duck_chain else "",
            output_path,
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0 and ":resampler=soxr" in af:
            # This ffmpeg build may lack libsoxr; retry once with the default
            # resampler rather than parsing version-specific error text.
            cmd[cmd.index("-af") + 1] = af.replace(":resampler=soxr", "")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg continuous assembly failed: {result.stderr[-800:]}")
    finally:
        os.unlink(list_path)
        if lead_path is not None:
            lead_path.unlink(missing_ok=True)
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
