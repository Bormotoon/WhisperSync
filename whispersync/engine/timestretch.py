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

    ``assemble_continuous`` folds this chain into its own filter graph so
    ducking happens in the same single ffmpeg pass as assembly, instead of a
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
    same ffmpeg invocation (see ``duck_filter_chain``) instead of a separate
    decode/encode pass over an intermediate file — one encode instead of two.
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


def mix_clips_on_timeline(
    clips: list[tuple[Path, float]],
    total_duration: float,
    sample_rate: int,
    output_path: Path,
    channels: int = _DEFAULT_CHANNELS,
    codec: str = _DEFAULT_CODEC,
) -> Path:
    """Mix already-rendered audio clips (path, timeline-offset-seconds) onto one
    continuous WAV spanning ``total_duration``, each delayed to its own offset
    over a silent bed — a single master track for users without an NLE.

    Every clip is resampled/upmixed to ``sample_rate``/``channels`` via
    ``aresample`` + ``aformat`` before mixing (they may come from different
    recorders with different native formats); ``amix`` sums without
    normalizing volume (``normalize=0``) so overlapping clips don't get
    quieter than a non-overlapping single clip would.
    """
    if not clips or total_duration <= 0:
        return generate_silence(output_path, max(total_duration, 0.0), sample_rate, channels, codec)

    layout = "mono" if channels == 1 else "stereo" if channels == 2 else f"{channels}c"
    inputs: list[str] = []
    filter_parts: list[str] = []
    mix_labels: list[str] = []
    for i, (path, offset) in enumerate(clips):
        inputs += ["-i", str(path)]
        delay_ms = max(0, round(offset * 1000))
        delay_arg = "|".join([str(delay_ms)] * channels) if channels > 1 else str(delay_ms)
        filter_parts.append(
            f"[{i}:a]aresample={sample_rate}:resampler=soxr,"
            f"aformat=sample_fmts=fltp:channel_layouts={layout},"
            f"adelay={delay_arg}:all=1[a{i}]"
        )
        mix_labels.append(f"[a{i}]")
    filter_parts.append(
        f"{''.join(mix_labels)}amix=inputs={len(clips)}:duration=longest:normalize=0,"
        f"apad,atrim=0:{total_duration:.6f},asetpts=PTS-STARTPTS[out]"
    )
    filter_complex = ";".join(filter_parts)

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
        str(channels),
        "-acodec",
        codec,
        str(output_path),
    ]
    logger.info("Mixing %d clip(s) onto master timeline -> %s", len(clips), output_path)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0 and ":resampler=soxr" in filter_complex:
        cmd[cmd.index("-filter_complex") + 1] = filter_complex.replace(":resampler=soxr", "")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg master-timeline mix failed: {result.stderr[-800:]}")
    return output_path
