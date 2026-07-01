"""End-to-end orchestration pipeline with progress signals.

The recorder is the continuous reference. Each camera clip is transcribed and
aligned to the recorder independently, so a clip's position on the master
timeline comes from matched timecodes — clips need not be contiguous.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from whispersync.config import WhisperSyncConfig
from whispersync.engine.acoustic import refine_piece_boundaries
from whispersync.engine.export import generate_fcpxml, validate_fcpxml
from whispersync.engine.matcher import align
from whispersync.engine.media import MediaInfo, extract_audio_to_wav, probe
from whispersync.engine.naming import natural_key
from whispersync.engine.strategies import get_strategy
from whispersync.engine.timestretch import (
    apply_pause_ducking,
    assemble_continuous,
    render_piece,
)
from whispersync.engine.transcriber import WhisperEngine
from whispersync.engine.transcript_export import save_transcript
from whispersync.models import AlignmentMap, MediaClip, SubClip, SyncResult, Transcript

logger = logging.getLogger(__name__)


@dataclass
class PipelineProgress:
    stage: str
    progress: float = 0.0
    message: str = ""
    # Optional timeline snapshot (list of clip dicts) for the GUI timeline view.
    clips: list[dict[str, Any]] | None = None


ProgressCallback = Callable[[PipelineProgress], None]


def make_timeline_entries(
    cameras: list[CameraGroup],
    clip_camera: list[int],
    video_clips: list[MediaClip],
    video_status: list[str],
    audio_clips: list[MediaClip],
    audio_speed: list[float],
    audio_track: list[str],
    audio_status: list[str],
) -> list[dict[str, Any]]:
    """Build a flat list of timeline clip dicts for the GUI.

    Video clips occupy one row per camera (row = camera index); audio lanes are
    stacked below in first-seen order. ``speed`` is the playback rate applied to
    the recorder audio (1.0 = untouched); ``status`` is pending/working/done.
    """
    entries: list[dict[str, Any]] = []
    for clip, ci, status in zip(video_clips, clip_camera, video_status, strict=True):
        entries.append(
            {
                "track": cameras[ci].name,
                "row": ci,
                "kind": "video",
                "lane": clip.lane,
                "offset": clip.offset,
                "duration": clip.duration,
                "in_point": clip.in_point,
                "speed": 1.0,
                "status": status,
                "name": clip.path.stem,
            }
        )

    audio_rows: dict[str, int] = {}
    next_row = len(cameras)
    for label in audio_track:
        if label not in audio_rows:
            audio_rows[label] = next_row
            next_row += 1

    for clip, speed, label, status in zip(
        audio_clips, audio_speed, audio_track, audio_status, strict=True
    ):
        entries.append(
            {
                "track": label,
                "row": audio_rows[label],
                "kind": "audio",
                "lane": clip.lane,
                "offset": clip.offset,
                "duration": clip.duration,
                "in_point": clip.in_point,
                "speed": speed,
                "status": status,
                "name": clip.path.stem,
            }
        )
    return entries


@dataclass
class CameraGroup:
    name: str
    lane: int
    infos: list[MediaInfo]
    clips: list[MediaClip]


def scan_cameras(video_dir: Path, config: WhisperSyncConfig) -> list[CameraGroup]:
    """Discover cameras and probe their clips.

    If ``video_dir`` contains sub-folders with video files, each sub-folder is
    treated as a separate camera placed on its own positive lane (1, 2, 3, …).
    Otherwise the flat folder is a single camera on lane 1. Clip offsets are
    left at 0 — real timeline positions come later from matched timecodes.
    """
    exts = tuple(config.video_exts)

    def videos_in(d: Path) -> list[Path]:
        return sorted(
            (p for p in d.iterdir() if p.suffix.lower() in exts),
            key=lambda p: natural_key(p.name),
        )

    subdirs = sorted(
        (d for d in video_dir.iterdir() if d.is_dir() and videos_in(d)),
        key=lambda d: natural_key(d.name),
    )
    sources: list[tuple[str, list[Path]]]
    if subdirs:
        sources = [(d.name, videos_in(d)) for d in subdirs]
    else:
        sources = [("camera", videos_in(video_dir))]

    if not any(paths for _, paths in sources):
        raise RuntimeError(f"No video files found in {video_dir}")

    cameras: list[CameraGroup] = []
    for cam_index, (name, paths) in enumerate(sources):
        infos: list[MediaInfo] = []
        clips: list[MediaClip] = []
        lane = cam_index + 1
        for path in paths:
            info = probe(path)
            infos.append(info)
            clips.append(
                MediaClip(
                    path=path,
                    kind="video",
                    offset=0.0,
                    in_point=0.0,
                    duration=info.duration,
                    lane=lane,
                    role="Video",
                )
            )
        cameras.append(CameraGroup(name=name, lane=lane, infos=infos, clips=clips))

    return cameras


def scan_video_clips(
    video_dir: Path, config: WhisperSyncConfig
) -> tuple[list[MediaInfo], list[MediaClip]]:
    """Flat view over all cameras (used by the dry-run path)."""
    cameras = scan_cameras(video_dir, config)
    infos: list[MediaInfo] = []
    clips: list[MediaClip] = []
    for cam in cameras:
        infos.extend(cam.infos)
        clips.extend(cam.clips)
    return infos, clips


def compute_master_offsets(
    alignments: list[AlignmentMap | None], durations: list[float]
) -> tuple[list[float], list[int]]:
    """Place each clip on the master timeline (recorder seconds) from its
    matched recorder start time ``-offset/k``, anchored so the earliest clip
    sits at 0. Clips that could not be aligned fall back to following the
    previous clip and are reported by index.

    Returns (offsets, unaligned_indices).
    """
    rec_starts: list[float | None] = []
    for am in alignments:
        if am is not None and am.k != 0:
            rec_starts.append(-am.offset / am.k)
        else:
            rec_starts.append(None)

    aligned = [r for r in rec_starts if r is not None]
    ref = min(aligned) if aligned else 0.0

    offsets: list[float] = []
    unaligned: list[int] = []
    prev_end = 0.0
    for i, (rs, dur) in enumerate(zip(rec_starts, durations, strict=True)):
        if rs is not None:
            off = rs - ref
        else:
            off = prev_end
            unaligned.append(i)
        offsets.append(off)
        prev_end = off + dur
    return offsets, unaligned


def preliminary_offsets(video_clips: list[MediaClip], clip_camera: list[int]) -> list[float]:
    """Rough timeline positions before alignment: lay each camera's clips
    end-to-end in (natural) name order. Used only to populate the GUI timeline
    while transcription runs; real positions come from matched timecodes."""
    running: dict[int, float] = {}
    offsets: list[float] = []
    for clip, cam in zip(video_clips, clip_camera, strict=True):
        start = running.get(cam, 0.0)
        offsets.append(start)
        running[cam] = start + clip.duration
    return offsets


def sequence_order_warnings(
    video_clips: list[MediaClip],
    clip_camera: list[int],
    aligned: list[bool],
) -> list[str]:
    """Flag clips whose matched timeline order contradicts their filename order
    within a camera — a strong hint of a bad alignment for that clip."""
    warnings: list[str] = []
    by_cam: dict[int, list[int]] = {}
    for i, cam in enumerate(clip_camera):
        by_cam.setdefault(cam, []).append(i)
    for indices in by_cam.values():
        prev_off: float | None = None
        prev_name = ""
        for i in indices:  # indices are already in natural name order
            if not aligned[i]:
                continue
            off = video_clips[i].offset
            if prev_off is not None and off < prev_off - 1e-6:
                warnings.append(
                    f"{video_clips[i].path.name}: placed before {prev_name} despite later "
                    "filename — possible misalignment"
                )
            prev_off = off
            prev_name = video_clips[i].path.name
    return warnings


def clip_pieces(
    am: AlignmentMap,
    clip_duration: float,
    rec_duration: float,
    strategy_id: int,
    config: WhisperSyncConfig,
) -> tuple[float, list[tuple[float, float, float]]]:
    """Contiguous recorder pieces that tile a camera clip, for a continuous warp.

    Returns ``(lead_silence, pieces)`` where each piece is
    ``(rec_start, rec_in_duration, atempo_factor)`` and pieces are in playback
    order with no gaps between them — the recorder span for the clip is simply
    time-stretched (globally or piecewise between sync points) so its speech
    lands under the picture. ``lead_silence`` is the silence (seconds) before the
    first piece, non-zero only when the recorder does not reach the clip start.

    Strategy controls the breakpoint density: 1 = one global stretch, 2 = a piece
    per anchor (tightest), 3/4 = a piece per phrase (smoother, fewer seams).
    """
    k = am.k or 1.0

    def l2r(t_local: float) -> float:
        return (t_local - am.offset) / k

    def r2l(t_rec: float) -> float:
        return am.offset + k * t_rec

    rec0 = min(max(l2r(0.0), 0.0), rec_duration)
    rec1 = min(max(l2r(clip_duration), 0.0), rec_duration)
    if rec1 - rec0 <= 1e-3:
        return 0.0, []

    # Interior breakpoints come from the REAL matched word times — each is
    # (recorder_time, local_clip_time), where local time is the anchor's camera
    # time. Warping between these tracks the actual (non-linear) drift instead of
    # a single global slope.
    pts = sorted(
        (a.rec_time, a.cam_time)
        for a in am.anchors
        if rec0 < a.rec_time < rec1 and 0.0 <= a.cam_time <= clip_duration
    )

    # Strategy 1 (or no usable anchors): one global stretch across the clip.
    if strategy_id == 1 or not pts:
        out_dur = r2l(rec1) - r2l(rec0)
        lead = max(0.0, r2l(rec0))
        if out_dur <= 1e-3:
            return lead, []
        return lead, [(rec0, rec1 - rec0, (rec1 - rec0) / out_dur)]

    # Strategies 3 & 4: fewer seams — thin anchors to roughly one per phrase.
    if strategy_id in (3, 4):
        spacing = max(config.phrase_gap_threshold, 1.0)
        thinned: list[tuple[float, float]] = []
        for rt, ct in pts:
            if not thinned or rt - thinned[-1][0] >= spacing:
                thinned.append((rt, ct))
        pts = thinned

    # Clip edges use the global line; interior uses matched word times. Keep only
    # strictly-increasing (rec, local) breakpoints so every piece is sane.
    raw_bps = [(rec0, r2l(rec0)), *pts, (rec1, r2l(rec1))]
    bps: list[tuple[float, float]] = []
    for rt, lt in raw_bps:
        if not bps or (rt > bps[-1][0] + 1e-3 and lt > bps[-1][1] + 1e-3):
            bps.append((rt, lt))

    pieces: list[tuple[float, float, float]] = []
    for i in range(len(bps) - 1):
        (ra, la), (rb, lb) = bps[i], bps[i + 1]
        in_dur = rb - ra
        out_dur = lb - la
        if in_dur <= 1e-4 or out_dur <= 1e-4:
            continue
        # Safety clamp: a stray anchor can never blow the stretch past atempo's range.
        factor = max(0.5, min(2.0, in_dur / out_dur))
        pieces.append((ra, in_dur, factor))

    lead = max(0.0, bps[0][1])
    if config.smooth_tempo and len(pieces) > 1:
        pieces = _smooth_piece_tempo(pieces, config.max_tempo_jump)
    return lead, pieces


def _smooth_piece_tempo(
    pieces: list[tuple[float, float, float]], max_jump: float
) -> list[tuple[float, float, float]]:
    """Limit the tempo (atempo factor) change between consecutive pieces.

    A large factor jump at a seam is heard as a stutter/tempo-break — especially
    when the seam falls mid-word ("подготовил" → "подга-га-товил"). We iteratively
    pull neighbouring factors toward each other until no adjacent pair differs by
    more than ``max_jump``. Each piece keeps its recorder start and IN duration; only
    its factor (hence output length) changes, so the timeline positions are barely
    affected while the audible tempo steps disappear.
    """
    in_durs = [p[1] for p in pieces]
    factors = [p[2] for p in pieces]
    orig_out = sum(d / f for d, f in zip(in_durs, factors, strict=True))

    for _ in range(100):
        changed = False
        for i in range(1, len(factors)):
            diff = factors[i] - factors[i - 1]
            if abs(diff) > max_jump:
                excess = (abs(diff) - max_jump) / 2.0
                sign = 1.0 if diff > 0 else -1.0
                factors[i] -= sign * excess
                factors[i - 1] += sign * excess
                changed = True
        if not changed:
            break

    # Smoothing changes each piece's output length; rescale all factors by a single
    # constant so the TOTAL output duration is preserved (else speech would drift).
    # A uniform scale keeps the smoothed shape and can't reintroduce a big jump.
    new_out = sum(d / f for d, f in zip(in_durs, factors, strict=True))
    scale = new_out / orig_out if orig_out > 0 else 1.0
    return [
        (ra, in_dur, max(0.5, min(2.0, f * scale)))
        for (ra, in_dur, _old), f in zip(pieces, factors, strict=True)
    ]


def pause_spans_local(
    am: AlignmentMap, clip_duration: float, gap_threshold: float, min_pause: float
) -> list[tuple[float, float]]:
    """Inter-phrase pause spans in LOCAL (camera) time for the rendered clip.

    Anchors are sorted by camera time; a gap between consecutive anchors larger than
    ``gap_threshold`` is a pause. Spans shorter than ``min_pause`` are dropped (not
    worth ducking). Also covers the head/tail of the clip before the first / after
    the last anchor. Returned spans are clamped to ``[0, clip_duration]`` and sorted.
    """
    cam_times = sorted(a.cam_time for a in am.anchors)
    if not cam_times:
        return []
    spans: list[tuple[float, float]] = []
    if cam_times[0] > min_pause:
        spans.append((0.0, cam_times[0]))
    for t0, t1 in zip(cam_times, cam_times[1:], strict=False):
        if t1 - t0 > gap_threshold:
            spans.append((t0, t1))
    if clip_duration - cam_times[-1] > min_pause:
        spans.append((cam_times[-1], clip_duration))
    out: list[tuple[float, float]] = []
    for a, b in spans:
        a = max(0.0, a)
        b = min(clip_duration, b)
        if b - a >= min_pause:
            out.append((a, b))
    return out


def resolve_workers(requested: int) -> int:
    """Number of parallel render processes: ``requested`` if >0, else os.cpu_count()."""
    if requested and requested > 0:
        return requested
    return max(1, os.cpu_count() or 1)


def _pool_context() -> Any:
    """A 'fork' multiprocessing context for the render pools, with a safe fallback.

    fork is ideal here: children inherit the already-imported modules (no costly
    re-import per task, which matters for many short ffmpeg calls) and it avoids the
    spawn/forkserver re-execution of __main__. On platforms without fork (Windows /
    macOS-spawn-only) we fall back to the default context, which still works because
    the worker functions are module-level and picklable.
    """
    try:
        return mp.get_context("fork")
    except (ValueError, RuntimeError):  # pragma: no cover - platform dependent
        return mp.get_context()


def render_pieces(
    pieces: list[tuple[float, float, float]],
    rec_path: Path,
    tmp_dir: Path,
    fade_ms: int,
    workers: int,
) -> list[Path]:
    """Render every piece to its own indexed WAV, in parallel across ``workers``
    processes, and return the paths in piece order.

    Each piece is an independent CPU-bound ffmpeg call writing a distinct
    ``segment_{index}.wav`` (ffmpeg has no GPU audio filters, so spreading the work
    across cores is the real speed-up). The result is identical to a sequential run:
    files are keyed by index and re-sorted before returning.
    """
    if not pieces:
        return []
    if workers <= 1 or len(pieces) == 1:
        return [
            render_piece(rec_path, tmp_dir, rs, rd, fac, k, fade_ms)
            for k, (rs, rd, fac) in enumerate(pieces)
        ]
    results: dict[int, Path] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=_pool_context()) as pool:
        futures = {
            pool.submit(render_piece, rec_path, tmp_dir, rs, rd, fac, k, fade_ms): k
            for k, (rs, rd, fac) in enumerate(pieces)
        }
        for fut in futures:
            k = futures[fut]
            results[k] = fut.result()  # re-raises any worker exception here
    return [results[k] for k in range(len(pieces))]


def build_subclips(
    pieces: list[tuple[float, float, float]],
    seg_paths: list[Path],
    lead: float,
) -> list[SubClip]:
    """Local (compound-clip-timeline) offset/duration for each rendered piece.

    Mirrors the placement ``assemble_continuous`` would use — pieces laid end-to-end,
    starting after ``lead`` seconds of silence — but instead of concatenating them
    into one WAV, each piece stays its own file/clip. It lands exactly where it would
    in a contiguous assembly, while remaining individually editable (crossfade/nudge)
    once wrapped in a compound clip. ``apply_atempo_segment``/``extract_segment``
    force each rendered piece to exactly ``in_dur / factor`` seconds, so this cursor
    math matches the actual rendered file lengths.
    """
    subclips: list[SubClip] = []
    cursor = lead
    for (rec_start, in_dur, factor), path in zip(pieces, seg_paths, strict=True):
        out_dur = in_dur / factor
        subclips.append(SubClip(path=path, offset=cursor, in_point=rec_start, duration=out_dur))
        cursor += out_dur
    return subclips


def _timeline_end(clips: list[MediaClip]) -> float:
    return max((c.offset + c.duration for c in clips), default=0.0)


def _anchor_count(am: AlignmentMap | None) -> int:
    return len(am.anchors) if am is not None else 0


def run_pipeline(
    config: WhisperSyncConfig,
    video_dir: Path,
    audio_files: list[Path],
    strategy_id: int,
    output_path: Path,
    progress_callback: ProgressCallback | None = None,
) -> SyncResult:
    def _notify(
        stage: str,
        progress: float = 0.0,
        message: str = "",
        clips: list[dict[str, Any]] | None = None,
    ) -> None:
        if progress_callback is not None:
            progress_callback(
                PipelineProgress(stage=stage, progress=progress, message=message, clips=clips)
            )

    if not audio_files:
        raise ValueError("At least one recorder audio file is required.")

    engine: WhisperEngine | None = None
    cleanup_paths: list[Path] = []
    warnings: list[str] = []

    try:
        # --- scanning (group clips by camera) ---
        _notify("scanning", 0.0, "Scanning video directory...")
        cameras = scan_cameras(video_dir, config)
        video_clips: list[MediaClip] = []
        video_infos: list[MediaInfo] = []
        clip_camera: list[int] = []  # camera index per clip
        for ci, cam in enumerate(cameras):
            video_infos.extend(cam.infos)
            for clip in cam.clips:
                video_clips.append(clip)
                clip_camera.append(ci)
        cam_names = ", ".join(c.name for c in cameras)
        _notify(
            "scanning",
            1.0,
            f"Found {len(video_clips)} clip(s) across {len(cameras)} camera(s): {cam_names}",
        )

        # Preliminary layout from filenames so the timeline is populated while
        # transcription runs; real positions replace it after alignment.
        video_status = ["pending"] * len(video_clips)
        prelim = preliminary_offsets(video_clips, clip_camera)
        for clip, off in zip(video_clips, prelim, strict=True):
            clip.offset = off

        def _video_snapshot() -> list[dict[str, Any]]:
            return make_timeline_entries(
                cameras, clip_camera, video_clips, video_status, [], [], [], []
            )

        _notify("scanning", 1.0, "Preliminary layout", clips=_video_snapshot())

        # --- transcribe every recorder once ---
        engine = WhisperEngine(config)
        transcripts_dir = output_path.parent / "transcripts"

        def _save_tx(transcript: Transcript, stem: str, audio_path: Path) -> None:
            if not config.save_transcripts:
                return
            save_transcript(
                transcript,
                transcripts_dir,
                stem,
                audio_path=audio_path,
                model=config.model,
                device=engine.device if engine else config.device,
                compute_type=engine.compute_type if engine else config.compute_type,
                mode=config.transcribe_mode,
            )

        rec_infos = [probe(p) for p in audio_files]
        rec_transcripts = []
        for ri, rp in enumerate(audio_files):
            _notify("transcribing_recorder", ri / len(audio_files), f"Recorder: {rp.name}")
            rt = engine.transcribe(rp, lambda p: _notify("transcribing_recorder", p))
            rec_transcripts.append(rt)
            _save_tx(rt, rp.stem, rp)

        # --- align each clip against each recorder ---
        # aligns[clip_idx][rec_idx] = AlignmentMap | None
        aligns: list[list[AlignmentMap | None]] = []
        n = len(video_clips)
        for idx, clip in enumerate(video_clips):
            video_status[idx] = "working"
            _notify(
                "transcribing_camera",
                idx / max(n, 1),
                f"Transcribing camera clip {idx + 1}/{n}: {clip.path.name}",
                clips=_video_snapshot(),
            )
            clip_audio = extract_audio_to_wav(clip.path)
            cleanup_paths.append(clip_audio)
            clip_transcript = engine.transcribe(clip_audio)
            cam_name = cameras[clip_camera[idx]].name
            clip_stem = clip.path.stem if len(cameras) == 1 else f"{cam_name}_{clip.path.stem}"
            _save_tx(clip_transcript, clip_stem, clip.path)
            row: list[AlignmentMap | None] = []
            for rt in rec_transcripts:
                try:
                    row.append(align(clip_transcript, rt, config))
                except ValueError:
                    row.append(None)
            video_status[idx] = "pending"
            if all(a is None for a in row):
                warnings.append(f"{clip.path.name}: not aligned to any recorder")
            aligns.append(row)

        # --- pick the primary recorder (most total anchors) for placement ---
        rec_anchor_total = [
            sum(_anchor_count(aligns[ci][ri]) for ci in range(n)) for ri in range(len(audio_files))
        ]
        if max(rec_anchor_total, default=0) == 0:
            raise RuntimeError("No camera clip could be aligned to any recorder audio.")
        primary = max(range(len(audio_files)), key=lambda ri: rec_anchor_total[ri])
        if len(audio_files) > 1:
            warnings.append(f"Timeline placement uses recorder '{audio_files[primary].name}'")

        # --- place clips on the master timeline from primary-recorder timecodes ---
        _notify("aligning", 1.0, "Placing clips on timeline...")
        primary_aligns = [aligns[ci][primary] for ci in range(n)]
        durations = [c.duration for c in video_clips]
        offsets, unaligned = compute_master_offsets(primary_aligns, durations)
        for clip, off in zip(video_clips, offsets, strict=True):
            clip.offset = off
        for i in unaligned:
            warnings.append(f"{video_clips[i].path.name}: placed by order (no primary anchors)")

        aligned_primary = [primary_aligns[i] is not None for i in range(n)]
        for i in range(n):
            video_status[i] = "done" if aligned_primary[i] else "pending"
        warnings.extend(sequence_order_warnings(video_clips, clip_camera, aligned_primary))

        # --- choose which camera the synced audio is derived from ---
        anchors_per_cam: dict[int, int] = {}
        for ci, row in zip(clip_camera, aligns, strict=True):
            best_in_row = max((_anchor_count(a) for a in row), default=0)
            anchors_per_cam[ci] = anchors_per_cam.get(ci, 0) + best_in_row
        if config.audio_source_camera:
            audio_ci = next(
                (i for i, c in enumerate(cameras) if c.name == config.audio_source_camera),
                max(anchors_per_cam, key=lambda i: anchors_per_cam[i]),
            )
        else:
            audio_ci = max(anchors_per_cam, key=lambda i: anchors_per_cam[i])
        if len(cameras) > 1:
            warnings.append(f"Audio synced from camera '{cameras[audio_ci].name}'")

        # --- plan + render synced audio (one WAV per audio-source video clip) ---
        # Video files are referenced untouched. For each one we render a single
        # recorder-audio WAV of identical length, with speech placed at its synced
        # position and silence filling the gaps — so the FCPXML carries ~2 clips
        # per video instead of thousands of segment clips.
        _notify("planning", 0.0, "Planning sync strategy...")
        strategy = get_strategy(strategy_id)
        audio_synced_dir = output_path.parent / "audio_synced"
        audio_synced_dir.mkdir(parents=True, exist_ok=True)

        out_sr: int
        if config.timebase_source == "recorder" and rec_infos[primary].audio_sample_rate:
            out_sr = int(rec_infos[primary].audio_sample_rate or 48000)
        elif video_infos and video_infos[0].audio_sample_rate:
            out_sr = int(video_infos[0].audio_sample_rate or 48000)
        else:
            out_sr = 48000

        audio_clips: list[MediaClip] = []
        audio_speed: list[float] = []
        audio_track: list[str] = []

        @dataclass
        class RenderJob:
            clip_idx: int
            rec_path: Path
            lead: float
            pieces: list[tuple[float, float, float]]
            duration: float
            cam_audio: Path | None  # camera clip audio (mono 16k), for Boundary Flex
            rec_duration: float
            pauses: list[tuple[float, float]]  # local-time pause spans, for ducking

        render_jobs: list[RenderJob] = []

        def _add_job(vclip: MediaClip, am: AlignmentMap, ri: int, lane: int) -> None:
            lead, pieces = clip_pieces(
                am, vclip.duration, rec_infos[ri].duration, strategy_id, config
            )
            if not pieces:
                return
            label = f"Audio: {audio_files[ri].stem}" if config.recorder_mode == "all" else "Audio"
            # Friendly name tied to the camera clip (e.g. "DJI_0830_voice"); with
            # several recorders, disambiguate by recorder stem. Role = Dialogue so
            # FCPX colours/groups the synced voice as dialogue.
            voice_name = f"{vclip.path.stem}_voice"
            if config.recorder_mode == "all":
                voice_name = f"{vclip.path.stem}_{audio_files[ri].stem}_voice"
            audio_clips.append(
                MediaClip(
                    path=audio_files[ri],
                    kind="audio",
                    offset=vclip.offset,
                    in_point=0.0,
                    duration=vclip.duration,
                    lane=lane,
                    display_name=voice_name,
                    role="Dialogue",
                )
            )
            audio_speed.append(1.0 / am.k if am.k else 1.0)
            audio_track.append(label)

            # Camera clip audio is only needed for Boundary Flex; extract once here.
            cam_audio: Path | None = None
            if config.boundary_flex:
                cam_audio = extract_audio_to_wav(vclip.path)
                cleanup_paths.append(cam_audio)
            pauses = (
                pause_spans_local(
                    am, vclip.duration, config.phrase_gap_threshold, config.pause_duck_min_pause_s
                )
                if config.pause_duck_enabled
                else []
            )
            render_jobs.append(
                RenderJob(
                    clip_idx=len(audio_clips) - 1,
                    rec_path=audio_files[ri],
                    lead=lead,
                    pieces=pieces,
                    duration=vclip.duration,
                    cam_audio=cam_audio,
                    rec_duration=rec_infos[ri].duration,
                    pauses=pauses,
                )
            )

        for ci in range(n):
            if clip_camera[ci] != audio_ci:
                continue
            vclip = video_clips[ci]
            row = aligns[ci]
            if config.recorder_mode == "all":
                for ri, am in enumerate(row):
                    if am is not None:
                        _add_job(vclip, am, ri, lane=-(ri + 1))
            else:  # "best": one lane, strongest recorder per clip
                candidates = [(ri, am) for ri, am in enumerate(row) if am is not None]
                if candidates:
                    best_ri, best_am = max(candidates, key=lambda t: len(t[1].anchors))
                    _add_job(vclip, best_am, best_ri, lane=-1)

        audio_status = ["pending"] * len(audio_clips)

        def _timeline_snapshot() -> list[dict[str, Any]]:
            return make_timeline_entries(
                cameras,
                clip_camera,
                video_clips,
                video_status,
                audio_clips,
                audio_speed,
                audio_track,
                audio_status,
            )

        # Show the full planned layout (video placed, audio pending) up front.
        _notify("planning", 1.0, "Timeline planned", clips=_timeline_snapshot())

        from whispersync.models import SyncPlan

        all_clips = video_clips + audio_clips
        plan = SyncPlan(
            strategy_id=strategy_id,
            clips=all_clips,
            audio_ops=[],
            total_duration=_timeline_end(all_clips),
        )
        _notify("planning", 1.0, f"Strategy {strategy_id}: {strategy.name}")

        # --- render synced audio per clip: one continuous WAV, or a compound clip
        # of separately-positioned pieces (config.audio_compound) ---
        _notify("processing", 0.0, "Rendering synced audio...")
        # Small length-preserving fades declick the seams between stretched pieces.
        fade_ms = config.crossfade_ms if config.crossfade_enabled else 0
        workers = resolve_workers(config.render_workers)
        n_jobs = max(len(render_jobs), 1)
        if config.audio_compound and config.pause_duck_enabled:
            # Pause ducking attenuates a continuous assembly; in compound mode the
            # gaps between pieces already carry no audio at all, so it's a no-op.
            warnings.append(
                "Pause ducking has no effect in compound mode (gaps between pieces "
                "are already silent)."
            )
        for j, job in enumerate(render_jobs):
            clip_idx = job.clip_idx
            audio_status[clip_idx] = "working"
            _notify(
                "processing",
                j / n_jobs,
                f"Rendering synced audio {j + 1}/{n_jobs}",
                clips=_timeline_snapshot(),
            )
            # Keep scratch segments on the OUTPUT volume (next to the result), not
            # the system /tmp — /tmp may be small or on a full disk.
            with tempfile.TemporaryDirectory(prefix="whispersync_seg_", dir=audio_synced_dir) as td:
                tdp = Path(td)
                pieces = job.pieces
                # Boundary Flex: acoustically nudge each piece's recorder start so
                # speech lands under the picture to sub-frame accuracy.
                if config.boundary_flex and job.cam_audio is not None:
                    pieces = refine_piece_boundaries(
                        pieces,
                        job.lead,
                        job.cam_audio,
                        job.rec_path,
                        job.duration,
                        job.rec_duration,
                        config,
                        tmp_dir=tdp,
                        workers=workers,
                    )
                # Name the output after the clip (e.g. "DJI_0832_voice.wav") so the
                # media file matches its timeline clip; fall back to an index.
                voice_name = audio_clips[clip_idx].display_name or f"synced_{clip_idx:03d}"
                if config.audio_compound:
                    # Compound mode: each piece is rendered exactly as before, but kept
                    # as its own persistent file (in a per-clip folder, not the scratch
                    # tempdir) at the position it would occupy in a contiguous
                    # assembly. No assemble_continuous, no forced seam handling — the
                    # editor crossfades/nudges the pieces by hand in Final Cut.
                    pieces_dir = audio_synced_dir / f"{voice_name}_pieces"
                    pieces_dir.mkdir(parents=True, exist_ok=True)
                    seg_paths = render_pieces(pieces, job.rec_path, pieces_dir, fade_ms, workers)
                    audio_clips[clip_idx].subclips = build_subclips(pieces, seg_paths, job.lead)
                    # Nominal path (no single backing file — audio lives in .subclips)
                    # kept name-consistent with the non-compound output for the GUI.
                    audio_clips[clip_idx].path = audio_synced_dir / f"{voice_name}.wav"
                else:
                    # Each piece is an independent ffmpeg cut/stretch; render them across
                    # the CPU pool (ffmpeg has no GPU audio filters). Order is preserved.
                    seg_paths = render_pieces(pieces, job.rec_path, tdp, fade_ms, workers)
                    out = audio_synced_dir / f"{voice_name}.wav"
                    # Duck inter-phrase pauses (if enabled) on an intermediate file, then
                    # the ducked result becomes the final output.
                    if config.pause_duck_enabled and job.pauses:
                        raw = tdp / f"raw_{clip_idx:03d}.wav"
                        assemble_continuous(seg_paths, job.lead, job.duration, out_sr, raw)
                        apply_pause_ducking(
                            raw,
                            out,
                            job.pauses,
                            config.pause_duck_db,
                            config.pause_duck_fade_ms,
                            out_sr,
                        )
                    else:
                        assemble_continuous(seg_paths, job.lead, job.duration, out_sr, out)
                    audio_clips[clip_idx].path = out
            audio_clips[clip_idx].in_point = 0.0
            audio_status[clip_idx] = "done"
            _notify(
                "processing",
                (j + 1) / n_jobs,
                f"Rendered {j + 1}/{n_jobs}",
                clips=_timeline_snapshot(),
            )

        # --- extract voice-free camera ambience onto its own lane (optional) ---
        # Strip the camera's own voice (which would double/echo the clean synced
        # voice) but keep the room tone, on a lane below the synced audio.
        if config.ambience_track:
            from whispersync.engine import separation

            repo_root = Path(__file__).resolve().parents[2]
            if not separation.is_available(repo_root):
                warnings.append(
                    "Ambience track requested but the separation environment "
                    "(.sep-venv) is missing — skipped."
                )
            else:
                ambience_dir = output_path.parent / "ambience"
                model_dir = repo_root / "models" / "separator"
                ambient_lane = min((c.lane for c in audio_clips), default=-1) - 1
                amb_clips: list[MediaClip] = []
                src_clips = [video_clips[ci] for ci in range(n) if clip_camera[ci] == audio_ci]
                for k, vclip in enumerate(src_clips):
                    _notify(
                        "processing",
                        k / max(len(src_clips), 1),
                        f"Extracting ambience {k + 1}/{len(src_clips)}: {vclip.path.name}",
                    )
                    cam_wav = extract_audio_to_wav(vclip.path, sample_rate=out_sr, mono=False)
                    cleanup_paths.append(cam_wav)
                    try:
                        amb = separation.extract_ambience(
                            cam_wav,
                            ambience_dir,
                            repo_root,
                            config.ambience_model,
                            model_dir=model_dir if model_dir.exists() else None,
                        )
                    except (RuntimeError, OSError) as e:
                        warnings.append(f"{vclip.path.name}: ambience extraction failed ({e})")
                        continue
                    # Rename the separator's "…_(Instrumental)_<model>.wav" to a clean
                    # clip-matched name ("DJI_0832_ambience.wav").
                    amb_final = ambience_dir / f"{vclip.path.stem}_ambience.wav"
                    with contextlib.suppress(OSError):
                        amb.replace(amb_final)
                        amb = amb_final
                    amb_clips.append(
                        MediaClip(
                            path=amb,
                            kind="audio",
                            offset=vclip.offset,
                            in_point=0.0,
                            duration=vclip.duration,
                            lane=ambient_lane,
                            display_name=f"{vclip.path.stem}_ambience",
                            role="Effects",
                        )
                    )
                plan.clips.extend(amb_clips)
                plan.total_duration = _timeline_end(plan.clips)

        # --- export ---
        _notify("exporting", 0.0, "Generating FCPXML...")
        generate_fcpxml(
            plan,
            video_infos,
            output_path,
            config.fcpxml_version,
            output_path.stem,
            audio_sample_rate=out_sr,  # matches the rendered synced WAVs
        )
        # Safety net: catch a broken export (e.g. a DTD-invalid attribute) here,
        # with a clear warning, instead of the user only finding out when Final
        # Cut's import dialog rejects the whole file.
        if not validate_fcpxml(output_path):
            warnings.append(
                "Generated FCPXML failed internal validation — Final Cut Pro may "
                "refuse to import it. Please report this as a bug."
            )

        # --- collect quality warnings from the best alignment overall ---
        all_aligned = [a for row in aligns for a in row if a is not None]
        best = max(all_aligned, key=lambda a: len(a.anchors))
        if best.residual_ms > 40:
            warnings.append(f"High residual alignment error: {best.residual_ms:.1f} ms")

        # count the best recorder per clip for a representative anchor total
        anchors_used = sum(max((_anchor_count(a) for a in row), default=0) for row in aligns)

        _notify("done", 1.0, "Pipeline complete")
        return SyncResult(
            fcpxml_path=output_path,
            alignment=best,
            plan=plan,
            anchors_used=anchors_used,
            warnings=warnings,
        )

    except InterruptedError:
        logger.info("Pipeline cancelled by user")
        raise
    except Exception:
        logger.exception("Pipeline failed")
        raise
    finally:
        if engine is not None:
            engine.unload()
        for p in cleanup_paths:
            with contextlib.suppress(OSError):
                os.unlink(p)
