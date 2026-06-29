"""End-to-end orchestration pipeline with progress signals.

The recorder is the continuous reference. Each camera clip is transcribed and
aligned to the recorder independently, so a clip's position on the master
timeline comes from matched timecodes — clips need not be contiguous.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bormosync.config import BormoSyncConfig
from bormosync.engine.export import generate_fcpxml
from bormosync.engine.matcher import align
from bormosync.engine.media import MediaInfo, extract_audio_to_wav, probe
from bormosync.engine.naming import natural_key
from bormosync.engine.strategies import get_strategy
from bormosync.engine.timestretch import (
    apply_atempo_segment,
    assemble_continuous,
    extract_segment,
)
from bormosync.engine.transcriber import WhisperEngine
from bormosync.engine.transcript_export import save_transcript
from bormosync.models import AlignmentMap, MediaClip, SyncResult, Transcript

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


def scan_cameras(video_dir: Path, config: BormoSyncConfig) -> list[CameraGroup]:
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
                )
            )
        cameras.append(CameraGroup(name=name, lane=lane, infos=infos, clips=clips))

    return cameras


def scan_video_clips(
    video_dir: Path, config: BormoSyncConfig
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


def _phrase_blocks(rec_times: list[float], gap: float) -> list[tuple[float, float]]:
    """Group sorted recorder-time word marks into [start, end] phrases, splitting
    wherever the gap between consecutive words exceeds ``gap`` seconds."""
    if not rec_times:
        return []
    blocks: list[tuple[float, float]] = []
    start = end = rec_times[0]
    for t in rec_times[1:]:
        if t - end > gap:
            blocks.append((start, end))
            start = t
        end = t
    blocks.append((start, end))
    return blocks


def clip_pieces(
    am: AlignmentMap,
    clip_duration: float,
    rec_duration: float,
    strategy_id: int,
    config: BormoSyncConfig,
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

    anchors = sorted(a.rec_time for a in am.anchors)
    if strategy_id == 1:
        breakpoints = [rec0, rec1]
    elif strategy_id == 2:
        breakpoints = [rec0, *[a for a in anchors if rec0 < a < rec1], rec1]
    else:  # 3, 4 — break only between phrases
        blocks = _phrase_blocks(anchors, config.phrase_gap_threshold)
        mids = [(blocks[i][1] + blocks[i + 1][0]) / 2 for i in range(len(blocks) - 1)]
        breakpoints = [rec0, *[m for m in mids if rec0 < m < rec1], rec1]

    bps = sorted(set(breakpoints))
    pieces: list[tuple[float, float, float]] = []
    for i in range(len(bps) - 1):
        ra, rb = bps[i], bps[i + 1]
        in_dur = rb - ra
        out_dur = r2l(rb) - r2l(ra)
        if in_dur <= 1e-4 or out_dur <= 1e-4:
            continue
        pieces.append((ra, in_dur, in_dur / out_dur))

    lead = max(0.0, r2l(rec0))
    return lead, pieces


def _timeline_end(clips: list[MediaClip]) -> float:
    return max((c.offset + c.duration for c in clips), default=0.0)


def _anchor_count(am: AlignmentMap | None) -> int:
    return len(am.anchors) if am is not None else 0


def run_pipeline(
    config: BormoSyncConfig,
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
        # (audio_clip_index, recorder_path, lead_silence, pieces, clip_duration)
        render_jobs: list[tuple[int, Path, float, list[tuple[float, float, float]], float]] = []

        def _add_job(vclip: MediaClip, am: AlignmentMap, ri: int, lane: int) -> None:
            lead, pieces = clip_pieces(
                am, vclip.duration, rec_infos[ri].duration, strategy_id, config
            )
            if not pieces:
                return
            label = f"Audio: {audio_files[ri].stem}" if config.recorder_mode == "all" else "Audio"
            audio_clips.append(
                MediaClip(
                    path=audio_files[ri],
                    kind="audio",
                    offset=vclip.offset,
                    in_point=0.0,
                    duration=vclip.duration,
                    lane=lane,
                )
            )
            audio_speed.append(1.0 / am.k if am.k else 1.0)
            audio_track.append(label)
            render_jobs.append(
                (len(audio_clips) - 1, audio_files[ri], lead, pieces, vclip.duration)
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

        from bormosync.models import SyncPlan

        all_clips = video_clips + audio_clips
        plan = SyncPlan(
            strategy_id=strategy_id,
            clips=all_clips,
            audio_ops=[],
            total_duration=_timeline_end(all_clips),
        )
        _notify("planning", 1.0, f"Strategy {strategy_id}: {strategy.name}")

        # --- render one continuous synced WAV per clip ---
        _notify("processing", 0.0, "Rendering synced audio...")
        # Small length-preserving fades declick the seams between stretched pieces.
        fade_ms = config.crossfade_ms if config.crossfade_enabled else 0
        n_jobs = max(len(render_jobs), 1)
        for j, (clip_idx, rec_path, lead, pieces, dur) in enumerate(render_jobs):
            audio_status[clip_idx] = "working"
            _notify(
                "processing",
                j / n_jobs,
                f"Rendering synced audio {j + 1}/{n_jobs}",
                clips=_timeline_snapshot(),
            )
            # Keep scratch segments on the OUTPUT volume (next to the result), not
            # the system /tmp — /tmp may be small or on a full disk.
            with tempfile.TemporaryDirectory(prefix="bormosync_seg_", dir=audio_synced_dir) as td:
                tdp = Path(td)
                seg_paths: list[Path] = []
                for k, (rec_start, rec_dur, factor) in enumerate(pieces):
                    if abs(factor - 1.0) > 1e-6:
                        sp = apply_atempo_segment(
                            rec_path, tdp, rec_start, rec_dur, factor, k, fade_ms=fade_ms
                        )
                    else:
                        sp = extract_segment(rec_path, tdp, rec_start, rec_dur, k, fade_ms=fade_ms)
                    seg_paths.append(sp)
                out = audio_synced_dir / f"synced_{clip_idx:03d}.wav"
                assemble_continuous(seg_paths, lead, dur, out_sr, out)
            audio_clips[clip_idx].path = out
            audio_clips[clip_idx].in_point = 0.0
            audio_status[clip_idx] = "done"
            _notify(
                "processing",
                (j + 1) / n_jobs,
                f"Rendered {j + 1}/{n_jobs}",
                clips=_timeline_snapshot(),
            )

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
