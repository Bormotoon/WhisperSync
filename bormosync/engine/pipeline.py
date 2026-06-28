"""End-to-end orchestration pipeline with progress signals.

The recorder is the continuous reference. Each camera clip is transcribed and
aligned to the recorder independently, so a clip's position on the master
timeline comes from matched timecodes — clips need not be contiguous.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bormosync.config import BormoSyncConfig
from bormosync.engine.export import generate_fcpxml
from bormosync.engine.matcher import align
from bormosync.engine.media import MediaInfo, extract_audio_to_wav, probe
from bormosync.engine.strategies import get_strategy
from bormosync.engine.timestretch import apply_atempo_segment, extract_segment
from bormosync.engine.transcriber import WhisperEngine
from bormosync.models import AlignmentMap, MediaClip, SyncResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineProgress:
    stage: str
    progress: float = 0.0
    message: str = ""


ProgressCallback = Callable[[PipelineProgress], None]


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
        return sorted((p for p in d.iterdir() if p.suffix.lower() in exts), key=lambda p: p.name)

    subdirs = sorted(
        (d for d in video_dir.iterdir() if d.is_dir() and videos_in(d)),
        key=lambda d: d.name,
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
    def _notify(stage: str, progress: float = 0.0, message: str = "") -> None:
        if progress_callback is not None:
            progress_callback(PipelineProgress(stage=stage, progress=progress, message=message))

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

        # --- transcribe every recorder once ---
        engine = WhisperEngine(config)
        rec_infos = [probe(p) for p in audio_files]
        rec_transcripts = []
        for ri, rp in enumerate(audio_files):
            _notify("transcribing_recorder", ri / len(audio_files), f"Recorder: {rp.name}")
            rec_transcripts.append(
                engine.transcribe(rp, lambda p: _notify("transcribing_recorder", p))
            )

        # --- align each clip against each recorder ---
        # aligns[clip_idx][rec_idx] = AlignmentMap | None
        aligns: list[list[AlignmentMap | None]] = []
        n = len(video_clips)
        for idx, clip in enumerate(video_clips):
            _notify(
                "transcribing_camera",
                idx / max(n, 1),
                f"Transcribing camera clip {idx + 1}/{n}: {clip.path.name}",
            )
            clip_audio = extract_audio_to_wav(clip.path)
            cleanup_paths.append(clip_audio)
            clip_transcript = engine.transcribe(clip_audio)
            row: list[AlignmentMap | None] = []
            for rt in rec_transcripts:
                try:
                    row.append(align(clip_transcript, rt, config))
                except ValueError:
                    row.append(None)
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

        # --- plan audio (from the audio-source camera's clips) ---
        _notify("planning", 0.0, "Planning sync strategy...")
        strategy = get_strategy(strategy_id)
        audio_clips: list[MediaClip] = []
        audio_ops: list[dict[str, object]] = []

        def _plan_one(clip: MediaClip, am: AlignmentMap, ri: int, lane: int) -> None:
            cs, ops = strategy.plan_clip(
                am, audio_files[ri], clip.offset, clip.duration, rec_infos[ri].duration, config
            )
            for c in cs:
                c.lane = lane
            audio_clips.extend(cs)
            audio_ops.extend(ops)

        for ci in range(n):
            if clip_camera[ci] != audio_ci:
                continue
            clip = video_clips[ci]
            row = aligns[ci]
            if config.recorder_mode == "all":
                for ri, am in enumerate(row):
                    if am is not None:
                        _plan_one(clip, am, ri, lane=-(ri + 1))
            else:  # "best": one lane, best recorder per clip
                candidates = [(ri, am) for ri, am in enumerate(row) if am is not None]
                if candidates:
                    best_ri, best_am = max(candidates, key=lambda t: len(t[1].anchors))
                    _plan_one(clip, best_am, best_ri, lane=-1)

        from bormosync.models import SyncPlan

        all_clips = video_clips + audio_clips
        plan = SyncPlan(
            strategy_id=strategy_id,
            clips=all_clips,
            audio_ops=audio_ops,
            total_duration=_timeline_end(all_clips),
        )
        _notify("planning", 1.0, f"Strategy {strategy_id}: {strategy.name}")

        # --- process audio operations ---
        _notify("processing", 0.0, "Processing audio operations...")
        audio_synced_dir = output_path.parent / "audio_synced"
        audio_synced_dir.mkdir(parents=True, exist_ok=True)

        audio_idx = 0
        seg_index = 0
        n_ops = max(len(plan.audio_ops), 1)

        def _assign(out_path: Path) -> None:
            nonlocal audio_idx
            if audio_idx < len(audio_clips):
                audio_clips[audio_idx].path = out_path
                audio_clips[audio_idx].in_point = 0.0
                audio_idx += 1

        for i, op in enumerate(plan.audio_ops):
            op_type = op["type"]
            if op_type == "atempo_segment":
                out = apply_atempo_segment(
                    Path(str(op["input"])),
                    audio_synced_dir,
                    float(op["start"]),
                    float(op["duration"]),
                    float(op["factor"]),
                    seg_index,
                )
                _assign(out)
                seg_index += 1
            elif op_type == "extract":
                out = extract_segment(
                    Path(str(op["input"])),
                    audio_synced_dir,
                    float(op["start"]),
                    float(op["duration"]),
                    seg_index,
                )
                _assign(out)
                seg_index += 1
            else:
                logger.warning("Unknown audio op type '%s' — skipping", op_type)
            _notify("processing", (i + 1) / n_ops)

        # --- export ---
        _notify("exporting", 0.0, "Generating FCPXML...")
        if config.timebase_source == "recorder" and rec_infos[primary].audio_sample_rate:
            audio_sr: int | None = rec_infos[primary].audio_sample_rate
        else:
            audio_sr = None  # export derives it from the camera
        generate_fcpxml(
            plan,
            video_infos,
            output_path,
            config.fcpxml_version,
            output_path.stem,
            audio_sample_rate=audio_sr,
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
