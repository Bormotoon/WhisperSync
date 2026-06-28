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


def scan_video_clips(
    video_dir: Path, config: BormoSyncConfig
) -> tuple[list[MediaInfo], list[MediaClip]]:
    """Probe every video in the folder (sorted by name). Offsets are left at 0
    here — real timeline positions are computed later from matched timecodes."""
    exts = tuple(config.video_exts)
    video_paths = sorted(
        (p for p in video_dir.iterdir() if p.suffix.lower() in exts),
        key=lambda p: p.name,
    )
    if not video_paths:
        raise RuntimeError(f"No video files found in {video_dir}")

    video_infos: list[MediaInfo] = []
    video_clips: list[MediaClip] = []
    for path in video_paths:
        info = probe(path)
        video_infos.append(info)
        video_clips.append(
            MediaClip(
                path=path,
                kind="video",
                offset=0.0,
                in_point=0.0,
                duration=info.duration,
                lane=1,
            )
        )

    return video_infos, video_clips


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


def run_pipeline(
    config: BormoSyncConfig,
    video_dir: Path,
    audio_file: Path,
    strategy_id: int,
    output_path: Path,
    progress_callback: ProgressCallback | None = None,
) -> SyncResult:
    def _notify(stage: str, progress: float = 0.0, message: str = "") -> None:
        if progress_callback is not None:
            progress_callback(PipelineProgress(stage=stage, progress=progress, message=message))

    engine: WhisperEngine | None = None
    cleanup_paths: list[Path] = []
    warnings: list[str] = []

    try:
        # --- scanning ---
        _notify("scanning", 0.0, "Scanning video directory...")
        video_infos, video_clips = scan_video_clips(video_dir, config)
        _notify("scanning", 1.0, f"Found {len(video_clips)} video clip(s)")

        # --- transcribe recorder once ---
        engine = WhisperEngine(config)
        rec_info = probe(audio_file)
        _notify("transcribing_recorder", 0.0, "Transcribing recorder audio...")
        rec_transcript = engine.transcribe(
            audio_file, lambda p: _notify("transcribing_recorder", p)
        )

        # --- transcribe + align each clip independently ---
        alignments: list[AlignmentMap | None] = []
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
            try:
                am = align(clip_transcript, rec_transcript, config)
            except ValueError as e:
                logger.warning("Clip %s could not be aligned: %s", clip.path.name, e)
                warnings.append(f"{clip.path.name}: not aligned ({e})")
                am = None
            alignments.append(am)

        if all(a is None for a in alignments):
            raise RuntimeError("No camera clip could be aligned to the recorder audio.")

        # --- place clips on the master timeline from timecodes ---
        _notify("aligning", 1.0, "Placing clips on timeline...")
        durations = [c.duration for c in video_clips]
        offsets, unaligned = compute_master_offsets(alignments, durations)
        for clip, off in zip(video_clips, offsets, strict=True):
            clip.offset = off
        for i in unaligned:
            warnings.append(f"{video_clips[i].path.name}: placed by order (no anchors)")

        # --- plan audio per clip with the chosen strategy ---
        _notify("planning", 0.0, "Planning sync strategy...")
        strategy = get_strategy(strategy_id)
        audio_clips: list[MediaClip] = []
        audio_ops: list[dict[str, object]] = []
        for clip, am in zip(video_clips, alignments, strict=True):
            if am is None:
                continue
            cs, ops = strategy.plan_clip(
                am, audio_file, clip.offset, clip.duration, rec_info.duration, config
            )
            audio_clips.extend(cs)
            audio_ops.extend(ops)

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
        if config.timebase_source == "recorder" and rec_info.audio_sample_rate:
            audio_sr: int | None = rec_info.audio_sample_rate
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

        # --- collect quality warnings from the best clip alignment ---
        best = max(
            (a for a in alignments if a is not None),
            key=lambda a: len(a.anchors),
        )
        if best.residual_ms > 40:
            warnings.append(f"High residual alignment error: {best.residual_ms:.1f} ms")

        _notify("done", 1.0, "Pipeline complete")
        return SyncResult(
            fcpxml_path=output_path,
            alignment=best,
            plan=plan,
            anchors_used=sum(len(a.anchors) for a in alignments if a is not None),
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
