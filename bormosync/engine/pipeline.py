"""End-to-end orchestration pipeline with progress signals."""

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
from bormosync.engine.media import extract_audio_to_wav, probe
from bormosync.engine.strategies import get_strategy
from bormosync.engine.timestretch import apply_atempo, apply_atempo_segment, extract_segment
from bormosync.engine.transcriber import WhisperEngine
from bormosync.models import MediaClip, SyncResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineProgress:
    stage: str
    progress: float = 0.0
    message: str = ""


ProgressCallback = Callable[[PipelineProgress], None]


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

    try:
        # --- scanning ---
        _notify("scanning", 0.0, "Scanning video directory...")
        exts = tuple(config.video_exts)
        video_paths = sorted(
            [p for p in video_dir.iterdir() if p.suffix.lower() in exts],
            key=lambda p: p.name,
        )
        if not video_paths:
            raise RuntimeError(f"No video files found in {video_dir}")

        video_infos = []
        video_clips: list[MediaClip] = []
        offset = 0.0
        for path in video_paths:
            info = probe(path)
            video_infos.append(info)
            video_clips.append(
                MediaClip(
                    path=path,
                    kind="video",
                    offset=offset,
                    in_point=0.0,
                    duration=info.duration,
                    lane=1,
                )
            )
            offset += info.duration

        # --- extracting ---
        _notify("extracting", 0.0, "Extracting camera audio...")
        cam_audio = extract_audio_to_wav(video_paths[0])
        cleanup_paths.append(cam_audio)

        # --- transcribing ---
        engine = WhisperEngine(config)

        def _make_transcribe_callback(stage: str) -> Callable[[float], None]:
            def _cb(progress: float) -> None:
                _notify(stage, progress)

            return _cb

        _notify("transcribing_camera", 0.0, "Transcribing camera audio...")
        cam_transcript = engine.transcribe(
            cam_audio, _make_transcribe_callback("transcribing_camera")
        )

        _notify("transcribing_recorder", 0.0, "Transcribing recorder audio...")
        rec_transcript = engine.transcribe(
            audio_file, _make_transcribe_callback("transcribing_recorder")
        )

        # --- aligning ---
        _notify("aligning", 0.0, "Aligning transcripts...")
        alignment = align(cam_transcript, rec_transcript, config)

        # --- planning ---
        _notify("planning", 0.0, "Generating sync plan...")
        strategy = get_strategy(strategy_id)
        plan = strategy.plan(alignment, audio_file, rec_transcript.duration, video_clips)

        # --- processing ---
        _notify("processing", 0.0, "Processing audio operations...")
        audio_synced_dir = output_path.parent / "audio_synced"
        audio_synced_dir.mkdir(parents=True, exist_ok=True)

        # Audio clips are filled in the same order their ops are emitted by the
        # strategy. Video clips live in plan.clips too, so we walk audio clips
        # explicitly rather than indexing plan.clips by position.
        audio_clips = [c for c in plan.clips if c.kind == "audio"]
        audio_idx = 0
        seg_index = 0
        n_ops = max(len(plan.audio_ops), 1)

        def _assign(out_path: Path) -> None:
            nonlocal audio_idx
            if audio_idx < len(audio_clips):
                clip = audio_clips[audio_idx]
                clip.path = out_path
                # The produced file is already trimmed/stretched, so the clip
                # must read from its start, not the original recorder offset.
                clip.in_point = 0.0
                audio_idx += 1

        for i, op in enumerate(plan.audio_ops):
            op_type = op["type"]
            if op_type == "atempo":
                out = apply_atempo(
                    Path(op["input"]), audio_synced_dir / "synced.wav", float(op["factor"])
                )
                _assign(out)
            elif op_type == "atempo_segment":
                out = apply_atempo_segment(
                    audio_file,
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
                    Path(op.get("input", str(audio_file))),
                    audio_synced_dir,
                    float(op["start"]),
                    float(op["duration"]),
                    seg_index,
                )
                _assign(out)
                seg_index += 1
            elif op_type == "silence":
                # Silence gaps are implicit in clip offsets for the clip-based
                # layout; nothing to render.
                pass
            else:
                logger.warning("Unknown audio op type '%s' — skipping", op_type)

            _notify("processing", (i + 1) / n_ops)

        # --- exporting ---
        _notify("exporting", 0.0, "Generating FCPXML...")
        generate_fcpxml(plan, video_infos, output_path, config.fcpxml_version, output_path.stem)

        warnings: list[str] = list(getattr(strategy, "warnings", []))
        if alignment.residual_ms > 40:
            warnings.append(f"High residual alignment error: {alignment.residual_ms:.1f} ms")
        if len(alignment.anchors) < config.min_anchors:
            warnings.append(
                f"Low anchor count: {len(alignment.anchors)} < {config.min_anchors} recommended"
            )

        _notify("done", 1.0, "Pipeline complete")
        return SyncResult(
            fcpxml_path=output_path,
            alignment=alignment,
            plan=plan,
            anchors_used=len(alignment.anchors),
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
