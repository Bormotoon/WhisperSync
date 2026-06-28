"""Synchronization strategies: Global Linear, Local Time-Stretch, Silence Padding."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from bormosync.models import AlignmentMap, MediaClip, SyncPlan

logger = logging.getLogger(__name__)


def _timeline_end(clips: list[MediaClip]) -> float:
    """Return the latest timeline position reached by any clip."""
    return max((c.offset + c.duration for c in clips), default=0.0)


class SyncStrategy(ABC):
    """Abstract base class for synchronization strategies."""

    @abstractmethod
    def plan(
        self,
        alignment: AlignmentMap,
        rec_audio_path: Path,
        rec_duration: float,
        video_clips: list[MediaClip],
    ) -> SyncPlan: ...

    @property
    @abstractmethod
    def strategy_id(self) -> int: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def diagram_data(self) -> dict[str, Any]: ...


class GlobalLinearStrategy(SyncStrategy):
    """Apply a single global atempo = 1/K to the entire recording."""

    strategy_id = 1
    name = "Global Linear"
    description = "Apply a single global tempo change to the entire audio recording."

    def __init__(self) -> None:
        self._diagram_data: dict[str, Any] = {}

    def plan(
        self,
        alignment: AlignmentMap,
        rec_audio_path: Path,
        rec_duration: float,
        video_clips: list[MediaClip],
    ) -> SyncPlan:
        k = alignment.k
        factor = 1.0 / k
        cam_duration = rec_duration * k

        clip = MediaClip(
            path=rec_audio_path,
            kind="audio",
            offset=alignment.offset,
            in_point=0.0,
            duration=cam_duration,
            lane=-1,
        )

        audio_ops: list[dict[str, Any]] = [
            {"type": "atempo", "factor": factor, "input": str(rec_audio_path)},
        ]

        self._diagram_data = {
            "type": "global",
            "blocks": [{"start": 0.0, "end": 1.0, "label": f"atempo={factor:.4f}"}],
        }

        clips = list(video_clips) + [clip]
        return SyncPlan(
            strategy_id=self.strategy_id,
            clips=clips,
            audio_ops=audio_ops,
            total_duration=_timeline_end(clips),
        )

    @property
    def diagram_data(self) -> dict[str, Any]:
        return self._diagram_data


class LocalTimeStretchStrategy(SyncStrategy):
    """Divide recording into chunks between consecutive anchors with per-segment atempo."""

    strategy_id = 2
    name = "Local Time-Stretch"
    description = (
        "Divide the recording into segments between anchors " "and apply per-segment tempo change."
    )

    def __init__(self) -> None:
        self._diagram_data: dict[str, Any] = {}

    def plan(
        self,
        alignment: AlignmentMap,
        rec_audio_path: Path,
        rec_duration: float,
        video_clips: list[MediaClip],
    ) -> SyncPlan:
        anchors = sorted(alignment.anchors, key=lambda a: a.rec_time)

        if not anchors:
            logger.warning("No anchors — falling back to global strategy")
            return GlobalLinearStrategy().plan(alignment, rec_audio_path, rec_duration, video_clips)

        boundaries: list[tuple[float, float]] = [(0.0, alignment.offset)]
        for a in anchors:
            boundaries.append((a.rec_time, a.cam_time))
        boundaries.append((rec_duration, alignment.rec_to_cam(rec_duration)))

        audio_span = boundaries[-1][1] - boundaries[0][1]

        clips: list[MediaClip] = list(video_clips)
        audio_ops: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []

        for i in range(len(boundaries) - 1):
            rec_start, cam_start = boundaries[i]
            rec_end, cam_end = boundaries[i + 1]
            rec_chunk_dur = rec_end - rec_start
            if rec_chunk_dur <= 0:
                continue

            cam_chunk_dur = cam_end - cam_start
            k_local = cam_chunk_dur / rec_chunk_dur
            factor = 1.0 / k_local

            clips.append(
                MediaClip(
                    path=rec_audio_path,
                    kind="audio",
                    offset=cam_start,
                    in_point=rec_start,
                    duration=cam_chunk_dur,
                    lane=-1,
                )
            )

            audio_ops.append(
                {
                    "type": "atempo_segment",
                    "factor": factor,
                    "start": rec_start,
                    "duration": rec_chunk_dur,
                    "index": i,
                }
            )

            norm_start = cam_start / audio_span if audio_span > 0 else 0.0
            norm_end = cam_end / audio_span if audio_span > 0 else 0.0
            blocks.append({"start": norm_start, "end": norm_end, "label": f"K={k_local:.3f}"})

        self._diagram_data = {"type": "local", "blocks": blocks}

        return SyncPlan(
            strategy_id=self.strategy_id,
            clips=clips,
            audio_ops=audio_ops,
            total_duration=_timeline_end(clips),
        )

    @property
    def diagram_data(self) -> dict[str, Any]:
        return self._diagram_data


class SilencePaddingStrategy(SyncStrategy):
    """Extract speech segments as-is, insert/remove silence between them for alignment."""

    strategy_id = 3
    name = "Silence Padding"
    description = (
        "Extract speech segments between anchors without resampling "
        "and pad or trim silence between them."
    )

    def __init__(self) -> None:
        self._diagram_data: dict[str, Any] = {}
        self.warnings: list[str] = []

    def plan(
        self,
        alignment: AlignmentMap,
        rec_audio_path: Path,
        rec_duration: float,
        video_clips: list[MediaClip],
    ) -> SyncPlan:
        anchors = sorted(alignment.anchors, key=lambda a: a.rec_time)
        self.warnings.clear()

        if not anchors:
            logger.warning("No anchors — falling back to global strategy")
            return GlobalLinearStrategy().plan(alignment, rec_audio_path, rec_duration, video_clips)

        clips: list[MediaClip] = list(video_clips)
        audio_ops: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        current_cam = alignment.offset

        # initial silence before first anchor
        first_gap = anchors[0].cam_time - current_cam
        if first_gap > 0:
            audio_ops.append({"type": "silence", "duration": first_gap})
            blocks.append({"start": 0.0, "end": first_gap, "kind": "silence"})
            current_cam = anchors[0].cam_time

        for i in range(len(anchors) - 1):
            a0 = anchors[i]
            a1 = anchors[i + 1]
            rec_dur = a1.rec_time - a0.rec_time
            if rec_dur <= 0:
                continue

            # speech segment
            clips.append(
                MediaClip(
                    path=rec_audio_path,
                    kind="audio",
                    offset=a0.cam_time,
                    in_point=a0.rec_time,
                    duration=rec_dur,
                    lane=-1,
                )
            )
            audio_ops.append(
                {
                    "type": "extract",
                    "input": str(rec_audio_path),
                    "start": a0.rec_time,
                    "duration": rec_dur,
                }
            )
            blocks.append({"start": a0.cam_time, "end": a0.cam_time + rec_dur, "kind": "speech"})
            seg_end_cam = a0.cam_time + rec_dur

            # gap to next anchor
            gap = a1.cam_time - seg_end_cam
            if gap > 0:
                audio_ops.append({"type": "silence", "duration": gap})
                blocks.append({"start": seg_end_cam, "end": a1.cam_time, "kind": "silence"})
                current_cam = a1.cam_time
            elif gap < 0:
                msg = (
                    f"Negative gap {gap:.3f}s between anchor "
                    f"{a0.token} and {a1.token} — possible overlap"
                )
                logger.warning(msg)
                self.warnings.append(msg)
                current_cam = a1.cam_time
            else:
                current_cam = a1.cam_time

        # trailing silence after last anchor
        final_cam = alignment.rec_to_cam(rec_duration)
        if len(anchors) >= 2:
            last_seg_end = anchors[-1].cam_time + (anchors[-1].rec_time - anchors[-2].rec_time)
        else:
            last_seg_end = anchors[-1].cam_time + (rec_duration - anchors[-1].rec_time)
        last_gap = final_cam - last_seg_end
        if last_gap > 0:
            audio_ops.append({"type": "silence", "duration": last_gap})
            blocks.append({"start": last_seg_end, "end": final_cam, "kind": "silence"})

        audio_span = final_cam - alignment.offset

        # normalize blocks to 0-1
        if audio_span > 0:
            for b in blocks:
                b["start"] = b["start"] / audio_span
                b["end"] = b["end"] / audio_span

        self._diagram_data = {"type": "padding", "blocks": blocks}

        return SyncPlan(
            strategy_id=self.strategy_id,
            clips=clips,
            audio_ops=audio_ops,
            total_duration=_timeline_end(clips),
        )

    @property
    def diagram_data(self) -> dict[str, Any]:
        return self._diagram_data


def get_strategy(strategy_id: int) -> SyncStrategy:
    """Return a SyncStrategy instance for the given strategy_id."""
    strategies: dict[int, type[SyncStrategy]] = {
        1: GlobalLinearStrategy,
        2: LocalTimeStretchStrategy,
        3: SilencePaddingStrategy,
    }
    cls = strategies.get(strategy_id)
    if cls is None:
        raise ValueError(f"Unknown strategy_id {strategy_id}. Valid ids: {list(strategies)}")
    return cls()
