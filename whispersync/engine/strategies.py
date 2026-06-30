"""Synchronization strategies (per camera clip).

Each strategy plans the synced recorder audio for ONE camera clip, given a
linear alignment between that clip's local time and the recorder time:

    t_local = alignment.offset + alignment.k * t_rec

so the recorder time for a local clip position is ``(t_local - offset) / k``.

The pipeline calls ``plan_clip`` once per clip and assembles the full timeline,
which means clip placement comes purely from matched timecodes — no assumption
that clips are contiguous.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from whispersync.config import WhisperSyncConfig
from whispersync.models import AlignmentMap, Anchor, MediaClip

logger = logging.getLogger(__name__)

AudioOp = dict[str, Any]
ClipPlan = tuple[list[MediaClip], list[AudioOp]]

# Minimum length of a speech block, so an isolated anchor still yields audible
# audio instead of a zero-length segment that gets dropped.
_MIN_BLOCK_DUR = 0.2


def _rec_to_local(am: AlignmentMap, t_rec: float) -> float:
    return am.offset + am.k * t_rec


def _local_to_rec(am: AlignmentMap, t_local: float) -> float:
    return (t_local - am.offset) / am.k


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _speech_blocks(anchors: list[Anchor], gap: float) -> list[tuple[float, float]]:
    """Group anchors (sorted by rec_time) into [rec_start, rec_end] speech
    blocks, splitting wherever the recorder-time gap exceeds ``gap`` seconds."""
    if not anchors:
        return []
    blocks: list[tuple[float, float]] = []
    start = anchors[0].rec_time
    end = anchors[0].rec_time
    for a in anchors[1:]:
        if a.rec_time - end > gap:
            blocks.append((start, end))
            start = a.rec_time
        end = a.rec_time
    blocks.append((start, end))
    return blocks


class SyncStrategy(ABC):
    """Base class. Subclasses set strategy_id/name/description and implement
    plan_clip for a single camera clip."""

    strategy_id: int
    name: str
    description: str

    @abstractmethod
    def plan_clip(
        self,
        alignment: AlignmentMap,
        rec_audio_path: Path,
        clip_offset: float,
        clip_duration: float,
        rec_duration: float,
        config: WhisperSyncConfig,
    ) -> ClipPlan: ...


class GlobalLinearStrategy(SyncStrategy):
    strategy_id = 1
    name = "Global Linear"
    description = "One tempo change for the whole clip. Best for linear clock drift."

    def plan_clip(
        self,
        alignment: AlignmentMap,
        rec_audio_path: Path,
        clip_offset: float,
        clip_duration: float,
        rec_duration: float,
        config: WhisperSyncConfig,
    ) -> ClipPlan:
        r0 = _clamp(_local_to_rec(alignment, 0.0), 0.0, rec_duration)
        r1 = _clamp(_local_to_rec(alignment, clip_duration), 0.0, rec_duration)
        in_dur = r1 - r0
        if in_dur <= 0:
            return [], []

        # local span actually covered by available recorder audio
        local_start = _rec_to_local(alignment, r0)
        out_dur = _rec_to_local(alignment, r1) - local_start
        factor = in_dur / out_dur if out_dur > 0 else 1.0

        clip = MediaClip(
            path=rec_audio_path,
            kind="audio",
            offset=clip_offset + local_start,
            in_point=0.0,
            duration=out_dur,
            lane=-1,
        )
        op: AudioOp = {
            "type": "atempo_segment",
            "input": str(rec_audio_path),
            "start": r0,
            "duration": in_dur,
            "factor": factor,
        }
        return [clip], [op]


class LocalTimeStretchStrategy(SyncStrategy):
    strategy_id = 2
    name = "Local Time-Stretch"
    description = "Per-segment tempo change between anchors. Handles non-linear drift."

    def plan_clip(
        self,
        alignment: AlignmentMap,
        rec_audio_path: Path,
        clip_offset: float,
        clip_duration: float,
        rec_duration: float,
        config: WhisperSyncConfig,
    ) -> ClipPlan:
        anchors = sorted(alignment.anchors, key=lambda a: a.rec_time)
        if len(anchors) < 2:
            return GlobalLinearStrategy().plan_clip(
                alignment, rec_audio_path, clip_offset, clip_duration, rec_duration, config
            )

        # boundaries in recorder time: clip start, each anchor, clip end
        r_start = _clamp(_local_to_rec(alignment, 0.0), 0.0, rec_duration)
        r_end = _clamp(_local_to_rec(alignment, clip_duration), 0.0, rec_duration)
        rec_points = [r_start] + [a.rec_time for a in anchors] + [r_end]
        rec_points = sorted({_clamp(r, 0.0, rec_duration) for r in rec_points})

        clips: list[MediaClip] = []
        ops: list[AudioOp] = []
        for i in range(len(rec_points) - 1):
            rs, re = rec_points[i], rec_points[i + 1]
            in_dur = re - rs
            if in_dur <= 1e-4:
                continue
            ls = _rec_to_local(alignment, rs)
            le = _rec_to_local(alignment, re)
            out_dur = le - ls
            if out_dur <= 0:
                continue
            factor = in_dur / out_dur
            clips.append(
                MediaClip(
                    path=rec_audio_path,
                    kind="audio",
                    offset=clip_offset + ls,
                    in_point=0.0,
                    duration=out_dur,
                    lane=-1,
                )
            )
            ops.append(
                {
                    "type": "atempo_segment",
                    "input": str(rec_audio_path),
                    "start": rs,
                    "duration": in_dur,
                    "factor": factor,
                }
            )
        return clips, ops


class SilencePaddingStrategy(SyncStrategy):
    strategy_id = 3
    name = "Silence Padding"
    description = "Speech left untouched (zero pitch shift); only inter-phrase gaps move."

    def plan_clip(
        self,
        alignment: AlignmentMap,
        rec_audio_path: Path,
        clip_offset: float,
        clip_duration: float,
        rec_duration: float,
        config: WhisperSyncConfig,
    ) -> ClipPlan:
        anchors = sorted(alignment.anchors, key=lambda a: a.rec_time)
        if not anchors:
            return GlobalLinearStrategy().plan_clip(
                alignment, rec_audio_path, clip_offset, clip_duration, rec_duration, config
            )

        clips: list[MediaClip] = []
        ops: list[AudioOp] = []
        for rs, re in _speech_blocks(anchors, config.phrase_gap_threshold):
            rs_c = _clamp(rs, 0.0, rec_duration)
            re_c = _clamp(max(re, rs + _MIN_BLOCK_DUR), 0.0, rec_duration)
            dur = re_c - rs_c
            if dur <= 1e-4:
                continue
            local_start = _rec_to_local(alignment, rs_c)
            clips.append(
                MediaClip(
                    path=rec_audio_path,
                    kind="audio",
                    offset=clip_offset + local_start,
                    in_point=0.0,
                    duration=dur,
                    lane=-1,
                )
            )
            ops.append(
                {
                    "type": "extract",
                    "input": str(rec_audio_path),
                    "start": rs_c,
                    "duration": dur,
                }
            )
        return clips, ops


class HybridStrategy(SyncStrategy):
    strategy_id = 4
    name = "Hybrid (Global + Silence)"
    description = (
        "Each phrase is tempo-corrected by the clip's global K, then placed at "
        "its anchor position with silence absorbing the rest. Robust + near pitch-perfect."
    )

    def plan_clip(
        self,
        alignment: AlignmentMap,
        rec_audio_path: Path,
        clip_offset: float,
        clip_duration: float,
        rec_duration: float,
        config: WhisperSyncConfig,
    ) -> ClipPlan:
        anchors = sorted(alignment.anchors, key=lambda a: a.rec_time)
        if not anchors:
            return GlobalLinearStrategy().plan_clip(
                alignment, rec_audio_path, clip_offset, clip_duration, rec_duration, config
            )

        factor = 1.0 / alignment.k  # global linear calibration
        clips: list[MediaClip] = []
        ops: list[AudioOp] = []
        for rs, re in _speech_blocks(anchors, config.phrase_gap_threshold):
            rs_c = _clamp(rs, 0.0, rec_duration)
            re_c = _clamp(max(re, rs + _MIN_BLOCK_DUR), 0.0, rec_duration)
            in_dur = re_c - rs_c
            if in_dur <= 1e-4:
                continue
            local_start = _rec_to_local(alignment, rs_c)
            out_dur = alignment.k * in_dur  # camera-equivalent length
            clips.append(
                MediaClip(
                    path=rec_audio_path,
                    kind="audio",
                    offset=clip_offset + local_start,
                    in_point=0.0,
                    duration=out_dur,
                    lane=-1,
                )
            )
            ops.append(
                {
                    "type": "atempo_segment",
                    "input": str(rec_audio_path),
                    "start": rs_c,
                    "duration": in_dur,
                    "factor": factor,
                }
            )
        return clips, ops


STRATEGIES: dict[int, type[SyncStrategy]] = {
    1: GlobalLinearStrategy,
    2: LocalTimeStretchStrategy,
    3: SilencePaddingStrategy,
    4: HybridStrategy,
}


def get_strategy(strategy_id: int) -> SyncStrategy:
    cls = STRATEGIES.get(strategy_id)
    if cls is None:
        raise ValueError(f"Unknown strategy_id {strategy_id}. Valid ids: {list(STRATEGIES)}")
    return cls()
