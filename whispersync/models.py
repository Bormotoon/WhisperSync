"""Core data models for WhisperSync."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float
    norm: str = ""


@dataclass
class Segment:
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    source_path: Path
    language: str
    duration: float
    segments: list[Segment]

    @property
    def words(self) -> list[Word]:
        return [w for seg in self.segments for w in seg.words]


@dataclass
class Anchor:
    cam_time: float
    rec_time: float
    token: str
    confidence: float


@dataclass
class AlignmentMap:
    anchors: list[Anchor]
    offset: float
    k: float
    residual_ms: float

    def rec_to_cam(self, t: float) -> float:
        return self.offset + self.k * t


@dataclass
class SubClip:
    """One rendered speech piece inside a compound audio clip: ``offset`` is its
    position on the compound's own (local) timeline, ``in_point`` the source start."""

    path: Path
    offset: float
    in_point: float
    duration: float


@dataclass
class MediaClip:
    path: Path
    kind: Literal["video", "audio"]
    offset: float
    in_point: float
    duration: float
    lane: int
    # Optional FCPXML display name (falls back to the file stem) and FCPX role
    # (e.g. "Dialogue", "Effects", "Video") so the editor colours/groups clips.
    display_name: str | None = None
    role: str | None = None
    # When set, this audio clip is a COMPOUND clip: instead of one media file it
    # holds these separately-placed speech pieces (each editable in Final Cut), so
    # the user can crossfade/nudge them. The compound spans ``duration`` (= video).
    subclips: list[SubClip] | None = None


@dataclass
class SyncPlan:
    strategy_id: int
    clips: list[MediaClip]
    audio_ops: list[dict[str, Any]]
    total_duration: float


@dataclass
class SyncResult:
    fcpxml_path: Path
    alignment: AlignmentMap
    plan: SyncPlan
    anchors_used: int
    warnings: list[str]
