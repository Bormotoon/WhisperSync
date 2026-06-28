"""Core data models for BormoSync."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


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
class MediaClip:
    path: Path
    kind: Literal["video", "audio"]
    offset: float
    in_point: float
    duration: float
    lane: int


@dataclass
class SyncPlan:
    strategy_id: int
    clips: list[MediaClip]
    audio_ops: list[dict[str, object]]
    total_duration: float


@dataclass
class SyncResult:
    fcpxml_path: Path
    alignment: AlignmentMap
    plan: SyncPlan
    anchors_used: int
    warnings: list[str]
