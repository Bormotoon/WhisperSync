"""Configuration management for BormoSync."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir

APP_NAME = "bormosync"

DEFAULT_VIDEO_EXTS = [".mp4", ".mov", ".mxf", ".avi", ".mkv"]
DEFAULT_AUDIO_EXTS = [".wav", ".mp3", ".m4a", ".flac"]

MIN_ANCHORS = 8
ANCHOR_MIN_CONFIDENCE = 0.6

# Minimum silence (seconds) between consecutive anchors to split speech blocks
# in the silence-padding / hybrid strategies.
PHRASE_GAP_THRESHOLD = 0.6

# Coarse-then-fine matching for long recordings: a clip is first roughly located
# in the (possibly multi-hour) reference by rare-word voting, then matched
# precisely only inside a window around that estimate.
MATCH_WINDOW_MARGIN = 90.0  # seconds of slack added around the coarse estimate
SEED_MAX_OCCURRENCES = 50  # ignore tokens appearing more than this in the reference
SEED_BIN_WIDTH = 2.0  # seconds — histogram bin width for the coarse-offset vote

WHISPER_BEAM_SIZE = 5
WHISPER_TEMPERATURE = 0.0

# Which source provides the audio sample-rate reference for FCPXML time values.
TIMEBASE_SOURCES = ("camera", "recorder")


@dataclass
class BormoSyncConfig:
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    vad_filter: bool = True
    video_exts: list[str] = field(default_factory=lambda: list(DEFAULT_VIDEO_EXTS))
    audio_exts: list[str] = field(default_factory=lambda: list(DEFAULT_AUDIO_EXTS))
    fcpxml_version: str = "1.9"
    default_strategy: int = 1
    cache_dir: str | None = None
    output_dir: str | None = None
    use_cache: bool = True
    timebase_source: str = "camera"
    min_anchors: int = MIN_ANCHORS
    anchor_min_confidence: float = ANCHOR_MIN_CONFIDENCE
    phrase_gap_threshold: float = PHRASE_GAP_THRESHOLD
    match_window_margin: float = MATCH_WINDOW_MARGIN
    seed_max_occurrences: int = SEED_MAX_OCCURRENCES
    seed_bin_width: float = SEED_BIN_WIDTH

    @property
    def resolved_cache_dir(self) -> Path:
        if self.cache_dir:
            return Path(self.cache_dir)
        return Path(user_cache_dir(APP_NAME))

    @property
    def resolved_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        return Path.cwd() / "output"

    @property
    def resolved_config_dir(self) -> Path:
        return Path(user_config_dir(APP_NAME))

    @classmethod
    def from_file(cls, path: Path) -> BormoSyncConfig:
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def merge_cli_args(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)


def load_config(config_path: Path | None = None, **cli_overrides: object) -> BormoSyncConfig:
    if config_path and config_path.exists():
        cfg = BormoSyncConfig.from_file(config_path)
    else:
        cfg = BormoSyncConfig()
    cfg.merge_cli_args(**cli_overrides)
    return cfg
