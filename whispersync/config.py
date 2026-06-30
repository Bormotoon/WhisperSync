"""Configuration management for WhisperSync."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir

APP_NAME = "whispersync"

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

# Anti-hallucination temperature fallback ladder (proven on real podcast audio):
# a segment with high compression_ratio (repeats) or low logprob is retried at
# the next temperature, which kills the endless "Спасибо." loop on silence/music.
WHISPER_TEMPERATURE_LADDER = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Which source provides the audio sample-rate reference for FCPXML time values.
TIMEBASE_SOURCES = ("camera", "recorder")

# Word-matching backend. "legacy" = single difflib LCS (original). "dtw" = banded
# DTW over word indices, robust to repeated takes on both sides (both camera and
# recorder may contain re-takes/drafts; a single LCS can latch onto the wrong one).
ALIGN_MODES = ("legacy", "dtw")

# DTW (Tier 1) cost/band parameters. Band is a Sakoe-Chiba half-width in *word
# indices* around a coarse diagonal; cost mixes token mismatch with a penalty for
# straying off that diagonal (the slope term is what defeats wrong-take latching).
DTW_BAND_MIN = 8
DTW_BAND_MAX = 400
DTW_BAND_MARGIN_FRAC = 1.0  # fraction of match_window_margin folded into the band
DTW_TOKEN_WEIGHT = 1.0
DTW_SLOPE_WEIGHT = 0.25
DTW_SUBST_COST = 1.0
DTW_GAP_COST = 0.6
DTW_MIN_ANCHOR_CONF = 0.5

# Acoustic refine (Tier 2, "Flex Time"): sample a grid across the clip, measure the
# residual recorder↔camera lag by GCC-PHAT cross-correlation, and correct the
# anchor times sub-sample. Off by default. Sharpness = peak/median(|cc|); measured
# speech windows score 240–335 and silence/mismatch ≈12, so 50 is a safe gate.
ACOUSTIC_GRID_S = 25.0
ACOUSTIC_WINDOW_S = 7.0
ACOUSTIC_MAX_LAG_S = 1.0
ACOUSTIC_MIN_SHARPNESS = 50.0
GCC_EPS = 1e-8


@dataclass
class WhisperSyncConfig:
    model: str = "large-v3"
    # "auto" resolves to cuda when available, else cpu. compute_type "auto" picks
    # float16 on modern CUDA (capability >= 7.0), int8_float16 on older GPUs,
    # float32 on CPU.
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None
    vad_filter: bool = True
    beam_size: int = WHISPER_BEAM_SIZE
    # Batched GPU inference — the main speed lever (esp. for multi-hour recorders).
    # On CUDA OOM the engine auto-halves the batch, then falls back to CPU.
    batch_size: int = 16
    best_of: int = 1
    patience: float = 1.0
    condition_on_previous_text: bool = False
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    # "fast" = batched pipeline (no cross-segment context, ~real-time on GPU).
    # "quality" = sequential pipeline with context + hallucination guard: more
    # accurate on hard/quiet audio but ~10x slower.
    transcribe_mode: str = "fast"
    quality_beam_size: int = 10
    # Optional domain context to bias vocabulary (helps both modes).
    initial_prompt: str = ""
    video_exts: list[str] = field(default_factory=lambda: list(DEFAULT_VIDEO_EXTS))
    audio_exts: list[str] = field(default_factory=lambda: list(DEFAULT_AUDIO_EXTS))
    fcpxml_version: str = "1.9"
    default_strategy: int = 1
    cache_dir: str | None = None
    output_dir: str | None = None
    use_cache: bool = True
    # Save full transcripts (JSON + SRT) of every recorder and camera clip next
    # to the output, under output/transcripts/.
    save_transcripts: bool = True
    timebase_source: str = "camera"
    # Multicam: name (subfolder) of the camera the synced audio is derived from.
    # None = auto-pick the camera with the strongest alignment.
    audio_source_camera: str | None = None
    # Multiple recorders (different devices): "best" = one audio lane, each clip
    # synced from its best-matching recorder; "all" = every recorder on its own
    # audio lane (-1, -2, …) for multi-mic / multi-speaker setups.
    recorder_mode: str = "best"
    # Short equal-power fades at audio segment seams to declick joints (mainly
    # for the Local Time-Stretch strategy). Length-preserving, so no extra drift.
    crossfade_enabled: bool = True
    crossfade_ms: int = 10
    min_anchors: int = MIN_ANCHORS
    anchor_min_confidence: float = ANCHOR_MIN_CONFIDENCE
    phrase_gap_threshold: float = PHRASE_GAP_THRESHOLD
    match_window_margin: float = MATCH_WINDOW_MARGIN
    seed_max_occurrences: int = SEED_MAX_OCCURRENCES
    seed_bin_width: float = SEED_BIN_WIDTH
    # Word-matching backend (see ALIGN_MODES). "legacy" preserves the original
    # difflib behaviour; "dtw" uses the repeat-robust banded DTW in engine/dtw.py.
    align_mode: str = "legacy"
    dtw_band_min: int = DTW_BAND_MIN
    dtw_band_max: int = DTW_BAND_MAX
    dtw_band_margin_frac: float = DTW_BAND_MARGIN_FRAC
    dtw_token_weight: float = DTW_TOKEN_WEIGHT
    dtw_slope_weight: float = DTW_SLOPE_WEIGHT
    dtw_subst_cost: float = DTW_SUBST_COST
    dtw_gap_cost: float = DTW_GAP_COST
    dtw_min_anchor_conf: float = DTW_MIN_ANCHOR_CONF
    # Acoustic refine (Tier 2). Off by default; enriches anchor times via GCC-PHAT.
    acoustic_refine: bool = False
    acoustic_grid_s: float = ACOUSTIC_GRID_S
    acoustic_window_s: float = ACOUSTIC_WINDOW_S
    acoustic_max_lag_s: float = ACOUSTIC_MAX_LAG_S
    acoustic_min_sharpness: float = ACOUSTIC_MIN_SHARPNESS
    gcc_eps: float = GCC_EPS

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
    def from_file(cls, path: Path) -> WhisperSyncConfig:
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def merge_cli_args(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)


def load_config(config_path: Path | None = None, **cli_overrides: object) -> WhisperSyncConfig:
    if config_path and config_path.exists():
        cfg = WhisperSyncConfig.from_file(config_path)
    else:
        cfg = WhisperSyncConfig()
    cfg.merge_cli_args(**cli_overrides)
    return cfg
