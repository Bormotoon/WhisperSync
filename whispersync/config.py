"""Configuration management for WhisperSync."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir

logger = logging.getLogger(__name__)

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

# GCC-PHAT cross-correlation parameters, shared by Boundary Flex (below).
# Sharpness = peak/median(|cc|); measured speech windows score 240-335 and
# silence/mismatch ~12.
ACOUSTIC_MAX_LAG_S = 1.0
GCC_EPS = 1e-8

# Boundary Flex: acoustically refine each rendered piece's recorder start time by
# GCC-PHAT (camera audio ↔ recorder audio) so speech lands under the picture to
# sub-frame accuracy, independent of Whisper's ±50–100ms word timings. A piece's
# start is nudged only when the measurement is confident (sharpness gate) AND the
# residual exceeds a deadband (below it, the correction is within GCC noise and
# would inject jitter rather than remove it). Off by default.
FLEX_WINDOW_S = 4.0  # cross-correlation window per boundary
FLEX_MIN_SHARPNESS = 80.0  # stricter than the grid gate — small windows need a clear peak
FLEX_DEADBAND_S = 0.025  # ignore corrections within measurement noise (~25 ms)
FLEX_MAX_SHIFT_S = 0.15  # clamp any single boundary nudge

# Pause ducking: attenuate the recorder audio during inter-phrase pauses (gaps
# between speech blocks in the transcript) so a slightly mis-synced ambience/room
# tone in those gaps is inaudible. Gain is in dB (0 = no change … -inf = full
# silence); a short equal-power fade at each pause edge avoids clicks. Off by
# default; the gap that counts as a pause reuses phrase_gap_threshold.
PAUSE_DUCK_DB = -18.0
PAUSE_DUCK_FADE_MS = 80
PAUSE_DUCK_MIN_PAUSE_S = 0.6  # don't duck gaps shorter than this

# Ambience track: run a source-separation model over the camera audio to strip the
# camera's own (echoey, slightly-mis-synced) voice and keep only the room tone /
# ambience, placed on its own lane next to the clean synced voice. Off by default.
# The separator lives in the isolated ".sep-venv" environment (see separation.py);
# MelBand-RoFormer Inst V2 is the chosen model (best ambience-detail retention).
AMBIENCE_MODEL = "melband_roformer_inst_v2.ckpt"


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
    # Hybrid (3) is the recommended default out of the box — near-perfect
    # alignment at roughly half the distortion of pure stretching. This is the
    # single source of truth for the default strategy: CLI (--strategy) and GUI
    # (the pre-checked radio) both read it instead of hard-coding their own
    # value, which used to disagree (CLI defaulted to 1, GUI pre-checked 4).
    # See PROJECT_ANALYSIS.md §4.4.
    default_strategy: int = 3
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
    # Constant per-camera lip-sync calibration (milliseconds), added to every
    # synced audio clip's timeline offset. Acoustic methods (Boundary Flex,
    # the coarse text match) align recorder audio to CAMERA AUDIO — they
    # cannot see or correct a fixed mic-to-lips delay baked into a specific
    # camera's own audio pipeline (e.g. an internal ADC/encoder latency).
    # That constant is invisible to any audio-only measurement and needs a
    # one-time calibration (e.g. a clap synced by eye, frame-stepped) to
    # find. `camera_av_offset_ms` is the default for every camera;
    # `camera_av_offset_ms_by_camera` overrides it per camera sub-folder
    # name. Positive = delay the synced audio (recorder audio arrives EARLY
    # relative to this camera's lips); negative = advance it.
    camera_av_offset_ms: float = 0.0
    camera_av_offset_ms_by_camera: dict[str, float] = field(default_factory=dict)
    # Multiple recorders (different devices): "best" = one audio lane, each clip
    # synced from its best-matching recorder; "all" = every recorder on its own
    # audio lane (-1, -2, …) for multi-mic / multi-speaker setups.
    recorder_mode: str = "best"
    # Short equal-power fades at audio segment seams to declick joints (mainly
    # for the Local Time-Stretch strategy). Length-preserving, so no extra drift.
    # Only applied where a seam is not acoustically contiguous with its neighbour
    # (see pipeline._piece_seam_fades) — a fade on every piece boundary would carve
    # an audible volume dip into otherwise-continuous recorder audio.
    crossfade_enabled: bool = True
    crossfade_ms: int = 10
    # Render-path audio quality (see PROJECT_ANALYSIS.md §2.0). "auto" preserves
    # the recorder's channel count and a lossless PCM codec matching its bit depth
    # end to end, instead of the old hard-coded 16-bit mono. Only "auto" is
    # supported today; the field exists so a future explicit override is additive.
    output_audio_format: str = "auto"
    # Tempo-conform method for each rendered piece. "auto" uses a transparent
    # resample ("varispeed") for small factors (|factor-1| <= a few tenths of a
    # percent — real clock drift) and falls back to atempo (WSOLA) for larger
    # corrections; "atempo"/"resample" force one method. See
    # timestretch.RESAMPLE_CONFORM_MAX_DEVIATION.
    stretch_method: str = "auto"
    # Seam-snap-to-silence: interior piece boundaries (Local Time-Stretch / Hybrid)
    # are nudged to the nearest inter-word silence in the recorder, within this
    # many seconds, so a seam never lands mid-word — the actual fix for the
    # mid-word tempo-break stutter ("подготовил" -> "подга-га-товил"). Unlike the
    # old factor-smoothing approach (removed: it fixed the stutter by averaging
    # atempo factors, but that redistributed each piece's output length and let
    # speech drift off the picture by up to ~1.4s), this only moves WHERE the cut
    # happens — every piece's tempo factor, and hence the sync, is unchanged.
    seam_snap_max_s: float = 0.4
    # Parallelism for the CPU-bound audio render (ffmpeg has no GPU audio filters):
    # each piece / Flex window is an independent ffmpeg call, spread across a process
    # pool. 0 = auto (os.cpu_count()); 1 = sequential. Output is identical regardless.
    render_workers: int = 0
    # ffprobe timeout (seconds) when reading each media file's metadata. The
    # default is generous for local files; raise it for clips on slow
    # network/NAS storage that can legitimately take longer to respond.
    probe_timeout_s: float = 30.0
    min_anchors: int = MIN_ANCHORS
    anchor_min_confidence: float = ANCHOR_MIN_CONFIDENCE
    phrase_gap_threshold: float = PHRASE_GAP_THRESHOLD
    match_window_margin: float = MATCH_WINDOW_MARGIN
    seed_max_occurrences: int = SEED_MAX_OCCURRENCES
    seed_bin_width: float = SEED_BIN_WIDTH
    # GCC-PHAT cross-correlation parameters shared by Boundary Flex and the
    # acoustic fallback below.
    acoustic_max_lag_s: float = ACOUSTIC_MAX_LAG_S
    gcc_eps: float = GCC_EPS
    # Acoustic fallback ("Strategy 0"): when a clip can't be aligned to any
    # recorder via the transcript (too little transcribable speech — music,
    # background noise, a language Whisper garbles, near-silence), fall back
    # to a coarse GCC-PHAT cross-correlation grid scan across the whole
    # recorder span to estimate offset/K directly from the waveforms. Turns
    # "no usable words" from a hard failure into a still-working (if less
    # precise) placement, as long as the same physical audio event reaches
    # both the camera and the recorder mic. On by default; the anchors list
    # is empty for a fallback match, so clip_pieces uses one global tempo
    # conform for that clip regardless of the chosen strategy. See
    # PROJECT_ANALYSIS.md §10.2.
    acoustic_fallback: bool = True
    acoustic_fallback_grid_s: float = 30.0
    acoustic_fallback_window_s: float = 8.0
    acoustic_fallback_min_sharpness: float = 50.0
    # Boundary Flex: acoustically nudge each piece's recorder start so speech
    # lands under the picture to sub-frame accuracy. On by default — it's the
    # best-out-of-the-box lip-sync setting (the GUI pre-checked this while the
    # CLI defaulted it off; this is the single source of truth both now read).
    # Costs a little extra processing; disable for the fastest run.
    boundary_flex: bool = True
    flex_window_s: float = FLEX_WINDOW_S
    flex_min_sharpness: float = FLEX_MIN_SHARPNESS
    flex_deadband_s: float = FLEX_DEADBAND_S
    flex_max_shift_s: float = FLEX_MAX_SHIFT_S
    # Pause ducking (off by default): attenuate inter-phrase pauses by pause_duck_db
    # (0 dB = off … -inf = full silence) to hide ambience desync in gaps.
    pause_duck_enabled: bool = False
    pause_duck_db: float = PAUSE_DUCK_DB
    pause_duck_fade_ms: int = PAUSE_DUCK_FADE_MS
    pause_duck_min_pause_s: float = PAUSE_DUCK_MIN_PAUSE_S
    # Ambience track (off by default): extract voice-free camera ambience onto its own
    # lane so the only voice is the clean synced one (no doubled/echoed voice).
    ambience_track: bool = False
    ambience_model: str = AMBIENCE_MODEL
    # Render a single WAV spanning the whole timeline (every synced voice clip,
    # and the ambience track if enabled, mixed at their timeline offsets over a
    # silent bed) next to the FCPXML, for users without an NLE to drop the
    # project into. Off by default (one extra full-length render).
    render_master_wav: bool = False

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
        # Silently dropping unknown keys used to hide typos (e.g. a config
        # written with "pause_duck_dB" instead of "pause_duck_db" would just
        # never take effect, with no indication why). Warn about anything that
        # isn't a real field. See PROJECT_ANALYSIS.md §2.9.
        known = cls.__dataclass_fields__
        unknown = sorted(set(data) - set(known))
        if unknown:
            logger.warning(
                "Unknown config key(s) in %s (ignored — check for typos): %s",
                path,
                ", ".join(unknown),
            )
        return cls(**{k: v for k, v in data.items() if k in known})

    def merge_cli_args(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)


def load_config(config_path: Path | None = None, **cli_overrides: object) -> WhisperSyncConfig:
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg = WhisperSyncConfig.from_file(config_path)
    else:
        cfg = WhisperSyncConfig()
    cfg.merge_cli_args(**cli_overrides)
    return cfg
