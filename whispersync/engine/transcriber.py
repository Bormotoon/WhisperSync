"""Whisper-based transcription engine with caching.

Hardware/decoding settings are ported from the production Podcast Reels Forge
pipeline (RTX 5060 Ti 16GB): auto device/compute-type resolution, batched GPU
inference with an OOM fallback ladder, and the anti-hallucination decoding
controls that keep the transcript clean (which directly improves anchor quality).
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

# HF Xet protocol hangs on some networks (CLOSE-WAIT socket); force plain HTTP.
# Must be set before faster_whisper/huggingface_hub import.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from whispersync.config import WHISPER_TEMPERATURE_LADDER, WhisperSyncConfig
from whispersync.models import Segment, Transcript, Word

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch optional at import time
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# CUDA compute capability at/above which float16 is the sensible default.
_CUDA_FLOAT16_MAJOR = 7


def _ct2_cuda_available() -> bool:
    """Whether CUDA is usable by faster-whisper (ctranslate2), independent of torch."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:  # pragma: no cover - ctranslate2 is always present with faster_whisper
        return False


def resolve_device(requested: str) -> str:
    """Resolve a requested device ("auto"/"cuda"/"cpu") to an available one.

    faster-whisper runs on ctranslate2, not torch, so CUDA can be used even when
    torch is absent — we fall back to ctranslate2's own device probe.
    """
    req = str(requested).strip().lower()
    cuda = (torch is not None and torch.cuda.is_available()) or _ct2_cuda_available()
    if req == "auto":
        return "cuda" if cuda else "cpu"
    if req == "cuda":
        if cuda:
            return "cuda"
        logger.warning("CUDA not available; falling back to CPU")
    return "cpu"


def _default_compute_type(device: str) -> str:
    if device != "cuda":
        return "float32"
    if torch is None:
        # No torch to probe capability; float16 is the right default for any modern
        # CUDA GPU (verified on the RTX 5060 Ti via ctranslate2 4.8).
        return "float16"
    try:
        major, _minor = torch.cuda.get_device_capability()
    except (RuntimeError, AttributeError):
        return "float32"
    return "float16" if major >= _CUDA_FLOAT16_MAJOR else "int8_float16"


def select_compute_type(device: str, requested: str) -> str:
    """Resolve compute_type, honouring an explicit value or "auto"."""
    if requested and requested.strip().lower() != "auto":
        return requested
    return _default_compute_type(device)


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and any(k in msg for k in ("cuda", "cudnn", "cublas", "gpu"))


class WhisperEngine:
    def __init__(self, config: WhisperSyncConfig) -> None:
        self.config = config
        self._model: Any = None
        self._device: str = resolve_device(config.device)
        self._compute_type: str = select_compute_type(self._device, config.compute_type)

    @property
    def device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type

    def _load(self, device: str, compute_type: str) -> Any:
        from faster_whisper import WhisperModel

        logger.info(
            "Loading whisper model=%s device=%s compute_type=%s",
            self.config.model,
            device,
            compute_type,
        )
        return WhisperModel(self.config.model, device=device, compute_type=compute_type)

    def _cleanup_cuda(self) -> None:
        gc.collect()
        if self._device == "cuda" and torch is not None:
            with contextlib.suppress(AttributeError, RuntimeError):
                torch.cuda.empty_cache()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            self._model = self._load(self._device, self._compute_type)
            return
        except RuntimeError as exc:
            if not (self._device == "cuda" and _is_cuda_oom(exc)):
                raise
            logger.warning("CUDA OOM at model init; trying smaller compute types")
            self._cleanup_cuda()

        for ct in ("float16", "int8_float16", "int8"):
            try:
                self._model = self._load("cuda", ct)
                self._compute_type = ct
                return
            except Exception:
                self._cleanup_cuda()
        logger.warning("CUDA OOM persists; switching to CPU")
        self._device, self._compute_type = "cpu", "float32"
        self._model = self._load(self._device, self._compute_type)

    def _decode_kwargs(self) -> dict[str, Any]:
        cfg = self.config
        kwargs: dict[str, Any] = {
            "language": cfg.language,
            "word_timestamps": True,
            "vad_filter": cfg.vad_filter,
            "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 400},
            "best_of": max(1, cfg.best_of),
            "patience": max(1.0, cfg.patience),
            "temperature": list(WHISPER_TEMPERATURE_LADDER),
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "repetition_penalty": max(1.0, cfg.repetition_penalty),
            "no_repeat_ngram_size": max(0, cfg.no_repeat_ngram_size),
        }
        if cfg.initial_prompt:
            kwargs["initial_prompt"] = cfg.initial_prompt
        return kwargs

    def _run(self, audio_path: Path, batch_size: int) -> Any:
        """Start a transcription run (returns a lazy segment generator + info)."""
        kwargs = self._decode_kwargs()
        if self.config.transcribe_mode.strip().lower() == "quality":
            kwargs["beam_size"] = max(1, self.config.quality_beam_size)
            kwargs["condition_on_previous_text"] = True
            return self._model.transcribe(str(audio_path), **kwargs)

        from faster_whisper import BatchedInferencePipeline

        kwargs["beam_size"] = self.config.beam_size
        kwargs["condition_on_previous_text"] = self.config.condition_on_previous_text
        batched = BatchedInferencePipeline(model=self._model)
        try:
            return batched.transcribe(str(audio_path), batch_size=max(1, batch_size), **kwargs)
        except TypeError:
            # Older faster-whisper without the batched API: degrade gracefully.
            return self._model.transcribe(
                str(audio_path),
                beam_size=self.config.beam_size,
                **self._decode_kwargs(),
            )

    def _materialize(
        self, audio_path: Path, batch_size: int, progress_callback: Callable[[float], None] | None
    ) -> tuple[list[Segment], str, float]:
        segments_gen, info = self._run(audio_path, batch_size)
        total = float(getattr(info, "duration", 0.0) or 0.0)
        result: list[Segment] = []
        for seg in segments_gen:
            words = [
                Word(
                    text=w.word.strip(),
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                )
                for w in (seg.words or [])
            ]
            result.append(Segment(start=seg.start, end=seg.end, words=words))
            if progress_callback and total > 0:
                progress_callback(min(seg.end / total, 1.0))
        language = str(getattr(info, "language", self.config.language or "") or "")
        return result, language, total

    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Callable[[float], None] | None = None,
    ) -> Transcript:
        key = self._cache_key(audio_path, self.config)
        cache_file = self._cache_path(self.config.resolved_cache_dir, key)
        if self.config.use_cache:
            cached = self._load_cache(cache_file)
            if cached is not None:
                logger.info("Loaded transcript from cache for %s", audio_path)
                return cached

        self._ensure_model()

        # OOM degradation ladder: GPU(batch) -> GPU(batch/2) -> ... -> CPU.
        cur_batch = (
            max(1, self.config.batch_size)
            if self.config.transcribe_mode.strip().lower() != "quality"
            else 1
        )
        logger.info("Transcribing %s (mode=%s)", audio_path, self.config.transcribe_mode)
        while True:
            try:
                segments, language, total = self._materialize(
                    audio_path, cur_batch, progress_callback
                )
                break
            except RuntimeError as exc:
                if not (self._device == "cuda" and _is_cuda_oom(exc)):
                    raise
                self._cleanup_cuda()
                if cur_batch > 1:
                    cur_batch = max(1, cur_batch // 2)
                    logger.warning("CUDA OOM; retrying on GPU with batch_size=%d", cur_batch)
                    continue
                logger.warning("CUDA OOM at batch_size=1; switching to CPU")
                self._device, self._compute_type = "cpu", "float32"
                self._model = self._load(self._device, self._compute_type)

        transcript = Transcript(
            source_path=audio_path.resolve(),
            language=language,
            duration=total,
            segments=segments,
        )
        if self.config.use_cache:
            self._save_cache(cache_file, transcript)
        return transcript

    @staticmethod
    def _cache_key(audio_path: Path, config: WhisperSyncConfig) -> str:
        stat = audio_path.stat()
        parts = "|".join(
            [
                str(audio_path.resolve()),
                str(stat.st_size),
                str(stat.st_mtime),
                config.model,
                config.compute_type,
                config.language or "",
                str(config.vad_filter),
                # decoding params that change the transcript
                config.transcribe_mode,
                str(config.beam_size),
                str(config.quality_beam_size),
                str(config.best_of),
                str(config.patience),
                str(config.condition_on_previous_text),
                str(config.repetition_penalty),
                str(config.no_repeat_ngram_size),
                config.initial_prompt,
            ]
        )
        return hashlib.sha256(parts.encode()).hexdigest()

    @staticmethod
    def _cache_path(cache_dir: Path, key: str) -> Path:
        return cache_dir / f"{key}.json"

    def _load_cache(self, cache_file: Path) -> Transcript | None:
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text())
            segments = []
            for seg in data["segments"]:
                words = [
                    Word(
                        text=w["text"],
                        start=w["start"],
                        end=w["end"],
                        probability=w["probability"],
                    )
                    for w in seg["words"]
                ]
                segments.append(Segment(start=seg["start"], end=seg["end"], words=words))
            return Transcript(
                source_path=Path(data["source_path"]),
                language=data["language"],
                duration=data["duration"],
                segments=segments,
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Corrupt cache file %s, will re-transcribe", cache_file)
            return None

    def _save_cache(self, cache_file: Path, transcript: Transcript) -> None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "source_path": str(transcript.source_path),
            "language": transcript.language,
            "duration": transcript.duration,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "words": [
                        {
                            "text": w.text,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in seg.words
                    ],
                }
                for seg in transcript.segments
            ],
        }
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info("Transcript cached to %s", cache_file)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            if self._device == "cuda" and torch is not None:
                with contextlib.suppress(AttributeError, RuntimeError):
                    torch.cuda.empty_cache()
            logger.info("Whisper model unloaded, VRAM freed")

    def __del__(self) -> None:
        self.unload()
