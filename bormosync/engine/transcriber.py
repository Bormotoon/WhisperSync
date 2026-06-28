from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path

from bormosync.config import WHISPER_BEAM_SIZE, WHISPER_TEMPERATURE, BormoSyncConfig
from bormosync.models import Segment, Transcript, Word

logger = logging.getLogger(__name__)


class WhisperEngine:
    def __init__(self, config: BormoSyncConfig) -> None:
        self.config = config
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info(
            "Loading whisper model=%s device=%s compute_type=%s",
            self.config.model,
            self.config.device,
            self.config.compute_type,
        )
        self._model = WhisperModel(
            self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    @staticmethod
    def _cache_key(audio_path: Path, config: BormoSyncConfig) -> str:
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

    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Callable[[float], None] | None = None,
    ) -> Transcript:
        key = self._cache_key(audio_path, self.config)
        cache_file = self._cache_path(self.config.resolved_cache_dir, key)
        cached = self._load_cache(cache_file)
        if cached is not None:
            logger.info("Loaded transcript from cache for %s", audio_path)
            return cached

        self._ensure_model()
        assert self._model is not None

        logger.info("Transcribing %s", audio_path)
        segments_gen, info = self._model.transcribe(
            str(audio_path),
            beam_size=WHISPER_BEAM_SIZE,
            temperature=WHISPER_TEMPERATURE,
            word_timestamps=True,
            vad_filter=self.config.vad_filter,
            language=self.config.language,
        )

        total_duration = info.duration
        result_segments: list[Segment] = []

        for seg in segments_gen:
            words = [
                Word(
                    text=word.word.strip(),
                    start=word.start,
                    end=word.end,
                    probability=word.probability,
                )
                for word in (seg.words or [])
            ]
            result_segments.append(Segment(start=seg.start, end=seg.end, words=words))

            if progress_callback and total_duration > 0:
                progress = min(seg.end / total_duration, 1.0)
                progress_callback(progress)

        transcript = Transcript(
            source_path=audio_path.resolve(),
            language=info.language,
            duration=total_duration,
            segments=result_segments,
        )

        self._save_cache(cache_file, transcript)
        return transcript

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Whisper model unloaded, VRAM freed")

    def __del__(self) -> None:
        self.unload()
