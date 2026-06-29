# BormoSync — Advanced Audio/Video Synchronization Tool

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

**BormoSync** — это инструмент для синхронизации звука и видео при dual-system recording: камера снимает видео с черновым звуком, а внешний рекордер (петличка, Zoom, Tascam) записывает чистый звук отдельно. BormoSync автоматически находит точное временное смещение между треками и генерирует FCPXML-проект для Final Cut Pro или DaVResolve.

Принцип работы: **транскрипция** → **поиск якорей** → **вычисление K/offset** → **стратегия синхронизации** → **FCPXML-экспорт**.

Используя [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2), BormoSync транскрибирует оба аудиопотока с word-level таймстемпами, находит совпадающие слова (якоря) через последовательностное выравнивание, применяет RANSAC-регрессию для устойчивого определения линейного дрейфа часов (K = скорость, offset = смещение), а затем выбирает оптимальную стратегию синхронизации в зависимости от условий записи.

Результат — FCPXML-файл с точной привязкой аудиоклипов к видео, готовый к импорту в Final Cut Pro или DaVinci Resolve. Никакого рендеринга: всё работает через ссылки на исходные файлы.

## Features

- **3 стратегии синхронизации** — Global Linear (линейный дрейф), Local Time-Stretch (нелинейный дрейф), Silence Padding (тональность-критичный)
- **PyQt6 GUI** с dark theme, drag-and-drop зонами, визуализацией стратегий и предпросмотром таймлайна
- **CLI headless mode** — полный контроль через командную строку, JSON-вывод для автоматизации
- **Кэш транскрипций** по SHA-256 — повторный запуск без перетранскрибации
- **FCPXML export** без рендеринга — готовый проект для FCP / DaVinci Resolve
- **RANSAC-регрессия** — устойчивое определение смещения и дрейфа даже с выбросами
- **NVIDIA GPU** ускорение (CUDA/cuDNN) с CPU fallback
- **Kроссплатформенность** — Windows, macOS, Linux

## Requirements

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| Python | >= 3.10 | 3.11+ |
| GPU | CPU fallback | NVIDIA GPU (CUDA/cuDNN) |
| ffmpeg/ffprobe | В PATH | Последняя версия |
| RAM | 4 GB | 8+ GB |
| Диск | 10 GB свободных | SSD |

- **NVIDIA GPU** — рекомендуется для быстрой транскрипции. Поддерживаются GPUs с Compute Capability >= 5.0. Без GPU работает CPU fallback (медленнее, но функционально).
- **ffmpeg / ffprobe** — должны быть доступны в `PATH`. Используются для извлечения аудио, time-stretch и конкатенации.
- **CUDA / cuDNN** — требуются для GPU-режима. Установите через [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

## Installation

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/your-username/BormoSync.git
cd BormoSync

# 2. Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# 3. Установите зависимости
pip install -r requirements.txt

# 4. Проверьте окружение
python bormosync/engine/system_check.py
```

Скрипт `system_check.py` проверит наличие ffmpeg, CUDA, Python, зависимостей и свободного места на диске. Вывод — цветная таблица с результатами + файл `report.json`.

## Usage

### GUI

```bash
python main.py
```

<!-- screenshot-placeholder -->

PyQt6 GUI с dark theme:

- **Drag-and-drop** видео-папки и аудиофайла
- Выбор стратегии (3 radio-кнопки)
- Кнопка **SYNC** — запуск синхронизации
- Визуализация выбранной стратегии (диаграмма + таймлайн)
- Лог в реальном времени с цветной подсветкой

### CLI

```bash
# Базовый запуск (Strategy 1 — Global Linear)
python main.py --cli \
  --video-dir ./videos \
  --audio-file rec.wav \
  --strategy 1 \
  --output out.fcpxml

# Strategy 2 — Local Time-Stretch
python main.py --cli \
  --video-dir ./videos \
  --audio-file rec.wav \
  --strategy 2 \
  --output out.fcpxml

# Strategy 3 — Silence Padding
python main.py --cli \
  --video-dir ./videos \
  --audio-file rec.wav \
  --strategy 3 \
  --output out.fcpxml

# Dry run — только выравнивание, без обработки аудио
python main.py --cli \
  --video-dir ./videos \
  --audio-file rec.wav \
  --dry-run

# JSON-вывод для автоматизации
python main.py --cli \
  --video-dir ./videos \
  --audio-file rec.wav \
  --json

# Язык транскрипции
python main.py --cli \
  --video-dir ./videos \
  --audio-file rec.wav \
  --language ru

# Конкретная модель Whisper
python main.py --cli \
  --video-dir ./videos \
  --audio-file rec.wav \
  --model large-v3 \
  --device cpu \
  --compute-type int8
```

### CLI Options

| Флаг | Тип | Описание |
|------|-----|----------|
| `--video-dir` | Path | **Обязательный.** Папка с видеофайлами |
| `--audio-file` | Path | **Обязательный.** Файл аудио с рекордера |
| `--strategy` | int | Стратегия: `1`, `2`, `3` или `4` (по умолчанию `1`) |
| `--output` | Path | Путь для FCPXML (по умолчанию `output/sync_output.fcpxml`) |
| `--model` | str | Модель Whisper (по умолчанию `large-v3`) |
| `--device` | str | `auto` / `cuda` / `cpu` (по умолчанию `auto`) |
| `--compute-type` | str | `auto` / `float16` / `int8` и т.д. (по умолчанию `auto`) |
| `--batch-size` | int | Размер батча для GPU-инференса, главный рычаг скорости (по умолчанию `16`) |
| `--mode` | str | `fast` (батчевый) или `quality` (последовательный, точнее, ~10× медленнее) |
| `--initial-prompt` | str | Подсказка темы для смещения словаря Whisper |
| `--language` | str | Код языка (`ru`, `en`, ...) или `None` для авто |
| `--fcpxml-version` | str | Версия FCPXML (по умолчанию `1.9`) |
| `--timebase-source` | str | `camera` или `recorder` — источник sample-rate для таймкодов FCPXML |
| `--audio-source-camera` | str | Мультикам: имя подпапки-камеры, с которой берётся звук (по умолчанию авто) |
| `--audio-file` (повтор) | Path | Можно указать несколько раз — несколько рекордеров |
| `--recorder-mode` | str | `best` (одна дорожка, лучший рекордер на клип) или `all` (каждый рекордер на свою дорожку) |
| `--crossfade` / `--no-crossfade` | flag | Микро-фейды на стыках аудиосегментов (declick), по умолчанию вкл. |
| `--crossfade-ms` | int | Длина фейда в мс (по умолчанию `10`) |
| `--config` | Path | Путь к JSON-конфигу |
| `--no-cache` | flag | Отключить кэш транскрипций |
| `--dry-run` | flag | Только выравнивание, без обработки |
| `--json` | flag | Вывод в формате JSON |
| `--verbose` | flag | Подробное логирование |

### Пример вывода (default)

```
═══════════════════════════════════════════════
  BormoSync — Results
═══════════════════════════════════════════════
  Anchors found: 42
  K (drift):     1.000237
  Offset:        12.847s
  Residual:      23.4 ms
  FCPXML:        output/sync_output.fcpxml
═══════════════════════════════════════════════
```

## Strategies Guide

| Стратегия | Название | Когда использовать |
|-----------|----------|--------------------|
| **1** | Global Linear | Линейный дрейф часов (наиболее частый случай). Простая коррекция: один `atempo` коэффициент на весь файл. |
| **2** | Local Time-Stretch | Нелинейный дрейф, меняющийся темп. Сегменты между якорями растягиваются/сжимаются локально. |
| **3** | Silence Padding | Важна тональность/интонация. Речь извлекается as-is (без pitch change), тишина между сегментами добавляется/обрезается. |
| **4** | Hybrid (Global + Silence) | Универсальный, рекомендуемый. Каждая фраза корректируется глобальным `K` клипа и ставится по своему якорю, остаток поглощается тишиной. Устойчив к нелинейному дрейфу и почти без pitch-артефактов. |

> **Размещение клипов.** Каждый видеоклип выравнивается к рекордеру независимо, и его позиция на таймлайне берётся из совпавших таймкодов. Клипы не обязаны идти встык — реальные паузы между записями сохраняются (работает для любых источников).

> **Мультикамера.** Положите клипы каждой камеры в отдельную подпапку внутри `--video-dir` (например, `videos/camA/`, `videos/camB/`). Каждая камера ляжет на свою дорожку (`lane 1, 2, …`). Чистый звук синхронизируется один раз с камеры-референса (`--audio-source-camera`, по умолчанию выбирается автоматически по лучшему выравниванию), чтобы не задваиваться на углах. Длинные источники (рекордер или видео на много часов) обрабатываются через оконный матчинг — клип сначала грубо локализуется по редким словам, затем точно матчится в узком окне.

> **Несколько рекордеров.** Передайте `--audio-file` несколько раз. Каждый клип выравнивается ко всем рекордерам; таймлайн строится по «основному» (с лучшим покрытием). `--recorder-mode best` (по умолчанию) — на каждый клип берётся лучший рекордер, одна аудиодорожка; `--recorder-mode all` — каждый рекордер кладётся на свою дорожку (для нескольких петличек/спикеров). **Важно:** если это просто куски одного устройства (рекордер бьёт запись по 15 мин) — это один источник с одними часами, их надо склеить заранее (`ffmpeg concat`, без потерь), а не передавать как разные рекордеры.

### Strategy 1: Global Linear

**Применение:** Один коэффициент `atempo = 1/K` на весь аудиофайл.

```
Видео:    |========================>
Аудио:    |========================>  × atempo(1/K)
```

- Быстро (одна операция ffmpeg)
- Минимальные артефакты
- Идеально для стабильных часов

### Strategy 2: Local Time-Stretch

**Применение:** Каждый сегмент между якорями получает свой `atempo`.

```
Видео:    |=== seg1 ===|=== seg2 ===|=== seg3 ===>
Аудио:    |== seg1 ==>|==== seg2 ====|== seg3 ==>  (каждый со своим atempo)
```

- Точнее при нестабильном дрейфе
- Несколько операций ffmpeg
- На стыках сегментов применяются микро-фейды (declick), включаются/выключаются галочкой «Crossfade segment seams» в GUI или `--crossfade`/`--no-crossfade` в CLI

### Strategy 3: Silence Padding

**Применение:** Речь извлекается без изменения pitch, тишина компенсирует смещение.

```
Видео:    |== речь ==| тишина |== речь ==| тишина |== речь ==>
Аудио:    |== речь ==|silence |== речь ==|silence |== речь ==>
```

- Сохраняет оригинальную тональность
- Подходит для интервью, подкастов
- Предупреждение при отрицательных gaps (перекрытие речи)

### Strategy 4: Hybrid (Global + Silence)

**Применение:** Каждая фраза корректируется глобальным `K` клипа (`atempo = 1/K`)
и ставится по своему якорю; промежутки между фразами поглощают остаток дрейфа.

```
Видео:    |== фраза ==| пауза |== фраза ==| пауза |== фраза ==>
Аудио:    |=×(1/K)===| silence|=×(1/K)===| silence|=×(1/K)==>
```

- Снимает линейный дрейф (как Strategy 1) и доводит по фразам (как Strategy 3)
- Устойчив к нелинейному дрейфу, pitch-сдвиг ≈ 0.1% (неслышимо)
- Рекомендуемый режим по умолчанию для длинных разговорных записей

## Configuration

### JSON Config

BormoSync поддерживает JSON-конфигурацию через `--config config.json`:

```json
{
    "model": "large-v3",
    "device": "auto",
    "compute_type": "auto",
    "language": "auto",
    "vad_filter": true,
    "beam_size": 5,
    "batch_size": 16,
    "best_of": 1,
    "patience": 1.0,
    "condition_on_previous_text": false,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "transcribe_mode": "fast",
    "quality_beam_size": 10,
    "initial_prompt": "",
    "video_exts": [".mp4", ".mov", ".mxf", ".avi", ".mkv"],
    "audio_exts": [".wav", ".mp3", ".m4a", ".flac"],
    "fcpxml_version": "1.9",
    "default_strategy": 1,
    "cache_dir": null,
    "output_dir": null,
    "use_cache": true,
    "timebase_source": "camera",
    "recorder_mode": "best",
    "crossfade_enabled": true,
    "crossfade_ms": 10,
    "min_anchors": 8,
    "anchor_min_confidence": 0.6,
    "phrase_gap_threshold": 0.6
}
```

**Приоритет:** CLI-флаги > JSON-конфиг > значения по умолчанию.

### Описание полей

| Поле | Тип | Описание |
|------|-----|----------|
| `model` | str | Модель Whisper (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `device` | str | Устройство: `cuda` или `cpu` |
| `compute_type` | str | Тип вычислений: `float16`, `int8`, `float32` |
| `language` | str/null | Код языка или `null` для автоопределения |
| `vad_filter` | bool | Фильтрация VAD (Voice Activity Detection) |
| `min_anchors` | int | Минимальное количество якорей (по умолчанию 8) |
| `anchor_min_confidence` | float | Минимальная уверенность якоря (0.0–1.0) |

## Troubleshooting

### CUDA not found

```
CUDA not available, falling back to CPU
```

**Решение:** Установите [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) и cuDNN. Проверьте совместимость версий PyTorch и CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### ffmpeg not in PATH

```
ffmpeg not found in PATH
```

**Решение:** Установите ffmpeg и добавьте в PATH:

- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Windows:** Скачайте с [ffmpeg.org](https://ffmpeg.org/download.html) и добавьте в PATH

### Мало якорей (low anchor count)

```
Warning: Only 3 anchors found (minimum: 8)
```

**Причины и решения:**
- Короткое видео — увеличьте длительность записи
- Много шума — включите `vad_filter: true`
- Разные языки — укажите `--language`
- Тихая речь — проверьте уровень сигнала на рекордере

### Большой residual

```
Residual: 156.2 ms (high — results may be inaccurate)
```

**Причины и решения:**
- Не идеально линейный дрейф — попробуйте Strategy 2
- Мало якорей — убедитесь что якори распределены по всей длительности
- Ошибки транскрипции — попробуйте модель `large-v3` вместо `small`

### Кэш транскрипций

Кэш хранится в `~/.cache/bormosync/` (Linux) или аналогичном каталоге. Для сброса:

```bash
# Через CLI
python main.py --cli --no-cache --video-dir ./videos --audio-file rec.wav

# Или удалите кэш вручную
rm -rf ~/.cache/bormosync/
```

## Architecture

```
BormoSync/
├── main.py                          # Entry point (GUI / CLI dispatch)
├── bormosync/
│   ├── cli.py                       # argparse CLI interface
│   ├── config.py                    # BormoSyncConfig dataclass + JSON loader
│   ├── models.py                    # Word, Segment, Transcript, Anchor, SyncPlan...
│   ├── engine/
│   │   ├── pipeline.py              # End-to-end orchestration
│   │   ├── transcriber.py           # WhisperEngine + SHA-256 cache
│   │   ├── matcher.py               # Anchor finding + RANSAC regression
│   │   ├── strategies.py            # 3 sync strategies
│   │   ├── timestretch.py           # ffmpeg atempo/segment/crossfade wrappers
│   │   ├── media.py                 # ffprobe, audio extraction, atempo chain
│   │   ├── export.py                # FCPXML generation
│   │   └── system_check.py          # Environment validation
│   └── gui/
│       ├── main_window.py           # PyQt6 MainWindow
│       ├── worker.py                # QObject background worker
│       ├── theme.qss                # Dark theme stylesheet
│       └── widgets/                 # DropZone, LogView, StrategyDiagram, TimelinePreview
└── tests/                           # pytest test suite
```

### Поток данных

```
Video files + Audio file
        │
        ▼
   probe() ──────────► MediaInfo
        │
        ▼
   extract_audio_to_wav() ──► 16kHz mono WAV
        │
        ▼
   WhisperEngine.transcribe() ──► Transcript (word-level timestamps)
        │
        ▼
   matcher.align() ──► find_anchors() → ransac_linear_fit() → AlignmentMap
        │
        ▼
   Strategy.plan() ──► SyncPlan (clips + audio_ops)
        │
        ▼
   Pipeline executes audio_ops ──► Processed audio segments
        │
        ▼
   generate_fcpxml() ──► .fcpxml (Final Cut Pro / DaVinci Resolve)
```

## License

MIT License — Copyright (c) 2024-2025 BormoSync Contributors.

See [LICENSE](LICENSE) for details.
