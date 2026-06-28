# 🤖 CLAUDE CODE SYSTEM PROMPT & ARCHITECTURE MANIFEST
**Project:** BormoSync — Advanced Audio/Video Synchronization Tool
**Role:** You are an expert Python Audio/Video Engineer and PyQt6 GUI Developer.
**Status:** Greenfield. Этот файл — единый источник правды (single source of truth) по архитектуре и плану.

---

## 🎯 ИНСТРУКЦИИ ДЛЯ АГЕНТА (READ FIRST)
1. **Итеративность:** Не пиши весь проект сразу. Выполняй задачи строго по фазам (Phase). Завершив фазу — дойди до маркера `[ОСТАНОВКА]`, выведи короткий отчёт и запроси подтверждение перед следующей фазой.
2. **Самопроверка:** Ты в среде выполнения. Любой написанный скрипт (`system_check.py`, тесты, демо) ОБЯЗАН быть запущен через терминал; читай `stdout`/`stderr` и чини ошибки до зелёного результата.
3. **Локальность и приватность:** Вся обработка аудио/видео — СТРОГО локально. Никаких облачных API для транскрибации/аналитики/телеметрии. Единственное допустимое сетевое обращение — однократная загрузка весов модели Whisper (с поддержкой полностью офлайн-режима при заранее скачанной модели).
4. **Структура:** Пакет `bormosync/`, логика разделена на `engine/` (вычисления) и `gui/` (интерфейс). Модульный, тестируемый код; бизнес-логика не зависит от Qt.
5. **Качество кода:** Type hints везде; форматирование `black`, линт `ruff`, типы `mypy`. Публичные функции — с docstring. Без «магических чисел» — выноси в `config.py`.
6. **Идемпотентность и кэш:** Транскрибация дорогая (GPU). Кэшируй результаты по хэшу входа; повторный прогон не должен пересчитывать то, что не менялось.
7. **Без деструктива:** Инструмент НЕ рендерит и НЕ перезаписывает исходные медиа. Выход — только `.fcpxml` (+ опционально новые аудиофайлы в отдельной папке).

---

## 🧠 CONTEXT / ПОСТАНОВКА ЗАДАЧИ (зачем это всё)

**Двухсистемная запись звука (dual-system sound):** видео снимается на камеру (с черновым «scratch» звуком встроенного микрофона), а чистый звук пишется отдельно на диктофон/рекордер (петличка, Zoom/Tascam, смартфон). В пост-продакшене чистый звук нужно точно совместить с видео.

**Проблема дрейфа часов (clock drift):** у камеры и рекордера разные кварцевые генераторы, реальная частота дискретизации немного отличается от номинальной (например, 48000 Гц против фактических ~47995 Гц). Простой сдвиг (offset) выравнивает только начало; за десятки минут набегает рассинхрон в секунды → нужна коррекция скорости (time-stretch), а не только смещение.

**Почему через транскрипцию, а не waveform cross-correlation (как PluralEyes):** корреляция волновых форм отлично ловит короткий стартовый сдвиг, но при длинном дрейфе и сильно разном характере дорожек (камерный микрофон vs петличка, разный шум/АЧХ) плывёт. Совпадающие *слова* дают устойчивые смысловые «якоря» по всей длине записи и работают даже когда дорожки звучат по-разному. (Waveform-корреляция остаётся отличным fallback/уточнением — см. Backlog.)

**Что делает BormoSync:** транскрибирует обе дорожки с word-level таймкодами → находит якоря → строит отображение времени `t_cam = offset + K · t_rec` → одной из трёх стратегий приводит звук диктофона к таймлайну камеры → выгружает `.fcpxml` (видео на `lane 1`, синхронизированный звук на `lane -1`), без рендеринга.

---

## 📖 ГЛОССАРИЙ
- **Anchor (якорь):** пара совпавших слов `(cam_word, rec_word)` с таймкодами; опора для построения отображения времени.
- **Drift coefficient `K`:** наклон линейной зависимости `t_cam = offset + K·t_rec`. `K=1` — дрейфа нет; `K≠1` — рекордер идёт быстрее/медленнее камеры.
- **`offset`:** глобальный стартовый сдвиг рекордера относительно камеры (сек).
- **VAD (Voice Activity Detection):** детектор речи/пауз (в `faster-whisper` — Silero VAD), используется для нарезки по паузам.
- **Word-level timestamp:** таймкод начала/конца каждого слова.
- **FCPXML:** XML-формат проекта Final Cut Pro. `spine` — основной «хребет» таймлайна; `gap` — пустой клип-подложка; `lane` — слой (положительные — над основным, отрицательные — под/аудио); `offset` — позиция клипа на таймлайне; `start`/`duration` — точка входа и длительность.
- **Scratch audio:** черновой звук, записанный камерой.
- **NLE:** Non-Linear Editor (Final Cut Pro; FCPXML также читает DaVinci Resolve).

---

## 🏗️ ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ И СТЕК
* **Язык:** Python >= 3.10 (рекомендуется 3.11/3.12).
* **Движок транскрибации:** `faster-whisper` (CUDA, `compute_type="float16"`, `word_timestamps=True`, `vad_filter=True`). Бэкенд `ctranslate2` (нужны cuDNN/cuBLAS). Модель: `large-v3` (точность) или `medium`/`distil-large-v3` (скорость). Fallback на CPU (`compute_type="int8"`) — медленно, но работает.
* **Медиа-процессинг:** `ffmpeg-python` + бинарник `ffmpeg`/`ffprobe` в `PATH`; `pydub`. Декодирование аудио в `faster-whisper` идёт через `av` (PyAV).
* **Числа:** `numpy` (регрессия, RANSAC-подобный отбор инлаеров).
* **GUI Фреймворк:** `PyQt6`. Строгий Dark Theme, агрессивный дизайн: фон `#0A0A0A`, акценты — красный `#D32F2F` и огненно-оранжевый `#FF5722`.
* **Результат:** Экспорт в `.fcpxml` (по умолчанию v1.9; версия — параметр, поддержать 1.10/1.11) **без рендеринга медиа**.
* **Качество time-stretch:** предпочтительно `ffmpeg atempo` (сохраняет высоту тона) или `rubberband` (opt); учесть, что `pydub.speedup` меняет высоту тона и непригоден для речи.
* **Dev-инструменты:** `pytest`, `ruff`, `black`, `mypy`; опц. `pre-commit`, `rich`/`tqdm` (CLI-прогресс), `platformdirs` (пути кэша/конфига), `pydantic` (валидация конфига).
* **Платформа:** Final Cut Pro — только macOS, но сам BormoSync кроссплатформенный (Win/Linux/macOS с NVIDIA GPU): он лишь генерирует `.fcpxml`, который открывают на Mac.

---

## 🗂️ СТРУКТУРА ПРОЕКТА (целевое дерево)
```
BormoSync/
├── bormosync/
│   ├── __init__.py
│   ├── config.py            # настройки, дефолты, пути (cache/output)
│   ├── models.py            # dataclasses: Word, Segment, Transcript, Anchor, AlignmentMap, MediaClip, SyncPlan, SyncResult
│   ├── logging_setup.py     # единая настройка логирования (файл + GUI)
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── system_check.py  # проверка окружения, JSON-отчёт
│   │   ├── media.py         # ffprobe-метаданные, извлечение аудио, file:// URI, atempo-хелперы
│   │   ├── transcriber.py   # WhisperEngine + кэш транскрипций
│   │   ├── matcher.py       # нормализация, anchor matching, отбор инлаеров, регрессия K/offset
│   │   ├── timestretch.py   # цепочка atempo / rubberband-обёртка
│   │   ├── strategies.py    # SyncStrategy (база) + 3 стратегии
│   │   ├── export.py        # генератор FCPXML
│   │   └── pipeline.py      # оркестрация end-to-end (с сигналами прогресса)
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── worker.py        # QObject-воркер (moveToThread) + сигналы
│   │   ├── theme.qss
│   │   └── widgets/
│   │       ├── drop_zone.py       # drag&drop папки видео и файла диктофона
│   │       ├── strategy_diagram.py# схема выбранной стратегии (QGraphicsView)
│   │       ├── timeline_preview.py# превью раскладки клипов (opt)
│   │       └── log_view.py        # цветной лог
│   └── cli.py               # argparse, headless-режим
├── tests/
│   ├── test_matcher.py
│   ├── test_strategies.py
│   ├── test_timestretch.py
│   ├── test_export.py
│   └── fixtures/            # синтетические данные и мини-медиа
├── main.py                  # точка входа: GUI по умолчанию, --cli → CLI
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml           # метаданные, entry points, конфиг ruff/black/mypy
├── bormosync.spec           # PyInstaller
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

---

## 🧱 МОДЕЛЬ ДАННЫХ (`bormosync/models.py`)
Единые структуры, которыми обмениваются модули (dataclasses):
```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

@dataclass
class Word:
    text: str
    start: float            # секунды
    end: float
    probability: float
    norm: str = ""          # нормализованный токен (для матчинга)

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
    # words -> плоский список всех слов (property)

@dataclass
class Anchor:
    cam_time: float         # время на таймлайне камеры (master)
    rec_time: float         # время на рекордере
    token: str
    confidence: float

@dataclass
class AlignmentMap:
    anchors: list[Anchor]
    offset: float           # t_cam = offset + K * t_rec
    k: float
    residual_ms: float      # оценка остаточной ошибки выравнивания
    # rec_to_cam(t) -> offset + k * t

@dataclass
class MediaClip:
    path: Path
    kind: Literal["video", "audio"]
    offset: float           # позиция на таймлайне (сек)
    in_point: float         # точка входа в исходник (сек)
    duration: float         # длительность на таймлайне (сек)
    lane: int               # 1 для видео, -1 для аудио диктофона

@dataclass
class SyncPlan:
    strategy_id: int
    clips: list[MediaClip]      # что и куда укладывать
    audio_ops: list[dict]       # инструкции обработки (atempo/паддинг/нарезка)
    total_duration: float

@dataclass
class SyncResult:
    fcpxml_path: Path
    alignment: AlignmentMap
    plan: SyncPlan
    anchors_used: int
    warnings: list[str]
```

---

## ⚙️ КОНФИГ (`bormosync/config.py`)
Дефолты + загрузка из файла/CLI. Пример (через `--config` или `bormosync.json`):
```jsonc
{
  "model": "large-v3",          // large-v3 | medium | distil-large-v3
  "device": "cuda",             // cuda | cpu
  "compute_type": "float16",    // float16 | int8_float16 | int8
  "language": null,             // null = автоопределение, или "ru"/"en"
  "vad_filter": true,
  "video_exts": [".mp4", ".mov", ".mxf", ".avi", ".mkv"],
  "audio_exts": [".wav", ".mp3", ".m4a", ".flac"],
  "fcpxml_version": "1.9",
  "default_strategy": 1,
  "cache_dir": null,            // null = platformdirs user cache
  "output_dir": null,
  "min_anchors": 8,             // ниже порога → предупреждение/fallback
  "anchor_min_confidence": 0.6
}
```
- [ ] Приоритет источников настроек: CLI-флаги > файл конфига > дефолты.
- [ ] Пути кэша/вывода — через `platformdirs` (кроссплатформенно).

---

## 🔁 СКВОЗНЫЕ ПРИНЦИПЫ (cross-cutting)
- [ ] **Логирование:** единый `logging_setup.py`; уровни; вывод в файл и в GUI-лог; в начале каждого прогона логировать все параметры (воспроизводимость).
- [ ] **Кэш транскрипций:** ключ = хэш от `(path, size, mtime, model, compute_type, language, vad)`; значение = JSON транскрипта. Повторный прогон с тем же входом не зовёт GPU.
- [ ] **Обработка ошибок:** ранняя валидация (наличие `ffmpeg`, файлов, GPU); понятные сообщения; GUI-поток никогда не падает (ошибки — через сигнал).
- [ ] **Приватность:** ноль сети при работе (кроме разовой загрузки модели); офлайн-режим; без телеметрии.
- [ ] **Детерминизм:** фиксировать параметры декодирования Whisper (beam_size, temperature) для стабильности результатов.
- [ ] **Производительность:** аудио для Whisper — 16 кГц моно; переиспользовать декодированный звук; чанкование длинных файлов; освобождать VRAM после прогона.

---

## 🚀 ROADMAP (TODO LIST)

### Phase 1: Environment & Scaffolding
- [x] Создать и активировать виртуальное окружение `venv`.
- [x] Создать `requirements.txt` (`faster-whisper`, `pydub`, `ffmpeg-python`, `PyQt6`, `numpy`) и `requirements-dev.txt` (`pytest`, `ruff`, `black`, `mypy`); зафиксировать версии; установить.
- [x] Развернуть дерево пакета `bormosync/` (`engine/`, `gui/`, `gui/widgets/`), `tests/`, `main.py` с пустыми модулями-заглушками.
- [x] Добавить `pyproject.toml` (метаданные, entry points `bormosync`/`bormosync-gui`, конфиг `ruff`/`black`/`mypy`), `.gitignore` (venv, `__pycache__`, кэш, сборки, `*.fcpxml` в output), скелеты `README.md`, `LICENSE`, `CHANGELOG.md`.
- [x] Написать `engine/system_check.py`. Скрипт проверяет:
   - Наличие `ffmpeg` и `ffprobe` в `PATH` (+ вывод версии).
   - `torch.cuda.is_available()` (критично) + имя GPU и объём VRAM.
   - Доступность бэкенда `ctranslate2`/cuDNN (пробная инициализация крошечной модели или явная проверка).
   - Версию Python, свободное место на диске под кэш модели.
   - Формирует человекочитаемый отчёт **и** машинный `report.json` (коды/булевы флаги).
- [x] **[ОСТАНОВКА]** Запусти `system_check.py` в терминале. Убедись, что всё зелёное. Выведи отчёт.
- **DoD:** окружение ставится с нуля по `requirements*.txt`; `system_check.py` проходит и печатает корректный отчёт; `ruff`/`black`/`mypy` на скелете без ошибок.

### Phase 2: Core Engine — Transcription & Alignment
**2.1 `engine/media.py`**
- [x] `ffprobe`-обёртка: длительность, fps (точная рациональная, напр. `30000/1001`), кодеки, число аудиоканалов, sample rate.
- [x] Извлечение аудио из видео через `ffmpeg-python` (16 кГц моно WAV/PCM — для Whisper; во временный файл или в память).
- [x] Хелпер `file://` URI с корректным percent-encoding путей/пробелов (для FCPXML).

**2.2 `engine/transcriber.py`**
- [x] Класс `WhisperEngine`: инициализация модели (`large-v3`/`medium`), параметры из конфига; ленивая загрузка; выгрузка/`del` для освобождения VRAM.
- [x] Метод транскрибации файла → `Transcript` с word-level таймкодами (`word_timestamps=True`, `vad_filter=True`).
- [x] Прогресс: итерировать ленивый генератор сегментов, считать прогресс от `info.duration` (callback/сигнал).
- [x] Кэш: до запуска GPU проверить кэш по хэшу входа; сохранять JSON-транскрипт.

**2.3 `engine/matcher.py`**
- [x] Нормализация токенов: lower-case, убрать пунктуацию, числа→слова (для одного языка), отбросить слова с `probability < anchor_min_confidence`.
- [x] Поиск якорей: выравнивание последовательностей слов (`difflib.SequenceMatcher` или Needleman–Wunsch), а не одиночные совпадения; предпочитать **редкие/уникальные** слова (лучше различимы).
- [x] Фильтры: монотонность таймкодов (время растёт), удаление дублей, минимальная плотность якорей по всей длине.
- [x] Регрессия: линейная аппроксимация `t_cam = offset + K·t_rec` (`numpy.polyfit`/lstsq) с RANSAC-подобным отбором инлаеров (отсев ошибочных совпадений по остаткам/медиане).
- [x] Вернуть `AlignmentMap` (offset, K, anchors, `residual_ms`). Если якорей < `min_anchors` или остаток велик — предупреждение и сигнал к fallback (см. Backlog: waveform-корреляция).
- [x] **[ОСТАНОВКА]** Прогон на тест-паре: вывести K, offset, число якорей, остаточную ошибку (мс).
- **DoD:** на синтетике с заданными `offset`/`K`/шумом matcher восстанавливает их в пределах допуска; кэш работает (второй прогон без GPU).

### Phase 3: The Three Synchronization Strategies
Создать `engine/strategies.py` с базой `SyncStrategy` (метод `plan(cam, rec, alignment, rec_audio_meta) -> SyncPlan`) и тремя реализациями. Хелперы time-stretch — в `engine/timestretch.py`.

**Связь дрейфа и atempo (важно):** из `t_cam = offset + K·t_rec` отрезок рекордера `Δrec` должен на таймлайне камеры занять `K·Δrec`. Фильтр `atempo=p` меняет длительность как `new = old/p`, значит для приведения **множитель `p = 1/K`**. (Проверка: рекордер «отстал», `K>1` → нужно растянуть звук → `p<1` → замедление. ✔)

1. **Global Linear Calibration**
   - [x] Взять инлаер-якоря → регрессия → один глобальный `K`, `offset`.
   - [x] Один `atempo`-проход на весь файл диктофона; разложить `p=1/K` в цепочку фильтров, если `p` вне `[0.5, 2.0]` (`atempo` чувствителен к диапазону — цепочка `atempo=a,atempo=b,...` всегда переносима).
   - [x] Результат: один обработанный аудиоклип на `offset`.
   - **Плюсы:** просто, без швов; **минусы:** предполагает линейный дрейф (обычно так и есть).

2. **Local Time-Stretch**
   - [x] Нарезать звук диктофона по паузам (VAD из Whisper).
   - [x] Для каждого сегмента взять охваченные якоря (или интерполировать локальный `K_i` от соседей/глобального) → целевые `start/duration` на таймлайне камеры.
   - [x] Растянуть/сжать каждый сегмент до целевой длительности. **Качество:** использовать `ffmpeg atempo` (сохраняет высоту тона) или `rubberband` (opt); `pydub.speedup` НЕ применять для речи (плывёт тон). Микро-кроссфейды 5–10 мс на стыках, привязка к нулевым пересечениям.
   - [x] Уложить сегменты по их целевым `offset`.
   - **Плюсы:** ловит нелинейный дрейф; **минусы:** возможны швы/артефакты.

3. **Silence Padding** (pitch-safe)
   - [x] Нарезать по фразам (VAD). Сегменты речи оставить **нетронутыми** (без ресемпла и изменения тона).
   - [x] Менять ТОЛЬКО длительность тишины между фразами: вычислить нужную паузу так, чтобы следующая фраза стартовала в своём cam-времени; добавлять/удалять миллисекунды подложки.
   - [x] Если требуемая пауза отрицательна (наложение) — обрезать тишину, при наложении речи — предупредить или локально слегка растянуть (fallback).
   - [x] Результат: множество мелких аудиоклипов на точных `offset` (без ресемпла) ИЛИ пересобранный WAV с правкой пауз.
   - **Плюсы:** идеальная тональность речи (ноль pitch-артефактов); **минусы:** корректирует лишь на границах фраз, нужны паузы.

- [x] Каждая стратегия отдаёт также метаданные для GUI-диаграммы (как схематично рисовать метод).
- **DoD:** для каждой стратегии запланированные `offset` отображают rec-время в ожидаемое cam-время в пределах ≤ 1 кадра (или ≤ ~40 мс) на всей длине (проверка на синтетике).

### Phase 4: FCPXML Generator
- [ ] `engine/export.py`: генерация через `xml.etree.ElementTree`; версия FCPXML — параметр (дефолт `1.9`), корректный `DOCTYPE`.
- [ ] Хелпер `to_rational(seconds, timebase)`: все тайм-значения — точные рациональные строки `"N/Ds"` (видео — по сетке кадров из fps; аудио — по sample rate, напр. timebase 48000), со снапом на сетку, иначе FCP округлит/отвергнет.
- [ ] `resources`: `format` (с `frameDuration`, напр. `1001/30000s`, шириной/высотой), `asset` для каждого видео и аудио + `media-rep kind="original-media" src="file:///…"`.
- [ ] `library → event → project → sequence(format, tcStart, tcFormat, duration) → spine`.
- [ ] Логика таймлайна:
   - `spine` содержит пустой `gap` (подложку) на всю длину записи.
   - Исходные видеофайлы — в `lane="1"` с точными `offset`/`start`/`duration`.
   - Аудио диктофона (нарезанное/обработанное) — в `lane="-1"` с точными `offset`.
- [ ] Уникальные `id` ресурсов; корректные `file://` URI.
- [ ] Валидация: well-formed XML; при наличии — проверка по FCPXML DTD; round-trip парсинг обратно и сверка тайм-значений.
- [ ] **[ОСТАНОВКА]** Сгенерировать `.fcpxml` из тестового `SyncPlan`; распарсить обратно; вывести сводку (число клипов, общая длительность).
- **DoD:** сгенерированный `.fcpxml` импортируется в Final Cut Pro / DaVinci Resolve; видео на `lane 1`, синхронизированный звук на `lane -1`; рассинхрон в пределах допуска по всей длине.

### Phase 5: Graphical User Interface (PyQt6)
Создать `gui/main_window.py`, `gui/worker.py`, `gui/theme.qss`, `main.py`.
- [ ] **Layout:** `QSplitter` — левая панель: Drag&Drop зоны (папка с видео + файл диктофона), список найденных клипов с метаданными; правая панель: визуализатор стратегии, лог, прогресс.
- [ ] **Стилизация:** вынести QSS в `theme.qss` (фон `#0A0A0A`, панели `#141414`, красные кнопки `#D32F2F`, оранжевые бордеры `#FF5722`, текст `#EEEEEE`, приглушённый `#888`).
- [ ] **Drag&Drop:** виджет с `dragEnterEvent`/`dropEvent`; принимать папки и файлы; валидировать расширения по конфигу.
- [ ] **Выбор стратегии:** три `QRadioButton` (Strategy 1/2/3).
- [ ] **Визуализация (КРИТИЧНО):** при переключении стратегий правый информационный виджет (`QGraphicsView`/`QLabel` с иконками) схематично показывает метод:
   - Strategy 1 — один длинный блок, равномерно масштабируемый.
   - Strategy 2 — несколько блоков, каждый сжимается/растягивается отдельно.
   - Strategy 3 — блоки речи фиксированы, меняются только промежутки-паузы.
   - (opt) лёгкая анимация перехода.
- [ ] **Поток:** запуск `pipeline` в фоне через паттерн QObject-воркер + `moveToThread` (не наследовать `QThread.run`); сигналы `progress(int)`, `stage(str)`, `log(str)`, `finished(SyncResult)`, `error(str)`; флаг отмены.
- [ ] **Прогресс/лог:** `QProgressBar` + статус этапа; цветной лог (autoscroll, цвет по уровню).
- [ ] **Settings dialog (opt):** модель, device, compute_type, язык, FCPXML-версия, папки кэша/вывода.
- [ ] **Результат:** показать число якорей, `K`, остаточную ошибку (мс), кнопки «Открыть папку»/«Reveal».
- [ ] **Persist:** запоминать последние пути/настройки через `QSettings`.
- **DoD:** перетащил папку видео + файл диктофона → выбрал стратегию (диаграмма обновилась) → запустил → UI не виснет, прогресс идёт, на выходе валидный `.fcpxml`; отмена работает.

### Phase 6: CLI & Packaging
- [ ] `bormosync/cli.py` + `main.py`: по умолчанию запуск GUI; при `--cli` — headless. Аргументы: `--cli`, `--video-dir`, `--audio-file`, `--strategy {1,2,3}`, `--output`, `--model`, `--device`, `--compute-type`, `--language`, `--fcpxml-version`, `--config`, `--cache/--no-cache`, `--dry-run`, `--json` (машинный отчёт), `--verbose`.
- [ ] Headless-прогресс в stdout (`rich`/`tqdm`), осмысленные коды возврата; пример вызова в `--help` и README.
- [ ] Поддержка конфиг-файла (`--config`), слияние с дефолтами; CLI перекрывает файл.
- [ ] `bormosync.spec` для `PyInstaller`: hidden imports (`ctranslate2`, `faster_whisper`, `av`, `tokenizers`); сбор data-файлов `faster_whisper`; учёт CUDA-библиотек (cuBLAS/cuDNN) — для CUDA рекомендовать `--onedir`; локация/бандл `ffmpeg`; модель грузится в кэш при первом запуске.
- [ ] **[ОСТАНОВКА]** Прогнать полный сценарий из CLI на мини-наборе; приложить команду и вывод.
- **DoD:** `python main.py --cli --video-dir … --audio-file … --strategy 1 --output out.fcpxml` отрабатывает end-to-end; `.spec` собирается (хотя бы `--onedir`).

### Phase 7: Testing & QA
- [ ] Юнит-тесты на синтетике:
   - `test_matcher.py`: два списка слов с известными `offset`/`K`/шумом → проверка восстановления и числа якорей.
   - `test_strategies.py`: для каждой стратегии — `offset` клипов отображают rec→cam в пределах допуска.
   - `test_timestretch.py`: разложение `atempo` (произведение факторов ≈ цель, каждый в `[0.5, 2.0]`).
   - `test_export.py`: генерация → парсинг обратно → сверка структуры/времён; (opt) валидация по DTD; golden-file.
- [ ] Интеграционный тест: мини-медиа на несколько секунд (генерировать `ffmpeg`-ом синус + TTS или хранить фикстуры) → end-to-end → валидный `.fcpxml`.
- [ ] (opt) CI (GitHub Actions): без GPU — быстрые CPU/мок-тесты Whisper; `ruff`/`black`/`mypy` как гейты.
- [ ] Ручная приёмка: импорт `.fcpxml` в FCP/DaVinci Resolve, проверка синхрона на слух/по волновой форме.
- **DoD:** `pytest` зелёный; покрыты matcher, стратегии, timestretch, export.

### Phase 8: Docs & Release
- [ ] `README.md`: что/зачем, требования (GPU/ffmpeg), установка, использование (GUI + CLI), гайд «какую стратегию когда» (1 — линейный дрейф; 2 — нелинейный; 3 — важна тональность речи и есть паузы), troubleshooting (CUDA/cuDNN/ffmpeg), скриншоты.
- [ ] `CHANGELOG.md` (Keep a Changelog), версионирование (SemVer), `LICENSE`.
- [ ] В приложении — tooltips/подсказки; демо-проект/сэмпл.
- **DoD:** новый пользователь по README ставит и запускает оба режима без устных пояснений.

---

## 🧊 BACKLOG / FUTURE IDEAS (после MVP; не блокируют релиз)
- [ ] **Waveform cross-correlation** (FFT) как Strategy 0 / fallback / уточнение якорей, когда транскрипт беден (музыка/шум/мало речи).
- [ ] **Auto-strategy:** анализ остатков линейной аппроксимации (высокий R² → Strategy 1; структурный остаток → 2; хорошие паузы + важен тон → 3).
- [ ] **Multi-recorder / multi-cam:** несколько аудиодорожек/лейнов, несколько камер.
- [ ] **Экспорт субтитров** SRT/VTT — транскрипт уже есть, почти бесплатно.
- [ ] **Другие форматы NLE:** OpenTimelineIO (OTIO) как универсальный обмен, Premiere XML/AAF, EDL; нативный профиль DaVinci Resolve.
- [ ] **«Render synced WAV» (opt):** запекать единый синхронный аудиофайл для не-FCP пользователей.
- [ ] **Loudness-нормализация** (EBU R128) и/или локальный denoise (RNNoise) как опции (это уже рендер — выносить аккуратно).
- [ ] **Ручная правка якорей** в GUI: волновая форма + перетаскивание якорей/ручной offset.
- [ ] **Speaker diarization:** сопоставление по говорящему.
- [ ] **Очередь/batch** прогонов с возобновлением; **plugin-архитектура** стратегий.
- [ ] **Дистрибуция:** macOS `.app` (+ нотаризация), Windows-инсталлятор (Inno Setup), i18n GUI (Qt translations).
- [ ] **Кусочно-линейная** модель дрейфа и взвешенная по confidence регрессия.

---

## ⚠️ RISKS & MITIGATIONS
- **CUDA/cuDNN-боль при установке** → надёжный `system_check` + раздел troubleshooting + CPU-fallback (`int8`).
- **Плохой транскрипт камерного scratch-звука → мало якорей** → fallback на waveform-корреляцию + ручной offset.
- **Мало/нет совпадающих слов** (музыка, тишина) → fallback; явное предупреждение.
- **Артефакты time-stretch** → `atempo`/`rubberband`, кроссфейды; Strategy 3 как pitch-safe вариант.
- **Несовпадение версии FCPXML с FCP пользователя** → версия-параметр + валидация + round-trip тест.
- **Баги кодирования путей/URI** → отдельные тесты `file://`/percent-encoding.
- **Длинные файлы → память** → чанкование транскрибации, потоковая обработка.

---

## ✅ DEFINITION OF DONE (общий критерий MVP)
Дана папка клипов камеры + один файл диктофона → BormoSync (GUI и CLI) строит валидный `.fcpxml`, который при импорте в FCP/DaVinci Resolve показывает видео на `lane 1` и синхронизированный звук диктофона на `lane -1`, с рассинхроном ≤ 1 кадра (≈ ≤ 40 мс) по всей длине для каждой из 3 стратегий; `system_check` проходит; `pytest` зелёный; исходные медиа не изменяются.

---

## ❓ OPEN QUESTIONS (уточнить по ходу)
- [ ] Целевая версия FCP/FCPXML у пользователя (1.9 vs 1.10/1.11)?
- [ ] Гарантирован ли единый язык речи на обеих дорожках (для нормализации/чисел)?
- [ ] Несколько видеоклипов камеры — встык по таймкодам или раскладывать как есть по порядку имён?
- [ ] Нужен ли пресет «максимальное качество речи» по умолчанию (Strategy 3) vs «максимальная простота» (Strategy 1)?
