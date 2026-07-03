#!/usr/bin/env bash
# Создаёт ИЗОЛИРОВАННОЕ окружение `.sep-venv` для извлечения эмбиента камеры
# (audio-separator + RoFormer). Держится отдельно от основного venv, потому что
# audio-separator требует более старый Python (3.12), чем основное приложение
# (3.14). Пайплайн вызывает его как подпроцесс — см. whispersync/engine/separation.py.
#
# Запускать из корня проекта. Нужен python3.12 в PATH (или uv).
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Удаляю старый .sep-venv"
rm -rf .sep-venv

if command -v uv >/dev/null 2>&1; then
    echo "==> Создаю .sep-venv (Python 3.12) через uv"
    uv venv --python python3.12 .sep-venv
    echo "==> Ставлю audio-separator[gpu]"
    VIRTUAL_ENV="$PWD/.sep-venv" uv pip install "audio-separator[gpu]"
else
    echo "==> Создаю .sep-venv через python3.12"
    python3.12 -m venv .sep-venv
    ./.sep-venv/bin/pip install --upgrade pip
    ./.sep-venv/bin/pip install "audio-separator[gpu]"
fi

echo "==> Проверка: RoFormer грузится в .sep-venv"
./.sep-venv/bin/python -c "
from audio_separator.separator import Separator
import torch
print('audio-separator OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())
"

echo ""
echo "Готово. Модель MelBand-RoFormer скачается автоматически при первом запуске"
echo "с включённой опцией 'Add camera-ambience track' (--ambience-track в CLI)."
