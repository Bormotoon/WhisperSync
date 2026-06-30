#!/usr/bin/env bash
# Пересоздаёт виртуальное окружение с нуля.
# Запускать ИЗ КОРНЯ проекта ПОСЛЕ того, как папка получила финальное имя
# (venv зашивает абсолютный путь, поэтому переименовывать папку после
#  создания venv нельзя — он сломается).
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Удаляю старый venv"
rm -rf venv

echo "==> Создаю новый venv"
python3 -m venv venv

echo "==> Обновляю pip и ставлю зависимости"
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements-dev.txt

echo "==> Проверка импортов"
./venv/bin/python -c "from whispersync.gui.main_window import main; print('OK: whispersync импортируется')"

echo "==> Тесты"
./venv/bin/python -m pytest -q

echo ""
echo "Готово. Запуск приложения:"
echo "  source venv/bin/activate && python3 main.py"
