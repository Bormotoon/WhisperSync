# Contributing to WhisperSync

Thanks for your interest in improving WhisperSync! This guide explains how to set
up a dev environment and the conventions we follow.

## Development setup

```bash
git clone https://github.com/Bormotoon/WhisperSync.git
cd WhisperSync
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt # includes runtime deps + tooling
python whispersync/engine/system_check.py
```

You also need **ffmpeg** and **ffprobe** in your `PATH`.

## Project layout

- `whispersync/engine/` — pure logic (transcription, matching, strategies, FCPXML).
  Business logic must not depend on Qt.
- `whispersync/gui/` — PyQt6 interface (window, worker, widgets, theme).
- `tests/` — pytest suite (no GPU/ffmpeg required; heavy paths are covered with
  synthetic data and mocks).

## Code style & quality gates

All of these must pass before a PR is merged (CI enforces them):

```bash
ruff check whispersync/ tests/      # lint
black --check whispersync/ tests/   # formatting (line length 100)
mypy whispersync/ main.py           # type checking
pytest                            # tests
```

- Type hints on all public functions; `from __future__ import annotations`.
- No magic numbers — put constants/defaults in `config.py`.
- Keep the engine import-light and Qt-free; do heavy/optional imports lazily.
- Add or update tests for any behavior change.

## Commit & PR guidelines

- Write focused commits with a clear subject line (imperative mood).
- Reference related issues (`Fixes #123`) where applicable.
- Describe **what** changed and **why** in the PR body; include test output for
  behavior changes and screenshots for GUI changes.
- Keep PRs scoped — smaller is easier to review.

## Reporting bugs / requesting features

Use the GitHub issue templates. For bugs, include OS, Python version, GPU/CPU,
ffmpeg version, the command you ran, and the full error/log output.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
