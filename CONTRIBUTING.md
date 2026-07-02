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
python -m whispersync.engine.system_check
```

You also need **ffmpeg** and **ffprobe** in your `PATH`.

### Reproducible installs (lock files)

`requirements.txt`/`requirements-dev.txt` use version ranges. `requirements.lock`/
`requirements-dev.lock` are a pinned, known-good snapshot (generated on Python
3.12 via `uv pip compile`) — CI installs from the lock file on 3.12 and from the
ranges on the other matrix versions, so both paths get exercised. Regenerate the
locks after changing a `requirements*.txt` file:

```bash
uv pip compile requirements.txt -o requirements.lock --python-version 3.12
uv pip compile requirements-dev.txt -o requirements-dev.lock --python-version 3.12
```

## Project layout

- `whispersync/engine/` — pure logic (transcription, matching, strategies, FCPXML).
  Business logic must not depend on Qt.
- `whispersync/gui/` — PyQt6 interface (window, worker, widgets, theme).
- `tools/` — standalone developer tools (e.g. `verify_sync.py`, the realized
  lip-sync-lag measurement harness), not part of the installed package.
- `tests/` — pytest suite. Most tests are pure logic (no GPU/ffmpeg required);
  a smaller set marked `@pytest.mark.integration` exercises real ffmpeg through
  the render path and is skipped automatically if ffmpeg isn't on `PATH`
  (`pytest -m "not integration"` to exclude explicitly).

## Code style & quality gates

All of these must pass before a PR is merged (CI enforces them):

```bash
ruff check whispersync/ tools/ tests/      # lint
black --check whispersync/ tools/ tests/   # formatting (line length 100)
mypy whispersync/ tools/ main.py           # type checking
pytest                                     # tests (incl. integration, if ffmpeg is present)
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
