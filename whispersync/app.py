"""WhisperSync entry point: GUI by default, --cli for headless mode.

Lives inside the package (not at repo root as a bare main.py) so it installs
correctly as a console_script/entry_point — a script wheel-installed via
pip has no reason to expect a top-level main.py module to exist on
sys.path, and pyproject.toml's whispersync-gui entry point previously
pointed at exactly that nonexistent-after-install module. See
PROJECT_ANALYSIS.md §9.5. The repo-root main.py is now a thin shim so
`python main.py` keeps working unchanged for anyone running from a checkout.
"""

from __future__ import annotations

import sys


def main() -> None:
    if "--cli" in sys.argv:
        # cli.main() parses sys.argv itself via argparse, which doesn't know
        # about --cli (it's this dispatcher's own flag, not a whispersync.cli
        # argument) and would reject it as "unrecognized arguments". Strip it
        # before handing off.
        sys.argv.remove("--cli")
        from whispersync.cli import main as cli_main

        cli_main()
    else:
        from whispersync.gui.main_window import main as gui_main

        gui_main()


if __name__ == "__main__":
    main()
