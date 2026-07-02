"""WhisperSync entry point: GUI by default, --cli for headless mode."""

import sys


def main() -> None:
    if "--cli" in sys.argv:
        # cli.main() parses sys.argv itself via argparse, which doesn't know
        # about --cli (it's this dispatcher's own flag, not a whispersync.cli
        # argument) and would reject it as "unrecognized arguments". Strip it
        # before handing off. See PROJECT_ANALYSIS.md §7 (CLI reliability).
        sys.argv.remove("--cli")
        from whispersync.cli import main as cli_main

        cli_main()
    else:
        from whispersync.gui.main_window import main as gui_main

        gui_main()


if __name__ == "__main__":
    main()
