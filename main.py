"""BormoSync entry point: GUI by default, --cli for headless mode."""

import sys


def main() -> None:
    if "--cli" in sys.argv:
        from bormosync.cli import main as cli_main

        cli_main()
    else:
        from bormosync.gui.main_window import main as gui_main

        gui_main()


if __name__ == "__main__":
    main()
