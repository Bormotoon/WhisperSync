"""Thin shim so `python main.py` works from a checkout; see whispersync/app.py."""

import multiprocessing

from whispersync.app import main

if __name__ == "__main__":
    # Required for PyInstaller builds now that the render pool may use
    # spawn/forkserver (see pipeline._pool_context): a frozen child process
    # re-runs this entry module and must hand control to multiprocessing
    # instead of launching a second GUI. No-op in a normal interpreter.
    multiprocessing.freeze_support()
    main()
