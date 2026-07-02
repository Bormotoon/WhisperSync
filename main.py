"""Thin shim so `python main.py` works from a checkout; see whispersync/app.py."""

from whispersync.app import main

if __name__ == "__main__":
    main()
