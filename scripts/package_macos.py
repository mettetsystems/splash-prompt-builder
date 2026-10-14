#!/usr/bin/env python3
"""Build a double-clickable Splash.app. This script must be run on macOS."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.macos.build_app import main


if __name__ == "__main__":
    raise SystemExit(main())
