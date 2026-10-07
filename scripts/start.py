"""Start Splash from its virtual environment on loopback only."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, help="Override the platform app-data directory")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    root = Path(__file__).resolve().parents[1]
    python = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists() or not (root / "frontend/dist/index.html").exists():
        parser.error("Run python scripts/setup.py first.")
    env = dict(os.environ)
    if args.data_dir:
        env["SPLASH_DATA_DIR"] = str(args.data_dir.expanduser().resolve())
    try:
        return subprocess.call([str(python), "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1",
            "--port", str(args.port), "--ws-max-size", "100000"], cwd=root, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
