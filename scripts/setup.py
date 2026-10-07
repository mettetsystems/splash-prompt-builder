"""Portable app setup. Model runtimes are installed separately for the chosen hardware."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", action="store_true", help="Include test dependencies")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or later is required.")
    npm = shutil.which("npm")
    if not npm:
        parser.error("Install Node.js 22.12 or later with npm, then retry.")
    python = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.create(root / ".venv", with_pip=True)
    requirements = "requirements-dev.txt" if args.dev else "requirements.txt"
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(root / "backend" / requirements)], check=True)
    subprocess.run([npm, "--prefix", str(root / "frontend"), "ci"], check=True)
    subprocess.run([npm, "--prefix", str(root / "frontend"), "run", "build"], check=True)
    print("Splash is ready. Run: python scripts/start.py")
    print("Projects and editing work immediately. Install a model runtime to enable enrichment; see README.md.")


if __name__ == "__main__":
    main()
