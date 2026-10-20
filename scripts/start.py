"""Start Splash from its virtual environment on loopback only."""
import argparse
import errno
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlopen


def port_status(port):
    """Check before starting Uvicorn, whose lifespan may already load a model."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(('127.0.0.1', port))
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
        else:
            return 'available'
    try:
        with urlopen(f'http://127.0.0.1:{port}/healthz', timeout=2) as response:
            health = json.load(response)
        if health.get('status') == 'healthy' and health.get('version') == '2.0.0' and health.get('engine') == 'diffusion':
            return 'splash'
    except (OSError, URLError, ValueError, AttributeError):
        pass
    return 'occupied'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=43827, help="Splash port (default: 43827)")
    parser.add_argument("--data-dir", type=Path, help="Override the platform app-data directory")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    try:
        status = port_status(args.port)
    except OSError as exc:
        parser.error(f"Could not check port {args.port}: {exc}")
    if status == 'splash':
        print(f"Splash is already running at http://127.0.0.1:{args.port}/. Open that address in your browser.\n"
              "No second server or model worker was started. Stop the existing server before restarting or changing its data directory.")
        return 0
    if status == 'occupied':
        print(f"Port {args.port} is occupied by another application or is unavailable. Stop that application or use --port with a free port.\n"
              "Splash was not started; no model weights were loaded.", file=sys.stderr)
        return 1
    root = Path(__file__).resolve().parents[1]
    python = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists() or not (root / "frontend/dist/index.html").exists():
        parser.error("Run python scripts/setup.py first.")
    env = dict(os.environ)
    if args.data_dir:
        env["SPLASH_DATA_DIR"] = str(args.data_dir.expanduser().resolve())
    options = {}
    if sys.platform == "win32":
        # A dedicated console group lets stop.py request graceful Uvicorn cleanup;
        # TerminateProcess would bypass the application's shutdown handlers.
        env["SPLASH_CONSOLE_GROUP"] = "1"
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen([str(python), "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1",
        "--port", str(args.port), "--ws-max-size", "100000"], cwd=root, env=env, **options)
    try:
        return process.wait()
    except KeyboardInterrupt:
        # POSIX terminals deliver Ctrl+C to both processes. Windows' separate
        # group needs forwarding. Keep the launcher alive until cleanup finishes.
        if sys.platform == "win32" and process.poll() is None:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            process.wait()
        except KeyboardInterrupt:
            print("Shutdown is still in progress; check the server log before restarting.", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
