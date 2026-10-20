"""Gracefully stop this checkout's Splash server (default port: 43827)."""

import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def option(arguments, name, default):
    result = default
    for index, value in enumerate(arguments):
        if value == name:
            result = arguments[index + 1] if index + 1 < len(arguments) else None
        if value.startswith(name + "="):
            result = value.split("=", 1)[1]
    return result


def matches_server(process, root, port):
    """Match the actual entry point, checkout, and port, never just a PID/name."""
    command = process.cmdline()
    if not command or Path(process.cwd()).resolve() != root.resolve():
        return False
    if command[1:3] == ["-m", "backend.app"]:
        return len(command) == 3 and port == 43827
    if command[1:3] == ["-m", "uvicorn"]:
        arguments = command[3:]
    elif Path(command[0]).name.lower() in ("uvicorn", "uvicorn.exe"):
        arguments = command[1:]
    elif len(command) > 1 and Path(command[1]).name.lower() in ("uvicorn", "uvicorn.exe"):
        arguments = command[2:]
    else:
        return False
    if "backend.app:app" not in arguments:
        return False
    selected_port = option(arguments, "--port", None)
    if selected_port is None:
        # Direct Uvicorn launches retain Uvicorn's own default, unlike Splash's launchers.
        selected_port = process.environ().get("UVICORN_PORT", "8000")
    app_dir = option(arguments, "--app-dir", str(root))
    if app_dir is None or selected_port != str(port):
        return False
    directory = Path(app_dir)
    if not directory.is_absolute():
        directory = root / directory
    return directory.resolve() == root.resolve()


def find_servers(psutil, root, port):
    username = psutil.Process().username()
    matches = []
    for process in psutil.process_iter():
        try:
            if process.username() == username and matches_server(process, root, port):
                matches.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    # A reload/multi-worker supervisor owns its children. Signal only the parent.
    ids = {process.pid for process in matches}
    return [process for process in matches if not any(parent.pid in ids for parent in process.parents())]


def running(psutil, process):
    try:
        return process.is_running() and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


def stop_server(psutil, process, root, port, timeout):
    # Process objects remember creation time, preventing signals to a reused PID.
    if not running(psutil, process):
        return
    if not matches_server(process, root, port):
        raise RuntimeError("The server identity changed; no shutdown signal was sent.")
    owned = [process, *process.children(recursive=True)]
    try:
        if sys.platform == "win32":
            if process.environ().get("SPLASH_CONSOLE_GROUP") != "1":
                raise RuntimeError("This Windows server was not started with the updated scripts/start.py. "
                                   "Stop it with Ctrl+C in its original terminal.")
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGTERM)
    except psutil.NoSuchProcess:
        pass  # Still wait for the children captured before the server exited.
    deadline = time.monotonic() + timeout
    while True:
        alive = [child for child in owned if running(psutil, child)]
        if not alive:
            return
        if time.monotonic() >= deadline:
            pids = ", ".join(str(child.pid) for child in alive)
            raise RuntimeError(f"Shutdown is still pending after {timeout:g}s (PID(s): {pids}). "
                               "No processes were force-killed. Check the server terminal/log, "
                               "or retry with a longer --timeout.")
        time.sleep(min(.1, max(0, deadline - time.monotonic())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=43827, help="Splash port to stop (default: 43827)")
    parser.add_argument("--timeout", type=int, default=30, help="Seconds to wait for cleanup (default: 30)")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    if args.timeout < 1:
        parser.error("Timeout must be at least one second.")
    try:
        import psutil
    except ImportError:
        python = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        # Compare prefixes: venv executables may be symlinks to the system Python.
        if python.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
            return subprocess.call([str(python), str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
        print("The shutdown script needs psutil. Run scripts/setup.py to install Splash's dependencies.", file=sys.stderr)
        return 1
    try:
        servers = find_servers(psutil, ROOT, args.port)
        if not servers:
            print(f"No matching Splash server found for this checkout on port {args.port}. Nothing was stopped.")
            return 0
        if len(servers) != 1:
            raise RuntimeError("Multiple matching servers found. Stop them from their original terminals.")
        server = servers[0]
        print(f"Stopping Splash on port {args.port} (PID {server.pid}); waiting for server and worker cleanup...", flush=True)
        try:
            stop_server(psutil, server, ROOT, args.port, args.timeout)
        except psutil.NoSuchProcess:
            pass  # It exited between discovery and the signal.
        print("Splash stopped. Saved projects, drafts, and downloaded models are preserved.")
        return 0
    except (psutil.Error, OSError, RuntimeError) as exc:
        print(f"Could not complete shutdown: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
