"""Launch the bundled Splash server and open it in the browser.

The macOS app executable starts this file with the bundled interpreter.
A second launch finds the running instance and opens its existing window.
Closing the parent app, or Quit, stops the server.
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def bundle_resources(script_path=None):
    return Path(script_path or __file__).resolve().parent


def instance_directory():
    return Path.home() / "Library" / "Application Support" / "Splash"


def log_directory():
    return Path.home() / "Library" / "Logs" / "Splash"


def server_command(python, port):
    return [str(python), "-m", "uvicorn", "backend.app:app", "--host", HOST,
            "--port", str(port), "--ws-max-size", "100000"]


def server_environment(base, app_root):
    env = dict(base)
    env.pop("PYTHONHOME", None)
    env.pop("SPLASH_PARENT_FD", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(app_root)
    return env


def acquire_lock(path):
    """Return an open file holding an exclusive lock, or None if another instance has it."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def read_port(directory):
    try:
        port = int((Path(directory) / "port").read_text().strip())
    except (OSError, ValueError):
        return DEFAULT_PORT
    return port if 1 <= port <= 65535 else DEFAULT_PORT


def write_port(directory, port):
    path = Path(directory) / "port"
    path.write_text(f"{port}\n")


def pick_port(start=DEFAULT_PORT, attempts=20):
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return port
    raise OSError(f"No free loopback port from {start} through {start + attempts - 1}.")


def healthy(port, timeout=0.5):
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/healthz", timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_until_healthy(port, timeout, stop=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop is not None and stop.is_set():
            return False
        if healthy(port):
            return True
        time.sleep(0.1)
    return False


def open_in_browser(url):
    if sys.platform == "darwin":
        subprocess.Popen(["/usr/bin/open", url])
        return
    import webbrowser
    webbrowser.open(url)


def notify(message):
    text = " ".join(message.split())
    if sys.platform != "darwin":
        print(text, file=sys.stderr)
        return
    script = f'display notification {json.dumps(text)} with title "Splash"'
    subprocess.run(["/usr/bin/osascript", "-e", script], check=False)


def watch_parent(fd, stop):
    os.set_inheritable(fd, False)
    try:
        while os.read(fd, 64):
            pass
    except OSError:
        pass
    stop.set()


def watch_parent_from_env(stop):
    raw = os.environ.get("SPLASH_PARENT_FD")
    if not raw:
        return
    try:
        fd = int(raw)
    except ValueError:
        return
    threading.Thread(target=watch_parent, args=(fd, stop), name="splash-parent", daemon=True).start()


def install_signal_handlers(stop):
    def handle(signum, frame):
        stop.set()
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def start_server(python, app_root, port, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", buffering=1)
    print(f"Starting Splash on http://{HOST}:{port}", file=log, flush=True)
    process = subprocess.Popen(
        server_command(python, port),
        cwd=app_root,
        env=server_environment(os.environ, app_root),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    return process


def stop_server(server, timeout=5):
    if server.poll() is not None:
        return
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        server.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        server.wait(timeout=2)


def main():
    resources = bundle_resources()
    app_root = resources / "app"
    python = resources / "python" / "bin" / "python3"
    support = instance_directory()
    log_path = log_directory() / "splash.log"
    support.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(support / "instance.lock")
    if lock is None:
        open_in_browser(f"http://{HOST}:{read_port(support)}")
        return 0
    server = None
    try:
        if not python.is_file() or not (app_root / "backend" / "app.py").is_file():
            notify(f"Splash is incomplete. Expected the runtime under {resources}.")
            return 1
        try:
            port = pick_port()
        except OSError as exc:
            notify(str(exc))
            return 1
        write_port(support, port)
        stop = threading.Event()
        install_signal_handlers(stop)
        watch_parent_from_env(stop)
        server = start_server(python, app_root, port, log_path)
        ready = wait_until_healthy(port, timeout=30, stop=stop)
        if stop.is_set() or not ready:
            stop_server(server)
            if stop.is_set():
                return 0
            message = f"Splash did not become ready. See {log_path}"
            with open(log_path, "a", buffering=1) as log:
                print(message, file=log, flush=True)
            notify(message)
            return 1
        open_in_browser(f"http://{HOST}:{port}")
        while server.poll() is None and not stop.is_set():
            stop.wait(0.2)
        user_stopped = stop.is_set()
        stop_server(server)
        if user_stopped:
            return 0
        return server.returncode or 0
    finally:
        if server is not None and server.poll() is None:
            stop_server(server)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
