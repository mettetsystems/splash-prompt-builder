import io
import os
import plistlib
import socket
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

from deploy.macos.build_app import PYTHON_ASSETS, asset_url, info_plist, main, safe_extract, stage_application
from deploy.macos.launch_splash import (
    acquire_lock, healthy, pick_port, read_port, server_command, server_environment, watch_parent, write_port,
)


class MacPackageTests(unittest.TestCase):
    def test_python_assets_are_pinned(self):
        self.assertIn("aarch64-apple-darwin", PYTHON_ASSETS["arm64"]["name"])
        self.assertIn("x86_64-apple-darwin", PYTHON_ASSETS["x86_64"]["name"])
        self.assertEqual(len(PYTHON_ASSETS["arm64"]["sha256"]), 64)

    def test_asset_url_encodes_the_version_plus(self):
        url = asset_url(PYTHON_ASSETS["arm64"]["name"])
        self.assertIn("%2B20261009", url)
        self.assertTrue(url.startswith("https://github.com/astral-sh/python-build-standalone/"))

    def test_plist_round_trip(self):
        encoded = plistlib.dumps(info_plist("14.0"))
        loaded = plistlib.loads(encoded)
        self.assertEqual(loaded["CFBundleExecutable"], "Splash")
        self.assertEqual(loaded["CFBundleIdentifier"], "local.splash.prompt-studio")
        self.assertEqual(loaded["LSMinimumSystemVersion"], "14.0")
        self.assertTrue(loaded["LSMultipleInstancesProhibited"])
        self.assertNotIn("CFBundleIconFile", info_plist(include_icon=False))

    def test_stage_copies_the_runtime_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = root / "backend"
            backend.mkdir()
            (backend / "app.py").write_text("print('ok')\n")
            cache = backend / "__pycache__"
            cache.mkdir()
            (cache / "app.cpython-314.pyc").write_bytes(b"x")
            dist = root / "frontend" / "dist"
            dist.mkdir(parents=True)
            (dist / "index.html").write_text("<html></html>")
            (root / "mobyMinutes").mkdir()
            (root / "mobyMinutes" / "note.txt").write_text("archive")
            destination = root / "app"
            stage_application(root, destination)
            self.assertEqual((destination / "backend" / "app.py").read_text(), "print('ok')\n")
            self.assertTrue((destination / "frontend" / "dist" / "index.html").is_file())
            self.assertFalse((destination / "backend" / "__pycache__").exists())
            self.assertFalse((destination / "mobyMinutes").exists())

    def test_extract_rejects_parent_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("../outside.txt")
                payload = b"no"
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            with self.assertRaises(SystemExit):
                safe_extract(archive, root / "out")
            self.assertFalse((root / "outside.txt").exists())

    def test_extract_keeps_a_normal_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "runtime.tar.gz"
            payload = b"#!/bin/sh\n"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("python/bin/python3")
                info.size = len(payload)
                info.mode = 0o755
                tar.addfile(info, io.BytesIO(payload))
            safe_extract(archive, root / "Resources")
            extracted = root / "Resources" / "python" / "bin" / "python3"
            self.assertEqual(extracted.read_bytes(), payload)

    def test_stage_requires_a_frontend_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "backend").mkdir()
            with self.assertRaises(SystemExit):
                stage_application(root, root / "app")

    def test_packager_refuses_to_build_off_macos(self):
        if sys.platform == "darwin":
            self.skipTest("the refusal path is for non-macOS hosts")
        with self.assertRaises(SystemExit) as caught:
            main([])
        self.assertIn("built on macOS", str(caught.exception))

    def test_help_is_available_off_macos(self):
        with patch("sys.stdout", io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_macos_requirements_are_not_the_cuda_stack(self):
        path = Path(__file__).resolve().parents[1] / "backend" / "requirements-macos.txt"
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]
        text = "\n".join(lines)
        self.assertIn("torch==2.14.1", text)
        self.assertNotIn("bitsandbytes", text)
        self.assertNotIn("cu130", text)


class LaunchSplashTests(unittest.TestCase):
    def test_server_stays_on_loopback(self):
        command = server_command("/Applications/Splash.app/Contents/Resources/python/bin/python3", 8000)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertNotIn("0.0.0.0", command)

    def test_server_environment_is_isolated(self):
        env = server_environment(
            {"PYTHONHOME": "/usr", "SPLASH_PARENT_FD": "4", "PATH": "/usr/bin"},
            Path("/Applications/Splash.app/Contents/Resources/app"),
        )
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("SPLASH_PARENT_FD", env)
        self.assertEqual(env["PYTHONPATH"], "/Applications/Splash.app/Contents/Resources/app")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")

    def test_single_instance_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "instance.lock"
            first = acquire_lock(path)
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_lock(path))
            first.close()
            second = acquire_lock(path)
            self.assertIsNotNone(second)
            second.close()

    def test_port_file_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_port(directory, 8001)
            self.assertEqual(read_port(directory), 8001)
            (directory / "port").write_text("nope")
            self.assertEqual(read_port(directory), 8000)

    def test_pick_port_skips_a_bound_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            busy = sock.getsockname()[1]
            chosen = pick_port(start=busy, attempts=5)
        self.assertNotEqual(chosen, busy)
        self.assertGreater(chosen, busy)

    def test_parent_close_requests_stop(self):
        read_fd, write_fd = os.pipe()
        stop = threading.Event()
        thread = threading.Thread(target=watch_parent, args=(read_fd, stop))
        thread.start()
        os.close(write_fd)
        self.assertTrue(stop.wait(2))
        thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_health_check_accepts_a_local_ok(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/healthz":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = b'{"status":"healthy"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            self.assertTrue(healthy(port))
            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                self.assertEqual(response.status, 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


class AppleHardwareTests(unittest.TestCase):
    def test_apple_silicon_reports_unverified_metal(self):
        from backend.models import hardware
        with patch("backend.models.platform.system", return_value="Darwin"), \
             patch("backend.models.platform.machine", return_value="arm64"), \
             patch("backend.models.subprocess.run", side_effect=FileNotFoundError):
            result = hardware()
        devices = [profile["device"] for profile in result["profiles"]]
        self.assertEqual(devices[:2], ["mps", "cpu"])
        self.assertEqual(result["metal"], "unverified")
        self.assertIn("Apple GPU present", result["gpu_status"])
        self.assertFalse(result["profiles"][0]["verified"])

    def test_intel_mac_does_not_offer_metal(self):
        from backend.models import hardware
        with patch("backend.models.platform.system", return_value="Darwin"), \
             patch("backend.models.platform.machine", return_value="x86_64"), \
             patch("backend.models.subprocess.run", side_effect=FileNotFoundError):
            result = hardware()
        devices = [profile["device"] for profile in result["profiles"]]
        self.assertNotIn("mps", devices)
        self.assertIn("cpu", devices)


if __name__ == "__main__":
    unittest.main()
