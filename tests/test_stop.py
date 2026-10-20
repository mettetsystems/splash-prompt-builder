import contextlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from urllib.request import urlopen

import psutil

from scripts import stop


def server_process(command=None, root=stop.ROOT):
    process = Mock(pid=12345)
    process.cmdline.return_value = command or [sys.executable, '-m', 'uvicorn', 'backend.app:app', '--port', '43827']
    process.cwd.return_value = str(root)
    process.environ.return_value = {}
    process.children.return_value = []
    process.is_running.return_value = True
    process.status.return_value = psutil.STATUS_RUNNING
    return process


class ShutdownTests(unittest.TestCase):
    def test_only_exact_app_checkout_and_port_match(self):
        self.assertTrue(stop.matches_server(server_process(), stop.ROOT, 43827))
        self.assertFalse(stop.matches_server(server_process(), stop.ROOT, 43828))
        self.assertFalse(stop.matches_server(server_process(root=stop.ROOT.parent), stop.ROOT, 43827))
        for command in ([sys.executable, '-m', 'uvicorn', 'other:app', '--port', '43827'],
                        [sys.executable, '-c', 'backend.app:app', '--port', '43827'],
                        [sys.executable, 'scripts/start.py'],
                        [sys.executable, '-m', 'uvicorn', 'backend.app:app', '--app-dir', '/tmp']):
            with self.subTest(command=command):
                self.assertFalse(stop.matches_server(server_process(command), stop.ROOT, 43827))

    def test_supported_entry_points_and_environment_port(self):
        for command in ([sys.executable, '-m', 'backend.app'],
                        ['uvicorn', 'backend.app:app', '--port=43827']):
            self.assertTrue(stop.matches_server(server_process(command), stop.ROOT, 43827))
        # Uvicorn without --port still uses its own default, not Splash's.
        direct = server_process([sys.executable, '/venv/bin/uvicorn', 'backend.app:app'])
        self.assertTrue(stop.matches_server(direct, stop.ROOT, 8000))
        self.assertFalse(stop.matches_server(direct, stop.ROOT, 43827))
        process = server_process(['uvicorn', 'backend.app:app'])
        process.environ.return_value = {'UVICORN_PORT': '43828'}
        self.assertFalse(stop.matches_server(process, stop.ROOT, 43827))
        self.assertTrue(stop.matches_server(process, stop.ROOT, 43828))
        process = server_process(['uvicorn', 'backend.app:app', '--port', '43827', '--port=43828'])
        self.assertFalse(stop.matches_server(process, stop.ROOT, 43827))
        self.assertTrue(stop.matches_server(process, stop.ROOT, 43828))

    @unittest.skipIf(sys.platform == 'win32', 'POSIX shutdown signal')
    def test_shutdown_signals_only_parent_and_waits_for_children(self):
        process, child = server_process(), server_process()
        process.children.return_value = [child]
        process.is_running.side_effect = [True, False, False]
        child.is_running.side_effect = [True, False]
        with patch.object(stop.time, 'sleep') as wait:
            stop.stop_server(psutil, process, stop.ROOT, 43827, 5)
        process.send_signal.assert_called_once_with(signal.SIGTERM)
        child.send_signal.assert_not_called()
        wait.assert_called_once()

    @unittest.skipIf(sys.platform == 'win32', 'POSIX shutdown signal')
    def test_timeout_reports_failure_without_force_killing(self):
        process = server_process()
        with patch.object(stop.time, 'monotonic', side_effect=[0, 31]):
            with self.assertRaisesRegex(RuntimeError, 'No processes were force-killed'):
                stop.stop_server(psutil, process, stop.ROOT, 43827, 30)
        process.kill.assert_not_called()

    def test_changed_identity_is_not_signaled(self):
        process = server_process(root=stop.ROOT.parent)
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            stop.stop_server(psutil, process, stop.ROOT, 43827, 30)
        process.send_signal.assert_not_called()

    def test_windows_refuses_unmanaged_server(self):
        process = server_process()
        with patch.object(stop.sys, 'platform', 'win32'):
            with self.assertRaisesRegex(RuntimeError, 'original terminal'):
                stop.stop_server(psutil, process, stop.ROOT, 43827, 30)
        process.send_signal.assert_not_called()

    def test_windows_uses_console_break_for_managed_server(self):
        process = server_process()
        process.environ.return_value = {'SPLASH_CONSOLE_GROUP': '1'}
        process.is_running.side_effect = [True, False]
        with patch.object(stop.sys, 'platform', 'win32'), patch.object(stop.signal, 'CTRL_BREAK_EVENT', 1, create=True):
            stop.stop_server(psutil, process, stop.ROOT, 43827, 30)
        process.send_signal.assert_called_once_with(1)
        process.terminate.assert_not_called()

    def test_already_stopped_is_successful_noop(self):
        with patch.object(stop, 'find_servers', return_value=[]), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(stop.main([]), 0)
        self.assertIn('port 43827. Nothing was stopped', output.getvalue())

    def test_ambiguous_servers_are_not_signaled(self):
        with patch.object(stop, 'find_servers', return_value=[server_process(), server_process()]), \
             patch.object(stop, 'stop_server') as shutdown, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(stop.main([]), 1)
        shutdown.assert_not_called()

    def test_invalid_arguments(self):
        for args in (['--port', '0'], ['--port', '65536'], ['--timeout', '0']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                stop.main(args)


@unittest.skipIf(sys.platform == 'win32', 'Live POSIX shutdown; Windows console behavior requires Windows qualification')
class LiveShutdownTests(unittest.TestCase):
    def test_stops_real_server_and_tool_children_and_preserves_database(self):
        self.check_shutdown()

    def test_stops_reload_supervisor_and_its_server(self):
        self.check_shutdown(['--reload'])

    def check_shutdown(self, extra_args=()):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ, SPLASH_DATA_DIR=directory)
            with tempfile.TemporaryFile(mode='w+') as log:
                process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'backend.app:app', '--host', '127.0.0.1',
                                            '--port', str(port), *extra_args], cwd=stop.ROOT, env=environment, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 20
                    children = []
                    while time.monotonic() < deadline:
                        try:
                            with urlopen(f'http://127.0.0.1:{port}/healthz', timeout=1) as response:
                                self.assertEqual(json.load(response)['status'], 'healthy')
                            children = psutil.Process(process.pid).children(recursive=True)
                            if any('backend.mcp_server' in child.cmdline() for child in children):
                                break
                        except OSError:
                            pass
                        if process.poll() is not None:
                            self.fail('Test Splash server exited during startup')
                        time.sleep(.1)
                    else:
                        self.fail('Test Splash server and MCP child did not start')
                    result = subprocess.run([sys.executable, str(stop.ROOT / 'scripts/stop.py'), '--port', str(port)],
                                            cwd=directory, capture_output=True, text=True, timeout=35)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('Splash stopped', result.stdout)
                    process.wait(timeout=5)
                    self.assertTrue(all(not stop.running(psutil, child) for child in children))
                    self.assertTrue(list(Path(directory).glob('*.sqlite3')))
                    log.seek(0)
                    output = log.read()
                    self.assertIn('Application shutdown complete', output)
                    self.assertNotIn('leaked semaphore', output)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
