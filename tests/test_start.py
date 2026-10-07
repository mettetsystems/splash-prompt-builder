import contextlib
import io
import socket
import unittest
from unittest.mock import patch

from scripts import start


class LauncherTests(unittest.TestCase):
    def test_existing_splash_does_not_spawn_another_server(self):
        output = io.StringIO()
        with patch.object(start, 'port_status', return_value='splash'), patch.object(start.subprocess, 'call') as spawn, contextlib.redirect_stdout(output):
            self.assertEqual(start.main([]), 0)
        spawn.assert_not_called()
        self.assertIn('already running at http://127.0.0.1:8000/', output.getvalue())

    def test_other_service_does_not_load_a_model(self):
        with patch.object(start, 'port_status', return_value='occupied'), patch.object(start.subprocess, 'call') as spawn, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(start.main(['--port', '8002']), 1)
        spawn.assert_not_called()

    def test_real_occupied_socket_and_splash_health_detection(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen()
            port = listener.getsockname()[1]
            with patch.object(start, 'urlopen', side_effect=OSError('unrelated service')):
                self.assertEqual(start.port_status(port), 'occupied')
            with patch.object(start, 'urlopen', return_value=io.BytesIO(b'{"status":"healthy","engine":"diffusion","version":"2.0.0"}')):
                self.assertEqual(start.port_status(port), 'splash')


if __name__ == '__main__':
    unittest.main()
