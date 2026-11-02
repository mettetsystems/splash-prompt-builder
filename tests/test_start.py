import contextlib
import io
import socket
import unittest
from unittest.mock import patch

from scripts import start


class LauncherTests(unittest.TestCase):
    @unittest.skipIf(start.os.name == 'nt', 'POSIX socket reuse behavior')
    def test_recently_closed_connection_does_not_block_restart(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            port = listener.getsockname()[1]
            with socket.create_connection(('127.0.0.1', port), timeout=2) as client:
                connection, _ = listener.accept()
                with connection:
                    connection.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b'')
        self.assertEqual(start.port_status(port), 'available')

    def test_existing_splash_does_not_spawn_another_server(self):
        output = io.StringIO()
        with patch.object(start, 'port_status', return_value='splash'), patch.object(start.subprocess, 'Popen') as spawn, contextlib.redirect_stdout(output):
            self.assertEqual(start.main([]), 0)
        spawn.assert_not_called()
        self.assertIn('already running at http://127.0.0.1:43827/', output.getvalue())

    def test_other_service_does_not_load_a_model(self):
        with patch.object(start, 'port_status', return_value='occupied'), patch.object(start.subprocess, 'Popen') as spawn, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(start.main(['--port', '8002']), 1)
        spawn.assert_not_called()

    def test_no_model_passed_only_to_launched_server(self):
        with patch.object(start, 'port_status', return_value='available'), \
             patch.object(start.Path, 'exists', return_value=True), \
             patch.dict(start.os.environ, {}, clear=True), \
             patch.object(start.subprocess, 'Popen') as spawn:
            spawn.return_value.wait.return_value = 0
            self.assertEqual(start.main(['--no-model']), 0)
            self.assertEqual(spawn.call_args.kwargs['env']['SPLASH_SKIP_MODEL_LOAD'], '1')
            self.assertNotIn('SPLASH_SKIP_MODEL_LOAD', start.os.environ)
            self.assertEqual(start.main([]), 0)
            self.assertNotIn('SPLASH_SKIP_MODEL_LOAD', spawn.call_args.kwargs['env'])

    def test_windows_ctrl_c_forwards_graceful_shutdown_to_server_group(self):
        with patch.object(start, 'port_status', return_value='available'), \
             patch.object(start.sys, 'platform', 'win32'), \
             patch.object(start.Path, 'exists', return_value=True), \
             patch.object(start.subprocess, 'CREATE_NEW_PROCESS_GROUP', 512, create=True), \
             patch.object(start.signal, 'CTRL_BREAK_EVENT', 1, create=True), \
             patch.object(start.subprocess, 'Popen') as spawn:
            server = spawn.return_value
            server.wait.side_effect = [KeyboardInterrupt, 0]
            server.poll.return_value = None
            self.assertEqual(start.main([]), 0)
        self.assertEqual(spawn.call_args.kwargs['creationflags'], 512)
        self.assertEqual(spawn.call_args.kwargs['env']['SPLASH_CONSOLE_GROUP'], '1')
        command = spawn.call_args.args[0]
        self.assertEqual(command[command.index('--port') + 1], '43827')
        server.send_signal.assert_called_once_with(1)
        self.assertEqual(server.wait.call_count, 2)

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
