import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app import app
from backend.engine import utf16_length


def update(text, request_id=1, **kwargs):
    return {'type': 'prompt_update', 'request_id': request_id, 'text': text,
            'cursor': utf16_length(text), **kwargs}


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        self.assertEqual(self.client.get('/healthz').json()['engine'], 'diffusion')

    def test_unbuilt_frontend_gives_actionable_error(self):
        with patch('backend.app.DIST', Path('/tmp/splash-nonexistent-dist-test')):
            self.assertEqual(self.client.get('/').status_code, 503)

    def test_request_identity_and_private_responses(self):
        with self.client.websocket_connect('/ws') as first, self.client.websocket_connect('/ws') as second:
            first.send_json(update('private first prompt', 11))
            self.assertEqual(first.receive_json()['request_id'], 11)
            second.send_json(update('private second prompt', 22))
            response = second.receive_json()
            # With the original broadcast bug, this receives client one's prompt.
            self.assertEqual(response['request_id'], 22)
            self.assertEqual(response['source'], 'private second prompt')
            first.send_json(update('', 12))
            self.assertEqual(first.receive_json()['request_id'], 12)

    def test_invalid_messages_do_not_kill_connection(self):
        invalid = ["not json", '[]', 'null', '{}', '{"type":"unknown"}',
                   '{"type":"prompt_update","request_id":true,"text":"x","cursor":1}']
        with self.client.websocket_connect('/ws') as ws:
            for raw in invalid:
                ws.send_text(raw)
                self.assertEqual(ws.receive_json()['type'], 'error')
            for message in [update('x', cursor=5), update('x', freeze_radius=-1),
                            update('x', commit='yes'), update('a' * 8001), update('🌙' * 4001),
                            update('x', request_id=-1), update('x', extra='bad')]:
                ws.send_json(message)
                self.assertEqual(ws.receive_json()['type'], 'error')
            ws.send_text('{"type":"prompt_update","request_id":1,"text":"\\ud800","cursor":1}')
            self.assertEqual(ws.receive_json()['type'], 'error')
            ws.send_bytes(b'binary')
            self.assertEqual(ws.receive_json()['type'], 'error')
            ws.send_json(update('valid 🌙\n  text', 99))
            self.assertEqual(ws.receive_json()['request_id'], 99)

    def test_oversized_frame_closes(self):
        with self.client.websocket_connect('/ws') as ws:
            ws.send_text('a' * 100001)
            with self.assertRaises(WebSocketDisconnect) as caught:
                ws.receive_json()
            self.assertEqual(caught.exception.code, 1009)

    def test_commit_and_clear(self):
        with self.client.websocket_connect('/ws') as ws:
            ws.send_json(update('a quiet lake'))
            output = ws.receive_json()['output']
            ws.send_json(update(output, 2, commit=True))
            committed = ws.receive_json()
            self.assertEqual(committed['output'], output)
            self.assertTrue(all(s['kind'] == 'anchor' for s in committed['segments']))
            ws.send_json(update('', 3))
            self.assertEqual(ws.receive_json()['segments'], [])


if __name__ == '__main__':
    unittest.main()
