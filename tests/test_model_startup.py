"""Startup must remain an editor even when a saved model cannot load."""
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from backend.app import app
from backend.storage import Store


class StartupTests(unittest.TestCase):
    def test_skip_preserves_saved_config_and_allows_manual_load(self):
        config = {'model': 'llada', 'device': 'cuda:1', 'path': '/unused/checkpoint', 'source': 'local'}
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, SPLASH_DATA_DIR=directory, SPLASH_SKIP_MODEL_LOAD='1'), \
             patch('backend.mcp_host.ToolHost.start'), \
             patch('backend.api.discover_endpoints', return_value=[]), \
             patch('backend.models.ModelService.load', new_callable=AsyncMock) as load:
            Store().set_config('model', config)
            with TestClient(app) as client:
                runtime = client.get('/api/models').json()['runtime']
                self.assertEqual(runtime['state'], 'unloaded')
                self.assertEqual(runtime['config'], config)
                self.assertIn('skipped for this session', runtime['notice'])
                load.assert_not_called()
                self.assertEqual(Store().config('model'), config)
                project = client.post('/api/projects', json={'agent': 'splash', 'text': 'Keep editing.'})
                self.assertEqual(project.status_code, 201)
                load.return_value = {'state': 'error', 'error': 'simulated failure'}
                with patch('backend.api.inspect_local'), patch('backend.api.dependency_status', return_value={'missing': [], 'packages': {'bitsandbytes': True}}):
                    response = client.post('/api/models/load', json=config)
                self.assertEqual(response.status_code, 200, response.text)
                load.assert_awaited_once()
                self.assertEqual(Store().config('model'), config)


if __name__ == '__main__':
    unittest.main()
