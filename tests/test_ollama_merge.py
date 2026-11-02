"""Regressions at the boundary between endpoint reuse and explicit Ollama loading."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from backend.app import app
from backend.models import ModelService, configure_torch_runtime
from backend.ollama import OllamaClient, select_ollama_model, EMBEDDING_MODEL
from backend.ollama_model import OllamaModel
from backend.local_index import _embed, directory_documents, index_status
from backend.research import retrieve_context, parse_document
from backend.storage import Store


class CatalogTests(unittest.TestCase):
    def test_reserve_and_cpu_fallback_use_free_memory(self):
        gpu = {'name': 'NVIDIA RTX 5090', 'free_mb': 10 * 1024}
        self.assertEqual(select_ollama_model([gpu], 64), 'qwen3.5:4b')
        self.assertEqual(select_ollama_model([{**gpu, 'free_mb': 12 * 1024}], 64), 'qwen3.5:9b')
        self.assertEqual(select_ollama_model([], 8), 'qwen3.5:4b')
        with self.assertRaises(ValueError): select_ollama_model([], 4)

    def test_management_url_stays_on_loopback_without_paths(self):
        for url in ('https://localhost', 'http://example.com', 'http://localhost/x', 'http://localhost?x', 'http://user@localhost'):
            with self.assertRaises(ValueError): OllamaClient(url).base_url
        self.assertEqual(OllamaClient('http://[::1]:11434/').base_url, 'http://[::1]:11434')

    def test_runtime_threads_are_bounded(self):
        torch = Mock()
        with patch.dict(os.environ, {'SPLASH_INFERENCE_THREADS': '2'}):
            self.assertLessEqual(configure_torch_runtime(torch, 'cpu'), 2)
        torch.set_num_interop_threads.assert_called_once_with(1)

    def test_bm25_excludes_unrelated_passages_and_retains_character_counts(self):
        project = {'documents': [parse_document('a.txt', b'Caching reduces network latency.'), parse_document('b.txt', b'Bananas are yellow.')]}
        result = retrieve_context(project, 'network latency')
        self.assertEqual([p['name'] for p in result['passages']], ['a.txt'])
        self.assertEqual(result['ranking'], 'bm25')
        self.assertGreater(result['total_characters'], len(result['passages'][0]['text']))

    def test_directory_recursion_and_missing_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.txt').write_text('Top level')
            (root / 'nested').mkdir()
            (root / 'nested' / 'b.md').write_text('Nested')
            self.assertEqual(len(directory_documents(directory)), 1)
            self.assertEqual(len(directory_documents(directory, True)), 2)
            self.assertFalse(index_status(root, 'project')['ready'])


class InferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_ollama_reconnect_does_not_start_weights_and_explicit_load_does(self):
        service = ModelService()
        config = {'source': 'ollama', 'model': 'qwen3.5:4b', 'base_url': 'http://localhost:11434/v1', 'context': 4096}
        try:
            with patch.object(OllamaModel, 'verify', AsyncMock(return_value={'connected': True})), \
                 patch.object(OllamaModel, 'load', AsyncMock()) as load, \
                 patch.object(service, 'exchange', AsyncMock()) as worker:
                self.assertEqual((await service.load(config))['state'], 'ready')
                load.assert_not_called(); worker.assert_not_called()
                self.assertEqual(service.status['engine'], 'ollama')
                await service.load(config, explicit_ollama_load=True)
                load.assert_awaited_once(); worker.assert_not_called()
        finally:
            await service.close()

    async def test_native_generation_uses_saved_server_and_preserves_cpu_and_truncation(self):
        model = OllamaModel({'source': 'ollama', 'model': 'qwen3.5:4b', 'base_url': 'http://localhost:12345/v1', 'context': 4096, 'cpu_only': True})
        self.assertEqual(model.client.base_url, 'http://localhost:12345')
        response = Mock(); response.json.return_value = {'message': {'content': 'partial'}, 'done_reason': 'length'}
        with patch.object(model, 'verify', AsyncMock()), patch.object(OllamaClient, 'request', AsyncMock(return_value=response)) as request:
            result = await model.generate([{'role': 'user', 'content': 'Rewrite'}], 100)
        self.assertTrue(result['truncated'])
        self.assertEqual(request.call_args.kwargs['json']['options']['num_gpu'], 0)

    async def test_embeddings_are_batched_and_cpu_only(self):
        async def respond(method, path, **kwargs):
            return SimpleNamespace(json=lambda: {'embeddings': [[1., 0.]] * len(kwargs['json']['input'])})
        models = SimpleNamespace(request=AsyncMock(side_effect=respond))
        with patch('psutil.virtual_memory', return_value=SimpleNamespace(available=4 * 2**30)):
            vectors = await _embed(models, ['text'] * 65)
        self.assertEqual(len(vectors), 65)
        self.assertEqual(models.request.await_count, 3)
        for call in models.request.call_args_list:
            self.assertEqual(call.kwargs['json']['options'], {'num_gpu': 0})
        models.request.reset_mock()
        with patch('psutil.virtual_memory', return_value=SimpleNamespace(available=2**30)):
            with self.assertRaises(ValueError): await _embed(models, ['text'])
        models.request.assert_not_called()


class ApiTests(unittest.TestCase):
    def test_saved_endpoint_with_ollama_catalog_name_keeps_its_server(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'SPLASH_DATA_DIR': directory, 'SPLASH_SKIP_MODEL_LOAD': ''}), \
             patch('backend.mcp_host.ToolHost.start'), \
             patch.object(ModelService, 'load', AsyncMock()) as load:
            config = {'source': 'endpoint', 'model': 'qwen3.5:4b', 'base_url': 'http://localhost:8080/v1'}
            Store().set_config('model', config)
            with TestClient(app):
                load.assert_awaited_once_with(config)


    def test_inspection_does_not_replace_active_engine_and_load_needs_confirmation_and_memory(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'SPLASH_DATA_DIR': directory}), \
             patch('backend.mcp_host.ToolHost.start'), \
             patch('backend.app.discover_endpoints', return_value=[]), \
             patch('backend.api.hardware', return_value={'gpus': [], 'available_ram_gb': 8}), \
             patch.object(OllamaClient, 'inspect', AsyncMock(return_value={'available': True, 'installed_models': ['qwen3.5:4b', EMBEDDING_MODEL]})), \
             patch.object(ModelService, 'load', AsyncMock(return_value={'state': 'ready'})) as load:
            with TestClient(app) as client:
                app.state.models.status = {'state': 'ready', 'engine': 'endpoint', 'model': 'maya'}
                self.assertEqual(client.get('/api/models/ollama').status_code, 200)
                self.assertEqual(app.state.models.status['model'], 'maya')
                self.assertEqual(client.post('/api/models/ollama/load', json={'model': 'qwen3.5:4b'}).status_code, 409)
                self.assertEqual(client.post('/api/models/ollama/load', json={'model': 'qwen3.5:9b', 'confirm_separate_model': True}).status_code, 400)
                load.assert_not_called()
                response = client.post('/api/models/ollama/load', json={'model': 'qwen3.5:4b', 'confirm_separate_model': True})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(load.call_args.kwargs['explicit_ollama_load'])
                self.assertTrue(Store().config('model')['cpu_only'])

    def test_nested_indexing_requires_confirmation_and_context_races_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'SPLASH_DATA_DIR': directory}), \
             patch('backend.mcp_host.ToolHost.start'), \
             patch('backend.app.discover_endpoints', return_value=[]), \
             patch.object(OllamaClient, 'inspect', AsyncMock(return_value={'available': True, 'installed_models': [EMBEDDING_MODEL]})):
            with TestClient(app) as client:
                p = client.post('/api/projects', json={'agent': 'splash', 'text': 'Explain caching.'}).json()
                path = f"/api/projects/{p['id']}/local-index"
                self.assertEqual(client.post(path, json={'recursive': True}).status_code, 400)
                client.post(f"/api/projects/{p['id']}/documents", files={'file': ('a.txt', b'Cache data')})
                async def concurrent_change(*args):
                    with app.state.store.edit(p['id']) as current: current['documents'] = []
                    return {'indexed': 1}
                with patch('backend.local_index.index_documents', AsyncMock(side_effect=concurrent_change)):
                    self.assertEqual(client.post(path, json={}).status_code, 409)
                self.assertEqual(app.state.store.get(p['id'])['documents'], [])
