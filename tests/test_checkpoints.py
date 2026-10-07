import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from backend.app import app
from backend.agents import Agents
from backend.checkpoints import (repository_id, source_spec, inspect_local, inspect_remote,
                                 fetch_checkpoint, downloaded_source)


def checkpoint(path, kind="a2d-qwen3"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": kind}))
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"test weights; model execution is mocked")
    return path


class CheckpointTests(unittest.TestCase):
    def test_repository_urls_and_revision_keys(self):
        self.assertEqual(repository_id(" https://huggingface.co/team/model/ "), "team/model")
        for bad in ("https://evil.example/team/model", "https://huggingface.co@evil.example/a/b",
                    "https://huggingface.co/team/model/tree/main", "../model", "/tmp/checkpoint", "team/../model"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                repository_id(bad)
        self.assertEqual(source_spec("compact")["source"], "catalog")
        self.assertEqual(source_spec("compact", "team/model")["requested_revision"], "main")
        self.assertNotEqual(source_spec("compact", "team/model", "main")["id"],
                            source_spec("compact", "team/model", "release")["id"])

    def test_local_complete_checkpoint_with_spaces_and_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory) / "Local 模型")
            result = inspect_local(path, "compact")
            self.assertEqual(result["path"], str(path))
            self.assertEqual(result["weight_files"], 1)
            with self.assertRaisesRegex(ValueError, "selected adapter requires"):
                inspect_local(path, "dream")

    def test_partial_download_lfs_and_invalid_shard_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = checkpoint(Path(directory))
            (path / "model.safetensors").write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc")
            with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
                inspect_local(path, "compact")
            (path / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "No Hugging Face model weights"):
                inspect_local(path, "compact")
            index = path / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"layer": "missing.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "Missing or empty"):
                inspect_local(path, "compact")
            index.write_text(json.dumps({"weight_map": {"layer": "../secret"}}))
            with self.assertRaisesRegex(ValueError, "invalid file path"):
                inspect_local(path, "compact")

    def test_remote_branch_is_resolved_and_downloaded_at_immutable_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = checkpoint(root / "metadata") / "config.json"
            api = Mock()
            api.model_info.return_value = SimpleNamespace(sha="a"*40, siblings=[SimpleNamespace(size=1000)])
            hub = SimpleNamespace(HfApi=Mock(return_value=api), hf_hub_download=Mock(return_value=str(config)), snapshot_download=Mock())
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                record = inspect_remote(root, source_spec("compact", "team/diffusion", "release"))
                self.assertEqual(record["revision"], "a"*40)
                self.assertEqual(hub.hf_hub_download.call_args.kwargs["revision"], "a"*40)
                self.assertIn("a"*40, record["path"])
                checkpoint(Path(record["path"]))
                fetch_checkpoint(record)
                self.assertEqual(hub.snapshot_download.call_args.kwargs["revision"], "a"*40)
                self.assertEqual(hub.snapshot_download.call_args.args[0], "team/diffusion")

    def test_saved_download_can_be_found_by_pinned_sha_after_restart(self):
        record = {**source_spec("compact", "team/model"), "state": "complete", "revision": "b"*40, "path": "/tmp/checkpoint"}
        saved = json.loads(json.dumps({record["id"]: record}))
        self.assertEqual(downloaded_source(saved, source_spec("compact", "team/model", "b"*40)), record)
        with self.assertRaises(ValueError):
            downloaded_source(saved, source_spec("compact", "other/model", "b"*40))

    def test_generation_provenance_uses_actual_repository_or_local_source(self):
        model = SimpleNamespace(status={"model": "compact", "config": {
            "source": "huggingface", "repo": "team/custom", "revision": "c"*40}})
        info = Agents(None, model, None).generation_info()
        self.assertEqual(info["model_repository"], "team/custom")
        self.assertEqual(info["model_revision"], "c"*40)
        model.status["config"] = {"source": "local", "repo": "", "revision": ""}
        self.assertIsNone(Agents(None, model, None).generation_info()["model_revision"])


class ModelSourceAPI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(os.environ, {"SPLASH_DATA_DIR": self.temp.name})
        env.start(); self.addCleanup(env.stop)
        host = patch("backend.mcp_host.ToolHost.start")
        host.start(); self.addCleanup(host.stop)
        self.client = TestClient(app).__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        dependencies = patch("backend.api.dependency_status", return_value={"python": "/app/.venv/bin/python", "missing": [], "packages": {"bitsandbytes": "0.50.2"}})
        dependencies.start(); self.addCleanup(dependencies.stop)

    def wait_download(self, key):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            record = self.client.get('/api/models').json()['downloads'][key]
            if record['state'] != 'downloading':
                return record
            time.sleep(.01)
        self.fail("Download task did not finish")

    def test_local_inspection_and_loading_preserve_actual_source(self):
        path = checkpoint(self.root / "my local model")
        response = self.client.post('/api/models/inspect', json={"model": "compact", "path": str(path)})
        self.assertEqual(response.status_code, 200, response.text)
        async def load(config):
            app.state.models.status = {"state": "ready", "model": "compact", "config": config}
            return app.state.models.status
        with patch.object(app.state.models, "load", side_effect=load):
            response = self.client.post('/api/models/load', json={"model": "compact", "source": "local", "path": str(path)})
            self.assertEqual(response.status_code, 200, response.text)
            loaded = self.client.get('/api/models').json()['runtime']['config']
        self.assertEqual(loaded["source"], "local")
        self.assertEqual(loaded["revision"], "")
        self.assertEqual(loaded["path"], str(path))

    def test_custom_download_is_persisted_and_reusable(self):
        spec = source_spec("compact", "team/custom", "main")
        path = checkpoint(self.root / "downloaded")
        record = {**spec, "path": str(path), "revision": "d"*40, "model_type": "a2d-qwen3"}
        with patch("backend.api.inspect_remote", return_value=record), patch("backend.api.fetch_checkpoint", return_value=inspect_local(path, "compact")):
            response = self.client.post('/api/models/download', json={"model": "compact", "repo": "team/custom"})
            self.assertEqual(response.status_code, 200, response.text)
            complete = self.wait_download(response.json()['id'])
        self.assertEqual(complete['state'], 'complete')
        saved = app.state.store.config("model_downloads")
        self.assertEqual(saved[spec['id']]['revision'], 'd'*40)
        async def load(config):
            app.state.models.status = {"state": "ready", "model": "compact", "config": config}
            return app.state.models.status
        with patch.object(app.state.models, "load", side_effect=load):
            response = self.client.post('/api/models/load', json={"model": "compact", "source": "huggingface", "repo": "team/custom", "revision": "d"*40})
            self.assertEqual(response.status_code, 200, response.text)
            config = self.client.get('/api/models').json()['runtime']['config']
        self.assertEqual(config['repo'], 'team/custom')
        self.assertEqual(config['revision'], 'd'*40)

    def test_missing_dependency_has_actionable_error_before_load(self):
        with patch("backend.api.dependency_status", return_value={"python": "/different/.venv/bin/python", "missing": ["torch"], "packages": {}}):
            response = self.client.post('/api/models/load', json={"model": "compact"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("/different/.venv/bin/python", response.json()['detail'])
        self.assertIn("--install-models", response.json()['detail'])
        self.assertIsNone(app.state.store.config('model'))

    def test_download_errors_are_visible_and_never_persist_as_complete(self):
        with patch("backend.api.inspect_remote", side_effect=ValueError("Repository not found or access denied")):
            response = self.client.post('/api/models/download', json={"model": "compact", "repo": "team/private"})
            record = self.wait_download(response.json()['id'])
        self.assertEqual(record['state'], 'error')
        self.assertIn('access denied', record['error'])
        self.assertIsNone(app.state.store.config('model_downloads'))

    def test_invalid_sources_do_not_change_saved_model(self):
        for body in ({"model": "compact", "repo": "https://evil.example/model"},
                     {"model": "dream", "path": str(checkpoint(self.root / "compact"))}):
            self.assertEqual(self.client.post('/api/models/inspect', json=body).status_code, 400)
        self.assertEqual(self.client.post('/api/models/load', json={"model": "compact", "source": "local"}).status_code, 400)
        self.assertIsNone(app.state.store.config('model'))


if __name__ == '__main__':
    unittest.main()
