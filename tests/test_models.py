import asyncio
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from backend.models import ModelService, DiffusionAdapter, cuda_memory_preflight, run_worker


class MemoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        (self.path / 'config.json').write_text(json.dumps({'dtype': 'bfloat16'}))
        self.config = {'model': 'llada', 'device': 'cuda:1', 'path': str(self.path), 'quantize': True}
        self.cuda = Mock()
        self.cuda.mem_get_info.return_value = (8 * 2**30, 32 * 2**30)
        self.cuda.get_device_name.return_value = 'Test 5090'
        self.cuda.get_device_capability.return_value = (12, 0)
        self.torch = SimpleNamespace(set_num_threads=Mock(), set_num_interop_threads=Mock(),
                                     backends=Mock(), cuda=self.cuda, float32='float32', float16='float16', bfloat16='bfloat16')
        checkpoint = patch('backend.checkpoints.inspect_local', return_value={'weight_gb': 32.51})
        self.inspect = checkpoint.start()
        self.addCleanup(checkpoint.stop)

    def test_busy_selected_gpu_rejected_before_tokenizer_or_weights(self):
        transformers = Mock()
        with patch.dict(sys.modules, torch=self.torch, transformers=transformers):
            with self.assertRaisesRegex(RuntimeError, '8.0 GiB free.*11.7 GiB'):
                DiffusionAdapter(self.config)
        self.cuda.mem_get_info.assert_called_once_with('cuda:1')
        transformers.AutoTokenizer.from_pretrained.assert_not_called()
        transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_free_gpu_loads_with_supported_dtype_and_memory_report(self):
        self.cuda.mem_get_info.return_value = (30 * 2**30, 32 * 2**30)
        transformers = Mock()
        with patch.dict(sys.modules, torch=self.torch, transformers=transformers):
            adapter = DiffusionAdapter({**self.config, 'model': 'dream'})
        kwargs = transformers.AutoModel.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs['dtype'], 'bfloat16')
        self.assertNotIn('torch_dtype', kwargs)
        self.assertEqual(adapter.memory_check['free_gib'], 30)
        self.assertEqual(adapter.memory_check['device'], 'cuda:1')

    def test_cpu_and_metal_do_not_query_cuda(self):
        for device in ('cpu', 'mps'):
            self.assertIsNone(cuda_memory_preflight({**self.config, 'device': device}, self.torch))
        self.cuda.mem_get_info.assert_not_called()
        self.inspect.assert_not_called()

    def test_checkpoint_precision_and_nf4_change_estimate(self):
        self.cuda.mem_get_info.return_value = (48 * 2**30, 48 * 2**30)
        fp16 = cuda_memory_preflight({**self.config, 'quantize': False}, self.torch)
        nf4 = cuda_memory_preflight(self.config, self.torch)
        self.assertGreater(fp16['estimated_required_gib'], 32)
        self.assertLess(nf4['estimated_required_gib'], 12)
        (self.path / 'config.json').write_text(json.dumps({'torch_dtype': 'float32'}))
        fp32_checkpoint = cuda_memory_preflight({**self.config, 'quantize': False}, self.torch)
        self.assertAlmostEqual(fp32_checkpoint['estimated_required_gib'] * 2, fp16['estimated_required_gib'], delta=.02)

    def test_prequantized_weights_are_not_discounted_twice(self):
        self.cuda.mem_get_info.return_value = (48 * 2**30, 48 * 2**30)
        (self.path / 'config.json').write_text(json.dumps({'dtype': 'bfloat16', 'quantization_config': {'load_in_4bit': True}}))
        report = cuda_memory_preflight(self.config, self.torch)
        self.assertGreater(report['estimated_required_gib'], 32)

    def test_offload_budget_checks_selected_gpu_without_full_weight_requirement(self):
        # The parser is supplied by Accelerate; model packages are optional in CI.
        utils = SimpleNamespace(convert_file_size_to_int=lambda text: int(text.removesuffix('GiB')) * 2**30)
        with patch.dict(sys.modules, {'accelerate.utils': utils}):
            report = cuda_memory_preflight({**self.config, 'quantize': False, 'offload': True, 'gpu_memory': '6GiB'}, self.torch)
            self.assertEqual(report['estimated_required_gib'], 6.75)
            self.cuda.mem_get_info.return_value = (6 * 2**30, 32 * 2**30)
            with self.assertRaisesRegex(RuntimeError, 'configured GPU offload budget'):
                cuda_memory_preflight({**self.config, 'offload': True}, self.torch)
        self.inspect.assert_not_called()

    def test_failed_load_releases_traceback_and_allows_retry(self):
        class Allocation: pass
        partials = []
        successful = Mock(memory_check=None)
        successful.generate.return_value = {'text': 'ok'}
        successful.sections.return_value = {'sections': []}
        def create(config):
            if not partials:
                allocation = Allocation()
                partials.append(weakref.ref(allocation))
                raise RuntimeError('simulated OOM')
            return successful
        def clean_cache():
            if partials:
                self.assertIsNone(partials[0](), 'failed constructor is still held by its traceback')
        self.cuda.empty_cache.side_effect = clean_cache
        inbox, outbox = queue.Queue(), queue.Queue()
        for job in [{'op': 'load', 'config': self.config}, {'op': 'load', 'config': self.config},
                    {'op': 'sections', 'text': '', 'system': '', 'suffix': '', 'max_tokens': 64}, None]:
            inbox.put(job)
        with patch.dict(sys.modules, torch=self.torch), patch('backend.models.DiffusionAdapter', side_effect=create):
            run_worker(inbox, outbox, Mock())
        self.assertIn('simulated OOM', outbox.get_nowait()['error'])
        self.assertTrue(outbox.get_nowait()['loaded'])
        self.assertEqual(outbox.get_nowait()['sections'], [])
        self.assertGreaterEqual(self.cuda.empty_cache.call_count, 3)

    def test_failed_execution_check_discards_loaded_model(self):
        references = []
        def create(config):
            adapter = Mock(memory_check=None)
            adapter.generate.side_effect = RuntimeError('qualification OOM')
            references.append(weakref.ref(adapter))
            return adapter
        inbox, outbox = queue.Queue(), queue.Queue()
        for job in [{'op': 'load', 'config': self.config}, {'op': 'sections'}, None]:
            inbox.put(job)
        with patch.dict(sys.modules, torch=self.torch), patch('backend.models.DiffusionAdapter', side_effect=create):
            run_worker(inbox, outbox, Mock())
        self.assertIn('qualification OOM', outbox.get_nowait()['error'])
        self.assertIn('Load a diffusion model first', outbox.get_nowait()['error'])
        self.assertIsNone(references[0]())


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_releases_ipc_and_is_idempotent(self):
        service = ModelService()
        await service.close()
        await service.close()
        self.assertIsNone(service.inbox)
        self.assertIsNone(service.outbox)
        self.assertIsNone(service.cancel)

    async def test_only_latest_pending_request_runs(self):
        service = ModelService()
        service.status = {"state":"ready"}
        started, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def exchange(job):
            calls.append(job["messages"])
            if len(calls) == 1:
                started.set(); await release.wait()
            return {"ok":True, "text":"result"}
        service.exchange = exchange
        first = asyncio.create_task(service.generate([{"content":"old"}]))
        await started.wait()
        middle = asyncio.create_task(service.generate([{"content":"middle"}]))
        await asyncio.sleep(0)
        latest = asyncio.create_task(service.generate([{"content":"latest"}]))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(first,middle,latest,return_exceptions=True)
        self.assertIsInstance(results[0],InterruptedError)
        self.assertIsInstance(results[1],InterruptedError)
        self.assertEqual(results[2]["text"],"result")
        self.assertEqual([c[0]["content"] for c in calls],["old","latest"])

    async def test_background_does_not_interrupt_editor(self):
        service = ModelService(); service.status={"state":"ready"}; service.active_priority=0
        with self.assertRaises(InterruptedError): await service.generate([],priority=2)
        self.assertEqual(service.sequence,0)

    async def test_cancellation_drains_worker_result(self):
        service=ModelService();service.status={"state":"ready"}
        started,finish=asyncio.Event(),asyncio.Event()
        async def exchange(job):
            started.set();await finish.wait();return {"ok":False,"error":"canceled"}
        service.exchange=exchange
        task=asyncio.create_task(service.generate([]));await started.wait();task.cancel()
        await asyncio.sleep(0)
        self.assertTrue(service.cancel.is_set())
        self.assertTrue(service.lock.locked())
        finish.set()
        await asyncio.gather(task,return_exceptions=True)
        self.assertFalse(service.lock.locked())


class SectionTests(unittest.TestCase):
    def test_spawned_worker_shutdown_has_no_resource_tracker_warning(self):
        # Use a real spawned process, without importing Torch or downloading weights.
        script = """
import asyncio
import multiprocessing as mp
from backend.models import ModelService, worker_main
async def check(force=False):
    service = ModelService()
    service.process = mp.get_context('spawn').Process(target=worker_main, args=(service.inbox, service.outbox, service.cancel))
    service.process.start()
    result = await service.exchange({'op':'sections'})
    assert result['ok'] is False
    if force:
        service.process.terminate()
        service.process.join(5)
        assert service.process.exitcode == 0
    await service.close()
if __name__ == '__main__':
    asyncio.run(check())
    import os
    if os.name != 'nt':
        asyncio.run(check(force=True))
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'check_shutdown.py'
            path.write_text(script)
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
            result = subprocess.run([sys.executable, str(path)], env=env, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('leaked semaphore', result.stderr)

    def test_exact_unicode_sections_and_token_budget(self):
        adapter=DiffusionAdapter.__new__(DiffusionAdapter)
        adapter.context=512
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs): return list(''.join(m['content'] for m in messages))
            def encode(self,text,**kwargs): return list(text)
        adapter.tokenizer=Tokenizer()
        text=('研究语言。 🌙 Preserve `name` and 42.\n\n'*30)
        result=adapter.sections(text,'Improve the prompt.','Reference text',128)
        self.assertEqual(''.join(s['text'] for s in result['sections']),text)
        self.assertGreater(len(result['sections']),1)
        self.assertTrue(all(len(s['text'])<=result['input_budget'] for s in result['sections']))


if __name__=='__main__':unittest.main()
