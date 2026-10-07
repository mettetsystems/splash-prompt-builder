import asyncio
import unittest
from unittest.mock import AsyncMock

from backend.models import ModelService, DiffusionAdapter


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
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
