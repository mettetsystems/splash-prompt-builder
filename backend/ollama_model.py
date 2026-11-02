"""Ollama inference with the same conservative review budgets as endpoint reuse."""
import time
from backend.endpoint_model import EndpointModel
from backend.ollama import OllamaClient


class OllamaModel(EndpointModel):
    @property
    def client(self):
        return OllamaClient(self.config['base_url'].removesuffix('/v1'))

    async def verify(self):
        saved = dict(self.config)
        result = await super().verify()
        self.config.update(source='ollama', cpu_only=saved.get('cpu_only', False))
        return result

    async def load(self):
        # Called only by the explicitly confirmed load action, never startup.
        await self.client.request('POST', '/api/chat', json=self.payload(
            [{'role': 'user', 'content': 'Say ready.'}], 16))

    def payload(self, messages, max_tokens):
        return {'model': self.config['model'], 'messages': messages, 'stream': False,
                'keep_alive': '10m', 'options': {'num_ctx': self.context, 'num_predict': max_tokens,
                'temperature': 0, **({'num_gpu': 0} if self.config.get('cpu_only') else {})}}

    async def generate(self, messages, max_tokens):
        await self.verify()
        count = len(self.tokenizer.apply_chat_template(messages))
        if count + max_tokens > self.context:
            raise ValueError('Request exceeds the endpoint context budget.')
        started = time.monotonic()
        response = await self.client.request('POST', '/api/chat', json=self.payload(messages, max_tokens))
        data = response.json()
        text = data.get('message', {}).get('content', '').strip()
        if not text:
            raise ValueError('Ollama returned no text. Choose a chat/instruct model.')
        finish = data.get('done_reason')
        return {'ok': True, 'text': text, 'truncated': finish != 'stop', 'stop_reason': finish or 'unknown',
                'max_output_tokens': max_tokens, 'output_tokens': data.get('eval_count'),
                'input_tokens': data.get('prompt_eval_count', count), 'context_limit': self.context,
                'latency_ms': round((time.monotonic() - started) * 1000),
                'engine': 'ollama', 'budget_method': 'utf8_byte_estimate'}
