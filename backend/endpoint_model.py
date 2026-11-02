"""Inference through a user-selected local OpenAI-compatible endpoint; no weights."""
import asyncio
import time
import httpx
from backend.endpoints import probe_endpoint, endpoint_config, headers


class BudgetTokenizer:
    """Conservative UTF-8 byte estimate, not an endpoint-specific tokenizer."""
    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))

    def apply_chat_template(self, messages, **kwargs):
        return [0] * (32 + sum(len(m['content'].encode('utf-8')) + 32 for m in messages))


class EndpointModel:
    def __init__(self, config):
        self.config = config
        self.context = config.get('context', 4096)
        self.tokenizer = BudgetTokenizer()

    async def verify(self):
        endpoint = await asyncio.to_thread(probe_endpoint, self.config['base_url'])
        if endpoint is None:
            raise ValueError('The saved model endpoint is unavailable. Start it or choose a model in Model & hardware. No local model was loaded.')
        self.config = endpoint_config(endpoint, self.config['model'], self.context)
        self.context = self.config['context']
        return {'connected': True, 'loaded': False, 'latency_ms': 0,
                'context_limit': self.context, 'budget_method': 'utf8_byte_estimate'}

    async def generate(self, messages, max_tokens):
        # Recheck known loaded inventories to avoid waking a different/unloaded
        # model. The server still owns memory allocation and concurrent requests.
        await self.verify()
        count = len(self.tokenizer.apply_chat_template(messages))
        if count + max_tokens > self.context:
            raise ValueError('Request exceeds the endpoint context budget. Shorten the prompt or increase its configured budget.')
        started = time.monotonic()
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=300) as client:
            response = await client.post(self.config['base_url'] + '/chat/completions', headers=headers(),
                json={'model': self.config['model'], 'messages': messages, 'max_tokens': max_tokens,
                      'temperature': 0, 'stream': False})
        if response.status_code != 200:
            raise ValueError(f'Model endpoint returned HTTP {response.status_code}. Check its logs and available memory; Splash did not load a fallback model.')
        data = response.json()
        try:
            choice = data['choices'][0]
            content = choice['message']['content']
            finish = choice.get('finish_reason')
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError('Endpoint returned an invalid chat-completion response.') from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Endpoint returned no text. Choose a chat/instruct model.')
        usage = data.get('usage') or {}
        return {'ok': True, 'text': content.strip(), 'truncated': finish != 'stop',
                'stop_reason': finish or 'unknown', 'max_output_tokens': max_tokens,
                'output_tokens': usage.get('completion_tokens'), 'input_tokens': usage.get('prompt_tokens', count),
                'latency_ms': round((time.monotonic() - started) * 1000), 'context_limit': self.context,
                'engine': 'endpoint', 'budget_method': 'utf8_byte_estimate'}

    def plan_review(self, text, system, references):
        from backend.enrichment import plan_review
        return {**plan_review(self.tokenizer, self.context, text, system, references),
                'budget_method': 'utf8_byte_estimate'}

    def sections(self, text, system, suffix, max_tokens):
        overhead = len(self.tokenizer.apply_chat_template([{'content': system}, {'content': 'Prompt section:\n' + suffix}]))
        budget = self.context - max_tokens - overhead - 256
        if budget < 64:
            raise ValueError('Endpoint context budget is too small for this review.')
        sections, start = [], 0
        while start < len(text):
            low, high = 0, len(text) - start
            while low < high:
                mid = (low + high + 1) // 2
                if len(text[start:start+mid].encode('utf-8')) <= budget:
                    low = mid
                else:
                    high = mid - 1
            if not low:
                raise ValueError('Endpoint context budget cannot fit this passage.')
            end = start + low
            sections.append({'text': text[start:end], 'start': start, 'end': end})
            start = end
        return {'sections': sections}
