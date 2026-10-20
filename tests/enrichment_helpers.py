"""Lightweight tokenizer and response fixtures; no model downloads in unit tests."""
import json
import re

from backend.comments import rewrite_comments
from backend.enrichment import plan_review


class WordTokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return re.findall(r'\w+|[^\w\s]', text)

    def apply_chat_template(self, messages, **kwargs):
        return ['chat'] * 12 + self.encode('\n'.join(m['content'] for m in messages))


async def prepare(text, system, references):
    return plan_review(WordTokenizer(), 4096, text, system, references)


def edit_response(before, after):
    return json.dumps({'suggestions': [{'quote': c['anchors'][0]['quote'], 'replacement': c['replacement'],
        'reason': 'Clarifies the requested action in this specific passage.', 'evidence': []}
        for c in rewrite_comments(before, after, 1, [], {}, limit=4)]})
