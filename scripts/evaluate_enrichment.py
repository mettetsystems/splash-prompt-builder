"""Evaluate grounded suggestions on repeatable cases without touching user projects.

Automatic checks cover structure/literal fidelity only. Inspect the saved model
responses and resulting edits to judge meaning, usefulness and unsupported facts.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.agents import Agents, POLICIES, RULES, apply_proposals, missing_constraints
from backend.models import DiffusionAdapter, CATALOG
from backend.storage import Store
from backend.research import parse_document

ROOT = Path(__file__).resolve().parents[1]


async def evaluate(args):
    os.environ.setdefault('HF_MODULES_CACHE', str(ROOT / '.models/modules'))
    config = {'model': args.model, 'path': str(args.path.resolve()), 'device': args.device,
              'context': args.context or CATALOG[args.model]['context'], 'steps': args.steps, 'quantize': args.quantize,
              'quality_mode': args.quality_mode}
    adapter = DiffusionAdapter(config)
    cancel = threading.Event()
    raw = []
    class LocalModel:
        status = {'state': 'ready', 'model': args.model, 'config': config}
        async def plan_review(self, text, system, references):
            return adapter.plan_review(text, system, references)
        async def generate(self, messages, max_tokens=256, priority=0):
            response = adapter.generate(messages, cancel, max_tokens)
            raw.append(response)
            return response
    cases = json.loads(args.cases.read_text())
    results = []
    with tempfile.TemporaryDirectory(prefix='splash-quality-') as directory:
        store = Store(directory)
        agents = Agents(store, LocalModel(), None)
        for case in cases[:args.limit]:
            raw.clear()
            p = store.create(case['agent'], case['prompt'])
            if case.get('context'):
                with store.edit(p['id']) as current:
                    current['documents'].append(parse_document('evaluation-brief.txt', case['context'].encode()))
            baseline = None
            if args.baseline:
                # Reproduce the former short rewrite request for direct human comparison.
                suffix = '\nReference context: ' + case.get('context', '')[:350]
                baseline = adapter.generate([{'role':'system','content':POLICIES[case['agent']]+' '+RULES},
                    {'role':'user','content':'Prompt section:\n'+case['prompt']+suffix}], cancel, 256)
            try:
                result = await agents.enrich(p['id'], 1)
                saved = store.get(p['id'])
                preview = apply_proposals(saved, [e['id'] for e in saved['proposals']])[0] if saved['proposals'] else case['prompt']
                record = {'case': case, 'config': config, 'result': result, 'preview': preview,
                          'comments': saved['comments'], 'responses': list(raw), 'baseline': baseline,
                          'checks': {'main_unchanged': saved['main_id']==p['main_id'], 'draft_unchanged': saved['draft']==p['draft'],
                                     'missing_literals': missing_constraints(case['prompt'], preview)},
                          'human_review': {'useful': None, 'preserves_meaning': None, 'unsupported_claims': None, 'notes': ''}}
            except Exception as exc:
                record = {'case': case, 'config': config, 'error': str(exc), 'responses': list(raw), 'baseline': baseline}
            results.append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            print(json.dumps({'case':case['id'], 'error':record.get('error'), 'quality':record.get('result',{}).get('quality'),
                              'generation_ms':sum(r.get('latency_ms',0) for r in raw)}, ensure_ascii=False), flush=True)
    return 1 if any('error' in r or r.get('result', {}).get('quality', {}).get('errors') for r in results) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=CATALOG, required=True)
    parser.add_argument('--path', type=Path, required=True, help='Existing local checkpoint; no downloads')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--context', type=int)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--quality-mode', choices=['quality','speed'], default='quality')
    parser.add_argument('--baseline', action='store_true', help='Also record the previous 256-token rewrite')
    parser.add_argument('--cases', type=Path, default=ROOT/'tests/fixtures/enrichment_cases.json')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/grounded-enrichment-evaluation.json')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be at least one')
    return asyncio.run(evaluate(args))


if __name__ == '__main__':
    raise SystemExit(main())
