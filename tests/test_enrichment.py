import json
import tempfile
import unittest
from unittest.mock import AsyncMock

from backend.agents import Agents, apply_proposals, missing_constraints
from backend.comments import rewrite_comments
from backend.enrichment import instructions, plan_review, output_budget, parse_suggestions, validate_suggestions
from backend.models import decoded_generation, generation_schedule
from backend.research import parse_document
from backend.storage import Store, Conflict
from tests.enrichment_helpers import WordTokenizer, prepare


def response(quote, replacement, reason='Clarifies which pages need performance work based on the project brief.', evidence=None):
    return json.dumps({'suggestions': [{'quote':quote,'replacement':replacement,'reason':reason,'evidence':evidence or []}]}, ensure_ascii=False)


class BudgetTests(unittest.TestCase):
    def test_adaptive_budgets_fit_small_and_large_profiles(self):
        self.assertEqual(output_budget(50,32768), (512,1024))
        self.assertEqual(output_budget(400,32768), (1024,2048))
        self.assertEqual(output_budget(1000,32768), (2048,4096))
        self.assertEqual(output_budget(5000,32768), (4096,4096))
        self.assertLessEqual(output_budget(5000,1024)[1],512)

    def test_unicode_long_sections_and_references_fit_actual_message_budget(self):
        tokenizer = WordTokenizer()
        text = ('研究语言 🌊 Keep `user_id` and 2015. Compare evidence carefully.\n' * 240)
        refs = [{'id':f'C{i}', 'kind':'document', 'name':f'brief {i} page 2', 'text':'Research context and evidence. '*80} for i in range(8)]
        for context in (1024,2048,8192,32768):
            with self.subTest(context=context):
                result = plan_review(tokenizer,context,text,instructions('splash-search','structured'),refs)
                self.assertEqual(''.join(s['text'] for s in result['sections']),text)
                for section in result['sections']:
                    self.assertLessEqual(section['input_tokens'] + section['retry_tokens'] + 80,context)
                    self.assertEqual(section['text'],text[section['start']:section['end']])
                if context >= 2048:
                    self.assertGreater(result['context_characters_used'],350)
                    self.assertGreater(len(result['references']),1)
        self.assertLessEqual(result['reference_tokens'],4096)

    def test_context_does_not_starve_uploaded_evidence(self):
        refs = [{'id':f'C{i}', 'kind':kind, 'name':kind, 'text':kind+' detail '*800}
                for i,kind in enumerate(['requirements','decisions','objectives','document','document','history'])]
        plan=plan_review(WordTokenizer(),2048,'Make the website faster.',instructions('splash-code','structured'),refs)
        self.assertTrue(any(r['kind']=='document' for r in plan['references']))
        self.assertGreater(plan['context_characters_excluded'],0)

    def test_small_context_fails_clearly(self):
        with self.assertRaisesRegex(ValueError,'context budget is too small'):
            plan_review(WordTokenizer(),512,'Test prompt',instructions('splash','prose'),[])

    def test_model_length_and_mask_outputs_are_reported_as_incomplete(self):
        class Tokenizer:
            eos_token_id=[0,9]
            mask_token_id=8
            def decode(self,tokens,**kwargs):return 'text'
        t=Tokenizer()
        self.assertEqual(decoded_generation(t,[1,2,3],3)['stop_reason'],'length')
        stopped=decoded_generation(t,[1,9,8,8],4)
        self.assertFalse(stopped['truncated'])
        self.assertEqual(stopped['output_tokens'],1)
        self.assertTrue(decoded_generation(t,[1,8,9],4)['truncated'])

    def test_quality_schedule_enables_llada_editing_and_scales_long_outputs(self):
        self.assertEqual(generation_schedule('llada',64,1024),{'threshold':.7,'editing_threshold':.5,'max_post_steps':16})
        self.assertEqual(generation_schedule('llada',64,1024,'speed')['editing_threshold'],0)
        self.assertEqual(generation_schedule('dream',64,512)['steps'],256)
        self.assertEqual(generation_schedule('dream',64,512,'speed')['steps'],64)
        self.assertEqual(generation_schedule('compact',512,128)['steps'],512)

    def test_chat_end_is_not_mistaken_for_truncation_or_another_turn(self):
        class Tokenizer:
            eos_token_id=0
            def convert_tokens_to_ids(self, text):return 7 if text=='<|role_end|>' else 99
            def convert_ids_to_tokens(self, value):return '<|role_end|>' if value==7 else '<unk>'
            def decode(self, values, **kwargs):return repr(values)
        result=decoded_generation(Tokenizer(),[1,2,7,3,4,5],6)
        self.assertFalse(result['truncated'])
        self.assertEqual(result['text'],'[1, 2]')
        self.assertEqual(result['stop_reason'],'message_end')


class SuggestionTests(unittest.TestCase):
    def parse(self, text, raw, refs=None):
        return parse_suggestions(raw,{'text':text,'start':0},1,{},refs or [])

    def test_exact_evidence_and_unicode_anchor(self):
        text='🌊 Make the website faster.'
        refs=[{'id':'C1','kind':'document','name':'Brief p7','text':'Focus on mobile product pages.'}]
        result=self.parse(text,response('Make the website faster.','Optimize the mobile product pages.',
            evidence=[{'id':'C1','quote':'mobile product pages'}]),refs)
        self.assertEqual(result[0]['anchors'][0]['start'],2)
        self.assertEqual(result[0]['evidence'][0]['name'],'Brief p7')
        accepted,rejected=validate_suggestions(text,result,refs,missing_constraints)
        self.assertEqual(len(accepted),1);self.assertEqual(rejected,[])

    def test_questions_do_not_become_edits(self):
        text='Make the website faster.'
        result=self.parse(text,response(text,None,'Which pages and performance measurements matter to your users?'))
        self.assertEqual(result[0]['kind'],'question')
        self.assertIsNone(result[0]['replacement'])

    def test_line_protocol_requires_complete_records_and_preserves_code_pipes(self):
        text='Use `a|b`. Make it faster.'
        raw='QUESTION\nQUOTE: Use `a|b`.\nWHY: Is `a|b` a literal identifier or a choice between alternatives?\nEND'
        self.assertIsNone(self.parse(text,raw)[0]['replacement'])
        self.assertEqual(self.parse(text,'NONE'),[])
        for bad in (raw.removesuffix('\nEND'), 'Here is my suggestion:\n'+raw,
                    raw.replace('QUESTION','EDIT'), raw.replace('WHY:','WHY: a\nWHY:')):
            with self.subTest(raw=bad),self.assertRaises(ValueError): self.parse(text,bad)

    def test_unsupported_evidence_and_ambiguous_anchors_are_rejected(self):
        for text,raw in [('Fast. Fast.',response('Fast.','Optimize performance.')),
                         ('Fast.',response('Fast.','Optimize performance.',evidence=[{'id':'secret','quote':'private'}])),
                         ('Fast.',response('Fast.','Optimize performance.',reason='Improves clarity and precision.'))]:
            with self.subTest(raw=raw), self.assertRaises(ValueError): self.parse(text,raw)
        with self.assertRaises(ValueError):self.parse('Fast.', '{"suggestions": [')

    def test_constraints_and_unsupported_targets_are_withheld(self):
        cases=[('Keep `user_id`.', 'Keep the user identifier.'),
               ('Do not use cloud APIs.', 'Use cloud APIs.'),
               ('Make the website faster. Budget is 15 dollars.', 'Make the website load in 5 seconds.'),
               ('Make the website faster.', 'The website must use caching.'),
               ('Improve performance.', 'Improve performance using `Redis`.')]
        for before,after in cases:
            with self.subTest(before=before):
                candidates=self.parse(before,response(before,after))
                accepted,rejected=validate_suggestions(before,candidates,[],missing_constraints)
                self.assertFalse(accepted);self.assertTrue(rejected)

    def test_historical_context_cannot_authorize_numeric_target(self):
        text='Make the website faster.'
        refs=[{'id':'C1','kind':'history','name':'Prior main','text':'Page target 2 seconds.'}]
        candidates=self.parse(text,response(text,'Make the website load in 2 seconds.',evidence=[{'id':'C1','quote':'2 seconds'}]),refs)
        self.assertFalse(validate_suggestions(text,candidates,refs,missing_constraints)[0])

    def test_overlapping_edits_and_noops_are_not_offered(self):
        text='Make the website faster.'
        first=self.parse(text,response(text,'Optimize website loading.'))
        overlap=self.parse(text,response('website','mobile website'))
        noop=self.parse(text,response(text,text))
        accepted,rejected=validate_suggestions(text,first+overlap+noop,[],missing_constraints)
        self.assertEqual(len(accepted),1)
        self.assertEqual(len(rejected),2)


class GroundedAgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=Store(self.temp.name)
        self.p=self.store.create('splash-code','Make the website faster. Do not change checkout behavior.')
        self.models=type('Model',(),{'status':{'state':'ready','model':'dream','config':{'context':4096}}})()
        self.models.plan_review=AsyncMock(side_effect=prepare)
        self.models.generate=AsyncMock(return_value={'text':response('Make the website faster.',None,'Which pages should be faster, and how will you measure success?')})
        self.agents=Agents(self.store,self.models,None)

    async def test_missing_details_become_questions_and_report_survives_reload(self):
        result=await self.agents.enrich(self.p['id'],1)
        saved=Store(self.temp.name).get(self.p['id'])
        self.assertEqual(saved['proposals'],[])
        self.assertEqual(saved['draft'],self.p['draft'])
        self.assertEqual(saved['main_id'],self.p['main_id'])
        self.assertEqual(saved['comments'][0]['kind'],'question')
        self.assertEqual(saved['enrichment_review']['quality']['questions'],1)
        self.assertGreaterEqual(result['context']['output_budget'],512)

    async def test_truncation_retries_once_with_larger_budget_then_keeps_previous_preview(self):
        with self.store.edit(self.p['id']) as p:
            p['proposals']=[{'id':'previous','rationale':'Language and instruction precision'}]
        self.models.generate.return_value={'text':'{"suggestions":[','truncated':True}
        result=await self.agents.enrich(self.p['id'],1)
        self.assertEqual(self.models.generate.await_count,2)
        budgets=[call.kwargs['max_tokens'] for call in self.models.generate.await_args_list]
        self.assertGreater(budgets[1],budgets[0])
        self.assertEqual(self.store.get(self.p['id'])['proposals'][0]['id'],'previous')
        self.assertTrue(result['quality']['previous_preview_retained'])
        self.assertTrue(result['quality']['errors'])

    async def test_malformed_output_gets_only_one_repair(self):
        self.models.generate.side_effect=[{'text':'I would rewrite everything.'},{'text':'{"suggestions":[]}'}]
        result=await self.agents.enrich(self.p['id'],1)
        self.assertEqual(self.models.generate.await_count,2)
        self.assertEqual(result['quality']['status'],'no_changes')
        self.assertEqual(self.models.generate.await_args_list[0].kwargs['max_tokens'],self.models.generate.await_args_list[1].kwargs['max_tokens'])

    async def test_current_context_changes_invalidate_review(self):
        async def changed(*args,**kwargs):
            with self.store.edit(self.p['id']) as p:
                p['documents'].append(parse_document('changed.txt',b'New scope'))
            return {'text':'{"suggestions":[]}'}
        self.models.generate.side_effect=changed
        with self.assertRaises(Conflict): await self.agents.enrich(self.p['id'],1)
        self.assertFalse(self.store.get(self.p['id'])['proposals'])

    async def test_decisions_are_still_enforced_and_noop_pass_retires_old_preview(self):
        with self.store.edit(self.p['id']) as p:
            p['comments']=rewrite_comments(p['draft']['text'],'Optimize loading. Do not change checkout behavior.',1,[],{})
        self.models.generate.return_value={'text':'{"suggestions":[]}'}
        await self.agents.enrich(self.p['id'],1)
        p=self.store.get(self.p['id'])
        self.assertTrue(p['comments'][0]['outdated'])
        self.assertEqual(p['draft']['text'],self.p['draft']['text'])


if __name__=='__main__': unittest.main()
