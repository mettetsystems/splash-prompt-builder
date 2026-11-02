import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.app import app
from backend.agents import Agents
from backend.comments import (anchor, thread, add_threads, reanchor_threads, consistency_comments,
    parse_review, rewrite_comments, review_document, answer_comment)
from backend.storage import Store, Conflict


class CommentTests(unittest.TestCase):
    def test_multiple_conflicts_quote_both_claims(self):
        text = 'Use Python. Do not use Python. The budget is $500. The budget is $900. Write exactly 200 words. Write exactly 300 words.'
        comments = consistency_comments(text, 3)
        self.assertEqual(len(comments), 3)
        for c in comments:
            self.assertEqual(c['kind'], 'conflict')
            self.assertIsNone(c['replacement'])
            self.assertEqual(c['messages'][0]['author'], 'consistency')
            self.assertEqual(len(c['anchors']), 2)
            self.assertTrue(all(text[a['start']:a['end']] == a['quote'] for a in c['anchors']))

    def test_compatible_counts_and_distinct_facts_do_not_conflict(self):
        self.assertEqual(consistency_comments('Write at least 200 words. Write at most 500 words. The budget is $500. The audience is students.', 1), [])

    def test_complementary_descriptions_and_different_units_are_not_conflicts(self):
        text = ('The website is fast. The website is colorful. '
                'The width is 1 m. The width is 100 cm. '
                'The budget is $500. The budget is $500.00.')
        self.assertEqual(consistency_comments(text, 1), [])

    def test_reanchor_unicode_without_losing_threads_or_replies(self):
        old = '🌊 Draft: Use Python.'
        comment = thread('question', 'Which version?', [anchor(old, 9, len(old), 1)])
        p = {'comments':[comment]}
        updated = 'Research notes. ' + old
        reanchor_threads(p, old, updated, 2)
        self.assertFalse(comment['outdated'])
        self.assertEqual(comment['anchors'][0]['quote'], 'Use Python.')
        self.assertEqual(comment['anchors'][0]['start'], 25)
        self.assertEqual(comment['anchors'][0]['version'], 2)
        reanchor_threads(p, updated, updated.replace('Python', 'Rust'), 3)
        self.assertTrue(comment['outdated'])
        self.assertEqual(comment['messages'][0]['text'], 'Which version?')

    def test_background_review_deduplicates_and_preserves_resolution(self):
        text = 'Use Python. Do not use Python.'
        p = {'comments':[]}
        add_threads(p, consistency_comments(text, 1))
        p['comments'][0]['status'] = 'resolved'
        p['comments'][0]['messages'].append({'author':'you','text':'Use Python for tooling only.'})
        add_threads(p, consistency_comments(text, 1))
        self.assertEqual(len(p['comments']), 1)
        self.assertEqual(len(p['comments'][0]['messages']), 2)
        self.assertEqual(p['comments'][0]['status'], 'resolved')

    def test_model_review_requires_exact_unique_anchors_and_two_conflict_quotes(self):
        text = 'Use Python. Do not use Python.'
        data = {'comments':[{'kind':'conflict','quote':'Use Python.','related_quote':'Do not use Python.','message':'Which direction should apply?'}]}
        result = parse_review(json.dumps(data), text, 1, {'model':'dream'})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['generation']['model'], 'dream')
        for changes in ({'quote':'Invented text'}, {'related_quote':''}, {'kind':'command'}, {'replacement':'execute me'}):
            bad = copy.deepcopy(data); bad['comments'][0].update(changes)
            with self.assertRaises(ValueError): parse_review(json.dumps(bad), text, 1, {})

    def test_suggestions_use_whole_words_and_exact_replacements(self):
        comments = rewrite_comments('🌊 write ai prompts', '🌊 Develop AI prompts', 1, [], {})
        self.assertTrue(comments)
        for comment in comments:
            a = comment['anchors'][0]
            self.assertEqual('🌊 write ai prompts'[a['start']:a['end']], a['quote'])
        self.assertEqual(comments[0]['anchors'][0]['quote'], 'write ai ')
        self.assertEqual(comments[0]['replacement'], 'Develop AI ')

    def test_numbered_model_review_resolves_quotes_on_the_host(self):
        text = 'Work offline. Every request calls a cloud API.'
        data = {'comments':[{'kind':'conflict','passages':[1,2],'message':'Which connectivity requirement applies?'}]}
        comments = parse_review(json.dumps(data), text, 1, {})
        self.assertEqual([a['quote'] for a in comments[0]['anchors']], ['Work offline.', 'Every request calls a cloud API.'])
        for ids in ([1,3],[1,1],[True,2],[]):
            data['comments'][0]['passages'] = ids
            with self.assertRaises(ValueError): parse_review(json.dumps(data),text,1,{})
        lines='CONFLICT 1,2 | Which requirement takes priority?\nQUESTION 1 | Is offline use mandatory?'
        self.assertEqual(len(parse_review(lines,text,1,{})),2)
        self.assertEqual(parse_review('NONE',text,1,{}),[])
        with self.assertRaises(ValueError): parse_review('CONFLICT 1,99 | Invented passage',text,1,{})


class CommentAPI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {'SPLASH_DATA_DIR':self.temp.name}); env.start(); self.addCleanup(env.stop)
        mcp = patch('backend.mcp_host.ToolHost.start'); mcp.start(); self.addCleanup(mcp.stop)
        self.client = TestClient(app).__enter__(); self.addCleanup(self.client.__exit__, None, None, None)
        self.p = self.client.post('/api/projects', json={'agent':'splash', 'text':'🌊 Use Python. Include tests.'}).json()
        self.base = '/api/projects/'+self.p['id']

    def comment(self):
        r = self.client.post(self.base+'/comments', json={'text':'Which version?','start':2,'end':13,'expected_version':1})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()['comments'][0]

    def test_comment_reply_resolution_reload_and_project_isolation(self):
        c = self.comment()
        r = self.client.post(self.base+f'/comments/{c["id"]}/reply',json={'text':'Use the version already installed.'})
        self.assertEqual(r.status_code,200)
        self.client.patch(self.base+f'/comments/{c["id"]}',json={'status':'resolved'})
        reloaded = Store(self.temp.name).get(self.p['id'])
        self.assertEqual(reloaded['comments'][0]['status'],'resolved')
        self.assertEqual(len(reloaded['comments'][0]['messages']),2)
        self.assertEqual(reloaded['draft'],self.p['draft'])
        other = self.client.post('/api/projects',json={'agent':'splash-code','text':'Another project'}).json()
        r = self.client.post('/api/projects/'+other['id']+f'/comments/{c["id"]}/reply',json={'text':'Wrong project'})
        self.assertEqual(r.status_code,404)

    def test_applying_suggestion_preserves_main_and_archives_input(self):
        with app.state.store.edit(self.p['id']) as p:
            p['comments'] = rewrite_comments(p['draft']['text'],'🌊 Use Python 3. Include tests.',1,[],{'model':'dream'})
        c = p['comments'][0]
        result = self.client.post(self.base+f'/comments/{c["id"]}/apply',json={'expected_version':1})
        self.assertEqual(result.status_code,200,result.text)
        result = result.json()
        self.assertEqual(result['draft']['text'],'🌊 Use Python 3. Include tests.')
        self.assertEqual(result['main_id'],self.p['main_id'])
        self.assertIn(self.p['draft'],result['revisions'])
        self.assertEqual(result['comments'][0]['status'],'accepted')
        self.assertEqual(self.client.post(self.base+f'/comments/{c["id"]}/apply',json={'expected_version':2}).status_code,409)

    def test_changed_passage_disables_old_edits_but_keeps_thread(self):
        with app.state.store.edit(self.p['id']) as p:
            p['comments'] = rewrite_comments(p['draft']['text'],'🌊 Use Python 3. Include tests.',1,[],{})
        c = p['comments'][0]
        self.client.put(self.base+'/draft',json={'text':'🌊 Use Rust. Include tests.','expected_version':1})
        self.assertTrue(self.client.get(self.base).json()['comments'][0]['outdated'])
        self.assertEqual(self.client.post(self.base+f'/comments/{c["id"]}/apply',json={'expected_version':2}).status_code,409)

    def test_conflicts_cannot_be_applied_and_bad_selections_rejected(self):
        with app.state.store.edit(self.p['id']) as p:
            c = thread('conflict','Which language?', [anchor(p['draft']['text'],2,13,1)],replacement='Use Rust.')
            p['comments']=[c]
        self.assertEqual(self.client.post(self.base+f'/comments/{c["id"]}/apply',json={'expected_version':1}).status_code,400)
        self.assertEqual(self.client.post(self.base+'/comments',json={'text':'bad','start':0,'end':10000,'expected_version':1}).status_code,400)
        self.assertEqual(self.client.post(self.base+'/comments',json={'text':'bad','start':0,'end':5,'expected_version':2}).status_code,409)


class CommentAgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=Store(self.temp.name)
        class Model:
            status={'state':'ready','model':'dream'}
            calls=0
            async def sections(self,text,*args):return [{'text':text,'start':0,'end':len(text)}]
            async def generate(self,*args,**kwargs):
                self.calls+=1
                return {'text':'{"comments":[]}'}
        self.model=Model();self.agents=Agents(self.store,self.model,None)

    async def test_conflicts_leave_draft_main_untouched_and_prevent_silent_rewrite(self):
        p=self.store.create('splash','Use Python. Do not use Python. The budget is $500. The budget is $900.')
        result=await self.agents.enrich(p['id'],1)
        self.assertTrue(result['review_required'])
        self.assertEqual(self.model.calls,0)
        saved=self.store.get(p['id'])
        self.assertEqual(len(saved['comments']),2)
        self.assertEqual(saved['draft'],p['draft'])
        self.assertEqual(saved['main_id'],p['main_id'])

    async def test_malformed_model_comments_get_one_retry_without_invented_threads(self):
        p=self.store.create('splash','Explain caching for beginners.')
        async def bad(*args,**kwargs):self.model.calls+=1;return {'text':'not valid JSON'}
        self.model.generate=bad
        result=await review_document(self.agents,p['id'],1)
        self.assertEqual(self.model.calls,2)
        self.assertTrue(result['errors'])
        self.assertEqual(self.store.get(p['id'])['comments'],[])
        self.assertEqual(result['review']['status'], 'needs_attention')
        self.assertEqual(result['review']['invalid_response_samples'], ['not valid JSON'])

    async def test_truncated_review_retries_with_more_room_and_rejects_partial_comments(self):
        p=self.store.create('splash','Explain caching for beginners.')
        budgets=[]
        async def generate(messages, max_tokens, **kwargs):
            self.assertNotIn('Return only the improved prompt', messages[0]['content'])
            budgets.append(max_tokens)
            return {'text':'QUESTION 1 | Which caching system should this cover?',
                    'truncated':len(budgets)==1}
        self.model.generate=generate
        result=await review_document(self.agents,p['id'],1)
        self.assertEqual(budgets,[384,768])
        self.assertEqual(result['comments_added'],1)
        self.assertEqual(result['errors'],[])
        self.assertEqual(self.store.get(p['id'])['draft'],p['draft'])

    async def test_two_truncated_responses_never_publish_partial_comments(self):
        p=self.store.create('splash','Explain caching for beginners.')
        async def generate(*args, **kwargs):
            return {'text':'QUESTION 1 | Which caching system should this cover?', 'truncated':True}
        self.model.generate=generate
        result=await review_document(self.agents,p['id'],1)
        self.assertEqual(result['comments_added'],0)
        self.assertIn('output limit',result['errors'][0])
        self.assertEqual(len(result['review']['metrics']),2)

    async def test_model_can_add_multiple_comments_and_reply_without_editing(self):
        p=self.store.create('splash','Use a modern stack. Launch soon.')
        data={'comments':[
            {'kind':'question','quote':'modern stack','related_quote':'','message':'Which runtime is available?'},
            {'kind':'question','quote':'Launch soon.','related_quote':'','message':'What is the target date?'}]}
        async def review(*args,**kwargs):return {'text':json.dumps(data)}
        self.model.generate=review
        await review_document(self.agents,p['id'],1)
        saved=self.store.get(p['id']);self.assertEqual(len(saved['comments']),2)
        c=saved['comments'][0]
        async def reply(*args,**kwargs):return {'text':'Specify the installed Python version in your instruction.'}
        self.model.generate=reply
        await answer_comment(self.agents,p['id'],c['id'],c['messages'][-1]['id'])
        saved=self.store.get(p['id'])
        self.assertEqual(len(saved['comments'][0]['messages']),2)
        self.assertEqual(saved['draft'],p['draft'])

    async def test_edited_draft_rejects_obsolete_model_review(self):
        p=self.store.create('splash','Launch soon.')
        async def change_during_review(*args,**kwargs):
            self.store.save_draft(p['id'],'Launch in December.',1)
            return {'text':json.dumps({'comments':[{'kind':'question','passages':[1],'message':'When is soon?'}]})}
        self.model.generate=change_during_review
        with self.assertRaises(Conflict): await review_document(self.agents,p['id'],1)
        self.assertEqual(self.store.get(p['id'])['comments'],[])

    async def test_semantic_conflict_removes_unreviewed_rewrite(self):
        from backend.agents import proposals
        p=self.store.create('splash','Work offline. Every request calls a cloud API.')
        with self.store.edit(p['id']) as current:
            current['proposals']=proposals(p['draft']['text'],'Use a cloud API.',1)
        async def review(*args,**kwargs):
            return {'text':json.dumps({'comments':[{'kind':'conflict','passages':[1,2],'message':'Which connectivity requirement should apply?'}]})}
        self.model.generate=review
        await review_document(self.agents,p['id'],1)
        saved=self.store.get(p['id'])
        self.assertEqual(saved['proposals'],[])
        self.assertEqual(saved['draft'],p['draft'])


if __name__=='__main__':unittest.main()
