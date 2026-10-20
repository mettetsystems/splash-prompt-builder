import asyncio
import os
import tempfile
import threading
import time
import unittest
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.app import app
from backend.storage import Store
from backend.state import claim, finish, allowed, remember, settle, initialize
from tests.enrichment_helpers import prepare, edit_response


class TestModel:
    def __init__(self):
        self.status={'state':'ready','model':'dream','config':{'revision':'pinned','steps':64}}
        self.calls=0
        self.output='Develop precise prompts.'
        self.failure=False
        self.release=threading.Event();self.release.set()
    async def sections(self,text,*args):
        return [{'text':text,'start':0,'end':len(text)}]
    async def plan_review(self, text, system, references):
        return await prepare(text, system, references)
    async def generate(self,*args,**kwargs):
        self.calls+=1
        while not self.release.is_set():await asyncio.sleep(.002)
        if self.failure:raise RuntimeError('Model unavailable')
        content = args[0][-1]['content']
        if not content.startswith('Prompt section:\n'):
            return {'text': self.output}
        before = content.split('\n\nReference excerpts',1)[0].removeprefix('Prompt section:\n')
        return {'text':edit_response(before, self.output)}
    async def close(self):pass


class ProcessingAPI(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        env=patch.dict(os.environ,{'SPLASH_DATA_DIR':self.temp.name});env.start();self.addCleanup(env.stop)
        mcp=patch('backend.mcp_host.ToolHost.start');mcp.start();self.addCleanup(mcp.stop)
        self.client=TestClient(app).__enter__();self.addCleanup(self.client.__exit__,None,None,None)
        self.model=TestModel();app.state.models=self.model;app.state.agents.models=self.model
        app.state.agents.tools=None  # Use deterministic local retrieval; no external tool process in these tests.
        self.p=self.client.post('/api/projects',json={'agent':'splash','text':'write precise prompts.'}).json()
        self.base='/api/projects/'+self.p['id']
    def get(self):return self.client.get(self.base).json()
    def request(self,force=False,kind='refine'):
        version=self.get()['draft']['version']
        r=self.client.post(self.base+'/'+kind+('?force=true' if force else ''),json={'expected_version':version})
        self.assertEqual(r.status_code,200,r.text)
        return r.json()
    def wait(self,kind='refine'):
        for _ in range(200):
            job=self.client.get(self.base+'/jobs').json().get(kind)
            if job and job['status']!='running':return job
            time.sleep(.005)
        self.fail('Job did not finish')
    def edit(self,text):
        p=self.get()
        r=self.client.put(self.base+'/draft',json={'text':text,'expected_version':p['draft']['version']})
        self.assertEqual(r.status_code,200,r.text)
        return r.json()

    def test_completed_input_survives_runtime_reset_and_manual_retry_is_explicit(self):
        self.assertEqual(self.request()['status'],'started');self.wait()
        original=self.get()['proposals']
        app.state.job_results.clear();app.state.store=Store(self.temp.name);app.state.agents.store=app.state.store
        self.assertEqual(self.request()['status'],'unchanged')
        self.assertEqual(self.model.calls,1)
        self.assertEqual(self.get()['proposals'],original)
        self.assertEqual(self.client.get(self.base+'/jobs').json()['refine']['status'],'complete')
        self.assertEqual(self.request(force=True)['status'],'started');self.wait()
        self.assertEqual(self.model.calls,2)

    def test_failed_input_does_not_retry_on_automatic_poll(self):
        self.model.failure=True
        self.request();self.assertEqual(self.wait()['status'],'error')
        for _ in range(3):self.assertEqual(self.request()['status'],'unchanged')
        self.assertEqual(self.model.calls,1)
        self.model.failure=False;self.request(force=True);self.assertEqual(self.wait()['status'],'complete')

    def test_concurrent_requests_and_edit_cancellation_do_not_duplicate_work(self):
        self.model.release.clear()
        self.request()
        self.assertEqual(self.request()['status'],'already_running')
        self.edit('write precise instructions.')
        self.assertEqual(self.wait()['status'],'canceled')
        self.model.release.set();self.model.output='Develop precise instructions.'
        self.assertEqual(self.request()['status'],'started');self.wait()
        self.assertEqual(len(self.get()['processing']['runs']['refine']),2)
        self.assertTrue(all(p['version']==2 for p in self.get()['proposals']))

    def test_new_model_or_context_reprocesses_once_without_output_invalidating_cache(self):
        self.request();self.wait()
        self.model.status['config']['steps']=32
        self.assertEqual(self.request()['status'],'started');self.wait()
        with app.state.store.edit(self.p['id']) as p:
            p['documents'].append({'id':'context','name':'context.txt','pages':[{'page':1,'text':'Use precise terms.'}]})
        self.assertEqual(self.request()['status'],'started');self.wait()
        self.assertEqual(self.request()['status'],'unchanged')
        self.assertEqual(self.model.calls,3)

    def test_system_prompt_changes_invalidate_preview_and_reprocess_each_configuration(self):
        self.request();self.wait()
        before=self.get()
        self.assertTrue(before['proposals'])
        with app.state.store.edit(self.p['id']) as p:
            settle(p, 'author_choice')
        app.state.agents.last_research[self.p['id']] = ('prior instructions', time.monotonic())
        response=self.client.put(self.base+'/system-prompts',json={'prompts':[
            {'name':'Voice','text':'Prefer short sentences.'}]})
        self.assertEqual(response.status_code,200,response.text)
        updated=response.json()
        self.assertNotIn(self.p['id'], app.state.agents.last_research)
        self.assertEqual(updated['draft'],before['draft'])
        self.assertEqual(updated['main_id'],before['main_id'])
        self.assertIsNone(updated['processing']['settled'])
        self.assertFalse(updated['proposals'])
        self.assertTrue(all(c['outdated'] for c in updated['comments'] if c['origin']=='enrichment'))
        self.assertEqual(self.request()['status'],'started');self.wait()
        self.assertEqual(self.request()['status'],'unchanged')
        # Saving the same instructions, or changing a disabled note, is not new input.
        prompts=updated['system_prompts']+[{'name':'Unused','text':'Use tables.','enabled':False}]
        self.client.put(self.base+'/system-prompts',json={'prompts':prompts}).raise_for_status()
        self.assertEqual(self.request()['status'],'unchanged')
        # Returning to the original default must regenerate the now-cleared preview.
        self.client.put(self.base+'/system-prompts',json={'prompts':[]}).raise_for_status()
        self.assertEqual(self.request()['status'],'started');self.wait()
        self.assertTrue(self.get()['proposals'])
        self.assertEqual(self.model.calls,3)

    def test_system_prompt_change_cancels_pending_generation(self):
        self.model.release.clear()
        self.request()
        response=self.client.put(self.base+'/system-prompts',json={'prompts':[
            {'name':'Audience','text':'Use terms familiar to biologists.'}]})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.wait()['status'],'canceled')
        self.model.release.set()
        self.assertFalse(self.get()['proposals'])
        self.assertEqual(self.request()['status'],'started');self.wait()
        self.assertEqual(self.get()['draft']['text'],self.p['draft']['text'])

    def test_dismissal_removes_preview_and_survives_manual_retry_and_unrelated_edit(self):
        self.request();self.wait();p=self.get();c=p['comments'][0]
        r=self.client.patch(self.base+'/comments/'+c['id'],json={'status':'dismissed'})
        self.assertEqual(r.json()['proposals'],[])
        self.request(force=True);self.wait()
        self.assertEqual(self.get()['proposals'],[])
        self.edit('write precise prompts. Give examples.')
        self.model.output='Develop precise prompts. Give examples.'
        self.request();self.wait()
        self.assertEqual(self.get()['proposals'],[])
        self.assertEqual(len(self.get()['comments']),1)
        self.client.patch(self.base+'/comments/'+c['id'],json={'status':'open'})
        self.request(force=True);self.wait()
        self.assertTrue(self.get()['proposals'])

    def test_discard_entire_preview_is_remembered(self):
        self.request();self.wait();p=self.get()
        self.client.post(self.base+'/proposals/reject',json={'ids':[c['id'] for c in p['proposals']],'expected_version':1})
        self.request(force=True);self.wait()
        self.assertEqual(self.get()['proposals'],[])
        self.assertTrue(all(c['status']=='dismissed' for c in self.get()['comments']))

    def test_dismissal_during_generation_wins_over_the_late_model_result(self):
        self.request();self.wait();c=self.get()['comments'][0]
        self.model.release.clear();self.request(force=True)
        self.client.patch(self.base+'/comments/'+c['id'],json={'status':'dismissed'})
        self.model.release.set();self.wait()
        self.assertEqual(self.get()['proposals'],[])
        self.assertEqual(self.get()['comments'][0]['status'],'dismissed')

    def test_acceptance_during_generation_cancels_obsolete_results_without_an_error(self):
        self.request();self.wait();c=self.get()['comments'][0]
        self.model.release.clear();self.request(force=True)
        self.client.post(self.base+'/comments/'+c['id']+'/apply',json={'expected_version':1})
        self.model.release.set()
        self.assertEqual(self.wait()['status'],'canceled')
        self.assertEqual(self.get()['proposals'],[])
        self.assertEqual(self.get()['draft']['text'],'Develop precise prompts.')

    def test_acceptance_settles_input_and_does_not_offer_the_inverse_later(self):
        self.request();self.wait();c=self.get()['comments'][0]
        self.client.post(self.base+'/comments/'+c['id']+'/apply',json={'expected_version':1})
        self.assertEqual(self.request()['status'],'unchanged')
        self.assertEqual(self.model.calls,1)
        self.edit('Develop precise prompts. Give examples.')
        self.model.output='write precise prompts. Give examples.'
        self.request();self.wait()
        self.assertEqual(self.get()['proposals'],[])
        self.assertEqual(self.get()['comments'][0]['status'],'accepted')
        self.edit('Create precise prompts. Give examples.')
        self.model.output='Develop precise prompts. Give examples.'
        self.request();self.wait()
        self.assertTrue(self.get()['proposals'])

    def test_merge_settles_new_main_without_losing_history(self):
        self.request();self.wait();p=self.get()
        r=self.client.post(self.base+'/merge-review',json={'expected_version':1,'proposal_ids':[c['id'] for c in p['proposals']]}).json()
        self.client.post(self.base+'/merge/'+r['id'])
        self.assertEqual(self.request()['status'],'unchanged')
        p=self.get()
        self.assertEqual(p['draft']['text'],'Develop precise prompts.')
        self.assertEqual(sum(r['kind']=='main' for r in p['revisions']),2)
        self.assertTrue(p['edit_decisions'])
        self.assertTrue(all(c['status']=='accepted' for c in p['comments']))

    def test_review_completion_and_resolved_discussion_survive_reload(self):
        self.model.output='QUESTION 1 | Who is the audience?'
        self.request(kind='comments/review');self.wait('comments')
        c=self.get()['comments'][0]
        self.client.patch(self.base+'/comments/'+c['id'],json={'status':'resolved'})
        app.state.job_results.clear()
        self.assertEqual(self.request(kind='comments/review')['status'],'unchanged')
        self.request(force=True,kind='comments/review');self.wait('comments')
        self.assertEqual(len(self.get()['comments']),1)
        self.assertEqual(self.get()['comments'][0]['status'],'resolved')


class ProcessingStore(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=Store(self.temp.name)
        self.p=self.store.create('splash','write precise prompts.')

    def test_processing_claim_is_project_scoped_and_imports_have_independent_memory(self):
        with self.store.edit(self.p['id']) as p:
            run,response=claim(p,'refine','key')
            self.assertIsNone(response)
            self.assertEqual(claim(p,'refine','key')[1]['status'],'already_running')
            finish(p,'refine',run,'complete')
            settle(p,'test')
        copied=self.store.create('splash-code','ignored',source=self.p['id'])
        self.assertEqual(copied['processing'],{'runs':{},'settled':None})
        self.assertEqual(copied['edit_decisions'],[])
        self.assertIsNone(claim(copied,'refine','key')[1])

    def test_dead_server_attempt_is_interrupted_without_automatic_retry(self):
        p=self.p
        run,_=claim(p,'refine','key')
        with patch('backend.state.os.kill',side_effect=ProcessLookupError):
            _,reply=claim(p,'refine','key')
        self.assertEqual(reply['status'],'unchanged')
        self.assertEqual(p['processing']['runs']['refine'][0]['status'],'canceled')
        self.assertIsNone(claim(p,'refine','key',force=True)[1])

    def test_remembered_word_moves_with_unicode_context_and_does_not_lock_other_occurrences(self):
        with self.store.edit(self.p['id']) as p:
            remember(p,{'start':0,'end':6,'before':'write ','after':'Develop '},'dismissed')
        p=self.store.save_draft(self.p['id'],'🌊 Notes. write precise prompts. Then write tests.',1)
        start=p['draft']['text'].index('write')
        self.assertFalse(allowed(p,{'start':start,'end':start+6}))
        other=p['draft']['text'].rindex('write')
        self.assertTrue(allowed(p,{'start':other,'end':other+6}))

    def test_decision_during_inference_filters_the_late_result(self):
        p=self.p
        edit={'start':0,'end':6,'before':'write ','after':'Develop '}
        remember(p,edit,'dismissed')
        self.assertFalse(allowed(p,edit))
        self.assertTrue(allowed(p,{'start':6,'end':13,'before':'precise','after':'clear'}))

    def test_existing_dismissed_comments_migrate_into_decision_memory(self):
        from backend.comments import rewrite_comments
        from backend.agents import proposals
        p=self.p
        p.pop('edit_decisions');p.pop('processing')
        p['comments']=rewrite_comments(p['draft']['text'],'Develop precise prompts.',1,[],{})
        p['comments'][0]['status']='dismissed'
        p['proposals']=proposals(p['draft']['text'],'Develop precise prompts.',1)
        initialize(p)
        self.assertTrue(p['edit_decisions'])
        self.assertEqual(p['proposals'],[])
