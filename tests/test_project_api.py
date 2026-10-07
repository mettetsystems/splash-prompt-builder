import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.app import app
from backend.agents import proposals


class ProjectAPI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {"SPLASH_DATA_DIR": self.temp.name})
        env.start(); self.addCleanup(env.stop)
        mcp = patch("backend.mcp_host.ToolHost.start")
        mcp.start(); self.addCleanup(mcp.stop)
        self.client = TestClient(app).__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        response = self.client.post('/api/projects', json={"agent":"splash-search", "text":"Research waves. Preserve consent."})
        self.assertEqual(response.status_code, 201, response.text)
        self.p = response.json(); self.base = '/api/projects/' + self.p['id']

    def test_merge_conflict_atomicity_and_retry(self):
        r = self.client.put(self.base+'/draft', json={"text":"Research waves and tides.", "expected_version":1})
        self.assertEqual(r.status_code, 200)
        review = self.client.post(self.base+'/merge-review', json={"expected_version":2}).json()
        p = self.client.get(self.base).json()
        self.assertEqual(p['main_id'], self.p['main_id'])
        self.assertNotIn('reviews', p)
        merged = self.client.post(self.base+'/merge/'+review['id']).json()
        again = self.client.post(self.base+'/merge/'+review['id']).json()
        self.assertEqual(merged['main_id'], again['main_id'])
        self.assertEqual(len(merged['revisions']), len(again['revisions']))
        bad = self.client.put(self.base+'/draft', json={"text":"stale edit", "expected_version":1})
        self.assertEqual(bad.status_code, 409)

    def test_partial_accept_keeps_other_edits(self):
        with app.state.store.edit(self.p['id']) as p:
            p['proposals'] = proposals(p['draft']['text'], 'Explore waves. Preserve informed consent.', 1)
        p = self.client.get(self.base).json()
        first = p['proposals'][0]['id']
        response = self.client.post(self.base+'/proposals/accept', json={'ids':[first], 'expected_version':1})
        self.assertEqual(response.status_code, 200, response.text)
        p = response.json(); self.assertGreater(len(p['proposals']), 0)
        self.assertTrue(all(c['version'] == p['draft']['version'] for c in p['proposals']))
        response = self.client.post(self.base+'/proposals/accept', json={'ids':[c['id'] for c in p['proposals']], 'expected_version':p['draft']['version']})
        self.assertEqual(response.json()['draft']['text'], 'Explore waves. Preserve informed consent.')

    def test_unicode_validation_settings_limits_and_agent_immutability(self):
        for body in ({'branch_count':6}, {'branch_count':0}, {'agent':'splash-code'}):
            self.assertEqual(self.client.patch(self.base,json=body).status_code,422)
        self.assertEqual(self.client.put(self.base+'/draft',json={'text':'x','expected_version':True}).status_code,422)
        self.assertEqual(self.client.put(self.base+'/draft',content=json.dumps({'text':'\ud800','expected_version':1}),headers={'Content-Type':'application/json'}).status_code,422)
        self.assertEqual(self.client.post('/api/projects',json={'agent':'splash','text':'  '}).status_code,400)

    def test_context_upload_export_roundtrip_preserves_pdf_pages(self):
        response = self.client.post(self.base+'/documents', files={'file':('notes.md','Evidence 🌙\n\nDo not infer causality.'.encode(),'text/plain')})
        self.assertEqual(response.status_code,200,response.text)
        with app.state.store.edit(self.p['id']) as p:
            p['documents'][0]['name'] = 'research.pdf'
            p['documents'][0]['pages'] = [{'page':3,'text':'Page three evidence'}]
        exported = self.client.get(self.base+'/export').json()
        response = self.client.post('/api/projects/import-file?agent=splash-code',files={'file':('project.json',json.dumps(exported).encode(),'application/json')})
        self.assertEqual(response.status_code,201,response.text)
        imported = response.json()
        self.client.delete(self.base)
        loaded = self.client.get('/api/projects/'+imported['id']).json()
        self.assertEqual(loaded['documents'][0]['pages'][0]['page'],3)
        self.assertEqual(loaded['agent'],'splash-code')

    def test_export_text_excludes_unmerged_draft(self):
        self.client.put(self.base+'/draft',json={'text':'Unapproved draft','expected_version':1})
        text=self.client.get(self.base+'/export?format=txt').text
        self.assertEqual(text,self.p['draft']['text'])

    def test_invalid_model_options_rejected_before_configuration_changes(self):
        for body in ({'model':'compact','device':'cpu','quantize':True},
                     {'model':'dream','device':'cuda:0','quantize':True,'offload':True}):
            self.assertEqual(self.client.post('/api/models/load',json=body).status_code,422)
        self.assertIsNone(app.state.store.config('model'))

    def test_malformed_archives_leave_no_partial_projects(self):
        good = self.client.get(self.base+'/export').json()
        missing = copy.deepcopy(good); missing['project']['main_id']='missing'
        invalid = copy.deepcopy(good); invalid['project']['revisions'][0]['text']='\ud800'
        for data in ([], {}, missing, invalid):
            response = self.client.post('/api/projects/import-file?agent=splash',files={'file':('bad.json',json.dumps(data).encode(),'application/json')})
            self.assertEqual(response.status_code,400,response.text)
        self.assertEqual(len(self.client.get('/api/projects').json()),1)

    def test_websocket_identity_and_origin_protection(self):
        with self.client.websocket_connect('/ws/projects/'+self.p['id']) as ws:
            connected=ws.receive_json()
            self.assertEqual(connected['project_id'],self.p['id'])
            self.assertEqual(connected['protocol'],2)
            ws.send_text('ping'); self.assertEqual(ws.receive_json()['type'],'pong')
        self.assertEqual(self.client.post('/api/projects',json={'agent':'splash','text':'x'},headers={'Origin':'https://hostile.example'}).status_code,403)


if __name__ == '__main__': unittest.main()
