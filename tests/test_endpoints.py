import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from backend.endpoints import discover_endpoints, probe_endpoint, normalize_url, endpoint_config
from backend.endpoint_model import EndpointModel
from backend.models import ModelService
from backend.storage import Store
from backend.app import app
from scripts import setup


@contextlib.contextmanager
def server(routes=None, finish='stop'):
    requests = []
    routes = {'/v1/models': {'data': [{'id': 'already-running'}]}} if routes is None else routes
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            requests.append(('GET', self.path, None))
            value = routes.get(self.path)
            self.send_response(404 if value is None else 401 if value == 'auth' else 200)
            self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(json.dumps(value).encode())
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(('POST', self.path, body))
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(json.dumps({'choices':[{'message':{'content':'NONE'}, 'finish_reason':finish}],
                                       'usage':{'prompt_tokens':40,'completion_tokens':1}}).encode())
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}/v1', requests
    finally:
        http.shutdown(); http.server_close(); thread.join()


class DiscoveryTests(unittest.TestCase):
    def test_generic_discovery_only_reads_and_preserves_model_id(self):
        with server() as (url, calls):
            found = discover_endpoints([url])
            self.assertEqual(found[0]['models'][0]['id'], 'already-running')
            self.assertFalse(found[0]['loaded_only'])
            self.assertTrue(all(call[0] == 'GET' for call in calls))
            self.assertEqual(endpoint_config(found[0], 'already-running')['base_url'], url)

    def test_ollama_lists_only_running_models_even_if_none(self):
        with server({'/api/ps':{'models':[]}, '/v1/models':{'data':[{'id':'on-disk-only'}]}}) as (url,calls):
            self.assertEqual(probe_endpoint(url)['models'], [])
            self.assertEqual([c[1] for c in calls], ['/api/ps'])

    def test_lmstudio_filters_unloaded_and_embedding_models(self):
        rows = [{'type':'llm','loaded_instances':[]},
                {'type':'embedding','loaded_instances':[{'id':'embedding'}]},
                {'type':'llm','loaded_instances':[{'id':'ready','config':{'context_length':2048}}]}]
        with server({'/api/v1/models':{'models':rows}}) as (url,_):
            found = probe_endpoint(url)
            self.assertEqual(found['models'],[{'id':'ready','context':2048}])
            self.assertEqual(endpoint_config(found,'ready')['context'],2048)

    def test_auth_server_is_detected_without_usable_model(self):
        with server({'/v1/models':'auth'}) as (url,_):
            found = probe_endpoint(url)
            self.assertTrue(found['auth_required'])
            with self.assertRaises(ValueError): endpoint_config(found,'guessed')

    def test_unrelated_server_and_invalid_urls_are_not_selected(self):
        with server({'/v1/models':{'status':'healthy'}}) as (url,_):
            self.assertIsNone(probe_endpoint(url))
        for url in ('http://example.com/v1','file:///tmp/models','http://user:pass@localhost:8000',
                    'http://127.0.0.1:8000/v1?token=x','http://localhost:bad'):
            with self.subTest(url=url), self.assertRaises(ValueError): normalize_url(url)
        self.assertEqual(normalize_url('http://[::1]:8000/'), 'http://[::1]:8000/v1')


class EndpointInferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_request_cancels_obsolete_endpoint_request(self):
        service=ModelService(); service.status={'state':'ready'}
        started=asyncio.Event()
        async def generate(messages, max_tokens):
            if messages[0]['content']=='old':
                started.set()
                await asyncio.Event().wait()
            return {'ok':True,'text':'new'}
        service.endpoint=SimpleNamespace(generate=generate)
        first=asyncio.create_task(service.generate([{'content':'old'}]))
        await started.wait()
        try:
            result=await asyncio.wait_for(service.generate([{'content':'new'}]),2)
            self.assertEqual(result['text'],'new')
            self.assertIsInstance((await asyncio.gather(first,return_exceptions=True))[0],asyncio.CancelledError)
            self.assertIsNone(service.process)
        finally:
            first.cancel()
            await service.close()

    async def test_endpoint_switch_stops_owned_worker_before_verification(self):
        service=ModelService()
        try:
            order=[]
            async def stop():order.append('stop')
            async def verify():order.append('verify');return {'connected':True}
            adapter=SimpleNamespace(config={'source':'endpoint','model':'existing'},verify=verify)
            with patch.object(service,'stop_worker',side_effect=stop), \
                 patch('backend.endpoint_model.EndpointModel',return_value=adapter), \
                 patch.object(service,'exchange',new_callable=AsyncMock) as worker:
                result=await service.load({'source':'endpoint','model':'existing'})
                self.assertEqual(order,['stop','verify'])
                self.assertEqual(result['state'],'ready')
                worker.assert_not_called()
        finally:await service.close()

    async def test_selection_and_planning_never_spawn_worker_and_generation_uses_endpoint(self):
        with server() as (url,calls):
            service=ModelService()
            try:
                with patch.object(service,'exchange',new_callable=AsyncMock) as worker:
                    result=await service.load({'source':'endpoint','base_url':url,'model':'already-running','context':8192})
                    self.assertEqual(result['state'],'ready')
                    self.assertFalse(any(c[0]=='POST' for c in calls))
                    self.assertIsNone(service.process)
                    plan=await service.plan_review('Explain caching.','Review the prompt.',[])
                    self.assertEqual(plan['budget_method'],'utf8_byte_estimate')
                    output=await service.generate([{'role':'user','content':'Explain caching.'}],64)
                    self.assertEqual(output['text'],'NONE')
                    self.assertFalse(output['truncated'])
                    worker.assert_not_called()
                    post=next(c for c in calls if c[0]=='POST')
                    self.assertEqual(post[1],'/v1/chat/completions')
                    self.assertEqual(post[2]['model'],'already-running')
            finally: await service.close()

    async def test_unavailable_saved_endpoint_never_falls_back_to_local(self):
        service=ModelService()
        try:
            with patch('backend.endpoint_model.probe_endpoint',return_value=None), \
                 patch.object(service,'exchange',new_callable=AsyncMock) as worker:
                result=await service.load({'source':'endpoint','base_url':'http://localhost:1/v1','model':'missing'})
                self.assertEqual(result['state'],'error')
                self.assertIn('No local model',result['error'])
                worker.assert_not_called()
        finally: await service.close()

    async def test_length_finish_is_withheld_and_unicode_sections_preserve_input(self):
        with server(finish='length') as (url,_):
            model=EndpointModel({'source':'endpoint','base_url':url,'model':'already-running','context':2048})
            self.assertTrue((await model.generate([{'role':'user','content':'Hello'}],64))['truncated'])
            text='研究 🌙\n'*500
            sections=model.sections(text,'Review','',512)['sections']
            self.assertEqual(''.join(s['text'] for s in sections),text)
            self.assertGreater(len(sections),1)


class EndpointAppTests(unittest.TestCase):
    def test_failed_connection_keeps_endpoint_choice_instead_of_old_local_checkpoint(self):
        found={'base_url':'http://localhost:8000/v1','provider':'openai-compatible','models':[{'id':'maya'}]}
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'SPLASH_DATA_DIR':directory,'SPLASH_SKIP_MODEL_LOAD':''}), \
             patch('backend.mcp_host.ToolHost.start'), \
             patch('backend.app.discover_endpoints',return_value=[found]), \
             patch('backend.api.probe_endpoint',return_value=found), \
             patch('backend.endpoint_model.EndpointModel.verify',new_callable=AsyncMock,side_effect=ValueError('Server stopped')):
            store=Store();store.set_config('model',{'model':'llada','source':'local'})
            with TestClient(app) as client:
                response=client.post('/api/models/endpoint',json={'base_url':found['base_url'],'model':'maya'})
                self.assertEqual(response.status_code,400)
                self.assertEqual(store.config('model')['source'],'endpoint')
                self.assertIsNone(app.state.models.process)

    def test_no_existing_endpoint_keeps_normal_startup_and_explicit_local_choice_works(self):
        found={'base_url':'http://localhost:8000/v1','provider':'openai-compatible','models':[{'id':'maya'}]}
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'SPLASH_DATA_DIR':directory,'SPLASH_SKIP_MODEL_LOAD':''}), \
             patch('backend.mcp_host.ToolHost.start'), \
             patch('backend.app.discover_endpoints',return_value=[]), \
             patch('backend.api.discover_endpoints',return_value=[found]), \
             patch('backend.models.ModelService.load',new_callable=AsyncMock,return_value={'state':'ready'}) as load:
            config={'model':'compact','source':'local','path':'/unused'}
            Store().set_config('model',config)
            with TestClient(app) as client:
                load.assert_awaited_once_with(config)
                load.reset_mock()
                with patch('backend.api.inspect_local'), patch('backend.api.dependency_status',return_value={'missing':[],'packages':{'bitsandbytes':True}}):
                    response=client.post('/api/models/load',json={**config,'confirm_separate_model':True})
                self.assertEqual(response.status_code,200,response.text)
                load.assert_awaited_once()
                self.assertNotIn('confirm_separate_model',load.call_args.args[0])

    def test_startup_prompts_before_local_load_and_endpoint_selection_persists(self):
        with server() as (url,calls), tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'SPLASH_DATA_DIR':directory,'SPLASH_SKIP_MODEL_LOAD':''}), \
             patch('backend.mcp_host.ToolHost.start'):
            store=Store(); local={'model':'llada','source':'local','path':'/unused','device':'cuda:0'}
            store.set_config('model',local)
            found=probe_endpoint(url)
            with patch('backend.app.discover_endpoints',return_value=[found]), \
                 patch('backend.api.discover_endpoints',return_value=[found]), \
                 patch('backend.models.ModelService.exchange',new_callable=AsyncMock) as worker:
                with TestClient(app) as client:
                    status=client.get('/api/models').json()['runtime']
                    self.assertEqual(status['state'],'unloaded')
                    self.assertIn('Would you like',status['notice'])
                    worker.assert_not_called()
                    self.assertEqual(client.post('/api/models/load',json={'model':'llada'}).status_code,409)
                    worker.assert_not_called()
                    response=client.post('/api/models/endpoint',json={'base_url':url,'model':'already-running','context':8192})
                    self.assertEqual(response.status_code,200,response.text)
                    self.assertEqual(store.config('model')['source'],'endpoint')
                    self.assertFalse(any(c[0]=='POST' for c in calls))
                    p=client.post('/api/projects',json={'agent':'splash','text':'Explain caching.'}).json()
                    app.state.agents.tools = None  # Exercise local context fallback, not MCP startup.
                    client.post(f"/api/projects/{p['id']}/refine",json={'expected_version':1})
                    for _ in range(200):
                        jobs=client.get(f"/api/projects/{p['id']}/jobs").json()
                        if jobs.get('refine',{}).get('status')!='running':break
                        time.sleep(.01)
                    self.assertEqual(jobs['refine']['status'],'complete',jobs)
                    self.assertEqual(client.get(f"/api/projects/{p['id']}").json()['draft'],p['draft'])
                    worker.assert_not_called()
                # Saved endpoint is restored; startup never scans/loads weights.
                with patch('backend.app.discover_endpoints') as scan, TestClient(app) as client:
                    for _ in range(100):
                        state=client.get('/api/models').json()['runtime']['state']
                        if state=='ready':break
                        time.sleep(.01)
                    self.assertEqual(state,'ready')
                    scan.assert_not_called(); worker.assert_not_called()


class EndpointSetupTests(unittest.TestCase):
    def args(self,**kwargs):
        return SimpleNamespace(endpoint=None,endpoint_model=None,endpoint_context=4096,local_model=False,**kwargs)

    def test_accept_decline_and_noninteractive_choice(self):
        found=[{'base_url':'http://localhost:8000/v1','provider':'openai-compatible',
                'models':[{'id':'maya','context':None}]}]
        with contextlib.redirect_stdout(io.StringIO()), patch('scripts.setup.sys.stdin.isatty',return_value=True):
            with patch('builtins.input',return_value='1'):
                config,editor=setup.choose_endpoint(found,self.args())
                self.assertTrue(editor); self.assertEqual(config['model'],'maya')
            with patch('builtins.input',return_value='l'):
                self.assertEqual(setup.choose_endpoint(found,self.args()),(None,False))
        with contextlib.redirect_stdout(io.StringIO()), patch('scripts.setup.sys.stdin.isatty',return_value=False), patch('builtins.input') as ask:
            self.assertEqual(setup.choose_endpoint(found,self.args()),(None,True))
            ask.assert_not_called()

    def test_endpoint_setup_skips_torch_and_persists_after_success_only(self):
        found=[{'base_url':'http://localhost:8000/v1','provider':'openai-compatible','models':[{'id':'maya'}]}]
        report={'ok':True,'app_ready':True,'model_ready':False,'checks':[],'install_commands':[]}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(setup,'discover_endpoints',return_value=found), \
             patch.object(setup.deps,'node_checks',return_value=[]), \
             patch.object(setup.deps,'inspect_hardware',return_value={}), \
             patch.object(setup.deps,'profile_problem',return_value=None), \
             patch.object(setup.deps,'probe_environment',return_value={'python_version':[3,11]}), \
             patch.object(setup.deps,'check_dependencies',return_value=report) as checks, \
             patch.object(setup.deps,'print_report'), patch.object(setup.venv,'create'), \
             patch.object(setup.shutil,'which',return_value='npm'), \
             patch.object(setup.subprocess,'run') as install, contextlib.redirect_stdout(io.StringIO()):
            code=setup.main(['--endpoint',found[0]['base_url'],'--endpoint-model','maya'],Path(directory))
            self.assertEqual(code,0)
            self.assertEqual(checks.call_args.args[1],'editor')
            commands=[c.args[0] for c in install.call_args_list]
            self.assertFalse(any('requirements-models' in str(c) or 'torch==' in str(c) for c in commands))
            saved=next(c for c in install.call_args_list if 'backend.configure_endpoint' in c.args[0])
            self.assertEqual(json.loads(saved.kwargs['input'])['model'],'maya')

    def test_check_is_read_only_and_does_not_probe_gpu_with_detected_endpoint(self):
        found=[{'base_url':'http://localhost:8000/v1','provider':'openai-compatible','models':[{'id':'maya'}]}]
        with patch.object(setup,'discover_endpoints',return_value=found), \
             patch.object(setup.deps,'check_dependencies',return_value={'ok':True}) as check, \
             patch.object(setup.subprocess,'run') as write, patch('builtins.input') as ask, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(setup.main(['--check','--json']),0)
            self.assertEqual(json.loads(output.getvalue())['endpoints'],found)
            self.assertEqual(check.call_args.args[1],'editor')
            write.assert_not_called(); ask.assert_not_called()
