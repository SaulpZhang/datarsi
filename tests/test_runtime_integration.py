from copy import deepcopy
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
import os
import threading
import uuid
import pytest
from datarsi.agent import run_task
from datarsi.configuration import Config,DEFAULTS
from datarsi.inference import Client
from datarsi.tasks import Task


@pytest.mark.skipif(os.environ.get('DATARSI_DOCKER_TESTS')!='1',reason='Opt-in Docker/OpenCode integration test')
def test_real_opencode_edit_verify_record_cleanup(tmp_path):
    import docker
    docker_client=docker.from_env()
    tag='datarsi-runtime-test:'+uuid.uuid4().hex[:8]
    dockerfile="""FROM datarsi-env:0.3
RUN mkdir -p /testbed && cd /testbed && git init && git config user.email test@example.invalid && git config user.name test && printf 'value = 0\\n' > app.py && git add app.py && git commit -m initial
"""
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            if not body.get('tools'):
                delta={'role':'assistant','content':'Fixture task complete'}
                finish='stop'
            elif any(message['role']=='tool' for message in body['messages']):
                delta={'role':'assistant','content':'The value is now fixed.'}
                finish='stop'
            else:
                assert any(tool['function']['name']=='bash' for tool in body['tools'])
                delta={'role':'assistant','content':None,'tool_calls':[{'index':0,'id':'call_fix',
                    'type':'function','function':{'name':'bash','arguments':json.dumps({
                        'command':"printf 'value = 1\\n' > app.py",'description':'Fix app value'})}}]}
                finish='tool_calls'
            self.send_response(200)
            if body.get('stream'):
                self.send_header('Content-Type','text/event-stream'); self.end_headers()
                for choice in ({'index':0,'delta':delta,'finish_reason':None},
                    {'index':0,'delta':{},'finish_reason':finish}):
                    chunk={'id':'chatcmpl-fixture','object':'chat.completion.chunk','created':0,
                        'model':'fixture','choices':[choice]}
                    self.wfile.write(('data: '+json.dumps(chunk)+'\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
            else:
                self.send_header('Content-Type','application/json'); self.end_headers()
                self.wfile.write(json.dumps({'id':'chatcmpl-fixture','object':'chat.completion','created':0,
                    'model':'fixture','choices':[{'index':0,'message':delta,'finish_reason':finish}],
                    'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    client=Client(f'http://127.0.0.1:{server.server_port}/v1','fixture')
    try:
        docker_client.images.build(fileobj=io.BytesIO(dockerfile.encode()),tag=tag,rm=True,forcerm=True)
        task=Task('fixture','train','Fix value',tag,'/testbed',"python3 -c 'import app; assert app.value == 1'")
        values=deepcopy(DEFAULTS)
        values['agent']['task_timeout']=90
        cfg=Config(values,tmp_path)
        result=run_task(task,client,cfg,'mock-model-real-opencode',1,private_instruction='PRIVATE_SENTINEL')
        assert result['environment_error'] is None,result
        assert result['resolved'],result
        assert '+value = 1' in result['patch']
        assert result['agent_backend']=='opencode'
        assert any(row['tool']=='bash' for row in result['actions'])
        assert len(result['model_calls'])>=2
        assert 'PRIVATE_SENTINEL' in json.dumps(requests)
        assert 'PRIVATE_SENTINEL' not in json.dumps(result['model_calls'])
        if os.environ.get('DATARSI_TOKENIZER_TESTS')=='1':
            from transformers import AutoTokenizer
            from datarsi.data import prepare
            from datarsi.storage import write_jsonl
            tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B',use_fast=True)
            source=tmp_path/'opencode.jsonl'
            write_jsonl(source,[result])
            prepared=prepare(cfg,source,tmp_path/'tokenized.jsonl',tokenizer=tokenizer)
            assert prepared['examples']==len(result['model_calls'])
        assert not docker_client.containers.list(filters={'ancestor':tag})
        # A stuck owned child must be bounded and its task container removed.
        binary=tmp_path/'sleep-agent'
        binary.write_text('#!/bin/bash\nsleep 60\n')
        values['agent']['opencode_binary']=str(binary)
        values['agent']['task_timeout']=1
        timed=run_task(task,client,Config(values,tmp_path),'mock-model',1)
        assert timed['environment_error'] and '124' in timed['environment_error'],timed
        assert not docker_client.containers.list(filters={'ancestor':tag})
    finally:
        client.close()
        server.shutdown(); server.server_close(); thread.join(timeout=5)
        try: docker_client.images.remove(tag,force=True)
        except docker.errors.ImageNotFound: pass
        docker_client.close()
