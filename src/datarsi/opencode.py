"""OpenCode integration and a local, recording OpenAI-compatible gateway."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def response_message(raw):
    """Reassemble the assistant message from JSON or streamed OpenAI chunks."""
    if not raw.lstrip().startswith(b'data:'):
        return json.loads(raw)['choices'][0]['message']
    message={'role':'assistant','content':''}
    calls={}
    for line in raw.decode().splitlines():
        if not line.startswith('data:') or line[5:].strip()=='[DONE]':
            continue
        chunk=json.loads(line[5:])
        for choice in chunk.get('choices',[]):
            delta=choice.get('delta',{})
            message['content']+=delta.get('content') or ''
            for item in delta.get('tool_calls',[]):
                call=calls.setdefault(item['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
                call['id']+=item.get('id') or ''
                for field in ('name','arguments'):
                    call['function'][field]+=item.get('function',{}).get(field) or ''
    if calls:
        message['tool_calls']=[calls[key] for key in sorted(calls)]
    return message


class RecordingGateway:
    """One task per gateway; credentials and privileged guidance stay out of artifacts."""
    def __init__(self,client,config,private_instruction=None):
        self.base_url=str(client.client.base_url).rstrip('/')
        self.api_key=client.client.api_key
        self.model=client.model
        self.timeout=config['inference']['request_timeout']
        self.private=private_instruction
        self.calls=[]
        self.errors=[]
        self.token=secrets.token_hex(32)
        self.config=config

    def __enter__(self):
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self): self.forward()
            def do_GET(self): self.forward()
            def forward(self):
                if self.headers.get('Authorization')!='Bearer '+owner.token:
                    self.send_error(401)
                    return
                suffix=self.path.removeprefix('/v1')
                if suffix not in ('/chat/completions','/models'):
                    self.send_error(404)
                    return
                body=None
                request_data=None
                try:
                    if self.command=='POST':
                        length=int(self.headers.get('Content-Length','0'))
                        if length>16_000_000:
                            self.send_error(413)
                            return
                        request_data=json.loads(self.rfile.read(length))
                        request_data['model']=owner.model
                        request_data['temperature']=owner.config['agent']['temperature']
                        request_data['max_tokens']=owner.config['agent']['max_new_tokens']
                        request_data['chat_template_kwargs']={'enable_thinking':False}
                        request_data.pop('max_completion_tokens',None)
                        body=json.dumps(request_data).encode()
                    request=Request(owner.base_url+suffix,data=body,method=self.command,
                        headers={'Authorization':'Bearer '+owner.api_key,'Content-Type':'application/json'})
                    with urlopen(request,timeout=owner.timeout) as response:
                        raw=response.read()
                        content_type=response.headers.get('Content-Type','application/json')
                    if request_data is not None:
                        messages=deepcopy(request_data['messages'])
                        for message in messages:
                            # Previous assistant responses are context, not duplicate targets.
                            if message['role']=='assistant': message['supervise']=False
                            content=message.get('content')
                            if owner.private and isinstance(content,str):
                                message['content']=content.replace(owner.private,'')
                            elif owner.private and isinstance(content,list):
                                for part in content:
                                    if part.get('type')=='text': part['text']=part['text'].replace(owner.private,'')
                        messages.append(response_message(raw))
                        owner.calls.append({'messages':messages,'tools':request_data.get('tools',[])})
                    self.send_response(200)
                    self.send_header('Content-Type',content_type)
                    self.send_header('Content-Length',str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except Exception as error:
                    owner.errors.append(type(error).__name__)
                    code=error.code if isinstance(error,HTTPError) else 502
                    self.send_error(code,'Model gateway request failed')
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
        self.url=f'http://127.0.0.1:{self.server.server_port}/v1'
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def configuration(config,model,base_url,steps):
    return {'$schema':'https://opencode.ai/config.json','autoupdate':False,
        'share':'disabled','enabled_providers':['datarsi'],
        'provider':{'datarsi':{'npm':'@ai-sdk/openai-compatible','name':'vLLM',
            'options':{'baseURL':base_url,'apiKey':'{env:DATARSI_GATEWAY_KEY}'},
            'models':{model:{'name':model,'tool_call':True,
                'limit':{'context':config['inference']['max_model_len'],'output':config['agent']['max_new_tokens']}}}}},
        'model':'datarsi/'+model,'small_model':'datarsi/'+model,
        'permission':{'*':'deny','bash':'allow','read':'allow','edit':'allow','write':'allow',
            'glob':'allow','grep':'allow','list':'allow'},
        'compaction':{'auto':False},
        'agent':{'title':{'disable':True},'summary':{'disable':True},'compaction':{'disable':True},'datarsi':{'description':'Solve the repository task using observed evidence',
            'mode':'primary','steps':steps,'temperature':config['agent']['temperature'],
            'prompt':'You are a coding agent. Investigate the issue, edit files and run tests. Use repository evidence.',
            'tools':{'task':False,'webfetch':False,'websearch':False,'question':False,'skill':False,'lsp':False}}}}
