from dataclasses import dataclass
from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from urllib.request import urlopen


def terminate(process):
    if process is None:
        return
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    # The launcher may have exited while its engine children remain alive.
    try:
        os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=20)


@dataclass
class Reply:
    content: str
    usage: dict
    reasoning: str | None=None


class Client:
    def __init__(self,base_url,model,timeout=300,api_key_env='DATARSI_MODEL_API_KEY'):
        from openai import OpenAI
        self.client=OpenAI(base_url=base_url,api_key=os.environ.get(api_key_env,'EMPTY'),timeout=timeout,max_retries=2)
        self.model=model

    def complete(self,messages,max_tokens=2048,temperature=0.2):
        result=self.client.chat.completions.create(model=self.model,messages=messages,max_tokens=max_tokens,
            temperature=temperature,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        message=result.choices[0].message
        data=message.model_dump()
        return Reply(message.content or '',result.usage.model_dump() if result.usage else {},data.get('reasoning') or data.get('reasoning_content'))

    def close(self):
        self.client.close()


class VLLMService:
    def __init__(self,config,checkpoint,log_dir):
        self.config=config
        self.checkpoint=checkpoint
        self.log_dir=Path(log_dir)
        self.process=None
        self.log=None
        self.client=None
        self.model=config['inference']['served_model_name']+'-'+uuid.uuid4().hex[:8]

    def command(self):
        options=self.config['inference']
        return [sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',self.checkpoint,
            '--served-model-name',self.model,'--host',options['host'],'--port',str(options['port']),
            '--tensor-parallel-size',str(options['tensor_parallel_size']),
            '--max-model-len',str(options['max_model_len']),
            '--gpu-memory-utilization',str(options['gpu_memory_utilization']),
            '--enable-auto-tool-choice','--tool-call-parser',options['tool_call_parser'],*options['extra_args']]

    def __enter__(self):
        options=self.config['inference']
        if not options['managed']:
            self.client=Client(options['base_url'],options['served_model_name'],options['request_timeout'],options['api_key_env'])
            return self.client
        ids=self.config['resources']['gpu_ids']
        if not ids or options['tensor_parallel_size']>len(ids):
            raise ValueError('Assign GPUs before starting managed vLLM')
        host='127.0.0.1' if options['host']=='0.0.0.0' else options['host']
        with socket.socket() as probe:
            probe.settimeout(1)
            if probe.connect_ex((host,options['port']))==0:
                raise RuntimeError('Inference port already occupied; choose a different port')
        self.log_dir.mkdir(parents=True,exist_ok=True)
        self.log=(self.log_dir/'vllm.log').open('a')
        environment={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,ids))}
        try:
            self.process=subprocess.Popen(self.command(),stdout=self.log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            url=f"http://{host}:{options['port']}/v1"
            deadline=time.monotonic()+options['startup_timeout']
            while time.monotonic()<deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'vLLM exited; see {self.log_dir / "vllm.log"}')
                try:
                    with urlopen(url+'/models',timeout=2) as response:
                        models=json.load(response)
                    if self.model in {item['id'] for item in models.get('data',[])}:
                        self.client=Client(url,self.model,options['request_timeout'],options['api_key_env'])
                        return self.client
                except (OSError,ValueError):
                    pass
                time.sleep(0.5)
            raise TimeoutError('vLLM did not become ready before startup timeout')
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def __exit__(self,*args):
        if self.client:
            self.client.close()
        terminate(self.process)
        if self.log:
            self.log.close()
