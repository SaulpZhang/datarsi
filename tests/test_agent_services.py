from copy import deepcopy
import io
import json
import sys
import pytest
from datarsi.agent import parse_action,run_task
from datarsi.configuration import Config,DEFAULTS
from datarsi.experience import repair
from datarsi.inference import Reply,VLLMService
from datarsi.runtime import Execution
from datarsi.tasks import Task,check_disjoint


class FakeRuntime:
    calls=[]
    def __init__(self,*args): pass
    def __enter__(self): return self
    def execute(self,command,limit=None):
        self.calls.append(command)
        return Execution(0,'real observation',0.01)
    def patch(self): return 'diff produced by execution'
    def verify(self): return {'passed':True,'exit_code':0,'seconds':0.01}
    def __exit__(self,*args): pass


class FakeClient:
    def __init__(self,*args):
        self.requests=[]
        self.closed=False
    def complete(self,messages,**kwargs):
        self.requests.append(messages)
        return Reply('{"done":true}',{'prompt_tokens':3,'completion_tokens':2})
    def close(self): self.closed=True


def task(split='train'):
    return Task('task-1',split,'fix bug','task-image','/testbed','pytest',gold_patch='PRIVATE_GOLD_PATCH')


def test_json_actions():
    assert parse_action('```json\n{"command":"ls"}\n```')=={'command':'ls'}
    with pytest.raises(ValueError): parse_action('{"done":true,"command":"ls"}')
    with pytest.raises(ValueError): parse_action('{"command":""}')


def test_repair_replays_state_without_storing_private_prompt(tmp_path):
    cfg=Config(deepcopy(DEFAULTS),tmp_path)
    client=FakeClient()
    failed={'actions':[{'command':'ls'}],'failure_kind':'verification'}
    row=repair(task(),failed,client,cfg,'checkpoint',1,runtime_factory=FakeRuntime)
    assert row['resolved']
    assert 'PRIVATE_GOLD_PATCH' in json.dumps(client.requests)
    assert 'PRIVATE_GOLD_PATCH' not in json.dumps(row)
    assert row['messages'][3]['content'].endswith('real observation')


def test_dev_repair_and_split_overlap_rejected(tmp_path):
    with pytest.raises(ValueError,match='Only training'):
        repair(task('dev'),{'actions':[]},FakeClient(),Config(deepcopy(DEFAULTS),tmp_path),'model',1)
    with pytest.raises(ValueError,match='overlap'):
        check_disjoint([task()],[task('dev')])


def test_runtime_error_is_not_a_model_failure(tmp_path):
    class BrokenRuntime(FakeRuntime):
        def __enter__(self): raise RuntimeError('image missing')
    row=run_task(task(),FakeClient(),Config(deepcopy(DEFAULTS),tmp_path),'model',1,runtime_factory=BrokenRuntime)
    assert not row['resolved']
    assert 'image missing' in row['environment_error']


def test_owned_service_ready_and_shutdown(tmp_path,monkeypatch):
    import datarsi.inference as module
    values=deepcopy(DEFAULTS)
    values['resources']['gpu_ids']=[0]
    values['inference']['tensor_parallel_size']=1
    service=VLLMService(Config(values,tmp_path),'model',tmp_path)
    monkeypatch.setattr(service,'command',lambda:[sys.executable,'-c','import time;time.sleep(60)'])
    monkeypatch.setattr(module,'urlopen',lambda *a,**k:io.BytesIO(json.dumps({'data':[{'id':service.model}]}).encode()))
    monkeypatch.setattr(module,'Client',FakeClient)
    with service as client:
        assert service.process.poll() is None
    assert client.closed
    assert service.process.poll() is not None


def test_startup_timeout_closes_own_process(tmp_path,monkeypatch):
    import datarsi.inference as module
    values=deepcopy(DEFAULTS)
    values['resources']['gpu_ids']=[0]
    values['inference'].update(tensor_parallel_size=1,startup_timeout=0.01)
    service=VLLMService(Config(values,tmp_path),'model',tmp_path)
    monkeypatch.setattr(service,'command',lambda:[sys.executable,'-c','import time;time.sleep(60)'])
    monkeypatch.setattr(module,'urlopen',lambda *a,**k:io.BytesIO(b'{"data":[{"id":"different-model"}]}'))
    with pytest.raises(TimeoutError):
        service.__enter__()
    assert service.process.poll() is not None
