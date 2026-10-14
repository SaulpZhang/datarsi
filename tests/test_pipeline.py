from copy import deepcopy
import json
from pathlib import Path
from datarsi.configuration import Config,DEFAULTS
from datarsi.pipeline import Pipeline
from datarsi.storage import State,write_jsonl


class Tracker:
    def log(self,*args): pass


def test_rounds_use_new_checkpoint_and_new_data(tmp_path,monkeypatch):
    import datarsi.pipeline as module
    values=deepcopy(DEFAULTS)
    values['experiment']['rounds']=2
    values['data'].update(train_tasks='train.jsonl',dev_tasks='dev.jsonl',train_file='bootstrap.jsonl')
    cfg=Config(values,tmp_path)
    task={'task_id':'train','split':'train','problem':'issue','image':'img','repo_dir':'/repo','verify_command':'pytest'}
    write_jsonl(tmp_path/'train.jsonl',[task])
    write_jsonl(tmp_path/'dev.jsonl',[{**task,'task_id':'dev','split':'dev'}])
    write_jsonl(tmp_path/'bootstrap.jsonl',[{'input_ids':[1,2],'attention_mask':[1,1],'labels':[-100,2]}])
    monkeypatch.setattr(module,'preflight',lambda config:None)
    pipeline=Pipeline(cfg,State(cfg.run_dir,cfg.fingerprint),Tracker())
    pipeline.initial_data=lambda:{'train_file':'initial'}
    calls=[]
    def training(round_index,data_file,checkpoint):
        calls.append((round_index,data_file,checkpoint))
        return {'checkpoint':f'checkpoint-{round_index}'}
    pipeline.training=training
    pipeline.collect=lambda r,c,t,s:{'trajectories':'rollout','metrics':{'evaluate/pass_rate':r/10}}
    raw=tmp_path/'next.jsonl'
    write_jsonl(raw,[{'input_ids':[1,2],'attention_mask':[1,1],'labels':[-100,2],'split':'train'}])
    pipeline.reconstruct=lambda *args:{'raw_data':str(raw)}
    result=pipeline.run()
    assert result['final_checkpoint']=='checkpoint-2'
    assert calls[1][2]=='checkpoint-1'
    assert calls[1][1].endswith('next.tokenized.jsonl')


def test_environment_failures_are_retried_on_resume(tmp_path,monkeypatch):
    import pytest
    import datarsi.pipeline as module
    from datarsi.tasks import Task
    class Service:
        def __init__(self,*args): pass
        def __enter__(self): return None
        def __exit__(self,*args): pass
    monkeypatch.setattr(module,'VLLMService',Service)
    calls=[]
    def episode(*args):
        calls.append(1)
        return {'task_id':'t','model_checkpoint':'checkpoint','environment_error':'missing image' if len(calls)==1 else None,
                'resolved':len(calls)>1,'usage':{'prompt_tokens':0,'completion_tokens':0}}
    monkeypatch.setattr(module,'run_task',episode)
    cfg=Config(deepcopy(DEFAULTS),tmp_path)
    pipeline=Pipeline(cfg,State(cfg.run_dir,cfg.fingerprint),Tracker())
    tasks=[Task('t','train','issue','img','/repo','pytest')]
    with pytest.raises(RuntimeError,match='No task'):
        pipeline.collect(1,'checkpoint',tasks,'rollout')
    resumed=Pipeline(cfg,State(cfg.run_dir,cfg.fingerprint,resume=True),Tracker())
    result=resumed.collect(1,'checkpoint',tasks,'rollout')
    assert result['metrics']['rollout/pass_rate']==1
    assert len(calls)==2
