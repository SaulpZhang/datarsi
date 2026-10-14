from pathlib import Path
import pytest
from datarsi.configuration import load_config
from datarsi.storage import State,run_lock


def test_yaml_inheritance_and_override(tmp_path):
    (tmp_path/'base.yaml').write_text('experiment:\n  name: base\ntraining:\n  num_processes: 1\n')
    (tmp_path/'child.yaml').write_text('extends: base.yaml\nexperiment:\n  name: child\n')
    cfg=load_config(tmp_path/'child.yaml',['resources.gpu_ids=[0,1]'],root=tmp_path)
    assert cfg['training']['num_processes']==1
    assert cfg['resources']['gpu_ids']==[0,1]
    assert cfg.run_dir==tmp_path/'runs'/'child'


def test_unknown_keys_and_gpu_limit(tmp_path):
    path=tmp_path/'config.yaml'
    path.write_text('training:\n  typo: 2\n')
    with pytest.raises(ValueError,match='Unknown keys'):
        load_config(path)
    path.write_text('{}')
    with pytest.raises(ValueError,match='at most two'):
        load_config(path,['resources.gpu_ids=[0,1,2]'])
    with pytest.raises(ValueError,match='Unknown override'):
        load_config(path,['agent.typo=2'])


def test_config_cycle(tmp_path):
    (tmp_path/'a.yaml').write_text('extends: b.yaml')
    (tmp_path/'b.yaml').write_text('extends: a.yaml')
    with pytest.raises(ValueError,match='Circular'):
        load_config(tmp_path/'a.yaml')


def test_state_failure_retry_and_completed_reuse(tmp_path):
    state=State(tmp_path,'fingerprint')
    artifact=tmp_path/'checkpoint'
    def fail():
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        state.run('train',fail)
    assert state.value['stages']['train']['status']=='failed'
    artifact.write_text('saved')
    resumed=State(tmp_path,'fingerprint',resume=True)
    resumed.run('train',lambda:{'files':[str(artifact)],'checkpoint':str(artifact)})
    assert resumed.run('train',fail)['checkpoint']==str(artifact)
    artifact.unlink()
    with pytest.raises(FileNotFoundError):
        resumed.run('train',fail)
    with pytest.raises(ValueError,match='Config changed'):
        State(tmp_path,'changed',resume=True)


def test_run_lock_excludes_duplicate_run(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError,match='already in use'):
            with run_lock(tmp_path):
                pass


def test_fingerprint_includes_input_content(tmp_path):
    (tmp_path/'config.yaml').write_text('data:\n  train_file: train.jsonl\n')
    (tmp_path/'train.jsonl').write_text('first')
    cfg=load_config(tmp_path/'config.yaml',root=tmp_path)
    before=cfg.fingerprint
    (tmp_path/'train.jsonl').write_text('changed')
    assert before!=cfg.fingerprint
