from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    'experiment': {'name': 'development', 'rounds': 3, 'seed': 42, 'run_dir': None},
    'resources': {'gpu_ids': []},
    'model': {'name_or_path': 'Qwen/Qwen3.5-9B', 'kind': 'image_text', 'source': 'pretrained'},
    'data': {'bootstrap_trajectories': 'data/bootstrap.jsonl', 'train_file': None,
             'train_tasks': 'data/tasks/train.jsonl', 'dev_tasks': 'data/tasks/dev.jsonl',
             'max_length': 8192, 'max_train_examples': 1000, 'mask_thinking': True},
    'training': {'num_processes': 2, 'allow_cpu': False, 'trainer': {
        'bf16': True, 'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 16,
        'gradient_checkpointing': True, 'learning_rate': 1e-5, 'num_train_epochs': 1,
        'logging_steps': 10, 'save_steps': 100, 'save_total_limit': 2}},
    'inference': {'managed': True, 'base_url': 'http://127.0.0.1:18080/v1',
        'host': '127.0.0.1', 'port': 18080, 'served_model_name': 'datarsi',
        'tensor_parallel_size': 2, 'max_model_len': 16384, 'gpu_memory_utilization': 0.6,
        'startup_timeout': 600, 'request_timeout': 300, 'extra_args': [], 'api_key_env': 'DATARSI_MODEL_API_KEY'},
    'agent': {'max_steps': 30, 'command_timeout': 120, 'max_observation_chars': 16000,
        'max_new_tokens': 2048, 'temperature': 0.2},
    'runtime': {'docker_host': 'unix:///var/run/docker.sock', 'pull_missing_images': False,
        'network_disabled': True, 'memory_limit': '8g', 'cpus': 2},
    'experience': {'teacher_checkpoint': None, 'repair_max_steps': 30, 'replay_ratio': 0.5},
    'tracking': {'mode': 'offline', 'project': 'datarsi', 'entity': None, 'tags': [], 'log_manifests': True},
}


def merge(base, update):
    result=deepcopy(base)
    for key,value in update.items():
        if isinstance(value,dict) and isinstance(result.get(key),dict):
            result[key]=merge(result[key],value)
        else:
            result[key]=deepcopy(value)
    return result


def read_yaml(path, seen=None):
    path=Path(path).resolve()
    seen=set() if seen is None else set(seen)
    if path in seen:
        raise ValueError(f'Circular config inheritance: {path}')
    seen.add(path)
    raw=yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw,dict):
        raise ValueError('Config must be a YAML mapping')
    parent=raw.pop('extends',None)
    return merge(read_yaml(path.parent / parent,seen),raw) if parent else raw


@dataclass
class Config:
    values: dict
    root: Path=ROOT

    def __getitem__(self,key):
        return self.values[key]

    def path(self,value):
        path=Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    @property
    def run_dir(self):
        return self.path(self['experiment']['run_dir'] or f"runs/{self['experiment']['name']}")

    @property
    def fingerprint(self):
        sources=[self['data']['train_file'] or self['data']['bootstrap_trajectories'], self['data']['train_tasks'], self['data']['dev_tasks']]
        inputs={}
        for value in sources:
            path=self.path(value)
            if path.is_file():
                with path.open('rb') as stream:
                    inputs[str(path)]=hashlib.file_digest(stream,'sha256').hexdigest()
            else:
                inputs[str(path)]=None
        return hashlib.sha256(json.dumps({'config':self.values,'inputs':inputs},sort_keys=True).encode()).hexdigest()

    def save(self,path):
        Path(path).write_text(yaml.safe_dump(self.values,sort_keys=False))


def load_config(path, overrides=(), root=ROOT):
    path=Path(path)
    if not path.exists() and len(path.parts)==1:
        path=Path(root) / 'config' / path
    supplied=read_yaml(path)
    for group,content in supplied.items():
        if group not in DEFAULTS or not isinstance(content,dict):
            raise ValueError(f'Unknown/invalid config section: {group}')
        unknown=set(content)-set(DEFAULTS[group])
        if unknown:
            raise ValueError(f'Unknown keys in {group}: {sorted(unknown)}')
    values=merge(DEFAULTS,supplied)
    for item in overrides:
        key,separator,text=item.partition('=')
        if not separator:
            raise ValueError('Overrides must be section.key=value')
        parts=key.split('.')
        destination=values
        for part in parts[:-1]:
            if part not in destination or not isinstance(destination[part],dict):
                raise ValueError(f'Unknown override: {key}')
            destination=destination[part]
        if parts[-1] not in destination and not key.startswith('training.trainer.'):
            raise ValueError(f'Unknown override: {key}')
        destination[parts[-1]]=yaml.safe_load(text)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',values['experiment']['name']):
        raise ValueError('Experiment name must be a safe directory name')
    for value in (values['experiment']['rounds'],values['training']['num_processes'],values['data']['max_length'],values['data']['max_train_examples'],values['agent']['max_steps']):
        if type(value) is not int or value < 1:
            raise ValueError('Round, process and budget counts must be positive integers')
    ids=values['resources']['gpu_ids']
    if not isinstance(ids,list) or len(ids)>2 or len(set(ids))!=len(ids) or any(type(i) is not int or i<0 for i in ids):
        raise ValueError('gpu_ids must contain at most two distinct container GPU indices')
    if values['tracking']['mode'] not in {'offline','online','disabled'}:
        raise ValueError('Invalid W&B mode')
    if values['model']['kind'] not in {'causal','image_text'} or values['model']['source'] not in {'pretrained','tiny'}:
        raise ValueError('Invalid model kind/source')
    if not 0<=values['experience']['replay_ratio']<=1:
        raise ValueError('replay_ratio must be in [0,1]')
    if not 0<values['inference']['gpu_memory_utilization']<1:
        raise ValueError('gpu_memory_utilization must be in (0,1)')
    if not 1024<=values['inference']['port']<=65535:
        raise ValueError('Invalid inference port')
    if any(key in values['training']['trainer'] for key in ('output_dir','report_to','seed','use_cpu')):
        raise ValueError('output_dir/report_to/seed/use_cpu are managed by the pipeline')
    return Config(values,Path(root))
