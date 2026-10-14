from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import tempfile
import time


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.tmp-')
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.tmp-')
    try:
        with os.fdopen(fd,'w') as stream:
            for row in rows:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def run_lock(directory):
    Path(directory).mkdir(parents=True,exist_ok=True)
    with (Path(directory)/'.lock').open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This run directory is already in use') from error
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


class State:
    def __init__(self,directory,fingerprint,resume=False):
        self.path=Path(directory)/'state.json'
        if self.path.exists():
            if not resume:
                raise FileExistsError('Run exists; use --resume or a new experiment name')
            self.value=json.loads(self.path.read_text())
            if self.value['fingerprint']!=fingerprint:
                raise ValueError('Config changed; use a new run directory')
        else:
            self.value={'fingerprint':fingerprint,'stages':{}}

    def run(self,key,operation):
        prior=self.value['stages'].get(key,{})
        if prior.get('status')=='completed':
            for path in prior['outputs'].get('files',[]):
                if not Path(path).exists():
                    raise FileNotFoundError(f'Completed stage artifact is missing: {path}')
            return prior['outputs']
        entry={'status':'running','started_at':time.time()}
        self.value['stages'][key]=entry
        atomic_json(self.path,self.value)
        try:
            outputs=operation()
            entry.update(status='completed',outputs=outputs,finished_at=time.time())
        except BaseException as error:
            entry.update(status='failed',error=str(error),finished_at=time.time())
            atomic_json(self.path,self.value)
            raise
        atomic_json(self.path,self.value)
        return outputs
