from dataclasses import dataclass
import base64
import io
import json
import os
from pathlib import Path
import shlex
import tarfile
import time


@dataclass
class Execution:
    exit_code: int
    output: str
    seconds: float


class DockerRuntime:
    def __init__(self,task,config):
        self.task=task
        self.config=config
        self.container=None
        self.client=None

    def __enter__(self):
        import docker
        options=self.config['runtime']
        self.client=docker.DockerClient(base_url=options['docker_host'],timeout=self.config['agent']['command_timeout']+60)
        try:
            self.client.ping()
            try:
                self.client.images.get(self.task.image)
            except docker.errors.ImageNotFound:
                if not options['pull_missing_images']:
                    raise RuntimeError(f'Task image missing: {self.task.image}')
                self.client.images.pull(self.task.image)
            self.container=self.client.containers.run(self.task.image,detach=True,entrypoint='/bin/bash',
                command=['-lc','sleep infinity'],working_dir=self.task.repo_dir,init=True,runtime='runc',
                network_mode='container:'+options['opencode_network_container'],mem_limit=options['memory_limit'],
                nano_cpus=int(options['cpus']*1e9),labels={'datarsi.managed':'task'},
                environment={'NVIDIA_VISIBLE_DEVICES':'void','CUDA_VISIBLE_DEVICES':'-1'})
            if self.task.base_commit:
                import shlex
                result=self.execute('git reset --hard '+shlex.quote(self.task.base_commit))
                if result.exit_code:
                    raise RuntimeError('Cannot reset task repository: '+result.output)
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def execute(self,command,limit=None):
        start=time.monotonic()
        seconds=self.config['agent']['command_timeout']
        # Timeout is enforced inside the task container, not by abandoning an exec request.
        result=self.container.exec_run(['timeout',str(seconds),'bash','-lc',command],workdir=self.task.repo_dir,demux=True)
        output=b''.join(part or b'' for part in result.output).decode(errors='replace')
        if limit is not None and len(output)>limit:
            output=output[:limit//2]+'\n[output truncated]\n'+output[-limit//2:]
        return Execution(result.exit_code,output,time.monotonic()-start)

    def put_file(self,path,data,mode=0o600):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as archive:
            entry=tarfile.TarInfo(path.rsplit('/',1)[-1])
            entry.size=len(data)
            entry.mode=mode
            archive.addfile(entry,io.BytesIO(data))
        if not self.container.put_archive(path.rsplit('/',1)[0],stream.getvalue()):
            raise RuntimeError('Cannot copy file into task container')

    def run_opencode(self,settings,prompt,token):
        binary=Path(self.config['agent']['opencode_binary'])
        self.put_file('/tmp/datarsi-opencode',binary.read_bytes(),0o755)
        self.put_file('/tmp/datarsi-opencode.json',json.dumps(settings).encode())
        self.put_file('/tmp/datarsi-prompt',prompt.encode())
        # No shell interpolation of prompts/credentials. This owned exec launches in the
        # background; deleting its task container also kills every descendant on timeout.
        argv=['timeout','--signal=TERM','--kill-after=10s',str(self.config['agent']['task_timeout']),
            '/tmp/datarsi-opencode','run','--pure','--format','json','--agent','datarsi',
            '--model',settings['model']]
        command=' '.join(shlex.quote(item) for item in argv)+' "$(cat /tmp/datarsi-prompt)"'
        script=command+' > /tmp/datarsi-events.jsonl 2> /tmp/datarsi-stderr & child=$!; wait "$child"'
        environment={'DATARSI_GATEWAY_KEY':token,'OPENCODE_CONFIG':'/tmp/datarsi-opencode.json',
            'OPENCODE_DISABLE_PROJECT_CONFIG':'true','OPENCODE_DISABLE_AUTOUPDATE':'true',
            'OPENCODE_DISABLE_MODELS_FETCH':'true','OPENCODE_DISABLE_LSP_DOWNLOAD':'true',
            'XDG_CONFIG_HOME':'/tmp/datarsi-config','XDG_DATA_HOME':'/tmp/datarsi-data',
            'XDG_CACHE_HOME':'/tmp/datarsi-cache','HOME':'/tmp/datarsi-home'}
        # Docker's detached exec is the background process interface; no host shell or
        # Docker socket is given to OpenCode itself.
        job=self.client.api.exec_create(self.container.id,['bash','-lc',script],
            workdir=self.task.repo_dir,environment=environment)
        self.client.api.exec_start(job['Id'],detach=True)
        deadline=time.monotonic()+self.config['agent']['task_timeout']+15
        while True:
            status=self.client.api.exec_inspect(job['Id'])
            if not status['Running']:
                break
            if time.monotonic()>deadline:
                raise TimeoutError('OpenCode task exceeded timeout')
            time.sleep(0.2)
        events=[]
        output=self.execute('cat /tmp/datarsi-events.jsonl').output
        for line in output.splitlines():
            if line.strip(): events.append(json.loads(line))
        stderr=self.execute('cat /tmp/datarsi-stderr',limit=8000).output
        return {'exit_code':status['ExitCode'],'events':events,'stderr':stderr}

    def patch(self):
        result=self.execute('git add -N . && git diff --binary HEAD')
        if result.exit_code:
            raise RuntimeError('Cannot extract patch: '+result.output)
        if len(result.output)>2_000_000:
            raise RuntimeError('Patch exceeds size budget')
        return result.output

    def apply_patch(self,patch):
        encoded=base64.b64encode(patch.encode()).decode()
        result=self.execute(f'printf %s {encoded} | base64 -d | git apply --whitespace=nowarn -')
        if result.exit_code:
            raise RuntimeError('Cannot apply verification patch: '+result.output)

    def verify(self):
        if self.task.test_patch:
            self.apply_patch(self.task.test_patch)
        result=self.execute(self.task.verify_command,limit=self.config['agent']['max_observation_chars'])
        return {'passed':result.exit_code==0,'exit_code':result.exit_code,'seconds':result.seconds,'output':result.output}

    def __exit__(self,*args):
        try:
            if self.container:
                self.container.remove(force=True)
        finally:
            if self.client:
                self.client.close()
