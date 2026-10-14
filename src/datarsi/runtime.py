from dataclasses import dataclass
import base64
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
                network_disabled=options['network_disabled'],mem_limit=options['memory_limit'],
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
