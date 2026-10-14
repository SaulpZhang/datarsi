from copy import deepcopy
import io
import json
import os
import uuid
import pytest
from datarsi.agent import run_task
from datarsi.configuration import Config,DEFAULTS
from datarsi.inference import Reply
from datarsi.tasks import Task


@pytest.mark.skipif(os.environ.get('DATARSI_DOCKER_TESTS')!='1',reason='Opt-in Docker daemon integration test')
def test_real_task_container_edit_verify_cleanup(tmp_path):
    import docker
    client=docker.from_env()
    tag='datarsi-runtime-test:'+uuid.uuid4().hex[:8]
    dockerfile="""FROM datarsi-env:0.2
RUN mkdir -p /testbed && cd /testbed && git init && git config user.email test@example.invalid && git config user.name test && printf 'value = 0\\n' > app.py && git add app.py && git commit -m initial
"""
    try:
        client.images.build(fileobj=io.BytesIO(dockerfile.encode()),tag=tag,rm=True,forcerm=True)
        class Client:
            count=0
            def complete(self,*args,**kwargs):
                self.count+=1
                action={'command':"printf 'value = 1\\n' > app.py"} if self.count==1 else {'done':True}
                return Reply(json.dumps(action),{})
        task=Task('fixture','train','Fix value',tag,'/testbed',"python3 -c 'import app; assert app.value == 1'")
        values=deepcopy(DEFAULTS)
        cfg=Config(values,tmp_path)
        result=run_task(task,Client(),cfg,'fake-action-provider',1)
        assert result['environment_error'] is None,result
        assert result['resolved'],result
        assert '+value = 1' in result['patch']
        assert not client.containers.list(filters={'ancestor':tag})
    finally:
        try: client.images.remove(tag,force=True)
        except docker.errors.ImageNotFound: pass
        client.close()
