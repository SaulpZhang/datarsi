import json
import re
from .runtime import DockerRuntime

SYSTEM="""You are a coding agent in an isolated repository. Investigate the issue, edit files and run tests.
Return exactly one JSON object per turn: {"command":"shell command"} or {"done":true}.
Use observed repository evidence. The environment executes one command and returns its output."""


def parse_action(content):
    content=re.sub(r'^```(?:json)?\s*|\s*```$','',content.strip())
    action=json.loads(content)
    if not isinstance(action,dict):
        raise ValueError('Action must be an object')
    if action.get('done') is True and set(action)=={'done'}:
        return action
    if set(action)=={'command'} and isinstance(action['command'],str) and action['command'].strip():
        return action
    raise ValueError('Expected command or done action')


def run_task(task,client,config,checkpoint,round_index,prefix=None,private_instruction=None,runtime_factory=DockerRuntime):
    record={'task_id':task.task_id,'split':task.split,'model_checkpoint':checkpoint,'round':round_index,
        'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':task.problem}],
        'actions':[],'observations':[],'resolved':False,'environment_error':None,'failure_kind':None,
        'usage':{'prompt_tokens':0,'completion_tokens':0},'patch':''}
    options=config['agent']
    steps=config['experience']['repair_max_steps'] if private_instruction else options['max_steps']
    try:
        with runtime_factory(task,config) as runtime:
            # Re-execute commands to restore the Student state; never reuse old observations.
            for action in prefix or []:
                if 'command' not in action:
                    continue
                result=runtime.execute(action['command'],limit=options['max_observation_chars'])
                observation=f'exit_code={result.exit_code}\n{result.output}'
                record['messages'].extend([{'role':'assistant','content':json.dumps(action),'supervise':False}, {'role':'user','content':observation}])
                record['actions'].append(action)
                record['observations'].append({'exit_code':result.exit_code,'output':result.output,'seconds':result.seconds})
            for _ in range(steps):
                request=[{'role':message['role'],'content':message['content']} for message in record['messages']]
                if private_instruction:
                    request[0]['content']+='\nPRIVATE TEACHER GUIDANCE (not part of training context):\n'+private_instruction
                reply=client.complete(request,max_tokens=options['max_new_tokens'],temperature=options['temperature'])
                for key in record['usage']:
                    record['usage'][key]+=reply.usage.get(key,0) or 0
                try:
                    action=parse_action(reply.content)
                except (ValueError,TypeError):
                    record['failure_kind']='format'
                    break
                record['messages'].append({'role':'assistant','content':json.dumps(action)})
                if action.get('done'):
                    break
                result=runtime.execute(action['command'],limit=options['max_observation_chars'])
                record['actions'].append(action)
                record['observations'].append({'exit_code':result.exit_code,'output':result.output,'seconds':result.seconds})
                record['messages'].append({'role':'user','content':f'exit_code={result.exit_code}\n{result.output}'})
            else:
                record['failure_kind']='budget'
            record['patch']=runtime.patch()
            record['verification']=runtime.verify()
            record['resolved']=record['verification']['passed']
            if not record['resolved'] and not record['failure_kind']:
                record['failure_kind']='verification'
    except Exception as error:
        record['environment_error']=f'{type(error).__name__}: {error}'
    return record
