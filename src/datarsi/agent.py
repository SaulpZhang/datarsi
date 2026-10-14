from .runtime import DockerRuntime
from .opencode import RecordingGateway,configuration


def run_task(task,client,config,checkpoint,round_index,prefix=None,private_instruction=None,
             runtime_factory=DockerRuntime,prefix_patch=None):
    record={'task_id':task.task_id,'split':task.split,'model_checkpoint':checkpoint,'round':round_index,
        'agent_backend':'opencode','messages':[],'model_calls':[],'actions':[],'observations':[],
        'resolved':False,'environment_error':None,'failure_kind':None,
        'usage':{'prompt_tokens':0,'completion_tokens':0},'patch':''}
    options=config['agent']
    steps=config['experience']['repair_max_steps'] if private_instruction else options['max_steps']
    try:
        with runtime_factory(task,config) as runtime:
            if prefix_patch:
                runtime.apply_patch(prefix_patch)
            else:
                for action in prefix or []:
                    if 'command' in action:
                        runtime.execute(action['command'],limit=options['max_observation_chars'])
            with RecordingGateway(client,config,private_instruction) as gateway:
                prompt=task.problem
                if private_instruction:
                    prompt+='\nPRIVATE TEACHER GUIDANCE:\n'+private_instruction
                result=runtime.run_opencode(configuration(config,client.model,gateway.url,steps),prompt,gateway.token)
                record['model_calls']=gateway.calls
                if gateway.errors:
                    raise RuntimeError('Model gateway errors: '+','.join(gateway.errors))
            record['opencode']=result
            if result['exit_code']!=0:
                raise RuntimeError(f"OpenCode exited with {result['exit_code']}: {result['stderr'][-2000:]}")
            events=result['events']
            if any(event.get('type')=='error' for event in events):
                raise RuntimeError('OpenCode emitted an error event')
            for event in events:
                part=event.get('part',{})
                if event.get('type')=='tool_use':
                    state=part.get('state',{})
                    record['actions'].append({'tool':part.get('tool'),'input':state.get('input',{})})
                    record['observations'].append({'output':state.get('output',''),'status':state.get('status')})
                if event.get('type')=='step_finish':
                    tokens=part.get('tokens',{})
                    record['usage']['prompt_tokens']+=tokens.get('input',0)+tokens.get('cache',{}).get('read',0)
                    record['usage']['completion_tokens']+=tokens.get('output',0)
            if not record['model_calls']:
                raise RuntimeError('OpenCode produced no recorded model calls')
            record['messages']=record['model_calls'][-1]['messages']
            if len(record['model_calls'])>=steps:
                record['failure_kind']='budget'
            record['patch']=runtime.patch()
            record['verification']=runtime.verify()
            record['resolved']=record['verification']['passed']
            if not record['resolved'] and not record['failure_kind']:
                record['failure_kind']='verification'
    except Exception as error:
        record['environment_error']=f'{type(error).__name__}: {error}'
    return record
