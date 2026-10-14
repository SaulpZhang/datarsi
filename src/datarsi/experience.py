from .agent import run_task

STRATEGIES={
    'format':'Use OpenCode tools to inspect the relevant source before editing.',
    'budget':'First use targeted searches to localize the smallest relevant code region. Avoid repeating broad searches.',
    'verification':'First reproduce or inspect the remaining test failure, then inspect its callers before editing.',
}


def repair(task,failed,client,config,checkpoint,round_index,runtime_factory=None):
    if task.split!='train':
        raise ValueError('Only training tasks may be repaired using privileged information')
    if not task.gold_patch:
        raise ValueError(f'{task.task_id}: gold_patch is required for gold-guided repair')
    kind=failed.get('failure_kind') or 'verification'
    strategy=STRATEGIES.get(kind,STRATEGIES['verification'])
    private=f'{strategy}\nReference patch (training privilege only):\n{task.gold_patch}\nUse tools to acquire the evidence necessary for the next action.'
    kwargs={'runtime_factory':runtime_factory} if runtime_factory else {}
    record=run_task(task,client,config,checkpoint,round_index,prefix=failed['actions'],prefix_patch=failed.get('patch'),private_instruction=private,**kwargs)
    record['repair_strategy']=kind
    record['source']='student-aware-repair'
    return record
