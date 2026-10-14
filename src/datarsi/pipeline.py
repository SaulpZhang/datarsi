from pathlib import Path
import json
import os
import subprocess
import sys
from .agent import run_task
from .data import prepare
from .experience import repair
from .inference import VLLMService,terminate
from .storage import State,atomic_json,read_jsonl,write_jsonl
from .tasks import load_tasks,check_disjoint


def plan(config,stage,round_index):
    return {'stage':stage,'round':round_index,'run_dir':str(config.run_dir),
        'rounds':config['experiment']['rounds'],'gpu_ids':config['resources']['gpu_ids'],
        'tracking':config['tracking']['mode'],
        'flow':['prepare','train','rollout','evaluate','repair','prepare-next'],
        'requirements':['Normalized train/dev task JSONL','Validated bootstrap trajectories or tokenized SFT data',
                        'GPU assignment for real training/vLLM','Docker socket for repository tasks']}


def preflight(config):
    if not config['resources']['gpu_ids']:
        raise ValueError('Set resources.gpu_ids after assigning GPUs to the container')
    if config['training']['num_processes']!=len(config['resources']['gpu_ids']):
        raise ValueError('num_processes must match the assigned GPU count')
    import docker
    with docker.DockerClient(base_url=config['runtime']['docker_host']) as client:
        client.ping()


class Pipeline:
    def __init__(self,config,state,tracker):
        self.config=config
        self.state=state
        self.tracker=tracker

    def folder(self,round_index,stage):
        path=self.config.run_dir/f'round-{round_index:03d}'/stage
        path.mkdir(parents=True,exist_ok=True)
        return path

    def initial_data(self):
        source=self.config['data']['train_file'] or self.config['data']['bootstrap_trajectories']
        target=self.config.run_dir/'initial.tokenized.jsonl'
        return self.state.run('prepare',lambda:prepare(self.config,self.config.path(source),target))

    def training(self,round_index,data_file,checkpoint):
        if not self.config['training']['allow_cpu'] and len(self.config['resources']['gpu_ids']) != self.config['training']['num_processes']:
            raise ValueError('Assign GPUs matching num_processes before training')
        folder=self.folder(round_index,'train')
        def execute():
            job={'round':round_index,'train_file':str(data_file),'checkpoint':checkpoint,'output_dir':str(folder)}
            job_path=folder/'job.json'
            atomic_json(job_path,job)
            argv=[str(self.config.root/'main.py'),'--config',str(self.config.run_dir/'resolved-config.yaml'),
                  '--stage','train','--round',str(round_index)]
            processes=self.config['training']['num_processes']
            command=[sys.executable,*argv] if processes==1 else [sys.executable,'-m','torch.distributed.run',
                '--standalone','--nproc_per_node',str(processes),*argv]
            environment={**os.environ,'DATARSI_JOB_SPEC':str(job_path),
                'CUDA_VISIBLE_DEVICES':','.join(map(str,self.config['resources']['gpu_ids']))}
            process=None
            with (folder/'train.log').open('a') as log:
                try:
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
                    code=process.wait()
                    if code:
                        raise RuntimeError(f'Training exited with {code}; see {folder / "train.log"}')
                finally:
                    terminate(process)
            metrics=json.loads((folder/'training-metrics.json').read_text())
            saved_checkpoint=folder/'final'
            if not (saved_checkpoint/'config.json').exists():
                raise FileNotFoundError('Training did not save a complete checkpoint')
            self.tracker.log({'round':round_index,'train/loss':metrics['train_loss']})
            return {'checkpoint':str(saved_checkpoint),'files':[str(saved_checkpoint),str(folder/'training-metrics.json')]}
        return self.state.run(f'r{round_index:03d}/train',execute)

    def collect(self,round_index,checkpoint,tasks,stage):
        folder=self.folder(round_index,stage)
        def execute():
            records=[]
            with VLLMService(self.config,checkpoint,folder) as client:
                for task in tasks:
                    target=folder/f'{task.task_id}.json'
                    if target.exists():
                        record=json.loads(target.read_text())
                        if record.get('model_checkpoint')!=checkpoint:
                            raise ValueError('Partial task checkpoint changed')
                        if record.get('environment_error'):
                            record=run_task(task,client,self.config,checkpoint,round_index)
                            atomic_json(target,record)
                    else:
                        record=run_task(task,client,self.config,checkpoint,round_index)
                        atomic_json(target,record)
                    records.append(record)
            path=folder/'trajectories.jsonl'
            write_jsonl(path,records)
            valid=[row for row in records if not row['environment_error']]
            if not valid:
                raise RuntimeError('No task produced a valid execution; inspect per-task environment errors and retry')
            successes=sum(row['resolved'] for row in valid)
            metrics={'round':round_index,f'{stage}/tasks':len(records),f'{stage}/valid_tasks':len(valid),
                f'{stage}/environment_errors':len(records)-len(valid),
                f'{stage}/pass_rate':successes/len(valid) if valid else 0.0,
                f'{stage}/coverage':len(valid)/len(records),
                f'{stage}/prompt_tokens':sum(row['usage']['prompt_tokens'] for row in records),
                f'{stage}/completion_tokens':sum(row['usage']['completion_tokens'] for row in records)}
            atomic_json(folder/'metrics.json',metrics)
            self.tracker.log(metrics)
            return {'trajectories':str(path),'metrics':metrics,'files':[str(path),str(folder/'metrics.json')]}
        return self.state.run(f'r{round_index:03d}/{stage}',execute)

    def reconstruct(self,round_index,checkpoint,tasks,rollout_path):
        folder=self.folder(round_index,'repair')
        def execute():
            task_map={task.task_id:task for task in tasks}
            rows=list(read_jsonl(rollout_path))
            failures=[row for row in rows if not row['resolved'] and not row['environment_error']]
            accepted=[row for row in rows if row['resolved'] and not row['environment_error']]
            attempted=0
            repaired=[]
            teacher=self.config['experience']['teacher_checkpoint'] or checkpoint
            if failures:
                with VLLMService(self.config,teacher,folder) as client:
                    for failure in failures:
                        task=task_map[failure['task_id']]
                        if not task.gold_patch:
                            continue
                        target=folder/f'{task.task_id}.json'
                        row=json.loads(target.read_text()) if target.exists() else repair(task,failure,client,self.config,teacher,round_index)
                        if not target.exists():
                            atomic_json(target,row)
                        attempted+=1
                        repaired.append(row)
                        if row['resolved'] and not row['environment_error']:
                            accepted.append(row)
            write_jsonl(folder/'repaired.jsonl',repaired)
            source=self.config['data']['train_file'] or self.config['data']['bootstrap_trajectories']
            replay=list(read_jsonl(self.config.path(source)))
            budget=self.config['data']['max_train_examples']
            fresh_budget=max(1,int(budget*(1-self.config['experience']['replay_ratio'])))
            selected=accepted[:fresh_budget]
            # Deterministic replay; the method does not claim a learned curriculum here.
            selected+=replay[:budget-len(selected)]
            if not selected:
                raise ValueError('No verified or replay data for the next round')
            path=folder/'next.raw.jsonl'
            write_jsonl(path,selected)
            metrics={'round':round_index,'repair/attempted':attempted,'repair/accepted':sum(row['resolved'] and not row['environment_error'] for row in repaired),
                     'repair/autonomous_successes':sum(row['resolved'] and not row['environment_error'] for row in rows)}
            self.tracker.log(metrics)
            return {'raw_data':str(path),'files':[str(path),str(folder/'repaired.jsonl')],'metrics':metrics}
        return self.state.run(f'r{round_index:03d}/repair',execute)

    def run(self):
        if not self.config['inference']['managed']:
            raise ValueError('Automatic pipeline requires managed vLLM checkpoint switching')
        preflight(self.config)
        train=load_tasks(self.config.path(self.config['data']['train_tasks']),'train')
        dev=load_tasks(self.config.path(self.config['data']['dev_tasks']),'dev')
        check_disjoint(train,dev)
        dev_ids={task.task_id for task in dev}
        source=self.config['data']['train_file'] or self.config['data']['bootstrap_trajectories']
        if any(row.get('task_id') in dev_ids for row in read_jsonl(self.config.path(source))):
            raise ValueError('Development tasks found in bootstrap training data')
        data_file=self.initial_data()['train_file']
        checkpoint=self.config['model']['name_or_path']
        previous_rate=None
        for round_index in range(1,self.config['experiment']['rounds']+1):
            trained=self.training(round_index,data_file,checkpoint)
            checkpoint=trained['checkpoint']
            rollout=self.collect(round_index,checkpoint,train,'rollout')
            evaluation=self.collect(round_index,checkpoint,dev,'evaluate')
            rate=evaluation['metrics']['evaluate/pass_rate']
            if previous_rate is not None:
                self.tracker.log({'round':round_index,'evaluate/gain_over_previous_round':rate-previous_rate})
            previous_rate=rate
            if round_index<self.config['experiment']['rounds']:
                reconstructed=self.reconstruct(round_index,checkpoint,train,rollout['trajectories'])
                next_file=self.folder(round_index,'prepare')/'next.tokenized.jsonl'
                prepared=self.state.run(f'r{round_index:03d}/prepare-next',
                    lambda:prepare(self.config,reconstructed['raw_data'],next_file))
                data_file=prepared['train_file']
        return {'final_checkpoint':checkpoint}
