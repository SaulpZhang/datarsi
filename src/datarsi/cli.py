import argparse
import json
import os
from .configuration import load_config
from .pipeline import Pipeline,plan
from .storage import State,run_lock


def main():
    parser=argparse.ArgumentParser(description='Data-centric experience learning')
    parser.add_argument('--config',required=True,help='YAML path or filename under config/')
    parser.add_argument('--set',action='append',default=[],metavar='SECTION.KEY=VALUE')
    parser.add_argument('--stage',choices=['pipeline','prepare','train','rollout','repair','evaluate'],default='pipeline')
    parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Print the plan without creating jobs or downloading models')
    args=parser.parse_args()
    if args.round<1:
        parser.error('--round must be positive')
    config=load_config(args.config,args.set)
    if args.dry_run:
        print(json.dumps(plan(config,args.stage,args.round),indent=2,ensure_ascii=False))
        return
    worker=os.environ.get('DATARSI_JOB_SPEC')
    if worker:
        if args.stage!='train':
            raise ValueError('Worker mode is only for training')
        from .training import train
        train(config,json.loads(open(worker).read()))
        return
    from .tracking import Tracker
    with run_lock(config.run_dir):
        state=State(config.run_dir,config.fingerprint,args.resume)
        config.save(config.run_dir/'resolved-config.yaml')
        tracker=Tracker(config,config['experiment']['name'])
        failed=True
        try:
            pipeline=Pipeline(config,state,tracker)
            if args.stage=='pipeline':
                output=pipeline.run()
            elif args.stage=='prepare':
                output=pipeline.initial_data()
            elif args.stage=='train':
                data=pipeline.initial_data()['train_file']
                output=pipeline.training(args.round,data,config['model']['name_or_path'])
            else:
                from .tasks import load_tasks
                trained=state.value['stages'].get(f'r{args.round:03d}/train',{}).get('outputs',{})
                checkpoint=trained.get('checkpoint',config['model']['name_or_path'])
                split='dev' if args.stage=='evaluate' else 'train'
                tasks=load_tasks(config.path(config['data']['dev_tasks' if split=='dev' else 'train_tasks']),split)
                if args.stage=='repair':
                    rollout=pipeline.folder(args.round,'rollout')/'trajectories.jsonl'
                    output=pipeline.reconstruct(args.round,checkpoint,tasks,rollout)
                else:
                    output=pipeline.collect(args.round,checkpoint,tasks,args.stage)
            tracker.manifest(config.run_dir/'state.json',config['experiment']['name']+'-state')
            print(json.dumps(output,indent=2))
            failed=False
        finally:
            tracker.finish(failed)
