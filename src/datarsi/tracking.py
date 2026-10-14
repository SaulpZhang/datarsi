from pathlib import Path
import json
import time


class Tracker:
    def __init__(self,config,name,group=None):
        import wandb
        self.directory=config.run_dir
        self.directory.mkdir(parents=True,exist_ok=True)
        options=config['tracking']
        self.run=wandb.init(project=options['project'],entity=options['entity'],
            mode=options['mode'],name=name,id=name,group=group,tags=options['tags'],
            config=config.values,dir=str(self.directory),
            resume='allow' if options['mode']=='online' else None)
        self.log_manifests=options['log_manifests']

    def log(self,metrics):
        self.run.log(metrics)
        with (self.directory/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'time':time.time(),'run':self.run.id,**metrics})+'\n')

    def manifest(self,path,name):
        if self.log_manifests:
            import wandb
            artifact=wandb.Artifact(name,type='manifest')
            artifact.add_file(str(path))
            self.run.log_artifact(artifact)

    def finish(self,failed=False):
        self.run.finish(exit_code=1 if failed else 0)
