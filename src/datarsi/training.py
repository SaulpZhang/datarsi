from pathlib import Path
import json
import os
from .data import MaskPreservingCollator,validate_record
from .storage import atomic_json


def tiny_model():
    from tokenizers import Tokenizer,models,pre_tokenizers
    from transformers import GPT2Config,GPT2LMHeadModel,PreTrainedTokenizerFast
    vocabulary={'<pad>':0,'<eos>':1,'a':2,'b':3,'c':4,'d':5,'e':6,'<unk>':7}
    backend=Tokenizer(models.WordLevel(vocabulary,unk_token='<unk>'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',eos_token='<eos>',unk_token='<unk>')
    model=GPT2LMHeadModel(GPT2Config(vocab_size=8,n_positions=64,n_embd=16,n_layer=1,n_head=2,pad_token_id=0,eos_token_id=1))
    return model,tokenizer


def train(config,job):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM,AutoModelForImageTextToText,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback,set_seed
    from .tracking import Tracker
    allow_cpu=config['training']['allow_cpu']
    if allow_cpu and config['model']['source']!='tiny':
        raise ValueError('CPU mode is reserved for the explicit tiny-model smoke test')
    if not allow_cpu and not torch.cuda.is_available():
        raise RuntimeError('No assigned CUDA device is visible')
    world_size=int(os.environ.get('WORLD_SIZE','1'))
    rank=int(os.environ.get('RANK','0'))
    options=dict(config['training']['trainer'])
    if world_size>1 and not options.get('fsdp') and not options.get('deepspeed'):
        raise ValueError('Full-parameter distributed SFT requires explicit state sharding')
    set_seed(config['experiment']['seed'])
    if allow_cpu:
        torch.set_num_threads(2)
    dataset=load_dataset('json',data_files=job['train_file'],split='train')
    if not len(dataset):
        raise ValueError('Empty training dataset')
    for row in dataset:
        validate_record(row,config['data']['max_length'])
    dataset=dataset.select_columns(['input_ids','attention_mask','labels'])
    if config['model']['source']=='tiny':
        model,tokenizer=tiny_model()
    else:
        factory=AutoModelForCausalLM if config['model']['kind']=='causal' else AutoModelForImageTextToText
        dtype=torch.bfloat16 if options.get('bf16') else torch.float32
        model=factory.from_pretrained(job['checkpoint'],dtype=dtype)
        tokenizer=AutoTokenizer.from_pretrained(job['checkpoint'],use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer requires a pad or EOS token')
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    model.config.get_text_config().use_cache=False
    output=Path(job['output_dir'])
    output.mkdir(parents=True,exist_ok=True)
    tracker=Tracker(config,f"{config['experiment']['name']}-r{job['round']:03d}-train",config['experiment']['name']) if rank==0 else None

    class LoggingCallback(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if tracker and logs:
                numeric={f'train/{k}':v for k,v in logs.items() if isinstance(v,(int,float))}
                tracker.log({'train/step':state.global_step,'round':job['round'],**numeric})

    failed=True
    try:
        args=TrainingArguments(output_dir=str(output),report_to='none',use_cpu=allow_cpu,seed=config['experiment']['seed'],**options)
        trainer=Trainer(model=model,args=args,train_dataset=dataset,processing_class=tokenizer,
            data_collator=MaskPreservingCollator(tokenizer.pad_token_id),callbacks=[LoggingCallback()])
        result=trainer.train(resume_from_checkpoint=job.get('resume_checkpoint'))
        if not torch.isfinite(torch.tensor(result.training_loss)):
            raise RuntimeError('Training loss is not finite')
        final=output/'final'
        trainer.save_model(str(final))
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(final)
            metrics={**result.metrics,'round':job['round'],'checkpoint':str(final),'examples':len(dataset)}
            atomic_json(output/'training-metrics.json',metrics)
        failed=False
    finally:
        if tracker:
            tracker.finish(failed)
    return {'files':[str(output/'final'),str(output/'training-metrics.json')],
            'checkpoint':str(output/'final'),'metrics':str(output/'training-metrics.json')}
