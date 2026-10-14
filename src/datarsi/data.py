from copy import deepcopy
import re
import uuid
from .storage import read_jsonl,write_jsonl


def encode_messages(tokenizer,messages,max_length,mask_thinking=True):
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer with offsets is required')
    def render(value):
        return tokenizer.apply_chat_template(value,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    full=render(messages)
    spans=[]
    def locate(probe,marker,optional=False):
        rendered=render(probe)
        if optional and marker not in rendered:
            return
        if rendered.count(marker)!=1:
            raise ValueError('Template does not preserve assistant content')
        prefix,suffix=rendered.split(marker)
        if not full.startswith(prefix) or (suffix and not full.endswith(suffix)):
            raise ValueError('Template changed context; cannot construct a reliable loss mask')
        start,end=len(prefix),len(full)-len(suffix)
        if end>start:
            spans.append((start,end))
    for index,message in enumerate(messages):
        if message['role']!='assistant' or message.get('supervise') is False:
            continue
        content=message.get('content')
        if not isinstance(content,str):
            raise ValueError('Training assistant content must be a string')
        if content.count('<think>')!=content.count('</think>'):
            raise ValueError('Unbalanced thinking tags')
        thinking=list(re.finditer(r'<think>(.*?)</think>',content,re.DOTALL))
        stripped=re.sub(r'<think>.*?</think>','',content,flags=re.DOTALL)
        if '<think>' in stripped or '</think>' in stripped:
            raise ValueError('Nested or invalid thinking tags')
        if len(thinking)>1:
            raise ValueError('Multiple thinking blocks in one message are unsupported')
        marker='DATARSI_'+uuid.uuid4().hex
        probe=deepcopy(messages)
        # Preserve reasoning-dependent branches of the original chat template.
        # Qwen rewrites the thinking wrapper; replacing the whole content changes its prefix.
        if thinking:
            probe[index]['content']=content[:thinking[0].end()]+marker
        else:
            probe[index]['content']=marker
        locate(probe,marker)
        if not mask_thinking:
            probe=deepcopy(messages)
            if thinking:
                match=thinking[0]
                probe[index]['content']=content[:match.start(1)]+marker+content[match.end(1):]
                locate(probe,marker,optional=True)
            elif message.get('reasoning_content'):
                probe[index]['reasoning_content']=marker
                locate(probe,marker,optional=True)
    encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded['input_ids']
    if len(ids)>max_length:
        raise ValueError(f'Trajectory has {len(ids)} tokens, exceeding {max_length}')
    labels=[token if end>start and any(a<=start and end<=b for a,b in spans) else -100
            for token,(start,end) in zip(ids,encoded['offset_mapping'])]
    if not any(label!=-100 for label in labels[1:]):
        raise ValueError('No supervised next-token targets')
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels}


def validate_record(row,max_length):
    fields=('input_ids','attention_mask','labels')
    if any(not isinstance(row.get(key),list) for key in fields):
        raise ValueError('Tokenized records need input_ids, attention_mask and labels')
    ids,mask,labels=(row[key] for key in fields)
    if not 2<=len(ids)<=max_length or len(ids)!=len(mask) or len(ids)!=len(labels):
        raise ValueError('Invalid tokenized record lengths')
    if any(type(i) is not int or i<0 for i in ids) or any(type(i) is not int or i not in (0,1) for i in mask):
        raise ValueError('Invalid token IDs/attention mask')
    if any(type(label) is not int or (label!=-100 and (label!=token or not visible)) for token,label,visible in zip(ids,labels,mask)):
        raise ValueError('Labels must match input tokens or -100')
    if not any(i!=-100 and m for i,m in zip(labels[1:],mask[1:])):
        raise ValueError('No supervised next-token targets')
    return {key:row[key] for key in fields}


class MaskPreservingCollator:
    def __init__(self,pad_token_id):
        self.pad_token_id=pad_token_id

    def __call__(self,rows):
        import torch
        length=max(len(row['input_ids']) for row in rows)
        pads={'input_ids':self.pad_token_id,'attention_mask':0,'labels':-100}
        return {key:torch.tensor([row[key]+[value]*(length-len(row[key])) for row in rows],dtype=torch.long) for key,value in pads.items()}


def prepare(config,source,target,tokenizer=None):
    records=[]
    for row in read_jsonl(source):
        if row.get('split','train')!='train':
            raise ValueError('Only train split trajectories may enter SFT')
        if row.get('resolved') is False or row.get('environment_error'):
            continue
        if 'input_ids' in row:
            records.append(validate_record(row,config['data']['max_length']))
        else:
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer=AutoTokenizer.from_pretrained(config['model']['name_or_path'],use_fast=True)
            records.append(encode_messages(tokenizer,row['messages'],config['data']['max_length'],config['data']['mask_thinking']))
        if len(records)>=config['data']['max_train_examples']:
            break
    if not records:
        raise ValueError('No valid training examples')
    write_jsonl(target,records)
    return {'files':[str(target)],'train_file':str(target),'examples':len(records)}
