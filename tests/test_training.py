from copy import deepcopy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datarsi.configuration import Config, DEFAULTS
from datarsi.storage import write_jsonl
from datarsi.training import train


def test_cpu_full_sft_checkpoint_reload(tmp_path):
    values=deepcopy(DEFAULTS)
    values['model'].update(source='tiny',kind='causal',name_or_path='tiny')
    values['tracking']['mode']='disabled'
    values['experiment']['name']='checkpoint-test'
    values['training'].update(num_processes=1,allow_cpu=True)
    values['training']['trainer'].update(bf16=False,gradient_checkpointing=False,
        gradient_accumulation_steps=1,max_steps=2,save_strategy='no',logging_steps=1)
    data=tmp_path/'data.jsonl'
    write_jsonl(data,[{'input_ids':[2,3,4,5,1],'attention_mask':[1]*5,'labels':[-100,-100,4,5,1]}]*2)
    config=Config(values,tmp_path)
    output=train(config,{'round':1,'train_file':str(data),'checkpoint':'tiny','output_dir':str(tmp_path/'output')})
    model=AutoModelForCausalLM.from_pretrained(output['checkpoint'])
    tokenizer=AutoTokenizer.from_pretrained(output['checkpoint'])
    assert tokenizer.pad_token_id==0
    assert model.config.vocab_size==8
    model.eval()
    with torch.no_grad():
        logits=model(input_ids=torch.tensor([[2,3]])).logits
    assert logits.shape==(1,2,8)
    assert torch.isfinite(logits).all()
