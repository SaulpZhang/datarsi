import os
import pytest
from datarsi.data import encode_messages


@pytest.mark.skipif(os.environ.get('DATARSI_TOKENIZER_TESTS')!='1',reason='Opt-in real Qwen tokenizer check')
def test_real_qwen_template_loss_mask():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B',use_fast=True)
    messages=[{'role':'system','content':'You are a coding agent.'},
        {'role':'user','content':'Inspect the repository.'},
        {'role':'assistant','content':'{"command":"ls"}','supervise':False},
        {'role':'user','content':'app.py'},
        {'role':'assistant','content':'<think>PRIVATE_REASONING_SENTINEL</think>{"command":"git status"}'}]
    row=encode_messages(tokenizer,messages,8192)
    targets=tokenizer.decode([i for i in row['labels'] if i!=-100])
    assert 'git status' in targets
    assert 'PRIVATE_REASONING_SENTINEL' not in targets
    assert 'app.py' not in targets
    assert '"ls"' not in targets
