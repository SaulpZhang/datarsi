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


@pytest.mark.skipif(os.environ.get('DATARSI_TOKENIZER_TESTS')!='1',reason='Opt-in real Qwen tokenizer check')
def test_real_qwen_opencode_tool_supervision():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B',use_fast=True)
    tools=[{'type':'function','function':{'name':'bash','description':'Execute shell command',
        'parameters':{'type':'object','properties':{'command':{'type':'string'}}}}}]
    messages=[{'role':'system','content':'You are a coding agent.'},
        {'role':'user','content':[{'type':'text','text':'USER_SENTINEL'}]},
        {'role':'assistant','content':None,'tool_calls':[{'id':'old','type':'function',
            'function':{'name':'bash','arguments':'{"command":"FAILED_SENTINEL"}'}}],'supervise':False},
        {'role':'tool','tool_call_id':'old','content':'ENV_SENTINEL'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'new','type':'function',
            'function':{'name':'bash','arguments':'{"command":"git status"}'}}]}]
    row=encode_messages(tokenizer,messages,8192,tools=tools)
    targets=tokenizer.decode([i for i in row['labels'] if i!=-100])
    assert 'bash' in targets and 'git status' in targets and 'tool_call' in targets
    assert not any(value in targets for value in ('USER_SENTINEL','FAILED_SENTINEL','ENV_SENTINEL'))
