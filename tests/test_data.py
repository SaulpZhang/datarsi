import pytest
from datarsi.data import encode_messages,MaskPreservingCollator,validate_record,prepare
from datarsi.configuration import Config,DEFAULTS
from datarsi.storage import write_jsonl
from copy import deepcopy


@pytest.fixture
def tokenizer():
    from tokenizers import Tokenizer,models,pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    backend=Tokenizer(models.WordLevel({'<unk>':0,'answer':1,'secret':2,'tool':3,'user':4,'assistant':5},unk_token='<unk>'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='<unk>')
    tok.chat_template="{% for m in messages %}{{m['role']}}: {{m['content']}}\n{% endfor %}"
    return tok


def test_mask_excludes_environment_and_think(tokenizer):
    messages=[{'role':'user','content':'answer secret'},{'role':'assistant','content':'<think>secret</think> answer'},
              {'role':'user','content':'tool secret'},{'role':'assistant','content':'answer'}]
    row=encode_messages(tokenizer,messages,100)
    targets=[i for i in row['labels'] if i!=-100]
    assert targets==[1,1]


def test_unbalanced_thinking_and_length_rejected(tokenizer):
    with pytest.raises(ValueError,match='Unbalanced'):
        encode_messages(tokenizer,[{'role':'assistant','content':'<think>secret answer'}],100)
    with pytest.raises(ValueError,match='exceeding'):
        encode_messages(tokenizer,[{'role':'assistant','content':'answer answer answer'}],1)


def test_collator_preserves_existing_masks():
    rows=[{'input_ids':[1,2,3],'attention_mask':[1,1,1],'labels':[-100,-100,3]},
          {'input_ids':[1,2],'attention_mask':[1,1],'labels':[-100,2]}]
    batch=MaskPreservingCollator(0)(rows)
    assert batch['labels'].tolist()==[[-100,-100,3],[-100,2,-100]]
    assert batch['attention_mask'].tolist()==[[1,1,1],[1,1,0]]


def test_invalid_labels_and_no_targets():
    with pytest.raises(ValueError,match='Labels'):
        validate_record({'input_ids':[1,2],'attention_mask':[1,1],'labels':[-100,3]},10)
    with pytest.raises(ValueError,match='No supervised'):
        validate_record({'input_ids':[1,2],'attention_mask':[1,1],'labels':[1,-100]},10)


def test_prepare_rejects_dev_data(tmp_path):
    source=tmp_path/'raw.jsonl'
    write_jsonl(source,[{'split':'dev','input_ids':[1,2],'labels':[-100,2],'attention_mask':[1,1]}])
    cfg=Config(deepcopy(DEFAULTS),tmp_path)
    with pytest.raises(ValueError,match='Only train'):
        prepare(cfg,source,tmp_path/'out.jsonl')


def test_nested_thinking_rejected(tokenizer):
    with pytest.raises(ValueError,match='Nested'):
        encode_messages(tokenizer,[{'role':'assistant','content':'<think><think>secret</think></think> answer'}],100)


def test_failed_prefix_is_context_only(tokenizer):
    row=encode_messages(tokenizer,[{'role':'assistant','content':'secret','supervise':False},
        {'role':'user','content':'tool'},{'role':'assistant','content':'answer'}],100)
    assert [i for i in row['labels'] if i!=-100]==[1]


def test_thinking_supervision_can_be_enabled(tokenizer):
    row=encode_messages(tokenizer,[{'role':'assistant','content':'<think>secret</think> answer'}],100,mask_thinking=False)
    targets=[i for i in row['labels'] if i!=-100]
    assert 2 in targets and 1 in targets
