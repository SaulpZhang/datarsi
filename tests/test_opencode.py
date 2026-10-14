from copy import deepcopy
import json
from datarsi.configuration import Config,DEFAULTS
from datarsi.opencode import response_message,configuration


def test_streamed_tool_calls_are_reassembled():
    chunks=[{'choices':[{'delta':{'tool_calls':[{'index':0,'id':'c1','function':{'name':'bash','arguments':'{"command":'}}]}}]},
        {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'arguments':'"ls"}'}}]}}]}]
    raw=('\n\n'.join('data: '+json.dumps(chunk) for chunk in chunks)+'\n\ndata: [DONE]\n\n').encode()
    message=response_message(raw)
    assert message['tool_calls'][0]['function']=={'name':'bash','arguments':'{"command":"ls"}'}


def test_opencode_config_uses_only_selected_vllm():
    cfg=Config(deepcopy(DEFAULTS))
    settings=configuration(cfg,'round-model','http://127.0.0.1:1234/v1',12)
    assert settings['enabled_providers']==['datarsi']
    assert settings['agent']['datarsi']['steps']==12
    assert settings['permission']['*']=='deny'
    assert settings['agent']['title']['disable']
    assert settings['share']=='disabled'
