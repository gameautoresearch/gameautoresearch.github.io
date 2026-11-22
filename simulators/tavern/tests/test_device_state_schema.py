import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('device_state_schema',Path(__file__).resolve().parents[1]/'device_state_schema.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_projection_preserves_every_game_and_rng_field_and_checks_metadata():
    game={'rng':{'a':2**64-1},'players':[{'health':37}],'arbitrary_rule_field':[4,9]}
    value={'game':game,'index':3,'finished':[False,True]+[False]*6,
           'seed':12345,'controlled':[3,0]+[0]*6,'k':2}
    projected=module.native_episode(value,seed=12345,controlled=(3,0))
    assert projected=={'game':game,'index':3,'finished':[False,True]}
    assert projected['game'] is game
    assert set(value)=={'game','index','finished','seed','controlled','k'}
    for patch in [{'seed':1},{'controlled':[0]*8},{'k':1},{'unexpected':True},
                  {'finished':[False,True,True]+[False]*5}]:
        with pytest.raises(ValueError):
            module.native_episode(dict(value,**patch),seed=12345,controlled=(3,0))
