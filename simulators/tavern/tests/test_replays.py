"""Metadata projection must use CardId, not the observation embedding offset."""
import importlib.util
import json
from pathlib import Path
import taverngym as tg

p=Path(__file__).resolve().parents[1]/'export_demo_replays.py'
spec=importlib.util.spec_from_file_location('tavern_replay_export',p)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_replay_card_names_resolve_cardid_and_instance_bits():
    assert module.name(0)==tg.card_info(1)[1]
    assert module.name(0x1000)==module.name(0)
    game=tg.Game(2026110821,json.dumps(module.RULES))
    for seat in range(8):
        while game.needs_action(seat) and json.loads(game.state_json())['players'][seat]['hero'] is None:
            game.step(seat,game.bot_action(seat))
    state=json.loads(game.state_json())
    projected=module.players(state)
    for seat in range(8):
        actual=json.loads(game.view(seat))['me']
        assert projected[seat]['hero']==actual['hero']['name']
