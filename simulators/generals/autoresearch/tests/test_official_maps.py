"""Fail-closed map admission, including current TypedArray JSON serialization."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from integrations.generals_autoresearch import official_maps as maps


def fixture(monkeypatch, edit=None):
    replay = dict(version=19, id="example", usernames=["red", "blue"], teams=None,
                  mapWidth=17, mapHeight=18, mountains=[1, 2], cities=[20],
                  cityArmies=[47], generals=[0, 305], player_transforms={"0": 0, "1": 0})
    raw = [None] * 39
    if edit:
        edit(replay, raw)
    monkeypatch.setattr(maps, "decoder", lambda: SimpleNamespace(
        deserialize=lambda data: deepcopy(replay),
        decompress_from_uint8array=lambda data: json.dumps(raw),
    ))
    return replay


def test_exact_terrain_and_current_transform_dictionary(monkeypatch):
    original = fixture(monkeypatch)
    result, replay = maps.decode_map(b"fixture", "example")
    assert (result["width"], result["height"]) == (17, 18)
    assert result["grid"][0] == 1 and result["grid"][305] == 2
    assert result["grid"][1:3] == [-2, -2] and result["grid"][20] == 47
    assert result["grid"].count(0) == 301
    assert replay == original


@pytest.mark.parametrize("edit,reason", [
    (lambda r, a: r.update(id="wrong"), "Replay ID"),
    (lambda r, a: r.update(version=20), "v19"),
    (lambda r, a: a.append(None), "39-field"),
    (lambda r, a: r.update(mountains=[0]), "Overlapping"),
    (lambda r, a: r.update(mountains=[306]), "off board"),
    (lambda r, a: r.update(mapWidth=24), "shape support"),
    (lambda r, a: r.update(cityArmies=[2]), "City armies"),
    (lambda r, a: r.update(cityArmies=[]), "City armies"),
    (lambda r, a: r.update(tunnels=[10]), "non-base-game"),
    (lambda r, a: a.__setitem__(37, {"ChessClock": {}}), "tail"),
    (lambda r, a: r.update(player_transforms={"0": 0, "1": 1}), "player transforms"),
    (lambda r, a: r.update(player_transforms={"0": 0}), "transform dictionary"),
])
def test_rejects_unrepresentable_or_unsupported_maps(monkeypatch, edit, reason):
    fixture(monkeypatch, edit)
    with pytest.raises(ValueError, match=reason):
        maps.decode_map(b"fixture", "example")
