"""Convert verified Polytopia movies (schema polytopia-replay-v1, written by the game's replay exporters) into the
compact per-action JSON the blog's replay viewer plays (the same format as thekevinz.com's Polytopia post).

Usage: python3 tools/convert_polytopia_replays.py MOVIE.json [MOVIE.json ...]
Writes public/polytopia/<id>.json and src/posts/autoresearch/polytopia-replays.json. Inputs are not modified. Only
the god-view game states, action labels, players, outcome and demo seed are kept (no hashes of private files).
"""
import json, sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parent
RULES = json.loads((ROOT / "simulators/polytopia/src/polytopia_bot/gpu/rules.json").read_text())
UNITS = {u["id"]: u["name"] for u in RULES["units"] if u["id"] > 0}
TECHS = {t["id"]: t["name"] for t in RULES["technologies"]}
STRUCTS = {s["id"]: s["name"] for s in RULES["structures"]}
split = lambda name: "".join(" " + c if c.isupper() and i else c for i, c in enumerate(name))

# RL action families (engine catalog order) -> the viewer's command kinds: 1 move, 2 attack, 3 unit or tile ability,
# 4 train, 5 harvest, 6 build, 7 research, 8 capture, 9 city reward, 10 end turn.
FAMILIES = ["Move", "Attack", "Train unit", "Build", "Research", "Harvest", "Forest action", "City reward", "Capture",
            "Recover", "Promote", "Disband", "Heal", "Destroy", "Diplomacy", "End turn", "Resign"]
KIND = {0: 1, 1: 2, 2: 4, 3: 6, 4: 7, 5: 5, 6: 6, 7: 9, 8: 8, 15: 10, 16: 10}


def label(family, target):
    if family == 2:
        return "Train " + split(UNITS.get(target, "unit"))
    if family == 3:
        return "Build " + split(STRUCTS.get(target, "structure"))
    if family == 4:
        return "Research " + split(TECHS.get(target, "a tech"))
    return FAMILIES[family] if family < len(FAMILIES) else "Action"


def convert(movie):
    assert movie["schema"] == "polytopia-replay-v1", movie.get("schema")
    n = movie["size"]
    tile = lambda t: [t[0], t[1], t[3], t[4], t[5], t[2]]  # v1 [terrain, owner, capital, structure, resource, road]
    init = movie["initial"]
    tiles = [tile(t) for t in init["tiles"]]
    units = {u[0]: [u[1], u[2], round(u[3], 1), u[4]] for u in init["units"]}
    cities = {c[0]: [c[1], c[2]] for c in init["cities"]}
    out_init = {"tiles": [list(t) for t in tiles], "units": [[k, *v] for k, v in units.items()],
                "cities": [[k, *v] for k, v in cities.items()]}
    labels, index, actions = [], {}, []
    for f in movie["frames"]:
        player, family, source, target = f["event"]
        name = label(family, target)
        if name not in index:
            index[name] = len(labels)
            labels.append(name)
        if family in (0, 1):
            src, dst = source, target
        elif family in (2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13):
            src, dst = (source if 0 <= source < n * n else -1), -1
        else:
            src, dst = -1, -1
        u2 = {u[0]: [u[1], u[2], round(u[3], 1), u[4]] for u in f["units"]}
        c2 = {c[0]: [c[1], c[2]] for c in f["cities"]}
        unit_diff = [[t] for t in units if t not in u2] + [[t, *u] for t, u in u2.items() if units.get(t) != u]
        tile_diff = []
        for t in f["tiles"]:
            new = tile(t[1:])
            if tiles[t[0]] != new:
                tiles[t[0]] = new
                tile_diff.append([t[0], *new])
        city_diff = [[t, *c] for t, c in c2.items() if cities.get(t) != c]
        actions.append([player, f["turn"], KIND.get(family, 3), src, dst, index[name], unit_diff, tile_diff, city_diff,
                        f["scores"], None])
        units, cities = u2, c2
    o = movie["outcome"]
    return {
        "title": movie["title"],
        "size": n,
        "names": {"1": movie["players"][0], "2": movie["players"][1]},
        "units": {str(k): split(v) for k, v in UNITS.items()},
        "labels": labels,
        "init": out_init,
        "actions": actions,
        "outcome": {"winner": o["winner"], "reason": o["reason"], "scores": o["scores"]},
        "seed": movie.get("provenance", {}).get("demo_seed"),
    }


def main():
    out_dir = SITE / "public/polytopia"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for path in sys.argv[1:]:
        movie = json.loads(Path(path).read_text())
        data = convert(movie)
        (out_dir / f"{movie['id']}.json").write_text(json.dumps(data, separators=(",", ":")))
        index.append({"id": movie["id"], "title": movie["title"], "players": movie["players"],
                      "winner": data["outcome"]["winner"], "reason": data["outcome"]["reason"],
                      "scores": data["outcome"]["scores"], "rounds": movie["outcome"].get("rounds", movie["frames"][-1]["turn"]),
                      "actions": len(movie["frames"])})
        print(f"{movie['id']}: {len(data['actions'])} actions, labels {len(data['labels'])}")
    (SITE / "src/posts/autoresearch/polytopia-replays.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
