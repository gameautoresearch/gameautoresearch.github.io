"""Publish the recorded Pokémon 3v3 battles for the post's replay viewer.

Input (unchanged): replay-site/public/replays/pokemon/showdown-20261115/*.json, fresh public-seed viewing games between
the November 14–15 final networks, each with a complete Showdown protocol log that was re-verified by replaying the
recorded joint actions (see replay-site/README.md). Output: public/pokemon/<id>.json (the battle's players, winner,
turn count and log), played in the page by <BattleReplay> (src/showdown/, Pokémon Showdown's battle engine), and
src/posts/autoresearch/pokemon-replays.json listing the battles. The full-size view is /replays/pokemon/<id>.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "replay-site/public/replays/pokemon/showdown-20261115"
SITE = Path(__file__).resolve().parents[1]
OUT = SITE / "public/pokemon"
INDEX = SITE / "src/posts/autoresearch/pokemon-replays.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for meta in json.loads((SRC / "index.json").read_text()):
        log = json.loads((SRC / f"{meta['id']}.json").read_text())["log"]
        assert "|win|" in log, meta["id"]
        winner = meta["players"][0 if meta["winner"] == "p1" else 1]
        entry = {"id": meta["id"], "title": meta["title"], "players": meta["players"], "winner": winner, "turns": meta["turns"]}
        (OUT / f"{meta['id']}.json").write_text(json.dumps({**entry, "log": log}, ensure_ascii=False) + "\n")
        entries.append(entry)
    INDEX.write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(entries)} battles -> {OUT}")


if __name__ == "__main__":
    main()
