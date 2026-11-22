"""Bounded, local-only pilot importing official ladder maps, without changing RL pins.

The download route is the public route used by the v31.4.6 website. The pinned
upstream decoder supplies decompression; v19 tail fields are checked explicitly.
No login, model calls, cloud workers, or alternative map generation are involved.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[2] if _HERE.parent.name == 'generals' else _HERE.parents[1]
SITE = "https://generals.io"
LIST_URL = SITE + "/api/replays?count={count}&offset=0&l=duel"
DECODER = ROOT / "candidates/generals-bots/paper/validation/gior.py"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def fetch(url: str, path: Path) -> dict:
    """At most two attempts, bounded bytes, no authentication or cookie handling."""
    attempts = []
    for attempt in range(2):
        tmp = path.with_name(path.name + f".attempt{attempt}")
        result = subprocess.run(
            ["curl", "--silent", "--show-error", "--location", "--fail",
             "--connect-timeout", "8", "--max-time", "20", "--max-filesize", "5242880",
             "--proto", "=https", "--proto-redir", "=https",
             "-A", "GameAutoresearchBench/0.1 (bounded public replay research)",
             "-o", str(tmp), "-w", "%{http_code}", url],
            capture_output=True, text=True, timeout=25,
        )
        attempts.append(dict(exit_code=result.returncode, http_status=result.stdout,
                             error=result.stderr[:1000]))
        if result.returncode == 0:
            data = tmp.read_bytes()
            if len(data) > 5242880:
                raise ValueError("Response exceeds pilot limit")
            tmp.rename(path)
            return dict(url=url, file=path.name, bytes=len(data), sha256=sha(data), attempts=attempts)
        if attempt == 0:
            time.sleep(1)
    return dict(url=url, file=None, attempts=attempts, error="Both bounded attempts failed")


def decoder():
    spec = importlib.util.spec_from_file_location("official_pilot_gior", DECODER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_map(data: bytes, expected_id: str) -> tuple[dict, dict]:
    module = decoder()
    raw = json.loads(module.decompress_from_uint8array(data))
    replay = module.deserialize(data)
    if replay["version"] != 19 or len(raw) != 39:
        raise ValueError("Pilot admits only the inspected v19 / 39-field layout")
    if replay["id"] != expected_id:
        raise ValueError("Replay ID does not match requested game")
    if len(replay["usernames"]) != 2 or len(replay["generals"]) != 2 or replay["teams"]:
        raise ValueError("Not ordinary two-player 1v1")
    for key in ("map", "modifiers", "neutrals", "neutralArmies", "swamps", "lights",
                "observatories", "lookouts", "deserts", "tunnels", "tunnelLimits",
                "strongholds", "strongholdStrengths", "chessClockTimingsByMove"):
        if replay.get(key):
            raise ValueError(f"Unsupported non-base-game field: {key}")
    if any(raw[36:]):
        raise ValueError("Unsupported chess-clock / modifier-options tail")
    transforms = replay.get("player_transforms") or [0, 0]
    # TypedArray serialization on the current site is {"0": 0, "1": 0}.
    if isinstance(transforms, dict):
        if set(transforms) != {"0", "1"}:
            raise ValueError("Malformed player transform dictionary")
        transforms = [transforms["0"], transforms["1"]]
    if not isinstance(transforms, list) or len(transforms) != 2 or any(type(t) is not int or t != 0 for t in transforms):
        raise ValueError("Unsupported player transforms")
    width, height = replay["mapWidth"], replay["mapHeight"]
    if type(width) is not int or type(height) is not int or not (17 <= width <= 23 and 17 <= height <= 23):
        raise ValueError("Outside current 17..23 shape support")
    groups = [replay[k] for k in ("mountains", "cities", "generals")]
    tiles = [tile for group in groups for tile in group]
    if any(type(t) is not int or t < 0 or t >= width * height for t in tiles):
        raise ValueError("Tile off board or non-integer")
    if len(tiles) != len(set(tiles)):
        raise ValueError("Overlapping or duplicate terrain/spawn positions")
    armies = replay["cityArmies"]
    if len(armies) != len(replay["cities"]) or any(type(a) is not int or not 2 < a < 2**31 for a in armies):
        raise ValueError("City armies cannot be represented exactly")
    board = [0] * (width * height)
    for tile in replay["mountains"]:
        board[tile] = -2
    for tile, army in zip(replay["cities"], armies):
        board[tile] = army
    for player, tile in enumerate(replay["generals"]):
        board[tile] = player + 1
    # Measure official geometry; do not repair it to match generated-map constraints.
    start, goal = replay["generals"]
    distances = {start: 0}
    queue = deque([start])
    while queue:
        tile = queue.popleft()
        row, col = divmod(tile, width)
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            neighbor = nr * width + nc
            if 0 <= nr < height and 0 <= nc < width and board[neighbor] != -2 and neighbor not in distances:
                distances[neighbor] = distances[tile] + 1
                queue.append(neighbor)
    if goal not in distances:
        raise ValueError("Disconnected generals")
    result = dict(schema_version=1, replay_id=expected_id, replay_version=replay["version"],
                  width=width, height=height, grid=board, mountains=replay["mountains"],
                  cities=replay["cities"], city_armies=armies, generals=replay["generals"],
                  bfs_spawn_distance=distances[goal], mountains_count=len(replay["mountains"]),
                  map_identity_sha256=sha(json.dumps([width, height, board], separators=(",", ":")).encode()))
    return result, replay


def validate_native(maps: list[dict], output: Path) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from generals.core import game
    from integrations.generals.environment import action_mask, decode_action, encode, observations, transition
    from integrations.generals.runtime import sources

    binding = sources()  # Normal strict pinned-engine and source checks.
    boards = np.full((len(maps), 24, 24), -2, np.int32)
    for i, record in enumerate(maps):
        h, w = record["height"], record["width"]
        boards[i, :h, :w] = np.array(record["grid"], np.int32).reshape(h, w)
    states = jax.vmap(game.create_initial_state)(jnp.asarray(boards))
    for i, record in enumerate(maps):
        w = record["width"]
        expected = np.array([divmod(t, w) for t in record["generals"]])
        np.testing.assert_array_equal(np.asarray(states.general_positions[i]), expected)
        for tile, army in zip(record["cities"], record["city_armies"]):
            row, col = divmod(tile, w)
            assert int(states.armies[i, row, col]) == army
        np.testing.assert_array_equal(np.asarray(states.mountains[i]), boards[i] == -2)
        assert int(states.castles[i].sum()) == len(record["cities"])
        assert int(states.generals[i].sum()) == 2
        assert int(states.armies[i].sum()) == 2 + sum(record["city_armies"])
    initial = states

    @jax.jit
    def smoke(states):
        def tick(carry, key):
            state, active = carry
            obs = observations(state)
            mask = action_mask(obs)
            logits = jnp.where(mask, 0.0, -jnp.inf)
            actions = jax.random.categorical(key, logits).astype(jnp.int32)
            legal = jnp.take_along_axis(mask, actions[..., None], -1)[..., 0]
            boundary = transition(state, decode_action(actions))
            state = jax.tree.map(lambda a, b: jnp.where(active.reshape(active.shape + (1,) * (a.ndim - 1)), a, b),
                                 boundary.final_state, state)
            return (state, active & ~boundary.terminated & ~boundary.truncated), (legal.all(), jnp.isfinite(encode(obs)).all())
        return jax.lax.scan(tick, (states, jnp.ones(len(maps), bool)), jax.random.split(jax.random.PRNGKey(20261108), 64))

    (_, active), (legal, finite) = smoke(states)
    assert bool(legal.all()) and bool(finite.all())
    np.savez_compressed(output / "maps.npz", grids=boards)
    # Also retain a source-bound native bank that can be loaded offline.
    np.savez_compressed(output / "native-initial-states.npz", **{k: np.asarray(v) for k, v in initial._asdict().items()})
    return dict(backend=jax.default_backend(), engine_commit=binding["commit"],
                integration_binding=binding["implementation_sha256"], maps_loaded=len(maps),
                checked_rounds_per_map=64, all_actions_legal=True, all_visible_encodings_finite=True,
                exact_initial_terrain_city_armies_and_spawns=True, ongoing_after_smoke=int(active.sum()),
                full_official_replay_parity_tested=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--from-collection", type=Path,
                        help="Reprocess retained downloads offline into a NEW directory")
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        parser.error("Pilot count must be 1..20")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "raw").mkdir()
    (out / "maps").mkdir()
    report = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                  requested=args.count, scope="Small public current-ladder feasibility pilot; not a benchmark release",
                  decoder_sha256=sha(DECODER.read_bytes()), importer_sha256=sha(Path(__file__).read_bytes()),
                  downloads=[], maps=[], exclusions=[])
    if args.from_collection:
        data = (args.from_collection / "recent-list.json").read_bytes()
        (out / "recent-list.json").write_bytes(data)
        listing = dict(url=LIST_URL.format(count=args.count), sha256=sha(data), bytes=len(data),
                       reused_from=str(args.from_collection), network_request=False)
    else:
        listing = fetch(LIST_URL.format(count=args.count), out / "recent-list.json")
    report["listing"] = listing
    write_json(out / "report.json", report)
    if listing.get("error"):
        raise RuntimeError("Current ladder listing unavailable; see report")
    rows = json.loads((out / "recent-list.json").read_text())
    seen = set()
    for row in rows[:args.count]:
        rid = row.get("id", "")
        if row.get("type") != "1v1" or row.get("ladder_id") != "duel" or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", rid) or rid in seen:
            continue
        seen.add(rid)
        destination = out / "raw" / (rid + ".gior")
        if args.from_collection and (source := args.from_collection / "raw" / (rid + ".gior")).is_file():
            data = source.read_bytes()
            destination.write_bytes(data)
            receipt = dict(url=SITE + "/api/replay/" + rid, file=destination.name,
                           bytes=len(data), sha256=sha(data), reused_from=str(source), network_request=False)
        elif args.from_collection:
            receipt = dict(error="No retained replay file", url=SITE + "/api/replay/" + rid)
        else:
            receipt = fetch(SITE + "/api/replay/" + rid, destination)
        report["downloads"].append(receipt)
        if receipt.get("error"):
            report["exclusions"].append(dict(id=rid, reason=receipt["error"]))
        else:
            try:
                record, _ = decode_map((out / "raw" / (rid + ".gior")).read_bytes(), rid)
                record["started_unix_ms"] = row["started"]
                record["replay_sha256"] = receipt["sha256"]
                write_json(out / "maps" / (rid + ".json"), record)
                report["maps"].append(record)
            except (ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                report["exclusions"].append(dict(id=rid, reason=str(error)))
        write_json(out / "report.json", report)
        print(rid, "downloaded" if not receipt.get("error") else "failed", flush=True)
    maps = report["maps"]
    report["unique_maps"] = len({m["map_identity_sha256"] for m in maps})
    if maps:
        report["native_validation"] = validate_native(maps, out)
        report["dimensions_width_height"] = dict(Counter(f"{m['width']}x{m['height']}" for m in maps))
    report["artifact_sha256"] = {str(p.relative_to(out)): sha(p.read_bytes()) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "report.json"}
    write_json(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("maps", "downloads", "artifact_sha256")}, indent=2))
    if not maps:
        raise RuntimeError("No usable maps collected; see retained downloads and exclusions")


if __name__ == "__main__":
    main()
