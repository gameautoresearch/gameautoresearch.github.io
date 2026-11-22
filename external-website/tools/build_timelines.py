"""Derive the Pokémon 3v3 session timelines shown in the post (src/posts/autoresearch/timelines.json).

Inputs (unchanged): reports/pokemon/poke3v3-compute-vs-intelligence-2026-11-15/research_runs.csv (every training run of
the 21 counted attempts: start/end hours, private lift, practice score), reports/pokemon/poke3v3-research-behavior-
2026-11-15/behavior.json (per-attempt summary), and each session's requests.jsonl (every `ar` call) plus its clock.
Time origin is the session start (deadline − wall seconds), the same as behavior.py. Only timings, lifts and
practice scores leave this script: no job ids, seeds, paths or recipe contents.
"""
import csv, json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "reports/pokemon/poke3v3-compute-vs-intelligence-2026-11-15/research_runs.csv"
BEHAVIOR = ROOT / "reports/pokemon/poke3v3-research-behavior-2026-11-15/behavior.json"
BASES = [ROOT / "runs/local/poke3v3-gpt6-20261115", ROOT / "runs/cloud/poke3v3-batch-20261114"]
OUT = Path(__file__).resolve().parents[1] / "src/posts/autoresearch/timelines.json"


def session_dir(model, att):
    for b in BASES:
        p = b / "pokemon" / model / att
        if (p / "requests.jsonl").exists():
            return p
    raise FileNotFoundError(model, att)


def main():
    behavior = {(b["model"], b["attempt"]): b for b in json.loads(BEHAVIOR.read_text())}
    runs = defaultdict(list)
    for r in csv.DictReader(open(RUNS)):
        runs[(r["model"], r["attempt"])].append(r)
    out = []
    for (model, att), b in behavior.items():
        sd = session_dir(model, att)
        state = json.loads((sd / "session.json").read_text())
        result = json.loads((sd / "session-result.json").read_text())
        start = state["deadline"] - result["wall_seconds"]
        h = lambda t: round((t - start) / 3600, 4)
        reqs = [json.loads(l) for l in open(sd / "requests.jsonl")]
        ok = [q for q in reqs if not q["error"]]
        times = lambda op: [h(q["time"]) for q in ok if q["request"].get("op") == op]
        rs = sorted(runs[(model, att)], key=lambda r: float(r["start_h"]))
        out.append({
            "model": b["label"],
            "attempt": att,
            "finalLift": round(b["final_lift"], 1),
            "bestLift": round(b["best_lift"], 1),
            "sessionH": round(b["session_h"], 3),
            "gpuBusyH": round(b["gpu_busy_h"], 2),
            "promptM": b["prompt_tokens_M"],
            "runs": [{
                "s": round(float(r["start_h"]), 3),
                "e": round(float(r["end_h"]), 3) if r["end_h"] else None,
                "reqMin": round(float(r["requested_seconds"]) / 60),
                "lift": float(r["lift"]) if r["lift"] else None,
                "practice": round(float(r["practice_score"]), 1) if r["practice_score"] else None,
            } for r in rs],
            "evals": times("eval"),
            "stops": times("stop"),
            "checks": times("check"),
            "submits": times("submit"),
            "polls": len(times("status")),
        })
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"{len(out)} attempts, {sum(len(a['runs']) for a in out)} runs -> {OUT}")


if __name__ == "__main__":
    main()
