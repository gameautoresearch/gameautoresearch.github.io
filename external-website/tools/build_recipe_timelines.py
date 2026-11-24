"""Derive the Polytopia $8.70 (b3) recipe session timelines for Figure 4 (src/posts/autoresearch/recipe-timelines.json).

Reads each model's `b3` ($8.70 budget, 7 h wall limit, up to 2 L40S GPUs) session from the two main Polytopia
research-budget campaigns:
  - poly-budget-20261118 (Claude Opus 5.5, GPT-6.2 Astra, GPT-6.2 Sol)
  - poly-budget-ext-20261119 (Gemini 4 Neon, Claude Fable 5.2, DeepSeek V4.1 Flash, Kimi K3.1 Pro)

Inputs: each session's `session.json`, `session-result.json`, `NOTES.md`, and `research/ledger.sqlite3`, plus the
8 h Elo from `src/posts/autoresearch/budget.json`. GPU leases under 1 minute (aborted launch setup, e.g. Neon's
29 s `v1w`) are excluded from `runs` (recorded in `aborted`) so the training-run count matches Figure 3's 11 runs.
Only timings, durations, dollar charges, 8 h Elo and two-line experiment summaries leave this script.

    python3 tools/build_recipe_timelines.py --bench ~/repos/game-autoresearch-bench --ext-dir /path/to/ext-ledgers
"""
import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET_JSON = ROOT / "src/posts/autoresearch/budget.json"
OUT = ROOT / "src/posts/autoresearch/recipe-timelines.json"

SESSIONS = [
    ("neon-b3", "Gemini 4 Neon", "ext", "polytopia-neon-b3"),
    ("fable-b3", "Claude Fable 5.2", "ext", "polytopia-fable-b3"),
    ("opus-b3", "Claude Opus 5.5", "main", "polytopia-opus-b3"),
    ("flash-b3", "DeepSeek V4.1 Flash", "ext", "polytopia-flash-b3"),
    ("astra62-b3", "GPT-6.2 Astra", "main", "polytopia-astra62-b3"),
    ("sol62-b3", "GPT-6.2 Sol", "main", "polytopia-sol62-b3"),
    ("kimi-b3", "Kimi K3.1 Pro", "ext", "polytopia-kimi-b3"),
]

# Grounded two-line descriptions (what was tested, outcome) sourced from each session's NOTES.md.
NOTES: dict[tuple[str, str], tuple[str, str]] = {
    # Gemini 4 Neon (polytopia-neon-b3)
    ("neon-b3", "base0w"): (
        "Unmodified starter baseline (2×32 net, plain PPO)",
        "0.0% practice win rate; baseline anchor",
    ),
    ("neon-b3", "v1bw"): (
        "Dual-view critic + turn-based GAE + attention net",
        "3.1% practice (+3.1%); bottlenecked by slow compile",
    ),
    ("neon-b3", "v2aw"): (
        "Spatial pyramid net (poly-v4), pure self-play",
        "Crashed in CPU practice eval (SIGFPE); discarded",
    ),
    ("neon-b3", "v2bw"): (
        "Spatial pyramid net + historical opponent pool",
        "CPU eval crash + seat-flip KL bug; fixed on CPU",
    ),
    ("neon-b3", "v3aw"): (
        "CPU-safe pyramid (poly-v5) + annealed shaping, self-play",
        "11.3% practice (+8.2% over v1b), 3× faster compile",
    ),
    ("neon-b3", "v3bw"): (
        "CPU-safe pyramid + fixed historical opponent pool (PFSP)",
        "15.2% practice (+12.1%); clearly beat pure self-play",
    ),
    ("neon-b3", "v4aw"): (
        "1-epoch updates + 36 scoreboard inputs + MSE critic",
        "18.4% practice (+3.1%) at 49k samples/s; kept",
    ),
    ("neon-b3", "v4bw"): (
        "2-epoch updates + 21-bin distributional critic (HL-Gauss)",
        "15.4% practice (+0.2%), slower throughput; dropped",
    ),
    ("neon-b3", "v5aw"): (
        "Smaller batches (32k/4k) for 2× updates + early PFSP evict",
        "22.7% practice (+4.3%), balanced on both seats; kept",
    ),
    ("neon-b3", "v5bw"): (
        "Larger batches (65k/4k) + stronger shaping + higher LR",
        "21.7% practice (+3.3%); trailed v5a vs starter-1h",
    ),
    ("neon-b3", "v6finalw"): (
        "1-hour validation of winning v5a recipe (20-opponent pool)",
        "47.9% practice (beat both starters); final submission",
    ),
    # Claude Fable 5.2 (polytopia-fable-b3)
    ("fable-b3", "base1w"): (
        "Starter baseline (2×32 net, constant LR 3e-4)",
        "27.3% practice at 22 min; short-horizon anchor",
    ),
    ("fable-b3", "big1w"): (
        "Larger 4×48 net + cosine LR decay (5e-4 → 0.1×)",
        "20.3% practice; trailed starter (size + LR confounded)",
    ),
    ("fable-b3", "cos1w"): (
        "Cosine LR decay ablation on starter 2×32 net",
        "14.6% practice; decay hurt sharply at 22 min",
    ),
    ("fable-b3", "size1w"): (
        "Larger 4×48 net ablation, constant LR 3e-4",
        "24.0% practice; tested at full 1 h next",
    ),
    ("fable-b3", "full1w"): (
        "Full 1-hour validation of 4×48 net, constant LR 3e-4",
        "43.4% practice (tied starter-1h); final submission",
    ),
    ("fable-b3", "base2w"): (
        "Starter baseline replicate (2×32, seed 802)",
        "30.9% practice vs 27.3% on seed 801 (~3.5 pt spread)",
    ),
    # Claude Opus 5.5 (polytopia-opus-b3)
    ("opus-b3", "shapeaw"): (
        "Score-difference shaping + linear LR decay on 2×32 net",
        "30.1% practice at 27 min; beat wider 3×48 candidate",
    ),
    ("opus-b3", "shapebw"): (
        "Wider 3×48 net + score shaping + linear LR decay",
        "26.8% practice; trailed 2×32 and dropped",
    ),
    ("opus-b3", "final1w"): (
        "Full 1-hour run of 2×32 + score shaping + LR decay",
        "43.1% practice (43.0% vs starter-1h); final submission",
    ),
    ("opus-b3", "abl0w"): (
        "No-shaping ablation (2×32, linear LR decay only)",
        "30.9% practice vs 30.1% shaped (no clear gain)",
    ),
    ("opus-b3", "lr6w"): (
        "2× higher learning rate (6e-4 linear decay) + shaping",
        "30.3% practice, matched 3e-4; kept original LR",
    ),
    # DeepSeek V4.1 Flash (polytopia-flash-b3)
    ("flash-b3", "b64x2w"): (
        "Wider 2×64 net (469k params), starter PPO config",
        "28.7% practice at 22 min; beat 3×48 at matched updates",
    ),
    ("flash-b3", "a48x3w"): (
        "Full 1-hour run of deeper 3×48 net (361k params)",
        "46.1% practice (51.6% vs starter-1h); fallback recipe",
    ),
    ("flash-b3", "c56x3w"): (
        "Mid-size 3×56 net (448k params), starter PPO config",
        "Short width/depth probe at ~67k samples/s",
    ),
    ("flash-b3", "d64x2w"): (
        "Full 1-hour run of 2×64 net, starter PPO config",
        "51.2% practice (57.8% vs starter-1h); final submission",
    ),
    # GPT-6.2 Astra (polytopia-astra62-b3)
    ("astra62-b3", "fieldw"): (
        "Starter 2×32 + additive action log-priors + higher LR/λ",
        "51.4% practice at 57 min (57.0% vs starter-1h)",
    ),
    ("astra62-b3", "macrow"): (
        "Shared 48-wide trunk + auxiliary progress shaping + priors",
        "30.9% practice; stalled well behind field",
    ),
    ("astra62-b3", "offerw"): (
        "48-wide stem + 24 tactic flags, separate critic, LR decay",
        "53.7% practice (64.1% vs starter-1h); final submission",
    ),
    ("astra62-b3", "audit"): (
        "CPU integrity audit of final actor/critic weights",
        "Verified finite losses, trained weights, 0.8 ms inference",
    ),
    # GPT-6.2 Sol (polytopia-sol62-b3)
    ("sol62-b3", "sharedshapew"): (
        "Shared actor/critic trunk + potential shaping (2×32)",
        "27.3% practice at 1 h (trailed starter-1h at 36.7%)",
    ),
    ("sol62-b3", "spatialshapew"): (
        "Spatial 4×48 net (spatial-v3) + shared critic + shaping",
        "33.4% practice (+6.1 pts over 2×32 despite slower speed)",
    ),
    ("sol62-b3", "turnspatialw"): (
        "Turn-based GAE (discount per End Turn) on spatial-v3",
        "44.1% practice (+10.7 pts, beat starters); final submission",
    ),
    # Kimi K3.1 Pro (polytopia-kimi-b3)
    ("kimi-b3", "basew"): (
        "Unmodified starter baseline (2×32, 344M samples)",
        "46.3% practice at 57 min; full-horizon anchor",
    ),
    ("kimi-b3", "widew"): (
        "Larger 3×48 net (361k params), starter PPO config",
        "47.1% practice (+0.8 pts over baseline); final submission",
    ),
    ("kimi-b3", "cosinew"): (
        "3×48 net + cosine LR decay to 10%",
        "41.3% practice; led mid-run but decayed too fast late",
    ),
}

CPU_FALLBACK: dict[str, tuple[str, str]] = {
    "neon-b3": (
        "CPU debug isolating poly-v4 SIGFPE eval crash",
        "Traced to strided conv/resize in XLA:CPU; fixed in poly-v5",
    ),
}


def load_session(session_dir: Path, key: str, label: str, elo8: int) -> dict:
    sess = json.loads((session_dir / "session.json").read_text())
    res = json.loads((session_dir / "session-result.json").read_text())
    start = sess["deadline"] - res["wall_seconds"]
    wall_h = round((sess["finalized_at"] - start) / 3600, 2)
    spent_usd = round(res["budget"]["spent_usd"], 2)

    con = sqlite3.connect(session_dir / "research/ledger.sqlite3")
    rows = con.execute(
        "SELECT name, status, started, ended, charged, payload FROM jobs ORDER BY started"
    ).fetchall()
    con.close()

    runs = []
    aborted = 0
    cpu = []
    evals = []

    for name, _status, s, e, ch, pl in rows:
        if not s or not e:
            continue
        p = json.loads(pl) if pl else {}
        kind = p.get("kind")
        sh = round((s - start) / 3600, 3)
        eh = round((e - start) / 3600, 3)
        dur_min = round((e - s) / 60, 1)

        if kind == "practice":
            evals.append({"s": sh, "e": eh, "min": dur_min})
        elif kind in ("lease", "run"):
            gpu = p.get("shape", {}).get("gpu", 1 if kind == "run" else 0)
            if gpu >= 1:
                if (e - s) < 60:
                    aborted += 1
                    continue
                what, note = NOTES[(key, name)]
                runs.append({
                    "s": sh,
                    "e": eh,
                    "min": dur_min,
                    "usd": round(float(ch or 0), 2),
                    "slot": 0,
                    "what": what,
                    "note": note,
                })
            else:
                what, note = NOTES[(key, name)] if (key, name) in NOTES else CPU_FALLBACK[key]
                cpu.append({
                    "s": sh,
                    "e": eh,
                    "min": dur_min,
                    "usd": round(float(ch or 0), 2),
                    "what": what,
                    "note": note,
                })

    # Greedy 2-slot interval assignment (harness enforces max 2 concurrent GPUs).
    slot_end = [-1.0, -1.0]
    for r in runs:
        if r["s"] >= slot_end[0] - 1e-4:
            r["slot"] = 0
            slot_end[0] = r["e"]
        else:
            r["slot"] = 1
            slot_end[1] = r["e"]

    gpu_h = round(sum(r["min"] for r in runs) / 60, 2)
    return {
        "key": key,
        "model": label,
        "elo8": elo8,
        "spentUsd": spent_usd,
        "wallH": wall_h,
        "gpuH": gpu_h,
        "aborted": aborted,
        "runs": runs,
        "cpu": cpu,
        "evals": evals,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, type=Path, help="path to game-autoresearch-bench")
    ap.add_argument("--ext-dir", required=True, type=Path, help="directory containing poly-budget-ext-20261119 session dirs")
    args = ap.parse_args()

    budget = json.loads(BUDGET_JSON.read_text())
    elo_by_key = {r["key"]: r["curve"][-1]["elo"] for r in budget["recipes"] if r.get("curve")}
    starter_8h = next(p["elo"] for p in budget["starter"] if p["hours"] == 8.0)

    main_base = args.bench.expanduser() / "runs/nanny/poly-budget-20261118/volume/campaign"
    ext_base = args.ext_dir.expanduser()

    sessions = []
    for key, label, src, folder in SESSIONS:
        sdir = (main_base if src == "main" else ext_base) / folder
        sessions.append(load_session(sdir, key, label, elo_by_key[key]))

    sessions.sort(key=lambda s: (-len(s["runs"]), -s["elo8"]))

    out = {
        "budgetUsd": 8.7,
        "wallLimitH": 7,
        "maxGpus": 2,
        "starterElo8": starter_8h,
        "sessions": sessions,
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {len(sessions)} sessions ({sum(len(s['runs']) for s in sessions)} GPU runs) -> {OUT}")


if __name__ == "__main__":
    main()
