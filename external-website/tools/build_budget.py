"""Polytopia research-budget data for the post: src/posts/autoresearch/budget.json.

Reads these readouts of the bench checkout (Elo numbers, dollars and forecasts only; nothing private):
  reports/polytopia/research-budget-2026-11-18/data.json         18 sessions, compute multiplier after a 1 h final
  reports/polytopia/recipe-curves-2026-11-18/data.json           the same recipes trained for 8 h
  reports/polytopia/recipe-curves-2026-11-18/spend-frontier/data.json   the starter yardstick, 0.5-48 h
  reports/polytopia/forecast-8h-2026-11-19/results-b1.json       forecasts of four recipes' 8 h Elo
  reports/pokemon/forecast-lift-2026-11-19/results-a1+a1b.json   forecasts of 35 Pokémon sessions' lift
  reports/forecast-skill-2026-11-19/combined.json                those forecasts plus 146 research-run forecasts, scored
                                                                 against a curve fit

and, with --astra-long, the long-run page data (`reports/polytopia/astra-long-2026-11-18/auto.py fetch`, or
auto/page/data.json on Volume poly-astra-long-20261118): GPT-6.2 Astra t2's run past 8 h and the starter's seed-811
run past the yardstick's last point. That page is fitted separately (the compute-frontier pool plus stages pal-a*), so
each run is moved onto the recipe-curves scale by its mean difference where the two fits share snapshots.

    python3 tools/build_budget.py --bench ~/repos/game-autoresearch-bench [--astra-long data.json]

`fill` holds the $8.70 recipes of the fill campaign (poly-budget-fill-20261122b: GPT-6 Astra, GPT-6.1 Sol, Gemini 4
Carbon; reports/polytopia/research-budget-fill-2026-11-22 and recipe-curves-fill-2026-11-22, same scales). It is kept
apart from `recipes` so Figures 2 and 3 are unchanged; only Figure 1's recipe tab reads it. `--fill-only` rewrites
just that key of the existing budget.json.
"""
import argparse
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src/posts/autoresearch/budget.json"
CELL = {"z": "zero-shot", "b1": "$2.90", "b3": "$8.70", "b8": "$23", "t2": "$8.70, 2 h limit", "t4": "$8.70, 4 h limit"}


def starter_hours(yard, elo):
    """Training hours at which the starter reaches `elo`, log-linear between yardstick points (None if beyond them)."""
    for (h0, e0), (h1, e1) in zip(yard, yard[1:]):
        if e0 <= elo <= e1:
            return 2 ** (math.log2(h0) + (elo - e0) / (e1 - e0) * (math.log2(h1) - math.log2(h0)))
    return None


def fill_recipes(reports, yard, early):
    """The fill campaign's recipes with an 8 h curve, in the `recipes` row format."""
    fb, fc = reports / "polytopia/research-budget-fill-2026-11-22/data.json", reports / "polytopia/recipe-curves-fill-2026-11-22/data.json"
    if not (fb.exists() and fc.exists()):
        return []
    budget = {r["name"]: r for r in json.loads(fb.read_text())["recipes"]}
    out = []
    for r in json.loads(fc.read_text())["recipes"]:
        if not r["curve"]:
            continue
        b = budget[r["entrant"]]
        pts = [dict(hours=round(p["hours"], 1), elo=round(p["elo"])) for p in r["curve"] if p["hours"] >= early]
        for p in pts:
            eq = starter_hours(yard, p["elo"])
            p["starterHours"] = None if eq is None else round(eq, 2)
        out.append(dict(
            key="fill-" + r["entrant"], model=r["model_label"], cell=CELL[r["cell"]], budgetUsd=r["research_usd"],
            spentUsd=r["spent_usd"], researchHours=b["research_hours"], broken=bool(r["broken"]),
            multiplier1h=r["multiplier_1h"], seedMultipliers=[round(s["hours"], 2) for s in b.get("seeds", []) if s.get("hours")],
            finals1h=[round(f["elo"]) for f in r["finals"]], curve=pts,
            early=[dict(hours=round(p["hours"], 3), elo=round(p["elo"])) for p in r["curve"] if p["hours"] < early],
            long=[], lo8=round(r["curve"][-1]["lo"]), hi8=round(r["curve"][-1]["hi"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--astra-long", type=Path, help="the astra-long page's data.json")
    ap.add_argument("--fill-only", action="store_true", help="rewrite only the `fill` key of the existing budget.json")
    args = ap.parse_args()
    reports = args.bench.expanduser() / "reports"
    load = lambda p: json.loads((reports / p).read_text())
    if args.fill_only:
        spend = load("polytopia/recipe-curves-2026-11-18/spend-frontier/data.json")
        yard = [(p["hours"], p["elo"]) for p in spend["starter"] if 0.45 <= p["hours"] <= 14]
        data = json.loads(OUT.read_text())
        data["fill"] = fill_recipes(reports, yard, 0.45)
        OUT.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        print(f"fill: {len(data['fill'])} recipes ({', '.join(r['model'] for r in data['fill'])}) -> {OUT}")
        return

    budget = load("polytopia/research-budget-2026-11-18/data.json")
    curves = load("polytopia/recipe-curves-2026-11-18/data.json")
    # The four models added on 2026-11-19 (campaign poly-budget-ext-20261119): their 1 h results, and their 8 h curves
    # once that comparison has been fitted (same schema as the originals, on the same scales).
    ext_budget = reports / "polytopia/research-budget-ext-2026-11-19/data.json"
    ext_curves = reports / "polytopia/recipe-curves-ext-2026-11-19/data.json"
    if ext_budget.exists():
        budget["recipes"] += json.loads(ext_budget.read_text())["recipes"]
    if ext_curves.exists():
        curves["recipes"] += json.loads(ext_curves.read_text())["recipes"]
    spend = load("polytopia/recipe-curves-2026-11-18/spend-frontier/data.json")
    poly_fc = load("polytopia/forecast-8h-2026-11-19/results-b1.json")
    poke_fc = load("pokemon/forecast-lift-2026-11-19/results-a1+a1b.json")
    combined = load("forecast-skill-2026-11-19/combined.json")
    rule_row = next(r for r in combined["rows"] if r["rule"])

    # The starter yardstick: fresh finals to 4 h, the mean of two 16 h runs' snapshots to 14.5 h, one continued run after.
    # Points before 30 minutes (10, 15 and 20 minutes, from second runs stopped at 20 minutes) are kept apart as
    # `starterEarly` and each recipe's `early`, so the 30 min to 8 h curves and everything computed from them stay as is.
    EARLY = 0.45
    starter = [dict(hours=p["hours"], elo=round(p["elo"]), n=p["n"]) for p in spend["starter"] if p["hours"] >= EARLY]
    starter_early = [dict(hours=round(p["hours"], 3), elo=round(p["elo"]), n=p["n"]) for p in spend["starter"] if p["hours"] < EARLY]
    yard = [(p["hours"], p["elo"]) for p in spend["starter"] if EARLY <= p["hours"] <= 14]   # monotone part, for equal-Elo hours

    # Long runs past the 8 h curves, on this scale (see the docstring). `long` stays empty without --astra-long.
    long_astra, asof_long = [], None
    if args.astra_long:
        page = json.loads(args.astra_long.read_text())
        asof_long = page["asof"]
        rc = {round(p["hours"], 2): p["elo"] for r in curves["recipes"] if r["entrant"] == "astra62-t2" for p in r["curve"]}
        run = page["astra"]["run"]
        off = sum(p["elo"] - rc[round(p["hours"], 2)] for p in run if round(p["hours"], 2) in rc)
        off /= sum(round(p["hours"], 2) in rc for p in run)
        long_astra = [dict(hours=round(p["hours"], 1), elo=round(p["elo"] - off), lo=round(p["lo"] - off), hi=round(p["hi"] - off))
                      for p in run if p["hours"] > 8.01]
        sp = {round(p["hours"], 2): p["elo"] for p in spend["starter"]}
        lg = page["longer"]
        off_s = sum(p["elo"] - sp[round(p["hours"], 2)] for p in lg if round(p["hours"], 2) in sp)
        off_s /= sum(round(p["hours"], 2) in sp for p in lg)
        last = starter[-1]["hours"]
        starter += [dict(hours=round(p["hours"], 1), elo=round(p["elo"] - off_s), n=1) for p in lg if p["hours"] > last + 0.01]
        print(f"astra long run: {len(long_astra)} points past 8 h (to {long_astra[-1]['hours']} h), shift {off:+.1f}; "
              f"starter: to {starter[-1]['hours']} h, shift {off_s:+.1f}")

    mult = {r["name"]: r for r in budget["recipes"]}
    recipes = []
    for r in curves["recipes"]:
        b = mult[r["entrant"]]
        pts = [dict(hours=round(p["hours"], 1), elo=round(p["elo"])) for p in r["curve"] if p["hours"] >= EARLY]
        early = [dict(hours=round(p["hours"], 3), elo=round(p["elo"])) for p in r["curve"] if p["hours"] < EARLY]
        for p in pts:
            eq = starter_hours(yard, p["elo"])
            p["starterHours"] = None if eq is None else round(eq, 2)
        seeds = [s.get("hours") for s in b.get("seeds", [])]
        recipes.append(dict(
            key=r["entrant"], model=r["model_label"], cell=CELL[r["cell"]], budgetUsd=r["research_usd"],
            spentUsd=r["spent_usd"], researchHours=b["research_hours"], broken=bool(r["broken"]),
            multiplier1h=r["multiplier_1h"], seedMultipliers=[round(s, 2) for s in seeds if s],
            finals1h=[round(f["elo"]) for f in r["finals"]], curve=pts, early=early,
            long=long_astra if r["entrant"] == "astra62-t2" else [],
            lo8=round(r["curve"][-1]["lo"]) if r["curve"] else None, hi8=round(r["curve"][-1]["hi"]) if r["curve"] else None))

    # Sessions of the added models whose 8 h curves are not in yet: their 1 h multiplier only, no curve.
    have = {r["key"] for r in recipes}
    for b in budget["recipes"]:
        if b["name"] in have or b["model"] in ("opus", "sol62", "astra62"):
            continue
        recipes.append(dict(
            key=b["name"], model=b["model_label"], cell=CELL[b["cell"]], budgetUsd=b["research_usd"],
            spentUsd=b["spent_usd"], researchHours=b["research_hours"], broken=bool(b["broken"]),
            multiplier1h=b["mean"]["hours"] if b["mean"] else None,
            seedMultipliers=[round(s["hours"], 2) for s in b.get("seeds", []) if s.get("hours")],
            finals1h=[], curve=[], early=[], long=[], lo8=None, hi8=None))

    def forecasters(res, keep):
        rows = [dict(key=k, model=m["label"].replace(" (agy)", ""), **{f: m[f] for f in keep}) for k, m in res["models"].items()]
        return rows

    data = dict(
        usdPerGpuHour=spend["usd_per_gpu_hour"],
        games=dict(budget=budget["games"], curves=curves["games"]),
        starter=starter, starterEarly=starter_early, recipes=recipes, fill=fill_recipes(reports, yard, EARLY), longAsOf=asof_long,
        # An untrained network: the random player's rating on the same fit, where every cost curve starts.
        zero=dict(elo=round(spend["zero"]["elo"]), lo=round(spend["zero"]["lo"]), hi=round(spend["zero"]["hi"]), games=spend["zero"]["games"]),
        polyForecast=dict(
            truth={k: dict(elo=round(v["elo"]), mean1h=round(v["mean_1h"])) for k, v in poly_fc["truth"].items()},
            best=poly_fc["best"],
            models=forecasters(poly_fc, ["train_further", "ranking", "p50", "mae_p50", "bias_p50", "coverage_80"]),
            baselines=[dict(name=k, **{f: v[f] for f in ["train_further", "p50", "mae_p50"]}) for k, v in poly_fc["baselines"].items()]),
        pokeForecast=dict(
            models=forecasters(poke_fc, ["sets", "pairwise", "top1", "rho35", "mae", "coverage80"]),
            baselines=[dict(name=k.replace("baseline: ", ""), **{f: v[f] for f in ["pairwise", "top1", "rho35", "mae"]}) for k, v in poke_fc["baselines"].items()]),
        # Relative forecast skill over the 35 fresh finals plus 146 research runs (every item once): each model's own
        # skill against a blind guess with its 90% interval (the figure's axis), and its
        # lead over a curve fit (forecast-skill-2026-11-19/algo.py, fitted to the other sessions), with 90% paired bootstrap intervals: whole research
        # sessions resampled. ciFinals is the same interval on the 35 finals alone.
        pokeSkill=dict(
            rule=rule_row["own"], items=combined["items"], finals=combined["n_finals"], runs=combined["n_runs"],
            width=combined["width"]["rule"], pairs=combined["pairs"],
            models=[dict(model=r["m"], skill=r["own"], skillCi=r["own_ci"], lead=r["vs_rule"], ci=r["vs_rule_ci"],
                         leadFinals=r["vs_rule_finals"], ciFinals=r["vs_rule_finals_ci"])
                    for r in combined["rows"] if not r["rule"]]),
    )
    OUT.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    done = [r for r in recipes if r["curve"]]
    print(f"{len(recipes)} recipes ({len(done)} with 8 h curves), {len(starter)} starter points, "
          f"{len(data['polyForecast']['models'])} + {len(data['pokeForecast']['models'])} forecasters -> {OUT}")


if __name__ == "__main__":
    main()
