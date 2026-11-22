#!/usr/bin/env python3
"""Offline planning arithmetic. Does not allocate workers or call a model API."""
import argparse
import json
import runpy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def estimate(config, prices):
    def dec(x):
        return Decimal(str(x))

    def usd(x):
        return float(x.quantize(Decimal("0.01")))

    attempt = config["attempt"]
    n_models = len(config["models"])
    n_games = len(config["games"])
    repeats = config["independent_research_attempts_per_model_game"]
    assert len({m["id"] for m in config["models"]}) == n_models
    assert len(set(config["games"])) == n_games
    sessions = n_models * n_games * repeats
    hours_each = dec(attempt["allocated_gpu_worker_seconds"]) / 3600
    hours = sessions * hours_each
    assert sessions == config["derived_matrix"]["attempt_count"]
    assert hours == dec(config["derived_matrix"]["max_main_h100_hours"])

    modal = prices["modal"]
    hourly = {
        "h100": dec(modal["h100_usd_per_second"]) * 3600,
        "sandbox_cpu": dec(modal["sandbox_cpu_usd_per_physical_core_second"]) * attempt["physical_cpu_cores"] * 3600,
        "sandbox_memory": dec(modal["sandbox_memory_usd_per_gib_second"]) * attempt["memory_gib"] * 3600,
    }
    hourly["full_worker"] = sum(hourly.values())
    main = hours * hourly["full_worker"]
    extra = prices["illustrative_additional_work"]
    baseline_hours = n_games * extra["baseline_training_seeds_per_game"] * hours_each
    grading_hours = sessions * len(config["evaluation"]["milestone_wall_hours"]) * dec(extra["grading_worker_hours_per_artifact"])
    calibration_hours = extra["calibration_models"] * extra["calibration_games"] * extra["calibration_attempts_per_cell"] * dec(extra["calibration_hours_per_attempt"])
    extra_hours = baseline_hours + grading_hours + calibration_hours
    extra_compute = extra_hours * hourly["full_worker"]
    fee = dec(prices["openrouter"]["standard_credit_purchase_fee_fraction"])
    contingency = dec(prices["planning_contingency_fraction"])

    results = {
        "benchmark_id": config["benchmark_id"],
        "price_observed_at": prices["observed_at_utc"],
        "forecast_only": True,
        "google_costed_as": "google/gemini-4-argon; Carbon unpriced and unresolved",
        "matrix": {"models": n_models, "games": n_games, "research_attempts_per_cell": repeats,
                   "sessions": sessions, "h100_hours_per_session": float(hours_each), "main_h100_hours": float(hours)},
        "hourly_usd": {k: float(v) for k, v in hourly.items()},
        "main_compute_usd": {**{k: usd(v * hours) for k, v in hourly.items() if k != "full_worker"}, "total": usd(main)},
        "compute_per_session_usd": usd(hourly["full_worker"] * hours_each),
        "compute_per_model_usd": usd(main / n_models),
        "compute_per_game_usd": usd(main / n_games),
        "additional_work_is_illustrative_not_authorized": True,
        "additional_work": {
            "baseline_h100_hours": float(baseline_hours), "grading_h100_hours": float(grading_hours),
            "calibration_h100_hours": float(calibration_hours), "total_h100_hours": float(extra_hours),
            "full_worker_usd": usd(extra_compute),
        },
        "token_scenarios": {},
    }
    for name, profile in prices["token_scenarios"].items():
        inp = dec(profile["aggregate_input_million_per_session"])
        out = dec(profile["output_including_reasoning_million_per_session"])
        cached = inp * dec(profile["cache_hit_fraction"])
        uncached = inp - cached
        writes = dec(profile["additional_cache_write_million_per_session"])
        per_model = []
        total_model = Decimal(0)
        for model in config["models"]:
            if model["route"] == "cli_proxy_subscription":
                per_model.append({"model": model["id"], "route": model["route"],
                                  "incremental_token_cash_usd_assuming_existing_plan_quota": 0,
                                  "subscription_and_extra_quota_cost_usd": None})
                continue
            p = prices["openrouter"]["selected_endpoint_prices"][model["id"]]
            rate = p["usd_per_million_tokens"]
            read_rate = rate.get("cache_read", rate["input"])
            one = uncached * dec(rate["input"]) + cached * dec(read_rate) + out * dec(rate["output_including_reasoning"])
            one += writes * dec(rate.get("additional_cache_write", 0))
            all_sessions = one * n_games * repeats
            total_model += all_sessions
            per_model.append({"model": model["id"], "route": "openrouter", "costed_model": p["model"],
                              "provider_tag": p["endpoint_tag"], "per_session_usd": usd(one),
                              "all_sessions_usd_before_credit_fee": usd(all_sessions)})
        funded = total_model * (1 + fee)
        subtotal = main + extra_compute + funded
        # Sensitivity: restore explicitly identified introductory/promotional prices.
        expiry_extra = Decimal(0)
        for model_id, multiplier in prices["openrouter"]["promotion_expiry_multipliers"].items():
            p = prices["openrouter"]["selected_endpoint_prices"][model_id]["usd_per_million_tokens"]
            cost = uncached * dec(p["input"]) + cached * dec(p.get("cache_read", p["input"]))
            cost += out * dec(p["output_including_reasoning"]) + writes * dec(p.get("additional_cache_write", 0))
            expiry_extra += cost * (dec(multiplier) - 1) * n_games * repeats
        expiry_extra *= 1 + fee
        results["token_scenarios"][name] = {
            "assumptions": profile, "models": per_model,
            "openrouter_inference_usd": usd(total_model), "openrouter_credit_purchase_fee_usd": usd(funded - total_model),
            "openrouter_cash_usd": usd(funded), "main_compute_plus_model_cash_usd": usd(main + funded),
            "including_illustrative_additional_compute_usd": usd(subtotal),
            "with_planning_contingency_usd": usd(subtotal * (1 + contingency)),
            "promotion_expiry_additional_model_cash_usd": usd(expiry_extra),
            "with_expired_promotions_and_contingency_usd": usd((subtotal + expiry_extra) * (1 + contingency)),
        }
    results["calendar_lower_bound_days_at_concurrency"] = {
        str(n): float(hours / (24 * n)) for n in (1, 5, 10, 20, 50)
    }
    results["grading_sensitivity"] = [
        {"hours_per_artifact": h, "total_grade_h100_hours": float(sessions * len(config["evaluation"]["milestone_wall_hours"]) * dec(h)),
         "total_grade_full_worker_usd": usd(sessions * len(config["evaluation"]["milestone_wall_hours"]) * dec(h) * hourly["full_worker"])}
        for h in (0.25, 0.5, 1)
    ]
    results["exclusions"] = prices["exclusions"]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "docs/benchmark-hero-2k-v2.json")
    parser.add_argument("--prices", type=Path, default=ROOT / "reports/benchmark-hero-2k-prices-2026-11-09.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config, prices = json.loads(args.config.read_text()), json.loads(args.prices.read_text())
    if config.get("schema_version") == 2:
        hero = runpy.run_path(str(ROOT / "tools/estimate_benchmark_hero_cost.py"))
        result = hero["estimate"](config, prices)
    else:
        result = estimate(config, prices)
    body = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as f:
            f.write(body)
        if "budget" in result:
            print(json.dumps({"output": str(args.output.resolve()), "matrix": result["matrix"], "budget": result["budget"]}, indent=2))
            return
        print(json.dumps({"output": str(args.output.resolve()), "main_h100_hours": result["matrix"]["main_h100_hours"],
                          "main_compute_usd": result["main_compute_usd"],
                          "scenarios": {k: {field: v[field] for field in ("openrouter_cash_usd", "main_compute_plus_model_cash_usd", "with_planning_contingency_usd", "with_expired_promotions_and_contingency_usd")} for k, v in result["token_scenarios"].items()}}, indent=2))
    else:
        print(body, end="")


if __name__ == "__main__":
    main()
