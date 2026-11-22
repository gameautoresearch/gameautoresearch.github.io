#!/usr/bin/env python3
"""Offline $2k hero-run costing and consistency checks; never starts paid work."""
import argparse
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def estimate(config, prices):
    D = lambda x: Decimal(str(x))
    money = lambda x: float(x.quantize(Decimal("0.01")))
    control = config["cost_control"]
    cap = D(control["campaign_incremental_cash_hard_cap_usd"])
    buckets = {k: D(v) for k, v in control["buckets_usd"].items()}
    if sum(buckets.values()) != cap:
        raise ValueError("Budget buckets must sum exactly to the global hard cap")
    nm, ng = len(config["models"]), len(config["games"])
    nr = config["independent_research_attempts_per_model_game"]
    if len({x["id"] for x in config["models"]}) != nm or len(set(config["games"])) != ng:
        raise ValueError("Duplicate model/game")
    n = nm * ng * nr
    attempt = config["attempt"]
    h = D(attempt["allocated_gpu_worker_seconds"]) / 3600
    main_hours = n * h
    extra = config["additional_work"]
    baseline_hours = ng * D(extra["baseline"]["allocated_h100_hours_per_game"])
    grade = extra["final_grading"]
    grade_artifacts = grade["researcher_artifacts"] + grade["baseline_artifacts"]
    if grade["researcher_artifacts"] != n:
        raise ValueError("Every researcher final must have a grade allocation")
    grade_hours = grade_artifacts * D(grade["allocated_h100_hours_per_artifact_ceiling"])
    if grade_hours != D(grade["total_hours_ceiling"]):
        raise ValueError("Final grading hours do not cover the declared artifacts")
    qualification_hours = ng * D(extra["qualification"]["allocated_h100_hours_per_game_ceiling"])
    all_hours = main_hours + baseline_hours + grade_hours + qualification_hours
    derived = config["derived_matrix"]
    for key, value in [("attempt_count", n), ("max_main_h100_hours", main_hours),
                       ("max_all_included_h100_hours", all_hours)]:
        if D(derived[key]) != value:
            raise ValueError("Stale derived matrix: " + key)
    modal = prices["modal"]
    gpu = D(modal["h100_usd_per_second"]) * 3600
    cpu = D(modal["sandbox_cpu_usd_per_physical_core_second"]) * attempt["physical_cpu_cores"] * 3600
    ram = D(modal["sandbox_memory_usd_per_gib_second"]) * attempt["memory_gib"] * 3600
    worker = gpu + cpu + ram
    workspace = attempt["cpu_workspace"]
    workspace_hours = n * D(workspace["max_seconds"]) / 3600
    if workspace_hours != D(derived["max_cpu_workspace_hours"]):
        raise ValueError("Stale workspace hours")
    workspace_rate = (D(modal["sandbox_cpu_usd_per_physical_core_second"]) * workspace["physical_cpu_cores"]
                      + D(modal["sandbox_memory_usd_per_gib_second"]) * workspace["memory_gib"]) * 3600
    resource = {"main_gpu_workers": main_hours * worker,
                "baseline_gpu_workers": baseline_hours * worker,
                "final_grading_gpu_workers": grade_hours * worker,
                "qualification_gpu_workers": qualification_hours * worker,
                "cpu_workspaces": workspace_hours * workspace_rate}
    margin = D(control["resource_quote_reservation_margin"])
    checks = {}
    for key, value in resource.items():
        held = value * margin
        if held > buckets[key]:
            raise ValueError(f"{key} cost including quote margin exceeds bucket")
        checks[key] = {"at_list_rates_usd": money(value), "with_quote_margin_usd": money(held),
                       "bucket_usd": money(buckets[key]), "fits": True}
    n_or = sum(m["route"] == "openrouter" for m in config["models"]) * ng * nr
    api_per_session = D(attempt["openrouter_cash_cap_usd_including_credit_fee"])
    if api_per_session * n_or > buckets["openrouter_cash_including_fees"]:
        raise ValueError("Per-session API caps exceed API bucket")
    fee = D(prices["openrouter"]["standard_credit_purchase_fee_fraction"])
    resource_total = sum(resource.values())
    scenarios = {}
    for name, profile in prices["token_scenarios"].items():
        inp = D(profile["aggregate_input_million_per_session"])
        out = D(profile["output_including_reasoning_million_per_session"])
        cached = inp * D(profile["cache_hit_fraction"])
        writes = D(profile["additional_cache_write_million_per_session"])
        costs, total, blocked = [], Decimal(0), []
        for m in config["models"]:
            if m["route"] != "openrouter":
                continue
            p = prices["openrouter"]["selected_endpoint_prices"][m["id"]]
            r = p["usd_per_million_tokens"]
            per = ((inp-cached) * D(r["input"]) + cached * D(r.get("cache_read", r["input"]))
                   + out * D(r["output_including_reasoning"])
                   + writes * D(r.get("additional_cache_write", 0))) * (1 + fee)
            cost = per * ng * nr
            total += cost
            if per > api_per_session:
                blocked.append(m["id"])
            costs.append({"model": m["id"], "per_session_cash_usd": money(per),
                          "all_sessions_cash_usd": money(cost), "fits_session_cap": per <= api_per_session})
        forecast = resource_total + total + buckets["builds_storage_egress_tax_and_hosting"]
        scenarios[name] = {"assumptions": profile, "models": costs,
                           "unconstrained_openrouter_cash_usd": money(total),
                           "exceeds_per_session_caps": blocked,
                           "full_volume_scenario_completable_under_model_caps": not blocked and total <= buckets["openrouter_cash_including_fees"],
                           "unconstrained_total_including_overhead_allowance_usd": money(forecast),
                           "meaning": "An over-cap scenario must stop paid calls; do not substitute the cap and pretend the same work completes."}
    return {
        "benchmark_id": config["benchmark_id"], "forecast_only": True, "launch_authorized": False,
        "price_observed_at": prices["observed_at_utc"],
        "matrix": {"models": nm, "games": ng, "research_attempts_per_cell": nr, "sessions": n,
                   "research_wall_hours": float(D(attempt["research_wall_seconds"])/3600),
                   "main_h100_hours": float(main_hours), "h100_hours_per_session": float(h),
                   "baseline_h100_hours": float(baseline_hours), "grading_h100_hours": float(grade_hours),
                   "qualification_h100_hours": float(qualification_hours), "all_h100_hours": float(all_hours),
                   "cpu_workspace_hours": float(workspace_hours)},
        "hourly_usd": {"h100": float(gpu), "gpu_worker_cpu": float(cpu), "gpu_worker_memory": float(ram),
                       "full_gpu_worker": float(worker), "cpu_workspace": float(workspace_rate)},
        "main_compute_usd": {"h100": money(main_hours*gpu), "sandbox_cpu": money(main_hours*cpu),
                             "sandbox_memory": money(main_hours*ram), "total": money(main_hours*worker)},
        "all_h100_gpu_only_usd": money(all_hours*gpu),
        "all_compute_before_model_and_overhead_usd": money(resource_total),
        "resource_budget_checks": checks,
        "budget": {"global_hard_cap_usd": money(cap), "buckets_usd": {k:money(v) for k,v in buckets.items()},
                   "all_buckets_sum_usd": money(sum(buckets.values())),
                   "openrouter_sessions": n_or, "openrouter_per_session_cash_cap_usd": money(api_per_session),
                   "per_session_inference_credit_cap_before_5_5_percent_fee_usd": float(api_per_session/(1+fee)),
                   "minimum_remaining_headroom_at_full_API_and_overhead_buckets_usd": money(cap-resource_total-buckets["openrouter_cash_including_fees"]-buckets["builds_storage_egress_tax_and_hosting"]),
                   "enforcement_status": control["runtime_integration_status"]},
        "token_scenarios": scenarios,
        "calendar_main_gpu_lower_bound_hours_at_two_workers": float(main_hours/2),
        "limitations": ["One researcher attempt per model/game: no research-seed variance estimate or reliable close ranking.",
                        "All OpenAI/Anthropic token cash assumes already-paid subscription quota; no extra seats or paid fallback.",
                        "Argon is the selected Google entry; Carbon is unavailable in the queried OpenRouter catalog.",
                        "Hard caps describe the new required launch protocol; no runtime ledger is provisioned or existing launcher rewritten by this estimator.",
                        "Smaller worker and bounded assessment runtimes must be qualified; unknown liabilities stay held and block new spend."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT/"docs/benchmark-hero-2k-v2.json")
    p.add_argument("--prices", type=Path, default=ROOT/"reports/benchmark-hero-2k-prices-2026-11-09.json")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = estimate(json.loads(a.config.read_text()), json.loads(a.prices.read_text()))
    body = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if a.output:
        with a.output.open("x") as f:
            f.write(body)
        print(json.dumps({"output":str(a.output.resolve()), "matrix":result["matrix"],
                          "main_compute_usd":result["main_compute_usd"],
                          "all_compute_usd":result["all_compute_before_model_and_overhead_usd"],
                          "budget":result["budget"],
                          "scenario_model_cash":{k:v["unconstrained_openrouter_cash_usd"] for k,v in result["token_scenarios"].items()}},indent=2))
    else:
        print(body,end="")


if __name__ == "__main__":
    main()
