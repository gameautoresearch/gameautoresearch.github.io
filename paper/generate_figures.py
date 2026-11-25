#!/usr/bin/env python3
"""Generate publication-quality vector PDF figures for the VibeResearch Bench paper
directly from the canonical JSON data files in external-website/src/posts/autoresearch/.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths & global style
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "external-website" / "src" / "posts" / "autoresearch"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Exact CMYK + model palette from colors.ts
CMYK = {
    "c": "#00a0d6",
    "cDeep": "#0081ad",
    "cLight": "#7fcfea",
    "m": "#e0218a",
    "mDeep": "#b8156e",
    "y": "#d4a000",
    "yDeep": "#8f6c00",
    "k": "#111111",
    "grey": "#bdbdbd",
}

MODEL_PALETTE = [
    ("6.2 astra", "#f08cc0"),
    ("neon", "#2f55c4"),
    ("carbon", "#7f9ae0"),
    ("hydrogen", "#1f9e89"),
    ("fable", "#e8740f"),
    ("muse", "#8a5cc4"),
    ("sol", CMYK["c"]),
    ("astra", CMYK["m"]),
    ("opus", CMYK["y"]),
    ("kimi", CMYK["k"]),
    ("deepseek", "#6b6b6b"),
    ("glm", "#9a9a9a"),
    ("luna", "#b5b5b5"),
    ("inkling", "#b5b5b5"),
    ("hy4", "#b5b5b5"),
]


def model_color(name: str) -> str:
    low = name.lower()
    for key, col in MODEL_PALETTE:
        if key in low:
            return col
    return CMYK["grey"]


def mark_color(name: str) -> str:
    c = model_color(name)
    return "#575752" if c in ("#b5b5b5", "#bdbdbd", "#9a9a9a") else c


def curve_color(name: str) -> str:
    c = model_color(name)
    return "#7d7d77" if c in ("#bdbdbd", "#b5b5b5") else c


def label_text_color(name: str) -> str:
    c = curve_color(name)
    if c == "#d4a000":
        return "#967000"
    if c == "#7f9ae0":
        return "#4762b0"
    if c == "#f08cc0":
        return "#c75091"
    return c


def display_model(mo: str) -> str:
    return "Claude Opus 5.5" if mo == "Opus 5.5" else mo


def win_pct(elo: float) -> float:
    return 100.0 / (1.0 + 10.0 ** (-elo / 400.0))


def fmt_win(elo: float) -> str:
    p = win_pct(elo)
    if p < 0.5:
        return "<1%"
    if p > 99.5:
        return ">99%"
    return f"{round(p)}%"


def layout_labels(items, min_gap, min_y, max_y):
    """Collision-free vertical label placement (y increases downward or upward; works for ascending y)."""
    out = sorted([{"key": k, "y": y} for k, y in items], key=lambda d: d["y"])
    if not out:
        return {}
    for _ in range(12):
        for i in range(1, len(out)):
            overlap = out[i - 1]["y"] + min_gap - out[i]["y"]
            if overlap > 0:
                out[i - 1]["y"] -= overlap / 2.0
                out[i]["y"] += overlap / 2.0
        if out[0]["y"] < min_y:
            shift = min_y - out[0]["y"]
            for d in out:
                d["y"] += shift
        if out[-1]["y"] > max_y:
            shift = out[-1]["y"] - max_y
            for d in out:
                d["y"] -= shift
    return {d["key"]: d["y"] for d in out}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
budget_data = json.loads((DATA_DIR / "budget.json").read_text())
top_scores_data = json.loads((DATA_DIR / "top-scores.json").read_text())
generals_data = json.loads((DATA_DIR / "generals-tournament.json").read_text())
poly13_data = json.loads((DATA_DIR / "polytopia13.json").read_text())
cap_data = json.loads((DATA_DIR / "capability-timeline.json").read_text())
rt_data = json.loads((DATA_DIR / "recipe-timelines.json").read_text())

RATE = budget_data["usdPerGpuHour"]

# Weighted PAVA starter curve + power-law extrapolation (matches CostMultiplier.tsx)
_blocks = []
for p in budget_data["starter"]:
    _blocks.append({"lt": math.log2(p["hours"]), "elo": p["elo"], "w": p["n"]})
    while len(_blocks) > 1 and _blocks[-2]["elo"] >= _blocks[-1]["elo"]:
        b = _blocks.pop()
        a = _blocks.pop()
        w = a["w"] + b["w"]
        _blocks.append({
            "lt": (a["lt"] * a["w"] + b["lt"] * b["w"]) / w,
            "elo": (a["elo"] * a["w"] + b["elo"] * b["w"]) / w,
            "w": w,
        })
KNOTS = [{"hours": 2.0 ** b["lt"], "elo": b["elo"]} for b in _blocks]
TOP_KNOT = KNOTS[-1]
FIT = {"einf": 568.43, "b": 522.51, "alpha": 0.28036}


def fit_hours(elo: float) -> float:
    return (FIT["b"] / max(FIT["einf"] - elo, 1e-9)) ** (1.0 / FIT["alpha"])


def starter_hours_for(elo: float):
    if elo >= TOP_KNOT["elo"]:
        return TOP_KNOT["hours"] * (fit_hours(elo) / fit_hours(TOP_KNOT["elo"])), True
    if elo <= KNOTS[0]["elo"]:
        s = (KNOTS[1]["elo"] - KNOTS[0]["elo"]) / math.log2(KNOTS[1]["hours"] / KNOTS[0]["hours"])
        return KNOTS[0]["hours"] * (2.0 ** ((elo - KNOTS[0]["elo"]) / s)), False
    for i in range(1, len(KNOTS)):
        a, b = KNOTS[i - 1], KNOTS[i]
        if elo <= b["elo"]:
            return a["hours"] * ((b["hours"] / a["hours"]) ** ((elo - a["elo"]) / (b["elo"] - a["elo"]))), False
    return TOP_KNOT["hours"], False


# ---------------------------------------------------------------------------
# Figure 1: Overall Ranking (TopScore.tsx)
# ---------------------------------------------------------------------------
def make_fig1():
    CELL_OFFSETS = {
        "1h": {
            "$2.90": 0.146952658206,
            "$8.70": -0.010457718437,
            "$23": -0.009175502062,
            "$8.70, 2 h limit": 0.020613940092,
            "$8.70, 4 h limit": -0.147933377799,
        },
        "8h": {
            "$2.90": 0.043603072642,
            "$8.70": 0.496266981517,
            "$23": -0.334910462831,
            "$8.70, 2 h limit": 0.087929094771,
            "$8.70, 4 h limit": -0.292888686099,
        },
    }
    NORMS = {
        "poly_empty": {"mean": 62.48, "sd": 11.394981351455},
        "poke": {"mean": 67.76, "sd": 7.682994207989},
        "forecast": {"mean": 0.2868, "sd": 0.061315250958},
        "recipe8": {"mean": -0.08217861795, "sd": 0.239269587071},
        "generals_t": {"mean": 1135.0, "sd": 221.360177087027},
    }

    obs1h, obs8h = [], []
    for r in budget_data["recipes"] + budget_data.get("fill", []):
        if r["cell"] == "zero-shot":
            continue
        mo = display_model(r["model"])
        m1 = 0.5 if (r["broken"] or r["multiplier1h"] is None) else r["multiplier1h"]
        end = r["curve"][-1] if r["curve"] else None
        m8 = 0.5 if (r["broken"] or not end) else starter_hours_for(end["elo"])[0] / end["hours"]
        obs1h.append((mo, r["cell"], math.log(m1)))
        obs8h.append((mo, r["cell"], math.log(m8)))

    recipe_all = {}
    for mo in set(o[0] for o in obs8h):
        e1 = [y - CELL_OFFSETS["1h"][c] for m, c, y in obs1h if m == mo and c in CELL_OFFSETS["1h"]]
        e8 = [y - CELL_OFFSETS["8h"][c] for m, c, y in obs8h if m == mo and c in CELL_OFFSETS["8h"]]
        if e1 and e8:
            recipe_all[mo] = math.exp((sum(e1) / len(e1) + sum(e8) / len(e8)) / 2.0)

    task_map = {t["id"]: t["scores"] for t in top_scores_data["tasks"]}
    task_map["recipe8"] = recipe_all
    task_map["generals_t"] = {r["model"]: r["elo"] for r in generals_data["rows"]}

    display_tasks = ["poly_empty", "recipe8", "poke", "forecast", "generals_t"]
    req_tasks = ["poly_empty", "poke", "forecast", "recipe8"]

    def standardize(tid, val):
        tv = math.log(val) if tid == "recipe8" else val
        return 50.0 + 10.0 * (tv - NORMS[tid]["mean"]) / NORMS[tid]["sd"]

    rows = []
    for mo in task_map["poly_empty"]:
        if all(mo in task_map[t] for t in req_tasks):
            each = [standardize(t, task_map[t][mo]) for t in display_tasks]
            sc = sum(each) / len(each)
            se = math.sqrt(sum((x - sc) ** 2 for x in each) / (len(each) - 1) / len(each))
            rows.append({"model": mo, "score": sc, "lo": sc - se, "hi": sc + se, "each": each})
    rows.sort(key=lambda r: -r["score"])

    # Layout using manual axes for crisp alignment
    fig = plt.figure(figsize=(6.8, 3.65))
    ax = fig.add_axes([0.23, 0.14, 0.42, 0.72])
    ax_tbl = fig.add_axes([0.66, 0.14, 0.33, 0.72])

    n = len(rows)
    ys = np.arange(n)[::-1]

    # Grid & axis on dot plot
    for t in [35, 40, 45, 50, 55, 60, 65]:
        lw = 0.9 if t == 50 else 0.55
        col = "#b8b8b0" if t == 50 else "#e6e6e2"
        ls = "--" if t == 50 else "-"
        ax.axvline(t, color=col, lw=lw, ls=ls, zorder=1)

    for i, r in enumerate(rows):
        y = ys[i]
        col = mark_color(r["model"])
        # ±1 SE thick line
        ax.plot([r["lo"], r["hi"]], [y, y], color=col, alpha=0.33, lw=4.8, solid_capstyle="round", zorder=2)
        # Main dot
        ax.scatter([r["score"]], [y], s=44, color=col, edgecolor="white", linewidth=1.1, zorder=3)
        # Subtle row guide
        if i < n - 1:
            ax.axhline(y - 0.5, color="#f2f2ef", lw=0.5, zorder=0)

    ax.set_xlim(34, 66)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([35, 40, 45, 50, 55, 60, 65])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{i + 1}   {r['model']}" for i, r in enumerate(rows)], fontsize=8.3, fontweight="medium")
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=3, color="#cccccc")
    ax.set_xlabel("Overall score (50 = reference average, +10 = +1 SD)", fontsize=8.0, color="#444444", labelpad=5)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

    # Right side: Score + 5 standardized task heatmap cells
    ax_tbl.set_xlim(0, 6.25)
    ax_tbl.set_ylim(-0.6, n - 0.4)
    ax_tbl.axis("off")

    # Headers
    ax_tbl.text(0.45, n - 0.15, "Score", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#333333")
    # Spanning Polytopia head
    ax_tbl.text(2.05, n + 0.30, "Polytopia", ha="center", va="bottom", fontsize=7.4, color="#555555")
    ax_tbl.plot([1.15, 2.95], [n + 0.19, n + 0.19], color="#c8c8c2", lw=0.8, clip_on=False)

    col_centers = [1.55, 2.55, 3.60, 4.65, 5.70]
    col_heads = [("", "empty"), ("", "recipes"), ("Pokémon", "3v3"), ("Fore-", "casting"), ("Generals", "tourney")]
    for cx, (h1, h2) in zip(col_centers, col_heads):
        if h1:
            ax_tbl.text(cx, n + 0.30, h1, ha="center", va="bottom", fontsize=7.3, color="#555555")
        ax_tbl.text(cx, n - 0.15, h2, ha="center", va="bottom", fontsize=7.3, color="#555555")

    cell_lo = np.array([247, 247, 248], dtype=float)
    cell_hi = np.array([184, 196, 212], dtype=float)

    for i, r in enumerate(rows):
        y = ys[i]
        ax_tbl.text(0.45, y, f"{r['score']:.1f}", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#111111")
        for cx, val in zip(col_centers, r["each"]):
            t = max(0.0, min(1.0, (val - 30.0) / 35.0))
            rgb = (cell_lo + (cell_hi - cell_lo) * t) / 255.0
            rect = patches.FancyBboxPatch(
                (cx - 0.43, y - 0.35), 0.86, 0.70,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=rgb, edgecolor="none", zorder=1
            )
            ax_tbl.add_patch(rect)
            ax_tbl.text(cx, y, f"{val:.0f}", ha="center", va="center", fontsize=7.8, color="#222222", zorder=2)

    out = OUT_DIR / "fig1_overall_ranking.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: 5-hour Polytopia empty-starter research race (Frontier13.tsx)
# ---------------------------------------------------------------------------
def make_fig2():
    models = poly13_data["models"]
    refs = poly13_data["refs"]

    labeled = [m for m in models if win_pct(m["best"]["elo"]) >= 10.0 and m["gpu"] >= 3.0]
    labeled_keys = {m["key"] for m in labeled}
    top_keys = {m["key"] for m in sorted(labeled, key=lambda m: -m["best"]["elo"])[:4]}

    fig, ax = plt.subplots(figsize=(6.8, 3.85))

    # Grid
    for y in [0, 25, 50, 75, 100]:
        ax.axhline(y, color="#222222" if y == 0 else "#eaeae6", lw=0.7 if y == 0 else 0.55, zorder=1)
    for x in [0, 1, 2, 3, 4, 5]:
        ax.axvline(x, color="#eaeae6", lw=0.55, zorder=1)

    # Reference lines
    y_tuned = win_pct(refs["starter-200m"])
    ax.axhline(y_tuned, color="#666660", lw=0.9, ls="--", zorder=2)
    ax.text(0.08, y_tuned + 1.8, f"Tuned PPO baseline (36 min, {round(y_tuned)}%)", fontsize=7.4, color="#555550", va="bottom")

    ax.axhline(50.0, color="#666660", lw=0.9, ls="--", zorder=2)
    ax.text(4.94, 48.2, "15-min PPO baseline (50% = even)", fontsize=7.4, color="#555550", va="top", ha="right")

    # Sort draw order: muted -> colored -> top
    def rank_key(m):
        r = 2 if m["key"] in top_keys else 1 if m["key"] in labeled_keys else 0
        return (r, m["best"]["elo"])

    for m in sorted(models, key=rank_key):
        best = -1e9
        xs, ys = [0.0], [0.0]
        for x, elo, _run, _sub in m["nets"]:
            if elo > best:
                best = elo
                xs.extend([x, x])
                ys.extend([ys[-1], win_pct(elo)])
        xs.append(m["gpu"])
        ys.append(ys[-1])

        colored = m["key"] in labeled_keys
        is_top = m["key"] in top_keys
        col = curve_color(m["label"]) if colored else "#c2c2bc"
        lw = 1.85 if is_top else 1.3 if colored else 0.9
        alpha = 0.95 if is_top else 0.65 if m["key"] == "sol" else 0.8 if colored else 0.48
        ax.plot(xs, ys, color=col, lw=lw, alpha=alpha, zorder=3 + (2 if is_top else 1 if colored else 0))

        if colored and m["gpu"] < 4.75:
            ax.scatter([m["gpu"]], [ys[-1]], s=16, color=col, edgecolor="white", lw=0.7, zorder=6)

    # Right-margin direct labels (collision-free)
    labeled_sorted = sorted(labeled, key=lambda m: -m["best"]["elo"])
    # In data coords (y from 0 to 100, ascending): higher elo = higher y
    y_targets = [(m["key"], win_pct(m["best"]["elo"])) for m in labeled_sorted]
    # layout_labels assumes ascending y; min_gap in win% units (~5.4%)
    y_pos = layout_labels(y_targets, min_gap=5.2, min_y=12.0, max_y=98.5)

    for m in labeled_sorted:
        col = curve_color(m["label"])
        tcol = label_text_color(m["label"])
        is_top = m["key"] in top_keys
        end_x = m["gpu"]
        end_y = win_pct(m["best"]["elo"])
        ly = y_pos[m["key"]]

        if end_x < 4.75:
            ax.plot([end_x + 0.04, 5.0], [end_y, end_y], color=col, lw=0.75, ls=":", alpha=0.55, clip_on=False)
        ax.plot([5.02 if end_x >= 4.75 else 5.0, 5.10, 5.17], [end_y, ly, ly], color=col, lw=0.8, alpha=0.7, clip_on=False)
        ax.text(
            5.20, ly, m["label"],
            color=tcol, fontsize=7.5, fontweight="bold" if is_top else "medium",
            va="center", ha="left", clip_on=False
        )
        ax.text(
            6.88, ly, fmt_win(m["best"]["elo"]),
            color="#555550", fontsize=7.4, va="center", ha="right", clip_on=False
        )

    other_n = len(models) - len(labeled)
    ax.text(5.20, 3.8, f"{other_n} other models", fontsize=7.4, color="#777770", va="center", clip_on=False)
    ax.text(5.20, 0.2, "(<10% win rate or early stop)", fontsize=6.7, color="#999992", va="center", clip_on=False)

    ax.set_xlim(0, 5.02)
    ax.set_ylim(-2, 102)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(["0h", "1h", "2h", "3h", "4h", "5h"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Research budget spent (GPU-hours on NVIDIA L40S)", fontsize=8.3, color="#444444", labelpad=5)
    ax.set_ylabel("Win rate vs. 15-min PPO baseline", fontsize=8.3, color="#444444", labelpad=5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    fig.subplots_adjust(left=0.09, right=0.72, top=0.94, bottom=0.14)
    out = OUT_DIR / "fig2_polytopia_race.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: Compute Multiplier — Research included vs. Training only
# ---------------------------------------------------------------------------
def make_fig3():
    recipes = [r for r in budget_data["recipes"] if r["curve"]]
    curves = []
    for r in recipes:
        pts = []
        for p in r["curve"]:
            sh, extrap = starter_hours_for(p["elo"])
            cost = r["spentUsd"] + p["hours"] * RATE
            pts.append({
                "r": r,
                "hours": p["hours"],
                "elo": p["elo"],
                "cost": cost,
                "sh": sh,
                "extrap": extrap,
                "x": (sh * RATE) / cost,
                "xTrain": sh / p["hours"],
            })
        curves.append((r, pts))

    at8 = sorted([pts[-1] for _, pts in curves], key=lambda q: -q["x"])
    top8 = at8[0]
    second8 = at8[1]

    # Training-only best recipe per model (6-8 h geometric mean)
    WINDOW = {6, 7, 8}
    bars = []
    for mo in list(dict.fromkeys(r["model"] for r, _ in curves)):
        cands = []
        for r, pts in curves:
            if r["model"] == mo and WINDOW.issubset({p["hours"] for p in pts}):
                win = [p for p in pts if p["hours"] in WINDOW]
                p8 = [p for p in win if p["hours"] == 8][0]
                gm = math.exp(sum(math.log(p["xTrain"]) for p in win) / len(win))
                cands.append({"r": r, "x": gm, "extrap": any(p["extrap"] for p in win), "sh8": p8["sh"]})
        bars.append(max(cands, key=lambda b: b["x"]))
    bars.sort(key=lambda b: -b["x"])

    fig = plt.figure(figsize=(6.8, 3.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.52, 1.0], wspace=0.34, top=0.80, bottom=0.16, left=0.08, right=0.98)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Shared legend on top
    legend_models = [b["r"]["model"] for b in bars]
    handles = [
        mpl.lines.Line2D([0], [0], color=model_color(mo), lw=2.2, label=display_model(mo))
        for mo in legend_models
    ] + [mpl.lines.Line2D([0], [0], color=CMYK["k"], lw=1.8, label="Starter (1×)")]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.99),
        ncol=4, frameon=False, fontsize=7.4, handlelength=1.4, columnspacing=1.2
    )

    # --- Panel (a): Research cost included ---
    ax1.set_title("(a) Research cost included (x-offset)", fontsize=8.8, fontweight="bold", loc="left", pad=6)
    ax1.set_yscale("log", base=2)
    y_ticks = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]
    y_labels = ["1/16", "1/8", "1/4", "1/2", "1×", "2×", "4×", "8×", "16×"]
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels)
    ax1.set_ylim(1 / 18, 20)
    ax1.set_xlim(0, 50)
    ax1.set_xticks([0, 10, 20, 30, 40, 50])
    ax1.set_xticklabels(["$0", "$10", "$20", "$30", "$40", "$50"])

    for y in y_ticks:
        ax1.axhline(y, color="#eaeae6", lw=0.55, zorder=1)
    for x in [0, 10, 20, 30, 40, 50]:
        ax1.axvline(x, color="#eaeae6", lw=0.55, zorder=1)
    ax1.axhline(1.0, color=CMYK["k"], lw=1.4, zorder=4)
    ax1.text(49.2, 1.07, "Starter, 1×", ha="right", va="bottom", fontsize=7.3, fontweight="bold", color="#111111", zorder=5)

    for r, pts in curves:
        is_top = r["key"] == top8["r"]["key"]
        is_sec = r["key"] == second8["r"]["key"]
        col = model_color(r["model"])
        xs = [p["cost"] for p in pts]
        ys = [p["x"] for p in pts]
        lw = 2.0 if is_top else 1.5 if is_sec else 1.05
        alpha = 1.0 if is_top else 0.85 if is_sec else 0.25
        z = 6 if is_top else 5 if is_sec else 2
        ax1.plot(xs, ys, color=col, lw=lw, alpha=alpha, zorder=z)
        for p in pts:
            fc = "white" if p["extrap"] else col
            ec = col if p["extrap"] else "white"
            ax1.scatter(
                [p["cost"]], [p["x"]],
                s=18 if is_top else 12 if is_sec else 8,
                facecolor=fc, edgecolor=ec, lw=1.0 if p["extrap"] else 0.5,
                alpha=alpha, zorder=z + 1
            )

    ax1.annotate(
        f"Gemini 4 Neon ($8.58): ≈{round(top8['x'])}×",
        xy=(top8["cost"], top8["x"]),
        xytext=(top8["cost"] - 1.5, 13.8),
        ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=model_color("Gemini 4 Neon"),
        arrowprops=dict(arrowstyle="-", color=model_color("Gemini 4 Neon"), lw=0.7, shrinkA=0, shrinkB=3),
        zorder=8
    )
    ax1.annotate(
        f"GPT-6.2 Astra ($8.67): {second8['x']:.1f}×",
        xy=(second8["cost"], second8["x"]),
        xytext=(38.5, 2.85),
        ha="center", va="bottom", fontsize=7.0, color="#b84a83",
        arrowprops=dict(arrowstyle="-", color="#c75091", lw=0.7, shrinkA=0, shrinkB=3),
        zorder=8
    )

    ax1.set_xlabel(f"Total cost: research + training (${RATE:.2f}/h)", fontsize=8.0, color="#444444", labelpad=4)
    ax1.set_ylabel("Compute multiplier (log scale)", fontsize=8.0, color="#444444", labelpad=4)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # --- Panel (b): Training compute only ---
    ax2.set_title("(b) Training compute only (6–8 h mean)", fontsize=8.8, fontweight="bold", loc="left", pad=6)
    ax2.set_xscale("log", base=2)
    nb = len(bars)
    yb = np.arange(nb)[::-1]

    for t in [1, 2, 4, 8, 16]:
        ax2.axvline(t, color=CMYK["k"] if t == 1 else "#eaeae6", lw=1.2 if t == 1 else 0.55, zorder=1)

    for i, b in enumerate(bars):
        y = yb[i]
        col = model_color(b["r"]["model"])
        # Draw bar from x=1 to x=b["x"]
        ax2.barh(y, b["x"] - 1.0, left=1.0, height=0.56, color=col, edgecolor="none", zorder=3)
        val_str = f"≈{round(b['x'])}×" if b["x"] >= 9.5 else f"{b['x']:.1f}×"
        extra = " (extrap.)" if b["extrap"] else ""
        ax2.text(
            b["x"] * 1.11, y, f"{val_str}{extra}",
            va="center", ha="left", fontsize=7.5, fontweight="bold" if i == 0 else "medium",
            color="#111111"
        )

    ax2.set_xlim(0.95, 36)
    ax2.set_ylim(-0.6, nb - 0.4)
    ax2.set_xticks([1, 2, 4, 8, 16])
    ax2.set_xticklabels(["1×", "2×", "4×", "8×", "16×"])
    ax2.set_yticks(yb)
    ax2.set_yticklabels([display_model(b["r"]["model"]) for b in bars], fontsize=7.6)
    ax2.tick_params(axis="y", length=0, pad=4)
    ax2.set_xlabel("Training compute multiplier (log)", fontsize=8.0, color="#444444", labelpad=4)
    for sp in ["top", "right", "left"]:
        ax2.spines[sp].set_visible(False)

    out = OUT_DIR / "fig3_cost_multiplier.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4: Recipe Timelines ($8.70 Polytopia budget)
# ---------------------------------------------------------------------------
def make_fig4():
    sessions = sorted(rt_data["sessions"], key=lambda s: -s["elo8"])
    starter_elo = rt_data["starterElo8"]
    starter_idx = sum(1 for s in sessions if s["elo8"] > starter_elo)

    rows = sessions[:starter_idx] + [None] + sessions[starter_idx:]
    n = len(rows)

    def run_color(minutes):
        if minutes < 20:
            return "#6e97ea"
        if minutes < 45:
            return "#3459ba"
        return "#19337d"

    fig = plt.figure(figsize=(6.8, 3.55))
    ax = fig.add_axes([0.21, 0.14, 0.63, 0.72])
    ax_r = fig.add_axes([0.85, 0.14, 0.14, 0.72])

    T = 4.5
    ys = np.arange(n)[::-1]

    for t in [0, 1, 2, 3, 4]:
        ax.axvline(t, color="#eaeae6", lw=0.6, zorder=1)

    for i, s in enumerate(rows):
        y = ys[i]
        if s is None:
            # Starter no-op reference row
            ax.axhline(y + 0.44, color="#cfcfc8", lw=0.8, ls="--", zorder=2)
            ax.axhline(y - 0.44, color="#cfcfc8", lw=0.8, ls="--", zorder=2)
            ax.text(0.08, y, "No experiments: unmodified starter recipe trained 8 h",
                    va="center", ha="left", fontsize=7.6, style="italic", color="#666660")
            continue

        # Session wall-time backdrop
        bar_h = 0.62
        ax.add_patch(patches.FancyBboxPatch(
            (0.0, y - 0.24), s["wallH"], bar_h - 0.12,
            boxstyle="round,pad=0.0,rounding_size=0.03",
            facecolor="#f4f4f0", edgecolor="none", zorder=1
        ))
        ax.plot([s["wallH"], s["wallH"]], [y - 0.24, y + 0.26], color="#a8a8a0", lw=1.0, zorder=2)

        # Two parallel GPU slots (slot 0 top, slot 1 bottom)
        for r in s["runs"]:
            slot_y = y + 0.03 if r["slot"] == 0 else y - 0.21
            rect = patches.FancyBboxPatch(
                (r["s"], slot_y), max(0.04, r["e"] - r["s"]), 0.20,
                boxstyle="round,pad=0.0,rounding_size=0.02",
                facecolor=run_color(r["min"]), edgecolor="white", lw=0.5, zorder=3
            )
            ax.add_patch(rect)

        # Practice evals micro-rail
        for ev in s["evals"]:
            rect = patches.Rectangle(
                (ev["s"], y - 0.36), max(0.025, ev["e"] - ev["s"]), 0.075,
                facecolor="#b9b8b0", edgecolor="none", zorder=3
            )
            ax.add_patch(rect)

        # CPU debug jobs
        for c in s["cpu"]:
            rect = patches.Rectangle(
                (c["s"], y - 0.37), max(0.035, c["e"] - c["s"]), 0.095,
                facecolor="#d4a000", edgecolor="none", zorder=4
            )
            ax.add_patch(rect)

        if i < n - 1 and rows[i + 1] is not None:
            ax.axhline(y - 0.5, color="#f0f0ec", lw=0.5, zorder=0)

    ax.set_xlim(0, T)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["0 h", "1 h", "2 h", "3 h", "4 h"])
    ax.set_yticks(ys)
    ax.set_yticklabels(
        ["Starter recipe" if s is None else s["model"] for s in rows],
        fontsize=8.1, fontweight="medium"
    )
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlabel("Real session time (7 h wall limit · $8.70 GPU budget · 2 L40S slots)", fontsize=8.0, color="#444444", labelpad=5)
    ax.set_title("Two L40S GPU slots over real session time", fontsize=8.1, color="#555550", loc="left", pad=6)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)

    # Legend above plot
    legend_handles = [
        patches.Patch(facecolor="#6e97ea", label="GPU run (<20m)"),
        patches.Patch(facecolor="#3459ba", label="GPU run (20–45m)"),
        patches.Patch(facecolor="#19337d", label="GPU run (≥45m)"),
        patches.Patch(facecolor="#d4a000", label="CPU debug"),
        patches.Patch(facecolor="#b9b8b0", label="Practice eval"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.54, 0.99),
        ncol=5, frameon=False, fontsize=7.3, handlelength=1.1, columnspacing=1.0
    )

    # Right summary columns: Runs and 8 h Elo
    ax_r.set_xlim(0, 2.0)
    ax_r.set_ylim(-0.55, n - 0.45)
    ax_r.axis("off")
    ax_r.text(0.55, n - 0.28, "Runs", ha="right", va="bottom", fontsize=7.8, fontweight="bold", color="#444444")
    ax_r.text(1.85, n - 0.28, "8 h Elo", ha="right", va="bottom", fontsize=7.8, fontweight="bold", color="#444444")
    ax_r.plot([0.0, 1.95], [n - 0.42, n - 0.42], color="#d8d8d3", lw=0.7)

    for i, s in enumerate(rows):
        y = ys[i]
        if s is None:
            ax_r.text(0.55, y, "0", ha="right", va="center", fontsize=8.1, color="#666660")
            ax_r.text(1.85, y, str(starter_elo), ha="right", va="center", fontsize=8.1, color="#666660")
            ax_r.axhline(y + 0.44, color="#cfcfc8", lw=0.8, ls="--")
            ax_r.axhline(y - 0.44, color="#cfcfc8", lw=0.8, ls="--")
        else:
            is_neon = s["key"] == "neon-b3"
            ax_r.text(0.55, y, str(len(s["runs"])), ha="right", va="center", fontsize=8.1, color="#222222")
            ax_r.text(
                1.85, y, str(s["elo8"]), ha="right", va="center", fontsize=8.2,
                fontweight="bold" if is_neon else "medium",
                color="#2f55c4" if is_neon else "#222222"
            )

    out = OUT_DIR / "fig4_recipe_timelines.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 5: Forecasting skill on 181 Pokémon runs (Budget.tsx ForecastLead)
# ---------------------------------------------------------------------------
def make_fig5():
    ps = budget_data["pokeSkill"]
    models = sorted(ps["models"], key=lambda m: -m["skill"])
    rule = ps["rule"]

    TONE = {"above": CMYK["c"], "same": "#6b6b6b", "below": CMYK["m"]}

    def tone_of(m):
        if m["ci"][0] > 0:
            return "above"
        if m["ci"][1] < 0:
            return "below"
        return "same"

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    n = len(models)
    ys = np.arange(n)[::-1]

    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        ax.axvline(t, color=CMYK["k"] if t == 0 else "#eaeae6", lw=0.9 if t == 0 else 0.55, zorder=1)
    ax.axvline(rule, color=CMYK["k"], lw=1.1, ls=":", zorder=2)

    ax.text(0.0, n - 0.15, "blind guess", ha="left", va="bottom", fontsize=7.4, color="#555550")
    ax.text(rule, n - 0.15, f"ridge regression (+{rule:.2f})", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#222222")

    for i, m in enumerate(models):
        y = ys[i]
        c = TONE[tone_of(m)]
        ax.plot([m["skillCi"][0], m["skillCi"][1]], [y, y], color=c, alpha=0.35, lw=4.2, solid_capstyle="round", zorder=3)
        ax.scatter([m["skill"]], [y], s=40, color=c, edgecolor="white", lw=1.0, zorder=4)
        ax.text(m["skillCi"][1] + 0.011, y, f"+{m['skill']:.2f}", va="center", ha="left", fontsize=7.6, color="#222222")
        if i < n - 1:
            ax.axhline(y - 0.5, color="#f4f4f0", lw=0.5, zorder=0)

    ax.set_xlim(-0.01, 0.56)
    ax.set_ylim(-0.6, n - 0.35)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticklabels(["0", "+0.10", "+0.20", "+0.30", "+0.40", "+0.50"])
    ax.set_yticks(ys)
    ax.set_yticklabels([m["model"] for m in models], fontsize=8.1)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlabel("Relative forecast skill (0 = blind guess, 1 = perfect; 90% paired bootstrap CI)", fontsize=8.0, color="#444444", labelpad=5)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)

    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=TONE["above"], markersize=6.5, label="Reliably better than ridge baseline"),
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=TONE["same"], markersize=6.5, label="Indistinguishable from baseline"),
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=TONE["below"], markersize=6.5, label="Reliably worse than baseline"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.99), ncol=3, frameon=False, fontsize=7.3, columnspacing=1.0)

    fig.subplots_adjust(left=0.25, right=0.96, top=0.86, bottom=0.14)
    out = OUT_DIR / "fig5_forecast_skill.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Figure 6: Capability Timeline (Feb–Nov 2026)
# ---------------------------------------------------------------------------
def make_fig6():
    models = sorted(cap_data["models"], key=lambda r: (r["date"], r["plotMult"]))
    floor = cap_data["floorMult"]

    def parse_dt(iso):
        return datetime.strptime(iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Running-best monotone frontier
    by_date = {}
    for row in models:
        d = parse_dt(row["date"])
        if d not in by_date or row["plotMult"] > by_date[d]["plotMult"]:
            by_date[d] = row
    frontier = []
    best = -1e9
    for d, row in sorted(by_date.items()):
        if row["plotMult"] > best:
            best = row["plotMult"]
            frontier.append((d, best))

    fig, ax = plt.subplots(figsize=(6.8, 3.35))

    t_min = datetime(2026, 1, 22, tzinfo=timezone.utc)
    t_max = datetime(2026, 11, 25, tzinfo=timezone.utc)
    end_t = datetime(2026, 11, 18, tzinfo=timezone.utc)
    y_max = 3.5

    # Background bands
    ax.axhspan(1.0, y_max, facecolor="#f4fafd", zorder=0)
    ax.axhspan(0.0, floor, facecolor="#f6f6f3", zorder=0)

    for y in [0, 1, 2, 3]:
        ax.axhline(y, color="#222222" if y == 0 else "#e6e6e2", lw=0.7 if y == 0 else 0.55, zorder=1)

    ax.axhline(1.0, color=CMYK["cDeep"], alpha=0.6, lw=1.05, ls="--", zorder=2)
    ax.text(datetime(2026, 1, 28, tzinfo=timezone.utc), 1.07, "1× — unmodified PPO starter", fontsize=7.5, color=CMYK["cDeep"], fontweight="medium")

    ax.axhline(floor, color="#d4d4ce", lw=0.8, ls=":", zorder=2)
    ax.text(datetime(2026, 1, 28, tzinfo=timezone.utc), floor + 0.05, "<0.43× — weaker than 5-min starter", fontsize=7.2, color="#888882")

    # Step frontier curve + subtle shaded area
    fx, fy = [frontier[0][0]], [frontier[0][1]]
    for d, v in frontier[1:]:
        fx.extend([d, d])
        fy.extend([fy[-1], v])
    fx.append(end_t)
    fy.append(fy[-1])

    ax.fill_between(fx, 0, fy, color=CMYK["c"], alpha=0.06, zorder=2)
    ax.plot(fx, fy, color=CMYK["cDeep"], lw=1.7, alpha=0.88, zorder=3)

    # Scatter points (weak first, then >=1x on top)
    weak = [r for r in models if (r["finalMult"] or 0) < 1.0]
    strong = [r for r in models if (r["finalMult"] or 0) >= 1.0]

    ax.scatter(
        [parse_dt(r["date"]) for r in weak],
        [r["plotMult"] for r in weak],
        s=26, color="#989892", alpha=0.7, edgecolor="white", lw=0.9, zorder=4
    )
    ax.scatter(
        [parse_dt(r["date"]) for r in strong],
        [r["plotMult"] for r in strong],
        s=40, color=CMYK["cDeep"], alpha=0.95, edgecolor="white", lw=1.0, zorder=5
    )

    def fmt_m(x):
        if x is None:
            return f"<{floor:.2f}×"
        return f"{x:.1f}×" if x >= 0.95 else f"{x:.2f}×"

    # Inline callouts matching CapabilityTimeline.tsx (collision-free)
    inline_cfg = {
        "opus5": (0, 8, "center", "bottom"),
        "astra": (-7, 0, "right", "center"),
        "sol": (7, 0, "left", "center"),
        "opus": (6, 8, "center", "bottom"),
    }
    for row in models:
        if row["key"] in inline_cfg:
            dx, dy, ha, va = inline_cfg[row["key"]]
            mult_str = f" {fmt_m(row['finalMult'])}" if row["finalMult"] is not None else ""
            ax.annotate(
                f"{row['short']}{mult_str}",
                xy=(parse_dt(row["date"]), row["plotMult"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha, va=va, fontsize=7.4,
                fontweight="bold" if (row["finalMult"] or 0) >= 1.0 else "normal",
                color="#222222" if (row["finalMult"] or 0) >= 1.0 else "#666660",
                zorder=6
            )

    # Right-margin stacked labels for Nov frontier cluster
    margin_keys = {"sol62", "gem4n", "fable"}
    margin_rows = sorted([r for r in models if r["key"] in margin_keys], key=lambda r: -r["plotMult"])
    my_map = layout_labels([(r["key"], r["plotMult"]) for r in margin_rows], min_gap=0.32, min_y=1.5, max_y=3.25)

    x_end_num = mdates.date2num(t_max)
    for r in margin_rows:
        cx = mdates.date2num(parse_dt(r["date"]))
        cy = r["plotMult"]
        ly = my_map[r["key"]]
        ax.plot([cx + 4, x_end_num + 5], [cy, ly], color=CMYK["cDeep"], lw=0.75, alpha=0.45, clip_on=False)
        ax.text(
            x_end_num + 8, ly, f"{r['short']} {fmt_m(r['finalMult'])}",
            va="center", ha="left", fontsize=7.5, fontweight="bold", color="#222222", clip_on=False
        )

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(-0.04, y_max)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0×", "1×", "2×", "3×"])

    tick_dates = [
        datetime(2026, 2, 1, tzinfo=timezone.utc),
        datetime(2026, 4, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        datetime(2026, 10, 1, tzinfo=timezone.utc),
        datetime(2026, 11, 18, tzinfo=timezone.utc),
    ]
    ax.set_xticks(tick_dates)
    ax.set_xticklabels(["Feb", "Apr", "Jun", "Aug", "Oct", "Nov 2026"])
    ax.set_ylabel("Compute multiplier vs. 15-min PPO starter", fontsize=8.2, color="#444444", labelpad=5)
    ax.set_xlabel("Model release date (February–November 2026)", fontsize=8.2, color="#444444", labelpad=5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    fig.subplots_adjust(left=0.09, right=0.80, top=0.94, bottom=0.16)
    out = OUT_DIR / "fig6_capability_timeline.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    print("All figures generated successfully.")
