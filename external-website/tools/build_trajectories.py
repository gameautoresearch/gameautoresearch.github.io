"""Build sanitized, reader-ready trajectory JSONs for the blog's Trajectory Viewer.

Reads ATIF `trajectory.json` files (plus `NOTES.md` and `session-result.json` where present)
from local benchmark session archives and emits:
  - `src/posts/autoresearch/trajectories-manifest.json` (lightweight index imported by the app)
  - `public/trajectories/<id>.json` (full trajectory payloads lazy-fetched on selection)

Preserves all model reasoning (`reasoning_content`), `Analysis:` / `Plan:` blocks, commands,
and cleaned terminal outputs so readers can inspect what the models were thinking and doing.

Usage:
    python3 external-website/tools/build_trajectories.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_OUT = SITE_ROOT / "src/posts/autoresearch/trajectories-manifest.json"
TRAJ_DIR = SITE_ROOT / "public/trajectories"

HOME = Path.home()
BENCH = HOME / "repos/game-autoresearch-bench"
DATA = HOME / "repos/game-autoresearch-data"
POKE = HOME / "repos/gab-poke-sub"

# Curated trajectories across the main benchmark tasks featured in the post.
SPECS = [
    # --- Polytopia · Empty File (5h, Figure 2) ---
    {
        "id": "poly-empty-fable",
        "taskGroup": "poly-empty",
        "taskLabel": "Polytopia · Empty File (5 h)",
        "model": "Claude Fable 5.2",
        "outcome": "220 best / 180 final Elo",
        "summary": "Built a PPO trainer from an empty file with full chain-of-thought recorded across 271 steps.",
        "sessionDir": DATA / "extracted/polytopia-frontier6-20261118/runs/cloud/polytopia-frontier6-20261118/polytopia/fable52/a4",
        "budget": "5 GPU-h",
    },
    {
        "id": "poly-empty-neon",
        "taskGroup": "poly-empty",
        "taskLabel": "Polytopia · Empty File (5 h)",
        "model": "Gemini 4 Neon",
        "outcome": "240 best / 186 final Elo",
        "summary": "Highest-Elo model in this archive on the empty-file Polytopia task; verbose step-by-step analysis across 22 training runs.",
        "sessionDir": DATA / "extracted/polytopia-frontier6-20261118/runs/cloud/polytopia-frontier6-20261118/polytopia/gem4neon/a1",
        "budget": "5 GPU-h",
    },
    {
        "id": "poly-empty-opus",
        "taskGroup": "poly-empty",
        "taskLabel": "Polytopia · Empty File (5 h)",
        "model": "Claude Opus 5.5",
        "outcome": "30 best / 18 final Elo",
        "summary": "Full chain-of-thought across 266 steps iterating through 23 empty-starter Polytopia experiments.",
        "sessionDir": DATA / "extracted/polytopia-frontier6-20261118/runs/cloud/polytopia-frontier6-20261118/polytopia/opus55/a4",
        "budget": "5 GPU-h",
    },
    {
        "id": "poly-empty-dsflash",
        "taskGroup": "poly-empty",
        "taskLabel": "Polytopia · Empty File (5 h)",
        "model": "DeepSeek V4.1 Flash",
        "outcome": "−39 best / −213 final Elo",
        "summary": "Complete internal monologue on every step (487 thinking traces) debugging training from scratch.",
        "sessionDir": DATA / "extracted/polytopia-empty15-20261118/runs/cloud/polytopia-empty15-20261118/polytopia/dsflash/a1",
        "budget": "5 GPU-h",
    },
    {
        "id": "poly-empty-kimi",
        "taskGroup": "poly-empty",
        "taskLabel": "Polytopia · Empty File (5 h)",
        "model": "Kimi K3.1 Pro",
        "outcome": "−55 best / −35 final Elo",
        "summary": "Systematic batch-size and hyperparameter sweeps from an empty Polytopia file.",
        "sessionDir": DATA / "extracted/polytopia-empty15-20261118/runs/cloud/polytopia-empty15-20261118/polytopia/kimi-k31/a1",
        "budget": "5 GPU-h",
    },
    {
        "id": "poly-empty-glm",
        "taskGroup": "poly-empty",
        "taskLabel": "Polytopia · Empty File (5 h)",
        "model": "GLM-5.3",
        "outcome": "−647 best / −717 final Elo",
        "summary": "318 thinking traces showing how an agent struggles to climb past early local optima.",
        "sessionDir": DATA / "extracted/polytopia-empty15-20261118/runs/cloud/polytopia-empty15-20261118/polytopia/glm53/a1",
        "budget": "5 GPU-h",
    },
    # --- Polytopia · $8.70 Budget Recipe (Figure 4 & Cost Multiplier) ---
    {
        "id": "poly-budget-astra-t2",
        "taskGroup": "poly-budget",
        "taskLabel": "Polytopia · $8.70 Recipe",
        "model": "GPT-6.2 Astra",
        "outcome": "344 Elo (8 h) · #2 multiplier",
        "summary": "2-hour wall-limit $8.70 recipe session that achieved the second-highest compute savings in the benchmark.",
        "sessionDir": BENCH / "runs/nanny/poly-budget-20261118/volume/campaign/polytopia-astra62-t2",
        "budget": "$8.70 · 2 h limit",
    },
    {
        "id": "poly-budget-opus-b3",
        "taskGroup": "poly-budget",
        "taskLabel": "Polytopia · $8.70 Recipe",
        "model": "Claude Opus 5.5",
        "outcome": "318 Elo (8 h)",
        "summary": "Tested score-difference shaping, wider networks, and learning-rate ablations under the $8.70 budget (Figure 5).",
        "sessionDir": BENCH / "runs/nanny/poly-budget-20261118/volume/campaign/polytopia-opus-b3",
        "budget": "$8.70 · 7 h limit",
    },
    {
        "id": "poly-budget-astra-b3",
        "taskGroup": "poly-budget",
        "taskLabel": "Polytopia · $8.70 Recipe",
        "model": "GPT-6.2 Astra",
        "outcome": "316 Elo (8 h)",
        "summary": "Explored additive action log-priors, tactical flags, and a 48-wide stem under the $8.70 budget (Figure 5).",
        "sessionDir": BENCH / "runs/nanny/poly-budget-20261118/volume/campaign/polytopia-astra62-b3",
        "budget": "$8.70 · 7 h limit",
    },
    {
        "id": "poly-budget-sol-b3",
        "taskGroup": "poly-budget",
        "taskLabel": "Polytopia · $8.70 Recipe",
        "model": "GPT-6.2 Sol",
        "outcome": "293 Elo (8 h)",
        "summary": "Three ~1-hour experiments moving from shared-trunk 2×32 to spatial-v3 with turn-based GAE (Figure 5).",
        "sessionDir": BENCH / "runs/nanny/poly-budget-20261118/volume/campaign/polytopia-sol62-b3",
        "budget": "$8.70 · 7 h limit",
    },
    # --- Pokémon 3v3 ---
    {
        "id": "poke-dspro-a2",
        "taskGroup": "pokemon",
        "taskLabel": "Pokémon 3v3",
        "model": "DeepSeek V4 Pro",
        "outcome": "+7.0 final / +14.1 best lift",
        "summary": "Showcase policy (a2): 2-layer SiLU MLPs with team-HP shaping and full thinking traces on 356 steps.",
        "sessionDir": POKE / "reports/pokemon/poke3v3-batch-2026-11-14/sessions/deepseek-v4pro/a2",
        "budget": "5 GPU-h",
    },
    {
        "id": "poke-inkling-b2",
        "taskGroup": "pokemon",
        "taskLabel": "Pokémon 3v3",
        "model": "Inkling 2 Small",
        "outcome": "+12.5 final lift",
        "summary": "Showcase policy (b2): wider 128w MLP with doubled entropy coefficient; includes 68 raw thinking traces.",
        "sessionDir": POKE / "reports/pokemon/poke3v3-batch-2026-11-14/sessions/inkling2s/b2",
        "budget": "5 GPU-h",
    },
    {
        "id": "poke-glm-a1",
        "taskGroup": "pokemon",
        "taskLabel": "Pokémon 3v3",
        "model": "GLM-5.3",
        "outcome": "−0.5 final / +2.3 best lift",
        "summary": "Explored larger cross-attention architectures on Pokémon 3v3 with 81 raw thinking traces.",
        "sessionDir": POKE / "reports/pokemon/poke3v3-batch-2026-11-14/sessions/glm53/a1",
        "budget": "5 GPU-h",
    },
    # --- Generals ---
    {
        "id": "gen-sol",
        "taskGroup": "generals",
        "taskLabel": "Generals",
        "model": "GPT-6.1 Sol",
        "outcome": "5 h research session",
        "summary": "Iterative Generals.io RL research session inspecting rollouts, editing policies, and launching leases.",
        "sessionDir": BENCH / "runs/nanny/suite-20261113/collected/campaign/generals-sol-1",
        "budget": "5 GPU-h",
    },
    {
        "id": "gen-astra",
        "taskGroup": "generals",
        "taskLabel": "Generals",
        "model": "GPT-6 Astra",
        "outcome": "5 h research session",
        "summary": "Concise 79-step Generals.io research session balancing fast experiments and evaluations.",
        "sessionDir": BENCH / "runs/nanny/suite-20261113/collected/campaign/generals-astra-1",
        "budget": "5 GPU-h",
    },
    {
        "id": "gen-flash",
        "taskGroup": "generals",
        "taskLabel": "Generals",
        "model": "DeepSeek V4 Flash",
        "outcome": "5 h research session",
        "summary": "192 full chain-of-thought reasoning traces across a 5-hour Generals.io training session.",
        "sessionDir": BENCH / "runs/nanny/suite-20261113/collected/campaign/generals-flash-1",
        "budget": "5 GPU-h",
    },
]

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[@-Z\\-_]")
SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|ak-[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]{20,}|/Users/[A-Za-z0-9._-]+)"
)
MAX_OUTPUT_CHARS = 24000


def parse_iso(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def find_trajectory(session_dir: Path) -> Path:
    harbor = session_dir / "harbor"
    cands = [p for p in harbor.rglob("agent/trajectory.json") if "/steps/" not in str(p)]
    if not cands:
        cands = list(harbor.rglob("agent/trajectory.json"))
    if not cands:
        raise FileNotFoundError(f"No trajectory.json under {harbor}")
    return cands[0]


def clean_terminal(raw: str) -> str:
    if not raw:
        return ""
    text = ANSI_RE.sub("", raw)
    # Strip boilerplate Terminus screen wrappers while keeping error/warning notices intact.
    text = re.sub(r"^(?:New Terminal Output|Current Terminal Screen):\s*\n?", "", text)
    # Trim trailing whitespace per line and collapse tmux empty-screen padding.
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    out: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln == "":
            blank_run += 1
            if blank_run <= 1:
                out.append("")
        else:
            blank_run = 0
            out.append(ln)
    cleaned = "\n".join(out)
    if len(cleaned) > MAX_OUTPUT_CHARS:
        omitted = len(cleaned) - MAX_OUTPUT_CHARS
        cleaned = cleaned[:MAX_OUTPUT_CHARS].rstrip() + f"\n\n… [output truncated: {omitted:,} more chars]"
    return cleaned


def split_analysis_plan(msg: str) -> tuple[str | None, str | None, str | None]:
    if not msg:
        return None, None, None
    s = msg.strip()
    m = re.match(r"^Analysis:\s*(.*?)(?:\n+Plan:\s*(.*))?$", s, re.DOTALL)
    if m:
        analysis = m.group(1).strip() or None
        plan = m.group(2).strip() if m.group(2) else None
        return analysis, plan, None
    m2 = re.match(r"^Plan:\s*(.*)$", s, re.DOTALL)
    if m2:
        return None, m2.group(1).strip() or None, None
    return None, None, s


def strip_heredocs(cmd_blob: str) -> str:
    """Keep shell command lines while omitting heredoc payloads so mentions inside NOTES/code don't misfire tags."""
    out: list[str] = []
    eof_marker: str | None = None
    for line in cmd_blob.splitlines():
        if eof_marker is not None:
            if line.strip() == eof_marker:
                eof_marker = None
            continue
        out.append(line)
        m = re.search(r"<<-?\s*['\"]?([A-Za-z0-9_]+)['\"]?", line)
        if m:
            eof_marker = m.group(1)
    return "\n".join(out)


def classify_step(
    commands: list[dict],
    reasoning: str | None,
    analysis: str | None,
    output: str,
) -> tuple[list[str], str | None]:
    """Return (tags, milestone_label_if_any)."""
    tags: list[str] = []
    if reasoning:
        tags.append("think")

    raw_blob = "\n".join(c.get("k", "") for c in commands)
    cmd_blob = strip_heredocs(raw_blob)
    cmd_stripped = cmd_blob.strip()
    has_complete = any(c.get("fn") == "mark_task_complete" for c in commands)

    # Detect submits / finalization
    is_submit = has_complete or bool(re.search(r"(?:^|[;&|\n]\s*)ar\s+(?:submit|finalize)\b", cmd_blob))
    # Detect training launches (training/launch.py, ar run, ar lease, or ar exec running train)
    is_train = bool(
        re.search(r"\bpython3?\s+(?:\./)?training/launch\.py\b", cmd_blob)
        or re.search(r"(?:^|[;&|\n]\s*)ar\s+(?:run|lease|train|start)\b", cmd_blob)
        or re.search(r"(?:^|[;&|\n]\s*)ar\s+exec\b[^\n]*(?:train|ppo|run_)", cmd_blob)
    )
    # Detect evaluations / practice
    is_eval = bool(re.search(r"(?:^|[;&|\n]\s*)ar\s+(?:eval|practice)\b", cmd_blob))
    # Detect NOTES.md updates
    is_notes = bool(re.search(r"NOTES\.md", cmd_blob) and re.search(r"(?:cat\s*<<|>>|>|tee|write_text|open\(|echo\b)", cmd_blob))
    # Detect code / config edits
    is_edit = bool(
        re.search(
            r"(?:cat\s*<<|tee\s+|sed\s+-i|apply_patch|write_text\(|open\([^)]*['\"][wa])",
            cmd_blob,
        )
        or re.search(r"python3?\s+-\s*<<", cmd_blob)
    )
    # Don't count pure NOTES.md writes as code edits unless another file is touched
    if is_edit and is_notes and not re.search(r"(?:\.py|\.json|\.sh|\.toml|\.yaml)", cmd_blob):
        is_edit = False

    is_error = bool(
        "Previous response had parsing errors:" in output
        or re.search(r"\b(?:Traceback \(most recent call last\)|SIGFPE|Segmentation fault|ModuleNotFoundError)\b", output)
    )

    if is_submit:
        tags.append("submit")
    if is_train:
        tags.append("train")
    if is_eval:
        tags.append("eval")
    if is_edit:
        tags.append("edit")
    if is_notes:
        tags.append("notes")
    if is_error:
        tags.append("error")

    # Classify routine wait/poll steps (only when there isn't substantive thinking or action)
    pure_wait_cmd = (
        cmd_stripped == ""
        or all(
            bool(
                re.match(
                    r"^(?:ar\s+(?:wait|poll|status|budget|log)\b|sleep\b|echo\b|date\b|ps\b|ls\b|tail\b)",
                    part.strip(),
                )
            )
            for part in re.split(r"[;&\n]+", cmd_stripped)
            if part.strip()
        )
    )
    has_deep_thought = len(reasoning or "") >= 450 or len(analysis or "") >= 360
    if pure_wait_cmd and not (is_submit or is_train or is_eval or is_edit or is_notes or has_deep_thought):
        tags.append("wait")

    # Milestone label for the timeline scrubber
    milestone: str | None = None
    if is_submit:
        m = re.search(r"ar\s+(submit|finalize)(?:\s+([^\s;&|\n]+))?", cmd_blob)
        milestone = f"Submit {m.group(2)}" if (m and m.group(2)) else "Submit / finalize"
    elif is_train:
        m = re.search(r"training/launch\.py\s+([^\s;&|\n]+)", cmd_blob)
        if not m:
            m = re.search(r"ar\s+(?:run|lease)\s+(?:--name\s+)?([^\s;&|\n]+)", cmd_blob)
        if not m:
            m = re.search(r"--name[=\s]+([^\s;&|\n]+)", cmd_blob)
        milestone = f"Train {m.group(1)}" if m else "Launch training"
    elif is_eval:
        m = re.search(r"ar\s+(?:eval|practice)\s+([^\s;&|\n]+)", cmd_blob)
        milestone = f"Eval {m.group(1)}" if m else "Run practice eval"
    elif is_edit:
        m = re.search(r"([a-zA-Z0-9_./-]+\.(?:py|json))", cmd_blob)
        milestone = f"Edit {m.group(1)}" if m else "Edit code"

    return tags, milestone


def check_no_secrets(payload_str: str, traj_id: str) -> None:
    hits = SECRET_RE.findall(payload_str)
    if hits:
        raise ValueError(f"Potential sensitive token in {traj_id}: {hits[:3]}")


def build_one(spec: dict) -> tuple[dict, dict]:
    sdir: Path = spec["sessionDir"]
    traj_path = find_trajectory(sdir)
    raw = json.loads(traj_path.read_text())

    raw_steps = raw.get("steps", [])
    start_ts = parse_iso(raw_steps[0].get("timestamp")) if raw_steps else None

    prompt_text = ""
    steps_out: list[dict] = []
    milestones: list[dict] = []

    thinking_steps = 0
    thinking_chars = 0
    analysis_chars = 0
    counts = {"train": 0, "eval": 0, "edit": 0, "submit": 0, "wait": 0}

    for st in raw_steps:
        src = st.get("source")
        step_id = st.get("step_id")
        ts = parse_iso(st.get("timestamp"))
        elapsed = max(0, int(round(ts - start_ts))) if (ts is not None and start_ts is not None) else None

        if src == "user" and not prompt_text:
            prompt_text = str(st.get("message") or "").strip()
            continue

        reasoning = (st.get("reasoning_content") or "").strip() or None
        analysis, plan, raw_msg = split_analysis_plan(str(st.get("message") or ""))

        cmds: list[dict] = []
        for tc in st.get("tool_calls") or []:
            fn = tc.get("function_name") or ""
            args = tc.get("arguments") or {}
            if fn == "bash_command":
                entry: dict = {"k": str(args.get("keystrokes") or "")}
                dur = args.get("duration")
                if dur is not None:
                    entry["d"] = dur
                cmds.append(entry)
            else:
                cmds.append({"fn": fn, "k": json.dumps(args) if args else ""})

        obs_parts: list[str] = []
        obs = st.get("observation") or {}
        for r in obs.get("results") or []:
            c = str(r.get("content") or "")
            if c.strip():
                obs_parts.append(clean_terminal(c))
        output = "\n\n".join(p for p in obs_parts if p)

        tags, milestone = classify_step(cmds, reasoning, analysis, output)

        if reasoning:
            thinking_steps += 1
            thinking_chars += len(reasoning)
        if analysis:
            analysis_chars += len(analysis)
        if plan:
            analysis_chars += len(plan)
        if raw_msg:
            analysis_chars += len(raw_msg)
        for k in counts:
            if k in tags:
                counts[k] += 1

        step_obj: dict = {
            "id": step_id,
            "src": src,
            "t": elapsed,
            "tags": tags,
        }
        if reasoning:
            step_obj["r"] = reasoning
        if analysis:
            step_obj["a"] = analysis
        if plan:
            step_obj["p"] = plan
        if raw_msg:
            step_obj["m"] = raw_msg
        if cmds:
            step_obj["c"] = cmds
        if output:
            step_obj["o"] = output

        metrics = st.get("metrics") or {}
        ptok = metrics.get("prompt_tokens")
        ctok = metrics.get("completion_tokens")
        if ptok is not None or ctok is not None:
            step_obj["tok"] = [ptok or 0, ctok or 0]

        idx = len(steps_out)
        steps_out.append(step_obj)

        if milestone:
            kind = (
                "submit" if "submit" in tags
                else "train" if "train" in tags
                else "eval" if "eval" in tags
                else "edit"
            )
            milestones.append({
                "i": idx,
                "id": step_id,
                "t": elapsed,
                "kind": kind,
                "label": milestone,
            })

    # Optional NOTES.md & session-result.json
    notes_path = sdir / "NOTES.md"
    notes_text = notes_path.read_text().strip() if notes_path.exists() else None
    res_path = sdir / "session-result.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    final_metrics = raw.get("final_metrics") or {}

    wall_s = None
    if steps_out and steps_out[-1].get("t"):
        wall_s = steps_out[-1]["t"]
    elif res.get("wall_seconds"):
        wall_s = int(round(res["wall_seconds"]))

    spent_usd = res.get("budget", {}).get("spent_usd")
    if spent_usd is not None:
        spent_usd = round(float(spent_usd), 2)

    stats = {
        "steps": len(steps_out),
        "thinkingSteps": thinking_steps,
        "thinkingChars": thinking_chars,
        "analysisChars": analysis_chars,
        "wallSec": wall_s,
        "spentUsd": spent_usd,
        "promptTokens": final_metrics.get("total_prompt_tokens"),
        "completionTokens": final_metrics.get("total_completion_tokens"),
        "trainSteps": counts["train"],
        "evalSteps": counts["eval"],
        "editSteps": counts["edit"],
        "submitSteps": counts["submit"],
        "waitSteps": counts["wait"],
        "hasNotes": bool(notes_text),
    }

    meta = {
        "id": spec["id"],
        "taskGroup": spec["taskGroup"],
        "taskLabel": spec["taskLabel"],
        "model": spec["model"],
        "outcome": spec["outcome"],
        "budget": spec["budget"],
        "summary": spec["summary"],
        "stats": stats,
    }

    full_payload = {
        **meta,
        "prompt": prompt_text,
        "notes": notes_text,
        "milestones": milestones,
        "steps": steps_out,
    }

    encoded = json.dumps(full_payload, ensure_ascii=False, separators=(",", ":"))
    check_no_secrets(encoded, spec["id"])
    return meta, full_payload


def main() -> None:
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    # Remove any stale JSONs from previous runs
    for old in TRAJ_DIR.glob("*.json"):
        old.unlink()

    manifest = []
    total_bytes = 0
    for spec in SPECS:
        meta, payload = build_one(spec)
        out_path = TRAJ_DIR / f"{spec['id']}.json"
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        out_path.write_text(encoded)
        kb = len(encoded.encode("utf-8")) / 1024
        total_bytes += len(encoded.encode("utf-8"))
        manifest.append(meta)
        st = meta["stats"]
        print(
            f"  {spec['id']:22s} {kb:7.1f} KB | steps={st['steps']:3d} "
            f"(think={st['thinkingSteps']:3d}, train={st['trainSteps']:2d}, eval={st['evalSteps']:2d}, "
            f"edit={st['editSteps']:3d}, wait={st['waitSteps']:3d}) notes={'Y' if st['hasNotes'] else 'N'}"
        )

    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"\nWrote {len(manifest)} trajectories ({total_bytes / (1024 * 1024):.2f} MB total) -> {TRAJ_DIR}")
    print(f"Wrote manifest -> {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
