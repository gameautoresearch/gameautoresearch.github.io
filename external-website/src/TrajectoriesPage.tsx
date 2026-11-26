// Standalone trajectory reader: /trajectories and /trajectories/<id>.
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { modelStyle } from "./posts/autoresearch/colors";
import manifestRaw from "./posts/autoresearch/trajectories-manifest.json";
import "./trajectories.css";

type StepTag = "think" | "train" | "eval" | "edit" | "submit" | "notes" | "wait" | "error";

type TrajectoryMeta = {
  id: string;
  taskGroup: "poly-empty" | "poly-budget" | "pokemon" | "generals" | "hex13" | "hex9";
  taskLabel: string;
  model: string;
  outcome: string;
  budget: string;
  summary: string;
  stats: {
    steps: number;
    thinkingSteps: number;
    thinkingChars: number;
    wallSec: number | null;
    spentUsd: number | null;
    trainSteps: number;
    evalSteps: number;
    editSteps: number;
    waitSteps: number;
    hasNotes: boolean;
  };
};

type StepCommand = { k?: string; d?: number; fn?: string };

type TrajectoryStep = {
  id: number;
  src: string;
  t: number | null;
  tags: StepTag[];
  r?: string;
  a?: string;
  p?: string;
  m?: string;
  c?: StepCommand[];
  o?: string;
};

type TrajectoryData = TrajectoryMeta & {
  prompt: string;
  notes: string | null;
  steps: TrajectoryStep[];
};

const MANIFEST = manifestRaw as TrajectoryMeta[];

const TASK_ORDER: { id: TrajectoryMeta["taskGroup"]; label: string }[] = [
  { id: "poly-empty", label: "Polytopia · Empty File (5 h)" },
  { id: "poly-budget", label: "Polytopia · $8.70 Recipe" },
  { id: "pokemon", label: "Pokémon 3v3" },
  { id: "generals", label: "Generals" },
  { id: "hex13", label: "Hex 13×13 · Recipe (3 h)" },
  { id: "hex9", label: "Hex 9×9 (5 h)" },
];

const BADGE_LABEL: Partial<Record<StepTag, string>> = {
  train: "Train",
  eval: "Eval",
  edit: "Edit",
  submit: "Submit",
  notes: "Notes",
  error: "Error",
};

const PAGE_SIZE = 40;
const CMD_PREVIEW_LINES = 16;
const OUT_PREVIEW_LINES = 8;

function fmtElapsed(sec: number | null): string {
  if (sec == null || !Number.isFinite(sec)) return "";
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rem = s % 60;
  if (h > 0) return `+${h}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `+${m}m ${String(rem).padStart(2, "0")}s`;
  return `+${rem}s`;
}

function formatCommands(cmds: StepCommand[]): string {
  return cmds
    .map((c) => {
      if (c.fn && c.fn !== "bash_command") return `[${c.fn}]${c.k ? ` ${c.k}` : ""}`;
      const k = (c.k ?? "").replace(/\n+$/, "");
      if (!k.trim()) return c.d ? `(wait ${c.d}s)` : "(wait)";
      return k
        .split("\n")
        .map((line, i) => (i === 0 ? `$ ${line}` : `  ${line}`))
        .join("\n");
    })
    .join("\n");
}

function stepMatches(step: TrajectoryStep, q: string): boolean {
  if (!q) return true;
  if (step.r?.toLowerCase().includes(q)) return true;
  if (step.a?.toLowerCase().includes(q)) return true;
  if (step.p?.toLowerCase().includes(q)) return true;
  if (step.m?.toLowerCase().includes(q)) return true;
  if (step.o?.toLowerCase().includes(q)) return true;
  return Boolean(step.c?.some((c) => (c.k || "").toLowerCase().includes(q)));
}

function highlight(text: string, q: string) {
  if (!q) return text;
  const re = new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
  const parts = text.split(re);
  if (parts.length === 1) return text;
  return parts.map((p, i) =>
    i % 2 === 1 ? (
      <mark key={i} className="tj-mark">
        {p}
      </mark>
    ) : (
      <Fragment key={i}>{p}</Fragment>
    ),
  );
}

type DisplayItem =
  | { kind: "step"; step: TrajectoryStep }
  | { kind: "polls"; id: string; steps: TrajectoryStep[] };

function StepView({
  step,
  query,
  showOutputs,
}: {
  step: TrajectoryStep;
  query: string;
  showOutputs: boolean;
}) {
  const [cmdExpanded, setCmdExpanded] = useState(false);
  const [outExpanded, setOutExpanded] = useState(false);

  const badges = step.tags.filter((t) => t in BADGE_LABEL);
  const cmdText = step.c?.length ? formatCommands(step.c) : "";
  const cmdLines = cmdText ? cmdText.split("\n") : [];
  const cmdClipped = !cmdExpanded && cmdLines.length > CMD_PREVIEW_LINES;

  const outLines = step.o ? step.o.split("\n") : [];
  const outClipped = !outExpanded && outLines.length > OUT_PREVIEW_LINES;

  return (
    <article className="tj-step">
      <header className="tj-step-head">
        <span className="tj-step-num">Step {step.id}</span>
        {step.t != null && <span className="tj-step-time">{fmtElapsed(step.t)}</span>}
        {badges.length > 0 && (
          <span className="tj-badges">
            {badges.map((t) => (
              <span key={t} className={`tj-badge tj-badge-${t}`}>
                {BADGE_LABEL[t]}
              </span>
            ))}
          </span>
        )}
      </header>

      {step.r && (
        <div className="tj-think">
          <div className="tj-label">Thinking</div>
          <div className="tj-text">{highlight(step.r, query)}</div>
        </div>
      )}

      {(step.a || step.p || step.m) && (
        <div className="tj-ap">
          {step.a && (
            <div>
              <span className="tj-label">Analysis</span>
              <p className="tj-text">{highlight(step.a, query)}</p>
            </div>
          )}
          {step.p && (
            <div className="tj-plan">
              <span className="tj-label">Plan</span>
              <p className="tj-text">{highlight(step.p, query)}</p>
            </div>
          )}
          {step.m && (
            <div>
              <span className="tj-label">Response</span>
              <p className="tj-text">{highlight(step.m, query)}</p>
            </div>
          )}
        </div>
      )}

      {cmdText && (
        <div className="tj-code">
          <pre>
            {highlight(
              cmdClipped ? cmdLines.slice(0, CMD_PREVIEW_LINES).join("\n") : cmdText,
              query,
            )}
          </pre>
          {cmdClipped && (
            <button type="button" className="tj-more-inline" onClick={() => setCmdExpanded(true)}>
              Show all {cmdLines.length} lines
            </button>
          )}
        </div>
      )}

      {showOutputs && step.o && (
        <div className="tj-out">
          <div className="tj-label">Output</div>
          <pre>
            {highlight(
              outClipped ? outLines.slice(0, OUT_PREVIEW_LINES).join("\n") : step.o,
              query,
            )}
          </pre>
          {outClipped && (
            <button type="button" className="tj-more-inline" onClick={() => setOutExpanded(true)}>
              Show all {outLines.length} lines
            </button>
          )}
        </div>
      )}
    </article>
  );
}

export default function TrajectoriesPage({ initialId }: { initialId?: string }) {
  const validInitial = MANIFEST.find((m) => m.id === initialId)?.id ?? MANIFEST[0].id;
  const [selectedId, setSelectedId] = useState(validInitial);
  const [data, setData] = useState<TrajectoryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const cacheRef = useRef<Map<string, TrajectoryData>>(new Map());

  const [queryInput, setQueryInput] = useState("");
  const [hidePolls, setHidePolls] = useState(true);
  const [showOutputs, setShowOutputs] = useState(true);
  const [expandedPolls, setExpandedPolls] = useState<Record<string, boolean>>({});
  const [limit, setLimit] = useState(PAGE_SIZE);

  const meta = MANIFEST.find((m) => m.id === selectedId) ?? MANIFEST[0];
  const base = import.meta.env.BASE_URL;

  useEffect(() => {
    document.title = `${meta.model} · ${meta.taskLabel} · Trajectories`;
  }, [meta]);

  useEffect(() => {
    const cached = cacheRef.current.get(selectedId);
    if (cached) {
      setData(cached);
      setLoading(false);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`${base}trajectories/${selectedId}.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<TrajectoryData>;
      })
      .then((json) => {
        if (cancelled) return;
        cacheRef.current.set(selectedId, json);
        setData(json);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to load");
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId, base]);

  const onSelect = (id: string) => {
    setSelectedId(id);
    setQueryInput("");
    setExpandedPolls({});
    setLimit(PAGE_SIZE);
    window.history.replaceState(null, "", `${base}trajectories/${id}`);
  };

  const query = queryInput.trim().toLowerCase();

  const { items, matchedCount } = useMemo(() => {
    if (!data) return { items: [] as DisplayItem[], matchedCount: 0 };
    const matched = query ? data.steps.filter((s) => stepMatches(s, query)) : data.steps;

    if (!hidePolls || query) {
      return {
        items: matched.map((step) => ({ kind: "step" as const, step })),
        matchedCount: matched.length,
      };
    }

    const out: DisplayItem[] = [];
    let buf: TrajectoryStep[] = [];
    const flush = () => {
      if (!buf.length) return;
      if (buf.length === 1) {
        out.push({ kind: "step", step: buf[0] });
      } else {
        const gid = `${buf[0].id}-${buf[buf.length - 1].id}`;
        if (expandedPolls[gid]) {
          for (const s of buf) out.push({ kind: "step", step: s });
        } else {
          out.push({ kind: "polls", id: gid, steps: buf });
        }
      }
      buf = [];
    };

    for (const s of matched) {
      if (s.tags.includes("wait")) buf.push(s);
      else {
        flush();
        out.push({ kind: "step", step: s });
      }
    }
    flush();
    return { items: out, matchedCount: matched.length };
  }, [data, query, hidePolls, expandedPolls]);

  const visible = items.slice(0, limit);
  const remaining = items.length - visible.length;
  const accent = modelStyle(meta.model).color;

  return (
    <div className="page-shell">
      <div className="site-layout tj-layout">
        <main className="profile tj-page">
          <p className="replay-page-back">
            <a href={`${base}#trajectories`}>← Back to the post</a>
          </p>

          <h1>Agent Trajectories</h1>
          <p className="tj-intro">
            Read what the models were thinking and doing during their research runs—raw
            chain-of-thought where available, step-by-step analysis and plans, commands, and
            terminal logs.
          </p>

          <div className="tj-picker">
            <label htmlFor="tj-select" className="tj-picker-label">
              Session
            </label>
            <select
              id="tj-select"
              className="tj-select"
              value={selectedId}
              onChange={(e) => onSelect(e.target.value)}
            >
              {TASK_ORDER.map((group) => (
                <optgroup key={group.id} label={group.label}>
                  {MANIFEST.filter((m) => m.taskGroup === group.id).map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.model} — {m.outcome} ({m.stats.steps} steps
                      {m.stats.thinkingSteps >= 5 ? ", CoT" : ""})
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>

          <section className="tj-summary" style={{ borderLeftColor: accent }}>
            <div className="tj-summary-top">
              <strong>{meta.model}</strong>
              <span className="tj-sep">·</span>
              <span>{meta.taskLabel}</span>
              <span className="tj-sep">·</span>
              <span>{meta.outcome}</span>
            </div>
            <p>{meta.summary}</p>
            <div className="tj-meta-line">
              <span>{meta.stats.steps} steps</span>
              {meta.stats.thinkingSteps > 0 && (
                <span>{meta.stats.thinkingSteps} thinking traces</span>
              )}
              <span>{meta.stats.trainSteps} train</span>
              <span>{meta.stats.evalSteps} eval</span>
              <span>{meta.stats.editSteps} edits</span>
              <a href={`${base}trajectories/${meta.id}.json`} target="_blank" rel="noreferrer">
                Raw JSON ↗
              </a>
            </div>
          </section>

          {data?.notes && (
            <details className="tj-disclosure">
              <summary>Agent&rsquo;s research notes (NOTES.md)</summary>
              <pre>{data.notes}</pre>
            </details>
          )}

          {data?.prompt && (
            <details className="tj-disclosure">
              <summary>Initial task prompt</summary>
              <pre>{data.prompt}</pre>
            </details>
          )}

          <div className="tj-controls">
            <input
              type="search"
              className="tj-search"
              value={queryInput}
              onChange={(e) => {
                setQueryInput(e.target.value);
                setLimit(PAGE_SIZE);
              }}
              placeholder="Search thinking, analysis, commands, output…"
              aria-label="Search trajectory"
            />
            <label className="tj-check">
              <input
                type="checkbox"
                checked={hidePolls}
                onChange={(e) => setHidePolls(e.target.checked)}
              />
              Collapse polls
            </label>
            <label className="tj-check">
              <input
                type="checkbox"
                checked={showOutputs}
                onChange={(e) => setShowOutputs(e.target.checked)}
              />
              Terminal output
            </label>
          </div>

          {!loading && !error && data && (
            <p className="tj-count">
              Showing {visible.length} of {items.length} entries ({matchedCount} steps)
            </p>
          )}

          <div className="tj-stream">
            {loading && <p className="tj-state">Loading {meta.model} trajectory…</p>}
            {error && <p className="tj-state">Could not load trajectory ({error}).</p>}
            {!loading && !error && data && items.length === 0 && (
              <p className="tj-state">No steps match “{queryInput}”.</p>
            )}

            {!loading &&
              !error &&
              visible.map((it) =>
                it.kind === "polls" ? (
                  <button
                    key={it.id}
                    type="button"
                    className="tj-polls"
                    onClick={() => setExpandedPolls((p) => ({ ...p, [it.id]: true }))}
                  >
                    <span>
                      ··· {it.steps.length} routine poll steps (Steps {it.steps[0].id}–
                      {it.steps[it.steps.length - 1].id})
                    </span>
                    <span>Show</span>
                  </button>
                ) : (
                  <StepView
                    key={it.step.id}
                    step={it.step}
                    query={query}
                    showOutputs={showOutputs}
                  />
                ),
              )}

            {!loading && !error && remaining > 0 && (
              <div className="tj-pager">
                <button type="button" onClick={() => setLimit((n) => n + PAGE_SIZE)}>
                  Load {Math.min(PAGE_SIZE, remaining)} more
                </button>
                <button
                  type="button"
                  className="tj-pager-all"
                  onClick={() => setLimit(items.length)}
                >
                  Show all ({remaining} left)
                </button>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
