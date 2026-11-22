import { useEffect, useMemo, useRef, useState } from "react";
import { CMYK, liftFill } from "./colors";
import { useAgentView } from "../../view";
import raw from "./timelines.json";

// Pokémon 3v3 research sessions, start to finish: one lane per model, every training run at its real interval,
// colored by the private lift its network reached, with practice evaluations, stops, code checks and submissions
// marked where they happened. Derived by tools/build_timelines.py from the session request logs and run ledger.

type Run = { s: number; e: number | null; reqMin: number; lift: number | null; practice: number | null };
export type Attempt = {
  model: string; attempt: string; finalLift: number; bestLift: number; sessionH: number; gpuBusyH: number; promptM: number;
  runs: Run[]; evals: number[]; stops: number[]; checks: number[]; submits: number[]; polls: number;
};
export const attempts = raw as Attempt[];

const MODELS = [...new Set(attempts.map((a) => a.model))];
const byModel = (m: string) => attempts.filter((a) => a.model === m);
const signed = (v: number) => (v > 0 ? `+${v}` : `${v}`);
const hm = (h: number) => `${Math.floor(h)}:${String(Math.round((h % 1) * 60)).padStart(2, "0")}`;

// Lift → color: magenta below the starter, grey around it, cyan as it climbs.
function liftColor(l: number | null) {
  if (l === null) return "#dcdcdc";
  return l >= 0 ? liftFill(Math.min(l / 20, 1)) : liftFill(-Math.min(-l / 12, 1));
}

function useWidth(ref: React.RefObject<HTMLElement>, dep: unknown) {
  const [w, setW] = useState(640);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setW(Math.max(280, Math.round(el.getBoundingClientRect().width)));
    measure();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, [ref, dep]);
  return w;
}

type Pick = { model: string; attempt: string; run?: number } | null;

export function SessionTimeline() {
  const agent = useAgentView();
  const [slot, setSlot] = useState(0);
  const [pick, setPick] = useState<Pick>(null);
  const [cursor, setCursor] = useState<number | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref, agent);
  const lanes = useMemo(() => MODELS.map((m) => byModel(m)[slot]), [slot]);

  if (agent) {
    return (
      <div className="agent-table">
        <table>
          <thead><tr><th>Attempt</th><th>Training runs, in order (minutes trained → private lift)</th><th>Stops</th><th>Practice</th><th>Final lift</th></tr></thead>
          <tbody>
            {attempts.map((a) => (
              <tr key={a.model + a.attempt}>
                <td>{a.model} {a.attempt}</td>
                <td>{a.runs.map((r) => `${r.e !== null ? Math.round((r.e - r.s) * 60) : "?"}m→${r.lift !== null ? signed(r.lift) : "?"}`).join(", ")}</td>
                <td>{a.stops.length}</td>
                <td>{a.evals.length}</td>
                <td>{signed(a.finalLift)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  const narrow = W < 520;
  const lane = narrow ? 52 : 44, m = { l: narrow ? 6 : 120, r: narrow ? 40 : 52, t: 22, b: 30 };
  const H = m.t + m.b + lanes.length * lane;
  const T = 5;
  const sx = (h: number) => m.l + (Math.min(h, T) / T) * (W - m.l - m.r);
  const toH = (clientX: number) => {
    const box = ref.current!.getBoundingClientRect();
    const x = ((clientX - box.left) / box.width) * W;
    return Math.max(0, Math.min(T, ((x - m.l) / (W - m.l - m.r)) * T));
  };

  const selected = pick && attempts.find((a) => a.model === pick.model && a.attempt === pick.attempt);
  const selRun = selected && pick?.run !== undefined ? selected.runs[pick.run] : null;

  return (
    <div className="ar-tl">
      <div className="ar-tabs" role="tablist" aria-label="Which attempt to show">
        {["Attempt 1", "Attempt 2", "Attempt 3"].map((t, i) => (
          <button key={t} role="tab" aria-selected={slot === i} onClick={() => { setSlot(i); setPick(null); }}>{t}</button>
        ))}
      </div>
      <div className="viz" ref={ref}
        onMouseMove={(e) => setCursor(toH(e.clientX))}
        onMouseLeave={() => setCursor(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Research session timelines">
          {[0, 1, 2, 3, 4, 5].map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={m.t - 6} y2={H - m.b} className="viz-grid" />
              <text x={sx(t)} y={H - m.b + 16} className="viz-axis" textAnchor="middle">{t} h</text>
            </g>
          ))}
          <text x={W - m.r + 6} y={m.t - 8} className="viz-axis">final</text>
          {lanes.map((a, i) => {
            const y = m.t + i * lane + (narrow ? 20 : lane / 2);
            const active = pick?.model === a.model && pick.attempt === a.attempt;
            return (
              <g key={a.model}>
                {narrow
                  ? <text x={m.l} y={y - 16} className="viz-label">{a.model} <tspan className="viz-axis">{a.attempt}</tspan></text>
                  : <text x={m.l - 10} y={y} className="viz-label" textAnchor="end" dominantBaseline="middle">{a.model}</text>}
                <line x1={sx(0)} x2={sx(a.sessionH)} y1={y} y2={y} stroke="#ececE8" strokeWidth={1} />
                <line x1={sx(a.sessionH)} x2={sx(a.sessionH)} y1={y - 9} y2={y + 9} stroke="#b9b8b2" strokeWidth={1.5} />
                {a.checks.map((t, j) => <line key={`c${j}`} x1={sx(t)} x2={sx(t)} y1={y + 10} y2={y + 14} stroke="#b9b8b2" strokeWidth={1} />)}
                {a.runs.map((r, j) => r.e !== null && (
                  <rect key={`r${j}`} x={sx(r.s)} y={y - 7} width={Math.max(sx(r.e) - sx(r.s), 2)} height={14} rx={3}
                    fill={liftColor(r.lift)}
                    stroke={active && pick?.run === j ? "var(--ink)" : "#fff"} strokeWidth={active && pick?.run === j ? 1.5 : 1}
                    className="ar-tl-run" tabIndex={0}
                    onClick={() => setPick({ model: a.model, attempt: a.attempt, run: j })}
                    onFocus={() => setPick({ model: a.model, attempt: a.attempt, run: j })} />
                ))}
                {a.evals.map((t, j) => <path key={`e${j}`} d={`M${sx(t)},${y - 15}l3.5,3.5l-3.5,3.5l-3.5,-3.5z`} fill={CMYK.k} />)}
                {a.stops.map((t, j) => (
                  <path key={`s${j}`} d={`M${sx(t) - 3},${y + 9}l6,6M${sx(t) + 3},${y + 9}l-6,6`} stroke={CMYK.m} strokeWidth={1.6} />
                ))}
                {a.submits.map((t, j) => <path key={`f${j}`} d={`M${sx(t)},${y - 9}v-9l6,2.5l-6,2.5`} fill={CMYK.y} stroke={CMYK.y} strokeWidth={1.2} />)}
                <text x={W - m.r + 6} y={y} className="viz-value" dominantBaseline="middle" fill={a.finalLift >= 10 ? CMYK.cDeep : "var(--ink)"}>{signed(+a.finalLift.toFixed(1)) === signed(a.finalLift) && Number.isInteger(a.finalLift) ? `${signed(a.finalLift)}.0` : signed(a.finalLift)}</text>
              </g>
            );
          })}
          {cursor !== null && (
            <g pointerEvents="none">
              <line x1={sx(cursor)} x2={sx(cursor)} y1={m.t - 10} y2={H - m.b} stroke="var(--ink)" strokeWidth={1} strokeDasharray="2 3" />
              <text x={sx(cursor)} y={m.t - 12} className="viz-axis" textAnchor="middle">{hm(cursor)}</text>
            </g>
          )}
        </svg>
      </div>
      <div className="ar-tl-key">
        <span><i className="run" />training run, by private lift</span>
        <span><svg width="10" height="10"><path d="M5,1l4,4l-4,4l-4,-4z" fill={CMYK.k} /></svg>practice eval</span>
        <span><svg width="10" height="10"><path d="M2,2l6,6M8,2l-6,6" stroke={CMYK.m} strokeWidth="1.6" /></svg>stopped a run</span>
        <span><svg width="10" height="12"><path d="M2,11v-10l7,3l-7,3" fill={CMYK.y} /></svg>submitted</span>
        <span><svg width="4" height="10"><path d="M2,2v6" stroke="#b9b8b2" /></svg>code/config check</span>
      </div>
      <div className="ar-tl-detail" aria-live="polite">
        {selected && selRun ? (
          <>
            <strong>{selected.model} {selected.attempt}, run {pick!.run! + 1} of {selected.runs.length}</strong>{" · "}
            {hm(selRun.s)}–{selRun.e !== null ? hm(selRun.e) : "?"} into the session · asked for {selRun.reqMin} min, trained{" "}
            {selRun.e !== null ? Math.round((selRun.e - selRun.s) * 60) : "?"} min · private lift{" "}
            <b>{selRun.lift !== null ? signed(+selRun.lift.toFixed(1)) : "n/a"}</b>
            {selRun.practice !== null && <> · practice {selRun.practice}/100</>}
          </>
        ) : (
          <span>Click a run for its timing and strength. Final = lift of the network the host retrained from the submitted recipe.</span>
        )}
      </div>
    </div>
  );
}

// Where each model's 4 GPU-hours went, mean of its three attempts, by how long each run actually trained.
const BANDS = [
  { id: "full", label: "Runs ≥ 45 min", color: CMYK.cDeep, test: (m: number) => m >= 45 },
  { id: "mid", label: "20–45 min", color: CMYK.cLight, test: (m: number) => m >= 20 && m < 45 },
  { id: "short", label: "< 20 min", color: CMYK.m, test: (m: number) => m < 20 },
];

export function GpuAllocation() {
  const agent = useAgentView();
  const rows = MODELS.map((model) => {
    const as = byModel(model);
    const h = BANDS.map((b) => as.reduce((n, a) => n + a.runs.reduce((k, r) => {
      if (r.e === null) return k;
      const min = (r.e - r.s) * 60;
      return k + (b.test(min) ? min / 60 : 0);
    }, 0), 0) / as.length);
    const runs = as.reduce((n, a) => n + a.runs.length, 0) / as.length;
    return { model, h, idle: Math.max(0, 4 - h.reduce((x, y) => x + y, 0)), runs };
  });
  if (agent) {
    return (
      <div className="agent-table">
        <table>
          <thead><tr><th>Model</th>{BANDS.map((b) => <th key={b.id}>{b.label} (h)</th>)}<th>Unused (h)</th><th>Runs per attempt</th></tr></thead>
          <tbody>{rows.map((r) => <tr key={r.model}><td>{r.model}</td>{r.h.map((v, i) => <td key={i}>{v.toFixed(2)}</td>)}<td>{r.idle.toFixed(2)}</td><td>{r.runs.toFixed(1)}</td></tr>)}</tbody>
        </table>
      </div>
    );
  }
  return (
    <div className="ar-alloc">
      <div className="viz-legend">
        {BANDS.map((b) => <span key={b.id}><i style={{ background: b.color }} />{b.label}</span>)}
        <span><i style={{ background: "#ecebe6" }} />Unused</span>
      </div>
      {rows.map((r, i) => (
        <div className="ar-alloc-row" key={r.model} style={{ ["--i" as string]: i }}>
          <span className="ar-alloc-label">{r.model}<small>{r.runs.toFixed(1)} runs per attempt</small></span>
          <span className="ar-alloc-bar">
            {r.h.map((v, j) => v > 0.01 && (
              <i key={j} style={{ width: `${(v / 4) * 100}%`, background: BANDS[j].color }} title={`${BANDS[j].label}: ${v.toFixed(2)} h`} />
            ))}
            <i style={{ width: `${(r.idle / 4) * 100}%`, background: "#ecebe6" }} title={`Unused: ${r.idle.toFixed(2)} h`} />
          </span>
        </div>
      ))}
      <div className="ar-alloc-axis"><span>0</span><span>1 h</span><span>2 h</span><span>3 h</span><span>4 h</span></div>
    </div>
  );
}
