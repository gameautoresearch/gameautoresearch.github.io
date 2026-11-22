import { useEffect, useRef, useState } from "react";
import { CMYK } from "./colors";
import { useAgentView } from "../../view";
import { DeltaBars } from "../shared/text";
import type { Bar } from "../shared/text";
import raw from "./generals-wave1.json";

// Generals, mid-size board, submit-a-network (configuration 1), wave 1: the night of Nov 18–19. One lane per
// session, every lease and practice run at its real interval from the session ledgers. Derived from
// runs/local/generals-medium-20261118/trajectory/normalized-runs.json; purposes are analyst-assigned from NOTES.md
// and the lease commands.

type Event = { s: number; e: number; k: string; t: string; d: string };
type Session = {
  id: string; model: string; note: string; endH: number; endLabel: string; summary: string;
  events: Event[]; used: string | null; submits: { at: number; t: string }[]; grades: { at: number; t: string }[];
};
const sessions = raw as Session[];

const KINDS = [
  { id: "new", label: "New recipe, from scratch", color: CMYK.cDeep },
  { id: "continue", label: "Continue from a checkpoint", color: CMYK.cLight },
  { id: "average", label: "Weight averaging", color: CMYK.y },
  { id: "own_eval", label: "Its own evaluation", color: CMYK.k },
  { id: "diagnose", label: "Smoke test or fix", color: "#bdbdbd" },
] as const;
const kindColor = (k: string) => KINDS.find((x) => x.id === k)?.color ?? "#dcdcdc";
const kindLabel = (k: string) => (k === "practice" ? "Public practice" : KINDS.find((x) => x.id === k)?.label ?? k);

const hm = (h: number) => `${Math.floor(h)}:${String(Math.round((h % 1) * 60)).padStart(2, "0")}`;
const workerH = (s: Session) => s.events.reduce((n, e) => n + (e.e - e.s), 0);

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

export function GeneralsTimeline() {
  const agent = useAgentView();
  const [pick, setPick] = useState<{ id: string; i: number } | null>(null);
  const [cursor, setCursor] = useState<number | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref, agent);

  if (agent) {
    return (
      <div className="agent-table">
        <table>
          <thead><tr><th>Session</th><th>Leases and practice runs, in order (start–end, purpose)</th><th>Submissions</th></tr></thead>
          <tbody>
            {sessions.map((s) => (
              <tr key={s.id}>
                <td>{s.model} ({s.note})</td>
                <td>{s.events.map((e) => `${hm(e.s)}–${hm(e.e)} ${kindLabel(e.k).toLowerCase()}: ${e.t}`).join("; ")}</td>
                <td>{s.submits.map((x) => hm(x.at)).join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  const narrow = W < 520;
  const lane = narrow ? 58 : 54, m = { l: narrow ? 6 : 132, r: narrow ? 44 : 56, t: 22, b: 30 };
  const H = m.t + m.b + sessions.length * lane;
  const T = 6;
  const sx = (h: number) => m.l + (Math.min(h, T) / T) * (W - m.l - m.r);
  const toH = (clientX: number) => {
    const box = ref.current!.getBoundingClientRect();
    const x = ((clientX - box.left) / box.width) * W;
    return Math.max(0, Math.min(T, ((x - m.l) / (W - m.l - m.r)) * T));
  };
  const sel = pick && sessions.find((s) => s.id === pick.id);
  const ev = sel ? sel.events[pick!.i] : null;

  return (
    <div className="ar-tl">
      <div className="viz-legend ar-gw-legend">
        {KINDS.map((k) => <span key={k.id}><i style={{ background: k.color, height: 8 }} />{k.label}</span>)}
      </div>
      <div className="viz" ref={ref} onMouseMove={(e) => setCursor(toH(e.clientX))} onMouseLeave={() => setCursor(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Generals research session timelines">
          {[0, 1, 2, 3, 4, 5, 6].map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={m.t - 6} y2={H - m.b} className="viz-grid" />
              <text x={sx(t)} y={H - m.b + 16} className="viz-axis" textAnchor="middle">{t} h</text>
            </g>
          ))}
          <text x={W - m.r + 6} y={m.t - 8} className="viz-axis">GPU</text>
          {sessions.map((s, i) => {
            const y = m.t + i * lane + (narrow ? 24 : lane / 2);
            const cut = s.id.startsWith("sol");
            return (
              <g key={s.id}>
                {narrow
                  ? <text x={m.l} y={y - 19} className="viz-label">{s.model} <tspan className="viz-axis">{cut ? "interrupted" : ""}</tspan></text>
                  : <>
                      <text x={m.l - 10} y={y - 4} className="viz-label" textAnchor="end">{s.model}</text>
                      <text x={m.l - 10} y={y + 11} className="viz-axis" textAnchor="end">{cut ? "interrupted, uncounted" : `${s.submits.length} submissions`}</text>
                    </>}
                <line x1={sx(0)} x2={sx(s.endH)} y1={y} y2={y} stroke="#ececE8" strokeWidth={1} />
                {cut
                  ? <path d={`M${sx(s.endH) - 4},${y - 4}l8,8M${sx(s.endH) + 4},${y - 4}l-8,8`} stroke={CMYK.m} strokeWidth={1.8} />
                  : <line x1={sx(s.endH)} x2={sx(s.endH)} y1={y - 9} y2={y + 9} stroke="#b9b8b2" strokeWidth={1.5} />}
                {s.events.map((e, j) => e.k === "practice"
                  ? <path key={j} d={`M${sx((e.s + e.e) / 2)},${y - 16}l3.5,3.5l-3.5,3.5l-3.5,-3.5z`} fill="#8a8a8a"
                      className="ar-tl-run" tabIndex={0} onClick={() => setPick({ id: s.id, i: j })} onFocus={() => setPick({ id: s.id, i: j })} />
                  : <rect key={j} x={sx(e.s)} y={y - 7} width={Math.max(sx(e.e) - sx(e.s), 2)} height={14} rx={2}
                      fill={kindColor(e.k)}
                      stroke={pick?.id === s.id && pick.i === j ? "var(--ink)" : "#fff"} strokeWidth={pick?.id === s.id && pick.i === j ? 1.5 : 1}
                      className="ar-tl-run" tabIndex={0}
                      onClick={() => setPick({ id: s.id, i: j })} onFocus={() => setPick({ id: s.id, i: j })} />)}
                {s.submits.map((x, j) => <path key={`f${j}`} d={`M${sx(x.at)},${y + 8}v10l6,-2.5l-6,-2.5`} fill={CMYK.m} stroke={CMYK.m} strokeWidth={1.2}><title>{`${hm(x.at)} ${x.t}`}</title></path>)}
                <text x={W - m.r + 6} y={y} className="viz-value" dominantBaseline="middle">{workerH(s).toFixed(1)} h</text>
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
        <span><svg width="10" height="10"><path d="M5,1l4,4l-4,4l-4,-4z" fill="#8a8a8a" /></svg>public practice</span>
        <span><svg width="10" height="12"><path d="M2,1v10l7,-3l-7,-3" fill={CMYK.m} /></svg>submitted a network</span>
        <span><svg width="10" height="10"><path d="M1,1l8,8M9,1l-8,8" stroke={CMYK.m} strokeWidth="1.6" /></svg>session lost</span>
      </div>
      <div className="ar-tl-detail" aria-live="polite">
        {sel && ev ? (
          <>
            <strong>{sel.model}</strong> · {hm(ev.s)}–{hm(ev.e)} into the session · {kindLabel(ev.k).toLowerCase()} · <b>{ev.t}</b>
            <span> {ev.d}</span>
          </>
        ) : (
          <span>Click a bar or diamond for what that job was. GPU = hours the L40S worker was held, leases plus practice.</span>
        )}
      </div>
    </div>
  );
}

// Where each session's worker time went, by purpose, against its 4.5-hour research allocation. Bars are worker
// lifetime from the ledger intervals; the percentage is the allocation the ledger charged, which also counts
// setup and packaging between recorded intervals.
const ALLOC_H = 4.5;
const PARTS = [...KINDS, { id: "practice", label: "Public practice", color: "#8a8a8a" }];

export function GeneralsAllocation() {
  const agent = useAgentView();
  const rows = sessions.map((s) => {
    const h = PARTS.map((p) => s.events.filter((e) => e.k === p.id).reduce((n, e) => n + (e.e - e.s), 0));
    return { s, h, total: h.reduce((a, b) => a + b, 0) };
  });
  if (agent) {
    return (
      <div className="agent-table">
        <table>
          <thead><tr><th>Session</th>{PARTS.map((p) => <th key={p.id}>{p.label} (h)</th>)}<th>Total (h)</th></tr></thead>
          <tbody>{rows.map((r) => <tr key={r.s.id}><td>{r.s.model}</td>{r.h.map((v, i) => <td key={i}>{v.toFixed(2)}</td>)}<td>{r.total.toFixed(2)}</td></tr>)}</tbody>
        </table>
      </div>
    );
  }
  return (
    <div className="ar-alloc">
      <div className="viz-legend">
        {PARTS.map((p) => <span key={p.id}><i style={{ background: p.color }} />{p.label}</span>)}
        <span><i style={{ background: "#ecebe6" }} />Unused</span>
      </div>
      {rows.map((r) => (
        <div className="ar-alloc-row" key={r.s.id}>
          <span className="ar-alloc-label">{r.s.model}<small>{r.s.id.startsWith("sol") ? "cut off at 3.4 h" : `${r.s.used} of the allocation used`}</small></span>
          <span className="ar-alloc-bar">
            {r.h.map((v, j) => v > 0.005 && (
              <i key={j} style={{ width: `${(v / ALLOC_H) * 100}%`, background: PARTS[j].color }} title={`${PARTS[j].label}: ${v.toFixed(2)} h`} />
            ))}
            <i style={{ width: `${Math.max(0, (ALLOC_H - r.total) / ALLOC_H) * 100}%`, background: "#ecebe6" }} title={`Unused: ${Math.max(0, ALLOC_H - r.total).toFixed(2)} h`} />
          </span>
        </div>
      ))}
      <div className="ar-alloc-axis"><span>0</span><span>1.5 h</span><span>3 h</span><span>4.5 h</span></div>
    </div>
  );
}

// Final tournament of every Generals mid-size network. Values are filled in once the wave 2 sessions (GPT-6.2 Sol,
// Claude Opus 5.5, Gemini 4 Neon) are graded and the five-way tournament is in; until every entrant has a rating
// the figure shows the field as pending. Set elo/lo/hi (Elo, 95% interval) per entrant to switch to the bar chart.
type Entrant = { label: string; wave: 1 | 2; elo: number | null; lo?: number; hi?: number };
const ENTRANTS: Entrant[] = [
  { label: "Kimi K3.1 Pro", wave: 1, elo: null },
  { label: "DeepSeek V4 Flash", wave: 1, elo: null },
  { label: "GPT-6.2 Sol", wave: 2, elo: null },
  { label: "Claude Opus 5.5", wave: 2, elo: null },
  { label: "Gemini 4 Neon", wave: 2, elo: null },
];

export function GeneralsElo() {
  const agent = useAgentView();
  if (ENTRANTS.every((e) => e.elo !== null)) {
    const bars: Bar[] = ENTRANTS.map((e) => ({
      label: e.label, value: e.elo!, lo: e.lo, hi: e.hi,
      tone: e.lo !== undefined && e.lo > 0 ? "good" : e.hi !== undefined && e.hi < 0 ? "bad" : "flat",
      note: `Wave ${e.wave}`,
    }));
    return <DeltaBars bars={bars} domain={[-300, 300]} unit="Elo" labelWidth={150} />;
  }
  if (agent) {
    return <ul>{ENTRANTS.map((e) => <li key={e.label}>{e.label} (wave {e.wave}): {e.elo ?? "pending"}</li>)}</ul>;
  }
  return (
    <div className="ar-pending">
      {ENTRANTS.map((e) => (
        <div className="ar-pending-row" key={e.label}>
          <span className="ar-pending-label">{e.label}<small>wave {e.wave}{e.wave === 2 ? " · running" : " · finished"}</small></span>
          <span className="ar-pending-track"><i /></span>
          <span className="ar-pending-value">{e.elo ?? "—"}</span>
        </div>
      ))}
      <div className="ar-pending-axis"><span>−300</span><span>0</span><span>+300</span></div>
      <p className="ar-pending-note">Final tournament pending. Ratings land here once all five networks have played every pairing.</p>
    </div>
  );
}
