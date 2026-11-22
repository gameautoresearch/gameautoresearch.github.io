import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useAgentView } from "../../view";
import { modelStyle } from "./colors";
import raw from "./polytopia13.json";

// Polytopia from an empty starter, 13 models at 5 GPU-hours (campaigns polytopia-empty15-20261118 and
// polytopia-frontier6-20261118; readout reports/polytopia/frontier6-2026-11-19.md on branch polytopia-frontier6-cloud).
// One Bradley–Terry fit over every network the 13 counted sessions sealed, their fresh 15-minute finals, the PPO
// starter (15 min = 0) and the ladder references. polytopia13.json is derived from that readout's elo.json:
// nets are [GPU-hours when sealed, Elo, run name, formally submitted].

type Net = [number, number, string, number];
type Model = {
  key: string; label: string; gpu: number; note: string | null;
  final: { elo: number; ci: number[] }; best: { elo: number; ci: number[]; run: string; gpu: number }; nets: Net[];
};
const DATA = raw as unknown as { games: number; pairs: number; models: Model[]; refs: Record<string, number> };
const MODELS = DATA.models;
const winPct = (elo: number) => 100 / (1 + 10 ** (-elo / 400));
const fmtWin = (elo: number) => {
  const p = winPct(elo);
  return p < 0.5 ? "<1%" : p > 99.5 ? ">99%" : `${Math.round(p)}%`;
};
const REFS = [
  { name: "200M starter (80%)", elo: DATA.refs["starter-200m"] },
  { name: "15-min starter (50%)", elo: 0 },
  { name: "20M starter (22%)", elo: DATA.refs["starter-20m"] },
  { name: "Untrained (<1%)", elo: DATA.refs["random"] },
];
const signed = (v: number) => (v > 0 ? `+${v}` : v < 0 ? `−${-v}` : "0");
const gpuText = (h: number) => (h < 1 ? `${Math.round(h * 60)} min` : `${h.toFixed(2)} GPU-h`);

function bestSoFar(m: Model) {
  const steps: [number, number][] = [];
  let best = -Infinity;
  for (const [x, y] of m.nets) if (y > best) { best = y; steps.push([x, y]); }
  steps.push([m.gpu, best]);
  return steps;
}

export function useWidth(ref: React.RefObject<HTMLElement>) {
  const [w, setW] = useState(690);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setW(Math.max(280, Math.round(el.getBoundingClientRect().width)));
    measure();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, [ref]);
  return w;
}

type Hit = { cx: number; cy: number; key: string; title: string; lines: string[] };

export function ResearchRace() {
  const agent = useAgentView();
  const plotId = useId();
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref), narrow = W < 560;
  const [log, setLog] = useState(true);
  const [hit, setHit] = useState<Hit | null>(null);
  const H = Math.round(Math.max(340, Math.min(500, W * 0.55)) * 1.2);
  const m = { l: 46, r: narrow ? 12 : 118, t: 22, b: 40 };
  const [yMin, yMax] = [-1550, 340];
  const xLo = 0.04, xHi = 5.1;
  const sx = (h: number) => {
    const t = log ? (Math.log10(Math.max(h, xLo)) - Math.log10(xLo)) / (Math.log10(xHi) - Math.log10(xLo)) : h / 5;
    return m.l + t * (W - m.l - m.r);
  };
  // Values below the shown range run off the bottom of the plot (clipped), rather than piling up on its edge.
  const sy = (e: number) => m.t + ((yMax - e) / (yMax - yMin)) * (H - m.t - m.b);
  const xTicks = log ? [0.05, 0.1, 0.25, 0.5, 1, 2, 5] : [0, 1, 2, 3, 4, 5];
  const yTicks = [-1500, -1000, -500, 0];

  if (agent) {
    return (
      <ul>
        {MODELS.map((x) => <li key={x.key}>{x.label}: {x.nets.length} networks, best {fmtWin(x.best.elo)} vs 15-min starter ({signed(x.best.elo)} Elo, {x.best.run} at {gpuText(x.best.gpu)}), fresh 15-min final {fmtWin(x.final.elo)} ({signed(x.final.elo)} Elo).</li>)}
      </ul>
    );
  }

  const top = MODELS.reduce((best, model) => model.best.elo > best.best.elo ? model : best);
  const highlighted = hit?.key ?? top.key;
  const rank = (model: Model) => Number(model.key === highlighted);
  const hits: Hit[] = MODELS.flatMap((model) => [
    ...model.nets.filter(([, elo]) => elo >= yMin).map(([hours, elo, run, submitted]) => ({
      cx: sx(hours), cy: sy(elo), key: model.key, title: `${model.label} · ${run}`,
      lines: [`${fmtWin(elo)} vs 15-min starter (${signed(elo)} Elo)`,
        `After ${gpuText(hours)} of research${submitted ? " · submitted" : ""}`],
    })),
    ...(model.final.elo >= yMin ? [{
      cx: sx(model.gpu), cy: sy(model.final.elo), key: model.key, title: `${model.label} · fresh 15-min final`,
      lines: [`${fmtWin(model.final.elo)} vs 15-min starter (${signed(model.final.elo)} Elo)`],
    }] : []),
  ]);
  const move = (ev: React.PointerEvent<SVGSVGElement>) => {
    const box = ev.currentTarget.getBoundingClientRect(), k = W / box.width;
    const px = (ev.clientX - box.left) * k, py = (ev.clientY - box.top) * k;
    let near: Hit | null = null, d = 14 * 14;
    for (const point of hits) {
      const distance = (point.cx - px) ** 2 + (point.cy - py) ** 2;
      if (distance < d) { d = distance; near = point; }
    }
    setHit(near);
  };
  const path = (pts: [number, number][]) => pts.map(([x, y], i) => (i ? `H${sx(x).toFixed(1)}V${sy(y).toFixed(1)}` : `M${sx(x).toFixed(1)},${sy(y).toFixed(1)}`)).join("");
  const tipLeft = hit ? Math.max(0, hit.cx + 12 + 250 > W ? hit.cx - 262 : hit.cx + 12) : 0;

  return (
    <div className="viz" ref={ref}>
      <div className="ar-tabs ar-tabs-wrap" role="tablist" aria-label="Time axis">
        <button role="tab" aria-selected={log} onClick={() => { setLog(true); setHit(null); }}>Log time</button>
        <button role="tab" aria-selected={!log} onClick={() => { setLog(false); setHit(null); }}>Linear time</button>
      </div>
      <div className="viz-legend">
        {MODELS.map(({ key, label }) => {
          const st = modelStyle(label);
          return <span key={key}><i className={st.dashed ? "dashed" : undefined} style={st.dashed ? { color: st.color } : { background: st.color }} />{label}</span>;
        })}
      </div>
      <div className="ar-cost-plot">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Best network so far against research GPU-hours, 13 models"
          onPointerMove={move} onPointerDown={move} onPointerLeave={() => setHit(null)} style={{ touchAction: "pan-y" }}>
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className="viz-grid" />
              <text x={m.l - 8} y={sy(t)} className="viz-axis" textAnchor="end" dominantBaseline="middle">{signed(t)}</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} className="viz-grid" />
              <text x={sx(t)} y={H - m.b + 16} className="viz-axis" textAnchor="middle">{t >= 1 || t === 0 ? `${t}h` : `${Math.round(t * 60)}m`}</text>
            </g>
          ))}
          <text x={m.l} y={m.t - 9} className="viz-axis">Elo relative to the 15-minute PPO starter</text>
          <text x={W - m.r} y={H - 4} className="viz-axis" textAnchor="end">research GPU-hours spent</text>
          {REFS.map((r) => (
            <g key={r.name}>
              <line x1={m.l} x2={W - m.r} y1={sy(r.elo)} y2={sy(r.elo)} className="viz-ref" />
              {!narrow && <text x={W - m.r + 6} y={sy(r.elo)} className="fr-ref" dominantBaseline="middle">{r.name}</text>}
            </g>
          ))}
          <clipPath id={plotId}><rect x={m.l - 6} y={m.t - 6} width={W - m.l - m.r + 12} height={H - m.t - m.b + 12} /></clipPath>
          <g clipPath={`url(#${plotId})`}>
            {[...MODELS].sort((a, b) => rank(a) - rank(b)).map((model) => {
              const st = modelStyle(model.label), lit = model.key === highlighted;
              return (
                <g key={model.key}>
                  <path d={path(bestSoFar(model))} fill="none" stroke={st.color} strokeWidth={lit ? 2.4 : 1.4}
                    strokeDasharray={st.dashed ? "5 3" : undefined} strokeLinejoin="round" opacity={lit ? 1 : hit ? 0.18 : 0.22} />
                  {model.nets.map(([hours, elo, , submitted], i) => elo >= yMin && (
                    <circle key={i} cx={sx(hours)} cy={sy(elo)} r={submitted ? 3.5 : 2.8} fill={st.color}
                      stroke="#fff" strokeWidth={1} opacity={lit ? 1 : hit ? 0.2 : 0.3} />
                  ))}
                  {model.final.elo >= yMin && (
                    <path d={`M${sx(model.gpu)},${sy(model.final.elo) - 5}l5,5l-5,5l-5,-5z`} fill={st.color}
                      stroke="#fff" strokeWidth={1.2} opacity={lit ? 1 : hit ? 0.2 : 0.3} />
                  )}
                </g>
              );
            })}
          </g>
        </svg>
        {hit && (
          <div className="viz-tip" style={hit.cy < 110 ? { left: tipLeft, top: hit.cy + 14, transform: "none" } : { left: tipLeft, top: hit.cy - 10 }} role="status">
            <strong>{hit.title}</strong>
            {hit.lines.map((line) => <span key={line}>{line}</span>)}
          </div>
        )}
      </div>
    </div>
  );
}

export function EloLeaderboard() {
  const agent = useAgentView();
  const [by, setBy] = useState<"best" | "final">("best");
  const rows = useMemo(() => [...MODELS].sort((a, b) => b[by].elo - a[by].elo), [by]);
  const lo = -1500, hi = 320, pct = (v: number) => ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * 100;
  if (agent) {
    return <ol>{rows.map((x) => <li key={x.key}>{x.label}: best network {fmtWin(x.best.elo)} vs 15-min starter ({signed(x.best.elo)} Elo), fresh 15-min final {fmtWin(x.final.elo)} ({signed(x.final.elo)} Elo)</li>)}</ol>;
  }
  return (
    <div className="fr-board">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Researcher</th>
            <th aria-sort={by === "best" ? "descending" : "none"}>
              <button type="button" onClick={() => setBy("best")}>Best network (win % vs starter) {by === "best" ? "▾" : ""}</button>
            </th>
            <th aria-sort={by === "final" ? "descending" : "none"}>
              <button type="button" onClick={() => setBy("final")}>15-min final {by === "final" ? "▾" : ""}</button>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((x, i) => {
            const st = modelStyle(x.label), v = x[by].elo;
            return (
              <tr key={x.key}>
                <td className="fr-rank">{i + 1}</td>
                <td>
                  <span className="fr-name"><i style={{ background: st.color }} />{x.label}</span>
                  {x.note && <small>{x.note}</small>}
                </td>
                <td className={by === "best" ? "fr-on" : ""} title={`${signed(x.best.elo)} Elo vs 15-min starter`}>
                  <span className="fr-num">{fmtWin(x.best.elo)}</span>
                  {by === "best" && <span className="fr-bar"><b style={{ left: `${pct(Math.min(0, v))}%`, width: `${Math.abs(pct(v) - pct(0))}%`, background: st.color }} /><em style={{ left: `${pct(0)}%` }} /></span>}
                </td>
                <td className={by === "final" ? "fr-on" : ""} title={`${signed(x.final.elo)} Elo vs 15-min starter`}>
                  <span className="fr-num">{fmtWin(x.final.elo)}</span>
                  {by === "final" && <span className="fr-bar"><b style={{ left: `${pct(Math.min(0, v))}%`, width: `${Math.abs(pct(v) - pct(0))}%`, background: st.color }} /><em style={{ left: `${pct(0)}%` }} /></span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export const frontierGames = DATA.games;
