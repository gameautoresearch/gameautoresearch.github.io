import { useEffect, useRef, useState } from "react";
import { useAgentView } from "../../view";
import { Table } from "../shared/text";
import raw from "./budget.json";
import { CMYK, modelStyle } from "./colors";

function usePanelWidth(ref: React.RefObject<HTMLElement>, fallback: number) {
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setW(Math.max(160, Math.round(el.getBoundingClientRect().width)));
    measure();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, [ref]);
  return w;
}

// Figure 2 in compute multipliers instead of Elo (owner request, 2026-11-20). Every recipe point of the Polytopia
// research-budget campaign is placed on the unmodified starter's compute curve: the multiplier is the starter's
// training cost to reach the same Elo divided by what the recipe cost in total (its research GPU spend plus its own
// training). 1x is the starter itself; 2x means the recipe got there for half the money, research included.
// The starter curve is its measured points (budget.json "starter", 0.5 h to 64 h) made monotone by weighted
// pool-adjacent-violators and read log-linearly between them, as the campaign's own multipliers were
// (reports/polytopia/research-budget-2026-11-18/fit.py). Above the longest starter run a value is a lower bound.

type CurvePoint = { hours: number; elo: number };
type Recipe = { key: string; model: string; cell: string; spentUsd: number; curve: CurvePoint[]; finals1h: number[]; lo8: number | null; hi8: number | null };
const DATA = raw as unknown as { usdPerGpuHour: number; starter: { hours: number; elo: number; n: number }[]; recipes: Recipe[] };
const RATE = DATA.usdPerGpuHour;

// Monotone starter curve: weighted PAVA over the measured points, each block placed at its weighted mean log-time.
const KNOTS = (() => {
  const blocks: { lt: number; elo: number; w: number }[] = [];
  for (const p of DATA.starter) {
    blocks.push({ lt: Math.log2(p.hours), elo: p.elo, w: p.n });
    while (blocks.length > 1 && blocks[blocks.length - 2].elo >= blocks[blocks.length - 1].elo) {
      const b = blocks.pop()!, a = blocks.pop()!, w = a.w + b.w;
      blocks.push({ lt: (a.lt * a.w + b.lt * b.w) / w, elo: (a.elo * a.w + b.elo * b.w) / w, w });
    }
  }
  return blocks.map((b) => ({ hours: 2 ** b.lt, elo: b.elo }));
})();
const TOP = KNOTS[KNOTS.length - 1];
// Past the starter's longest run the curve is extended with the shape of the power law fitted to all 82 starter
// networks, Elo(t) = E_inf - B t^-alpha (reports/polytopia/compute-multipliers-2026-11-20/curve.json, frontier
// scale), joined to the measured curve at its last point. That fit tracks the 25-54 h snapshots to within 5 Elo.
const FIT = { einf: 568.43, b: 522.51, alpha: 0.28036 };
const fitHours = (elo: number) => (FIT.b / Math.max(FIT.einf - elo, 1e-9)) ** (1 / FIT.alpha);
// Starter training hours to reach `elo`; `extrap` when the Elo is beyond the starter's longest run.
export function starterHoursFor(elo: number): { hours: number; extrap: boolean } {
  const k = KNOTS;
  if (elo >= TOP.elo) return { hours: TOP.hours * (fitHours(elo) / fitHours(TOP.elo)), extrap: true };
  if (elo <= k[0].elo) {
    const s = (k[1].elo - k[0].elo) / Math.log2(k[1].hours / k[0].hours);
    return { hours: k[0].hours * 2 ** ((elo - k[0].elo) / s), extrap: false };
  }
  for (let i = 1; i < k.length; i++) {
    const a = k[i - 1], b = k[i];
    if (elo <= b.elo) return { hours: a.hours * (b.hours / a.hours) ** ((elo - a.elo) / (b.elo - a.elo)), extrap: false };
  }
  return { hours: TOP.hours, extrap: false };
}

const usd = (v: number) => `$${v < 10 ? v.toFixed(2) : v.toFixed(0)}`;
const hrs = (h: number) => (h < 1 ? `${Math.round(h * 60)} min` : `${+h.toFixed(h < 10 ? 1 : 0)} h`);
const fx = (v: number) => (v >= 9.5 ? `${Math.round(v)}×` : v >= 0.95 ? `${v.toFixed(1)}×` : `${v.toFixed(2)}×`);

type Pt = { r: Recipe; hours: number; elo: number; cost: number; final: boolean; sh: number; extrap: boolean; x: number; xTrain: number; xTrainLo: number | null; xTrainHi: number | null };
const point = (r: Recipe, hours: number, elo: number, final: boolean): Pt => {
  const cost = r.spentUsd + hours * RATE, s = starterHoursFor(elo);
  const at8 = !final && hours === 8 && r.lo8 !== null && r.hi8 !== null;
  return { r, hours, elo, cost, final, sh: s.hours, extrap: s.extrap, x: (s.hours * RATE) / cost, xTrain: s.hours / hours,
    xTrainLo: at8 ? starterHoursFor(r.lo8!).hours / hours : null, xTrainHi: at8 ? starterHoursFor(r.hi8!).hours / hours : null };
};
const recipes = DATA.recipes.filter((r) => r.curve.length);
// The fresh 1 h finals (finals1h) are left off the plot: they only added noise (owner request, 2026-11-23).
const curves = recipes.map((r) => ({ r, pts: r.curve.map((p) => point(r, p.hours, p.elo, false)) }));
const at8 = curves.map((c) => c.pts[c.pts.length - 1]).sort((a, b) => b.x - a.x);
export const TOP8 = at8[0];
export const costMult = {
  n: at8.length,
  above: at8.filter((p) => p.x > 1).length,
  trainAbove: at8.filter((p) => p.xTrain > 1).length,
  top: { model: TOP8.r.model, spent: TOP8.r.spentUsd, x: TOP8.x, xTrain: TOP8.xTrain, extrap: TOP8.extrap, starterHours: TOP8.sh, starterUsd: Math.round(TOP8.sh * RATE) },
  second: { model: at8[1].r.model, spent: at8[1].r.spentUsd, x: at8[1].x },
  median: at8[at8.length >> 1].x,
};

// Training compute only: each model's best recipe, scored by the geometric mean of its training multiplier over the
// 6, 7 and 8 h checkpoints (research spend not counted). Averaging the end of training keeps one lucky checkpoint
// from setting a model's bar; the 95% range is the 8 h checkpoint's, shown only where the value is extrapolated.
const WINDOW = [6, 7, 8];
type Bar = { r: Recipe; x: number; extrap: boolean; sh8: number; lo: number | null; hi: number | null };
const gmean = (xs: number[]) => Math.exp(xs.reduce((a, v) => a + Math.log(v), 0) / xs.length);
const bars: Bar[] = [...new Set(curves.map((c) => c.r.model))].map((model) =>
  curves.filter((c) => c.r.model === model && WINDOW.every((h) => c.pts.some((p) => p.hours === h)))
    .map(({ r, pts }): Bar => {
      const win = pts.filter((p) => WINDOW.includes(p.hours)), p8 = win.find((p) => p.hours === 8)!;
      return { r, x: gmean(win.map((p) => p.xTrain)), extrap: win.some((p) => p.extrap), sh8: p8.sh, lo: p8.xTrainLo, hi: p8.xTrainHi };
    })
    .reduce((a, b) => (b.x > a.x ? b : a)),
).sort((a, b) => b.x - a.x);
export const trainBars = { n: bars.length, top: bars[0] };

function TrainBars({ W }: { W: number }) {
  const compact = W < 420;
  const hasBelow = bars.some((p) => p.x < 1);
  const row = compact ? 38 : 30;
  const m = compact ? { l: 10, r: 44, t: 22, b: 36 } : { l: 148, r: 96, t: 22, b: 36 };
  const H = m.t + m.b + bars.length * row;
  const maxBar = Math.max(...bars.map((p) => p.x));
  const topTick = 2 ** Math.max(4, Math.ceil(Math.log2(maxBar)));
  const L0 = Math.log2(hasBelow ? 1 / 2 : 1), L1 = Math.log2(topTick);
  const sx = (x: number) => m.l + ((Math.log2(Math.min(Math.max(x, 2 ** L0), 2 ** L1)) - L0) / (L1 - L0)) * (W - m.l - m.r);
  const ticks: number[] = hasBelow ? [1 / 2] : [];
  for (let t = 1; t <= topTick; t *= 2) ticks.push(t);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training-only compute multiplier of each model's best recipe">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={sx(t)} x2={sx(t)} y1={m.t - 4} y2={H - m.b} className="viz-grid" />
          <text x={sx(t)} y={H - m.b + 16} className="viz-axis" textAnchor="middle">{t >= 1 ? `${t}×` : `1/${1 / t}`}</text>
        </g>
      ))}
      <line x1={sx(1)} x2={sx(1)} y1={m.t - 4} y2={H - m.b} stroke={CMYK.k} strokeWidth={1.6} />
      <text x={m.l} y={m.t - 9} className="viz-axis">
        {compact ? "best recipe, 6–8 h mean" : "best recipe, mean of 6–8 h: starter's training ÷ recipe's (log scale)"}
      </text>
      <text x={W - m.r} y={H - 4} className="viz-axis" textAnchor="end">
        {compact ? "training multiplier (log)" : "training compute multiplier"}
      </text>
      {bars.map((p, i) => {
        const st = modelStyle(p.r.model);
        const y = m.t + i * row;
        const x0 = sx(1), x1 = sx(p.x);
        const barY = compact ? y + 19 : y + 5;
        const barH = compact ? 11 : row - 10;
        const midY = barY + barH / 2;
        return (
          <g key={p.r.key}>
            <title>{`${p.r.model}, best recipe (${p.r.cell}): ${p.extrap ? "≈" : ""}${fx(p.x)} averaged over 6, 7 and 8 h of training; at 8 h as strong as the starter after ${p.extrap ? "about " : ""}${hrs(p.sh8)}${p.extrap ? ` (extrapolated past the starter's ${hrs(TOP.hours)} run)` : ""}`}</title>
            <rect x={0} y={y} width={W} height={row} fill="transparent" />
            {compact ? (
              <text
                x={x0 + 5}
                y={y + 13}
                className="viz-label"
                style={{ fontSize: 11.5, paintOrder: "stroke", stroke: "#fff", strokeWidth: 3 }}
              >
                {p.r.model}
                {p.extrap && <tspan className="viz-axis"> · extrap.</tspan>}
              </text>
            ) : (
              <text x={m.l - 10} y={midY} className="viz-label" textAnchor="end" dominantBaseline="middle">{p.r.model}</text>
            )}
            <rect x={Math.min(x0, x1)} y={barY} width={Math.max(1, Math.abs(x1 - x0))} height={barH} rx={2} fill={st.color} />
            <text x={Math.max(x0, x1) + 6} y={midY} className="viz-label" dominantBaseline="middle" style={{ fontSize: compact ? 11.5 : 12.5 }}>
              <tspan style={{ fontWeight: 600 }}>{p.extrap ? "≈" : ""}{fx(p.x)}</tspan>
              {!compact && p.extrap && <tspan className="viz-axis" dx={6}>extrapolated</tspan>}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

type Hit = { cx: number; cy: number; key: string; model: string; title: string; lines: string[] };

export function CostMultiplier() {
  const agent = useAgentView();
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const W = usePanelWidth(leftRef, 446);
  const WR = usePanelWidth(rightRef, 224);
  const sideBySide = WR < 420;
  const compact = W < 520;
  const [hit, setHit] = useState<Hit | null>(null);
  if (agent) {
    return (
      <Table
        head={["Recipe", "Research spend", "Total cost at 8 h", "Starter training to match", "Compute multiplier, research included", "Training only"]}
        align={["l", "r", "r", "r", "r", "r"]}
        rows={at8.map((p) => [`${p.r.model}, ${p.r.cell}`, usd(p.r.spentUsd), usd(p.cost), `${p.extrap ? "about " : ""}${hrs(p.sh)}${p.extrap ? " (extrapolated)" : ""}`,
          `${p.extrap ? "≈" : ""}${fx(p.x)}`, `${p.extrap ? "≈" : ""}${fx(p.xTrain)}`])}
      />
    );
  }
  const H = sideBySide ? 22 + 36 + bars.length * 38 : Math.round(Math.max(300, Math.min(400, W * 0.52)));
  const m = { l: compact ? 38 : 46, r: compact ? 12 : 88, t: 22, b: 36 };
  const all = curves.flatMap((c) => c.pts);
  const cx = (q: Pt) => q.cost, vy = (q: Pt) => q.x;
  const COST_MAX = Math.ceil(Math.max(...all.map(cx)) / 5) * 5;
  const L0 = Math.log2(0.0625), L1 = Math.log2(16);
  const sx = (c: number) => m.l + (c / COST_MAX) * (W - m.l - m.r);
  const sy = (x: number) => m.t + (1 - (Math.log2(Math.max(x, 2 ** L0)) - L0) / (L1 - L0)) * (H - m.t - m.b);
  const yTicks = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16];
  const xTicks: number[] = [];
  for (let x = 0; x <= COST_MAX; x += W < 380 ? 10 : 5) xTicks.push(x);
  const models = [...bars.map((b) => b.r.model), ...recipes.map((r) => r.model)].filter((mo, i, a) => a.indexOf(mo) === i);
  const top = TOP8;
  // Hovering a point brings forward every recipe from that model; the hovered recipe is drawn heaviest.
  const rank = (r: Recipe) => (r.key === hit?.key ? 3 : r.model === hit?.model ? 2 : r === top.r ? 1 : 0);
  const hits: Hit[] = all.map((q) => ({
    cx: sx(cx(q)), cy: sy(vy(q)), key: q.r.key, model: q.r.model, title: `${q.r.model}, ${q.r.cell}`,
    lines: [
      `${hrs(q.hours)} trained: as strong as the starter after ${q.extrap ? "about " : ""}${hrs(q.sh)}${q.extrap ? ` (extrapolated past its ${hrs(TOP.hours)} run)` : ""}`,
      `${q.extrap ? "≈" : ""}${fx(q.x)} less compute than the starter, research included`,
      `research ${usd(q.r.spentUsd)} + training ${usd(q.hours * RATE)} = ${usd(q.cost)}; training alone ${q.extrap ? "≈" : ""}${fx(q.xTrain)}`,
    ],
  }));
  const move = (ev: React.PointerEvent<SVGSVGElement>) => {
    const box = ev.currentTarget.getBoundingClientRect(), k = W / box.width;
    const px = (ev.clientX - box.left) * k, py = (ev.clientY - box.top) * k;
    let near: Hit | null = null, d = 14 * 14;
    for (const h of hits) { const dd = (h.cx - px) ** 2 + (h.cy - py) ** 2; if (dd < d) { d = dd; near = h; } }
    setHit(near);
  };
  const path = (pts: Pt[]) => pts.map((p, i) => `${i ? "L" : "M"}${sx(cx(p)).toFixed(1)},${sy(vy(p)).toFixed(1)}`).join("");
  const tipW = compact ? 220 : 250;
  const tipLeft = hit ? Math.max(0, hit.cx + 12 + tipW > W ? Math.max(0, hit.cx - tipW - 12) : hit.cx + 12) : 0;
  return (
    <div className="ar-cost-wrap">
      <div className="viz-legend ar-cost-legend">
        {models.map((mo) => {
          const st = modelStyle(mo);
          return (
            <span key={mo}>
              <i style={{ background: st.color }} />
              {mo}
            </span>
          );
        })}
        <span><i style={{ background: CMYK.k }} />Starter (1×)</span>
      </div>
      <div className="ar-cost-pair">
        <figure className="pt-figure ar-cost-fig">
          <div className="ar-cost-head">Research cost included</div>
          <div className="viz ar-cost" ref={leftRef}>
            <div className="ar-cost-plot">
              <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Compute multiplier against total cost, research plus training, for every recipe"
                onPointerMove={move} onPointerDown={move} onPointerLeave={() => setHit(null)} style={{ touchAction: "pan-y" }}>
                {yTicks.map((t) => (
                  <g key={t}>
                    <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className="viz-grid" />
                    <text x={m.l - 6} y={sy(t)} className="viz-axis" textAnchor="end" dominantBaseline="middle">{t >= 1 ? `${t}×` : `1/${1 / t}`}</text>
                  </g>
                ))}
                {xTicks.map((t) => (
                  <g key={t}>
                    <line x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} className="viz-grid" />
                    <text x={sx(t)} y={H - m.b + 16} className="viz-axis" textAnchor="middle">${t}</text>
                  </g>
                ))}
                <text x={W - m.r} y={H - 4} className="viz-axis" textAnchor="end">
                  {compact ? "total cost: research + training" : `total cost: research GPU spend + training ($${RATE.toFixed(2)}/h)`}
                </text>
                <text x={m.l} y={m.t - 9} className="viz-axis">
                  {compact ? "compute multiplier (log scale)" : "compute multiplier: starter's cost ÷ recipe's total cost (log scale)"}
                </text>
                {[...curves].sort((a, b) => rank(a.r) - rank(b.r)).map(({ r, pts }) => {
                  const st = modelStyle(r.model), focus = hit?.key === r.key, hi = r === top.r;
                  const lit = hit ? r.model === hit.model : hi;
                  return (
                    <g key={r.key}>
                      <path d={path(pts)} fill="none" stroke={st.color} strokeWidth={focus ? 2.8 : lit ? 2.2 : 1.4} strokeLinejoin="round" opacity={lit ? 1 : hit ? 0.15 : 0.25} />
                      {pts.map((p) => <circle key={p.hours} cx={sx(cx(p))} cy={sy(vy(p))} r={focus ? 4 : 2.6} fill={p.extrap ? "#fff" : st.color} stroke={p.extrap ? st.color : "#fff"} strokeWidth={p.extrap ? 1.6 : 1} opacity={lit ? 1 : hit ? 0.2 : 0.35} />)}
                    </g>
                  );
                })}
                <line x1={m.l} x2={W - m.r} y1={sy(1)} y2={sy(1)} stroke={CMYK.k} strokeWidth={2} />
                {compact ? (
                  <text x={W - m.r} y={sy(1) - 6} className="viz-axis" textAnchor="end" style={{ fill: "var(--ink)", fontWeight: 600, paintOrder: "stroke", stroke: "#fff", strokeWidth: 3 }}>
                    Starter, 1×
                  </text>
                ) : (
                  <text x={W - m.r + 6} y={sy(1)} className="viz-label" dominantBaseline="middle" style={{ fontWeight: 600 }}>Starter, 1×</text>
                )}
                <text
                  x={compact ? sx(cx(top)) - 8 : sx(cx(top)) + 9}
                  y={compact ? sy(vy(top)) + 1 : sy(vy(top)) - 10}
                  className="viz-label"
                  textAnchor={compact ? "end" : "start"}
                  dominantBaseline={compact ? "middle" : "auto"}
                  style={{ fontSize: compact ? 11.5 : 12.5, fontWeight: 600, paintOrder: "stroke", stroke: "#fff", strokeWidth: 4, strokeLinejoin: "round" }}
                >
                  {compact
                    ? `${top.r.model}: ${top.extrap ? "≈" : ""}${fx(vy(top))}`
                    : `${top.r.model}, ${usd(top.r.spentUsd)} session: ${top.extrap ? "≈" : ""}${fx(vy(top))}`}
                </text>
              </svg>
              {hit && (
                <div className="viz-tip" style={hit.cy < 110 ? { left: tipLeft, top: hit.cy + 14, transform: "none" } : { left: tipLeft, top: hit.cy - 10 }} role="status">
                  <strong>{hit.title}</strong>
                  {hit.lines.map((l) => <span key={l}>{l}</span>)}
                </div>
              )}
            </div>
          </div>
        </figure>
        <figure className="pt-figure ar-cost-fig">
          <div className="ar-cost-head">Training compute only</div>
          <div className="viz ar-cost" ref={rightRef}>
            <TrainBars W={WR} />
          </div>
        </figure>
      </div>
    </div>
  );
}
