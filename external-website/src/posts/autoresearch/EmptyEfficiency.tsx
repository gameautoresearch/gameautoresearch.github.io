import { useMemo, useRef, useState } from "react";
import { useAgentView } from "../../view";
import { Table } from "../shared/text";
import { CMYK, modelStyle } from "./colors";
import { useWidth } from "./Frontier13";
import raw from "./empty-efficiency.json";

// Polytopia empty-starter campaign (reports/polytopia/frontier6-2026-11-19/spend-frontier/):
// 13 models built an RL trainer from boilerplate (no starter recipe) with 5 L40S GPU-hours
// (+2 h continuation for the six night-1 sessions) and submitted a recipe for a 15-min fresh final.
// Compares every network and fresh final against the unmodified PPO starter trained by the host.

type ModelInfo = {
  key: string;
  label: string;
  networks: number;
  continued: boolean;
  gpuH: number;
  best: { run: string; elo: number; lo: number; hi: number; gpu_h: number };
  final: number | null;
};
type Net = {
  model: string;
  run: string;
  cont: boolean;
  elo: number;
  lo: number;
  hi: number;
  gpuH: number;
  trainH: number;
};
type Final = {
  model: string;
  run: string;
  cont: boolean;
  elo: number;
  lo: number;
  hi: number;
  trainH: number;
};
type StarterRated = { label: string; seconds: number; elo: number; lo: number; hi: number };
type StarterMapped = { hours: number; kind: string; elo: number; seeds: number[] };

const DATA = raw as unknown as {
  usdPerGpuHour: number;
  games: number;
  models: ModelInfo[];
  networks: Net[];
  finals: Final[];
  references: { name: string; elo: number }[];
  starter: { rated: StarterRated[]; mapped: StarterMapped[]; offset: number };
};

const RATE = DATA.usdPerGpuHour;
const TMIN = 1 / 60;
const TMAX = 10;
const MODEL_BY_KEY = Object.fromEntries(DATA.models.map((m) => [m.key, m]));
const TOP_KEYS = DATA.models.slice(0, 5).map((m) => m.key);

const fmtE = (v: number) => {
  const r = Math.round(v);
  return r > 0 ? `+${r}` : r < 0 ? `−${-r}` : "0";
};
const fmtUsd = (v: number) => `$${v < 10 ? v.toFixed(2) : v.toFixed(1)}`;
const fmtT = (h: number) =>
  h < 1 ? (h * 60 < 1 ? `${Math.round(h * 3600)} s` : `${Math.round(h * 60)} min`) : `${+h.toFixed(2)} h`;
const winPct = (deltaElo: number) => 100 / (1 + 10 ** (-deltaElo / 400));
const fmtWin = (deltaElo: number) => {
  const p = winPct(deltaElo);
  return p < 0.5 ? "<1%" : p > 99.5 ? ">99%" : `${Math.round(p)}%`;
};

function starterHoursFor(elo: number): number | null {
  const curve = [{ hours: 0.25, elo: 0 }, ...DATA.starter.mapped];
  if (elo <= 0) return null;
  for (let i = 1; i < curve.length; i++) {
    const a = curve[i - 1], b = curve[i];
    if (elo <= b.elo) {
      const f = (elo - a.elo) / (b.elo - a.elo);
      return Math.exp(Math.log(a.hours) + f * (Math.log(b.hours) - Math.log(a.hours)));
    }
  }
  return null;
}

const STARTER_PTS = [
  ...DATA.starter.rated.map((p) => ({ hours: p.seconds / 3600, elo: p.elo, lo: p.lo, hi: p.hi, rated: true, label: p.label, seeds: [] as number[] })),
  ...DATA.starter.mapped.map((p) => ({ hours: p.hours, elo: p.elo, lo: p.elo, hi: p.elo, rated: false, label: `${p.hours} h of host training, ${p.kind}`, seeds: p.seeds })),
];

const REFS = [
  { name: "200M starter", elo: DATA.references.find((r) => r.name === "starter-200m")!.elo },
  { name: "15-min starter", elo: 0 },
];

// Export headline numbers for prose in Post.tsx.
const bestFinal5h = [...DATA.finals].filter((f) => !f.cont).sort((a, b) => b.elo - a.elo)[0];
const bestFinalAll = [...DATA.finals].sort((a, b) => b.elo - a.elo)[0];
const betterModels = new Set(DATA.finals.filter((f) => f.elo > 0).map((f) => f.model));
export const emptyEff = {
  models: DATA.models.length,
  beatingStarter: betterModels.size,
  bestFinal5hModel: MODEL_BY_KEY[bestFinal5h.model].label,
  bestFinal5hElo: Math.round(bestFinal5h.elo),
  bestFinal5hWinPct: Math.round(winPct(bestFinal5h.elo)),
  bestFinal5hStarterMin: Math.round((starterHoursFor(bestFinal5h.elo) ?? 0.25) * 60),
  bestFinal5hMult: +((starterHoursFor(bestFinal5h.elo) ?? 0.25) / 0.25).toFixed(1),
  bestFinalAllElo: Math.round(bestFinalAll.elo),
  bestFinalAllStarterMin: Math.round((starterHoursFor(bestFinalAll.elo) ?? 0.25) * 60),
  bestFinalAllMult: +((starterHoursFor(bestFinalAll.elo) ?? 0.25) / 0.25).toFixed(1),
};

type PlotItem =
  | { kind: "net"; n: Net }
  | { kind: "final"; f: Final };

type Hit = {
  cx: number;
  cy: number;
  key: string;
  title: string;
  lines: string[];
};

export default function EmptyEfficiency() {
  const agent = useAgentView();
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref);
  const [view, setView] = useState<"train" | "spend">("train");
  const [logAxis, setLogAxis] = useState<boolean>(true);
  const [fullY, setFullY] = useState<boolean>(false);
  const [hl, setHl] = useState<string | null>(null);
  const [hit, setHit] = useState<Hit | null>(null);

  const switchView = (next: "train" | "spend") => {
    setView(next);
    setLogAxis(next === "train");
    setHit(null);
  };

  const train = view === "train";
  const items: PlotItem[] = useMemo(
    () =>
      train
        ? [...DATA.networks.map((n) => ({ kind: "net" as const, n })), ...DATA.finals.map((f) => ({ kind: "final" as const, f }))]
        : DATA.networks.map((n) => ({ kind: "net" as const, n })),
    [train],
  );

  if (agent) {
    return (
      <Table
        head={["Researcher", "GPU-h used", "Best network (Elo)", "Fresh 15-min final (Elo)", "Win % vs 15-min starter"]}
        align={["l", "r", "r", "r", "r"]}
        rows={DATA.models.map((m) => [
          m.label,
          m.gpuH.toFixed(2),
          `${fmtE(m.best.elo)} (${m.best.run})`,
          m.final === null ? "—" : fmtE(m.final),
          m.final === null ? "—" : fmtWin(m.final),
        ])}
      />
    );
  }

  const narrow = W < 540;
  const H = Math.round(Math.max(350, Math.min(520, W * 0.58)));
  const m = { l: 48, r: narrow ? 12 : 106, t: 18, b: 42 };
  const yMin = fullY ? -1450 : -700;
  const yMax = 560;
  const yStep = fullY ? 250 : 100;

  const itemX = (it: PlotItem) =>
    train ? Math.max(it.kind === "net" ? it.n.trainH : it.f.trainH, TMIN) : (it.kind === "net" ? it.n.gpuH : 0) * RATE;
  const itemElo = (it: PlotItem) => (it.kind === "net" ? it.n.elo : it.f.elo);
  const itemModel = (it: PlotItem) => (it.kind === "net" ? it.n.model : it.f.model);
  const stX = (h: number) => (train ? Math.max(h, TMIN) : h * RATE);

  const xMax = train ? TMAX : 24;
  const xLo = train ? TMIN : 0.1;

  const sx = (v: number) =>
    logAxis
      ? m.l + (Math.log(Math.max(v, xLo) / xLo) / Math.log(xMax / xLo)) * (W - m.l - m.r)
      : m.l + (v / xMax) * (W - m.l - m.r);
  const sy = (e: number) => m.t + (1 - (e - yMin) / (yMax - yMin)) * (H - m.t - m.b);

  const xTicks: [number, string][] = logAxis
    ? (train
        ? (narrow ? [1, 5, 15, 60, 240] : [1, 2, 5, 15, 30, 60, 120, 240, 480]).map((x) => x / 60)
        : narrow
          ? [0.1, 1, 10]
          : [0.1, 0.25, 0.5, 1, 2, 5, 10, 20]
      )
        .filter((x) => x <= xMax)
        .map((x) => [x, train ? fmtT(x) : `$${x}`])
    : (() => {
        const out: [number, string][] = [];
        const step = train ? (narrow ? 2 : 1) : narrow ? 5 : 4;
        for (let x = 0; x <= xMax + 1e-9; x += step) out.push([x, train ? (x === 0 ? "0" : fmtT(x)) : `$${x}`]);
        return out;
      })();

  const yTicks: number[] = [];
  for (let y = Math.ceil(yMin / yStep) * yStep; y <= yMax; y += yStep) yTicks.push(y);

  // Frontier records across all models.
  const recs: PlotItem[] = [];
  {
    let best = -Infinity;
    for (const it of [...items].sort((a, b) => itemX(a) - itemX(b) || itemElo(b) - itemElo(a))) {
      if (itemElo(it) > best) {
        best = itemElo(it);
        recs.push(it);
      }
    }
  }
  const recSet = new Set(recs);

  // Per-model best-so-far step curves for the spend view.
  const activeKey = hit?.key ?? hl;
  const lineKeys = train ? [] : [...new Set([...TOP_KEYS, ...(activeKey && activeKey !== "starter" ? [activeKey] : [])])];
  const stepLines = lineKeys.map((k) => {
    const own = DATA.networks.filter((n) => n.model === k).sort((a, b) => a.gpuH - b.gpuH || b.elo - a.elo);
    const pts: [number, number][] = [];
    let best = -Infinity;
    for (const n of own) {
      if (n.elo > best) {
        best = n.elo;
        pts.push([n.gpuH * RATE, n.elo]);
      }
    }
    const endX = MODEL_BY_KEY[k].gpuH * RATE;
    const d = pts.length
      ? pts.map(([x, y], i) => (i ? `H${sx(x).toFixed(1)}V${sy(Math.max(y, yMin)).toFixed(1)}` : `M${sx(x).toFixed(1)},${sy(Math.max(y, yMin)).toFixed(1)}`)).join("") +
        `H${sx(endX).toFixed(1)}`
      : "";
    return { key: k, d, color: modelStyle(MODEL_BY_KEY[k].label).color };
  });

  // Build hover targets.
  const hits: Hit[] = [];
  for (const it of items) {
    const x = itemX(it);
    const e = itemElo(it);
    const k = itemModel(it);
    const label = MODEL_BY_KEY[k].label;
    const cx = sx(x);
    const cy = e < yMin ? H - m.b - 4 : sy(e);
    if (it.kind === "final") {
      const f = it.f;
      const sh = starterHoursFor(f.elo);
      hits.push({
        cx,
        cy,
        key: k,
        title: `${label} · ${f.run}`,
        lines: [
          `Elo ${fmtE(f.elo)} (${fmtWin(f.elo)} vs 15-min starter)`,
          `15 min fresh retrain from scratch ($0.73)`,
          ...(sh ? [`starter needs ${fmtT(sh)} for this Elo (${(sh / 0.25).toFixed(1)}× efficiency)`] : []),
        ],
      });
    } else {
      const n = it.n;
      hits.push({
        cx,
        cy,
        key: k,
        title: `${label} · ${n.run}${n.cont ? " (+2 h)" : ""}`,
        lines: [
          `Elo ${fmtE(n.elo)} (${fmtWin(n.elo)} vs 15-min starter)${recSet.has(it) ? " · frontier" : ""}`,
          `lease ${fmtT(n.trainH)} · cumulative research ${fmtUsd(n.gpuH * RATE)} (${n.gpuH.toFixed(2)} GPU-h)`,
        ],
      });
    }
  }
  for (const p of STARTER_PTS) {
    if (stX(p.hours) > xMax) continue;
    hits.push({
      cx: sx(stX(p.hours)),
      cy: sy(p.elo),
      key: "starter",
      title: `Unmodified starter (${p.rated ? "rated here" : "mapped"})`,
      lines: [
        p.label,
        `Elo ${fmtE(p.elo)} · ${fmtT(p.hours)} (${fmtUsd(p.hours * RATE)})`,
      ],
    });
  }

  const move = (ev: React.PointerEvent<SVGSVGElement>) => {
    const box = ev.currentTarget.getBoundingClientRect();
    const k = W / box.width;
    const px = (ev.clientX - box.left) * k;
    const py = (ev.clientY - box.top) * k;
    let near: Hit | null = null;
    let bd = 15 * 15;
    for (const h of hits) {
      const dd = (h.cx - px) ** 2 + (h.cy - py) ** 2;
      if (dd < bd) {
        bd = dd;
        near = h;
      }
    }
    setHit(near);
  };

  const starterRated = STARTER_PTS.filter((p) => p.rated && stX(p.hours) <= xMax);
  const starterMapped = STARTER_PTS.filter((p) => !p.rated && stX(p.hours) <= xMax);
  const linePath = (pts: typeof STARTER_PTS) =>
    pts.map((p, i) => `${i ? "L" : "M"}${sx(stX(p.hours)).toFixed(1)},${sy(p.elo).toFixed(1)}`).join("");
  const lastStarter = starterMapped[starterMapped.length - 1] ?? starterRated[starterRated.length - 1];

  // Non-overlapping labels on key positive frontier points.
  const labels: { x: number; y: number; anchor: "start" | "end"; text: string }[] = [];
  if (!narrow) {
    const boxes: [number, number, number][] = [];
    for (const it of [...recs].reverse()) {
      const e = itemElo(it);
      if (e < Math.max(yMin, 120)) continue;
      const k = itemModel(it);
      const run = it.kind === "final" ? "15m final" : it.n.run;
      const text = `${MODEL_BY_KEY[k].label} (${run})`;
      const w = text.length * 5.8;
      let x = sx(itemX(it)) - 8;
      let y = sy(e) - 9;
      let anchor: "start" | "end" = "end";
      if (x - w < m.l + 4) {
        x = sx(itemX(it)) + 8;
        y = sy(e) + 14;
        anchor = "start";
      }
      const bx: [number, number] = anchor === "end" ? [x - w, x] : [x, x + w];
      if (boxes.some((b) => bx[0] < b[1] && bx[1] > b[0] && Math.abs(y - b[2]) < 15)) continue;
      boxes.push([bx[0], bx[1], y]);
      labels.push({ x, y, anchor, text });
    }
  }

  const orderOf = (it: PlotItem) => {
    const k = itemModel(it);
    return (activeKey === k ? 4 : TOP_KEYS.includes(k) ? 2 : 1) + (it.kind === "final" ? 0.5 : 0);
  };
  const clippedCount = items.filter((it) => itemElo(it) < yMin).length;
  const tipLeft = hit ? Math.max(0, hit.cx + 12 + 245 > W ? hit.cx - 255 : hit.cx + 12) : 0;

  return (
    <div className="viz ar-cost" ref={ref}>
      <div className="fr-controls">
        <div className="ar-tabs ar-tabs-wrap" role="tablist" aria-label="Efficiency view">
          <button role="tab" aria-selected={train} onClick={() => switchView("train")}>
            Training time only
          </button>
          <button role="tab" aria-selected={!train} onClick={() => switchView("spend")}>
            Cumulative research spend
          </button>
        </div>
        <div className="ar-tabs" role="tablist" aria-label="Axis options">
          <button role="tab" aria-selected={logAxis} onClick={() => setLogAxis(true)}>Log x</button>
          <button role="tab" aria-selected={!logAxis} onClick={() => setLogAxis(false)}>Linear x</button>
          <button role="tab" aria-selected={!fullY} onClick={() => setFullY(false)}>Zoomed</button>
          <button role="tab" aria-selected={fullY} onClick={() => setFullY(true)}>Full range</button>
        </div>
      </div>

      <div className="viz-legend">
        <span><i style={{ background: CMYK.k }} />Unmodified starter (rated / mapped)</span>
        {train && <span><i className="sq" />Fresh 15-min final (from scratch)</span>}
        <span><i className="ring" />Frontier record</span>
        {!train && <span><i style={{ background: CMYK.m }} />Model best-so-far frontier</span>}
      </div>

      <div className="ar-cost-plot">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={
            train
              ? "Empty-starter Polytopia Elo against GPU training time, compared to the unmodified starter"
              : "Empty-starter Polytopia Elo against cumulative research GPU spend"
          }
          onPointerMove={move}
          onPointerDown={move}
          onPointerLeave={() => setHit(null)}
          style={{ touchAction: "pan-y" }}
        >
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className="viz-grid" />
              <text x={m.l - 8} y={sy(t)} className="viz-axis" textAnchor="end" dominantBaseline="middle">
                {fmtE(t)}
              </text>
            </g>
          ))}
          {xTicks.map(([x, label]) => (
            <g key={`${x}-${label}`}>
              <line x1={sx(x)} x2={sx(x)} y1={m.t} y2={H - m.b} className="viz-grid" />
              <text x={sx(x)} y={H - m.b + 16} className="viz-axis" textAnchor="middle">
                {label}
              </text>
            </g>
          ))}

          {train ? (
            <g>
              <line x1={sx(0.25)} x2={sx(0.25)} y1={m.t} y2={H - m.b} stroke={CMYK.k} strokeDasharray="2 3" opacity={0.4} />
              <text x={sx(0.25) + 5} y={H - m.b - 8} className="viz-axis">15-min cap</text>
            </g>
          ) : (
            <g>
              <line x1={sx(5 * RATE)} x2={sx(5 * RATE)} y1={m.t} y2={H - m.b} stroke={CMYK.k} strokeDasharray="2 3" opacity={0.4} />
              <text x={sx(5 * RATE) + 5} y={H - m.b - 8} className="viz-axis">5 GPU-h (+2 h cont. →)</text>
            </g>
          )}

          <text x={W - m.r} y={H - 4} className="viz-axis" textAnchor="end">
            {train
              ? `GPU training time behind the network (one L40S, ${logAxis ? "log" : "linear"} scale)`
              : `cumulative research GPU spend at $${RATE.toFixed(2)}/h (${logAxis ? "log" : "linear"} scale)`}
          </text>
          <text x={m.l} y={m.t - 6} className="viz-axis">
            Elo (15-min PPO starter = 0)
          </text>

          {REFS.filter((r) => r.elo >= yMin && r.elo <= yMax).map((r) => (
            <g key={r.name}>
              <line x1={m.l} x2={W - m.r} y1={sy(r.elo)} y2={sy(r.elo)} className="viz-ref" />
              {!narrow && (
                <text x={W - m.r + 6} y={sy(r.elo)} className="fr-ref" dominantBaseline="middle">
                  {r.name} ({fmtE(r.elo)})
                </text>
              )}
            </g>
          ))}

          {stepLines.map((ln) => (
            <path
              key={ln.key}
              d={ln.d}
              fill="none"
              stroke={ln.color}
              strokeWidth={activeKey === ln.key ? 2.4 : 1.7}
              strokeLinejoin="round"
              opacity={0.9}
            />
          ))}

          {[...items].sort((a, b) => orderOf(a) - orderOf(b)).map((it, idx) => {
            const k = itemModel(it);
            const e = itemElo(it);
            const cx = sx(itemX(it));
            const clipped = e < yMin;
            const cy = clipped ? H - m.b - 4 : sy(e);
            const col = modelStyle(MODEL_BY_KEY[k].label).color;
            const isTop = TOP_KEYS.includes(k);
            const isHl = activeKey === k;
            if (clipped) {
              return (
                <path
                  key={`c-${idx}`}
                  d={`M${cx - 3.5},${cy - 3}L${cx + 3.5},${cy - 3}L${cx},${cy + 3}Z`}
                  fill={col}
                  opacity={0.7}
                />
              );
            }
            if (it.kind === "final") {
              const s = isHl ? 5.6 : 4.8;
              return (
                <rect
                  key={`f-${idx}`}
                  x={cx - s}
                  y={cy - s}
                  width={2 * s}
                  height={2 * s}
                  rx={1.4}
                  fill={col}
                  stroke={CMYK.k}
                  strokeWidth={1.2}
                />
              );
            }
            return (
              <circle
                key={`n-${idx}`}
                cx={cx}
                cy={cy}
                r={isHl ? 4.2 : isTop ? 3.5 : 2.8}
                fill={col}
                stroke="#fff"
                strokeWidth={1}
                opacity={isTop ? 0.9 : 0.55}
              />
            );
          })}

          <g>
            <path d={linePath(starterRated)} fill="none" stroke={CMYK.k} strokeWidth={2.4} strokeLinejoin="round" />
            {starterMapped.length > 0 && (
              <path
                d={linePath(starterMapped)}
                fill="none"
                stroke={CMYK.k}
                strokeWidth={2}
                strokeDasharray="6 4"
                strokeLinejoin="round"
              />
            )}
            {[...starterRated, ...starterMapped].map((p, i) => {
              const cx = sx(stX(p.hours)), cy = sy(p.elo), q = 4.2;
              return (
                <path
                  key={i}
                  d={`M${cx},${cy - q}L${cx + q},${cy}L${cx},${cy + q}L${cx - q},${cy}Z`}
                  fill={p.rated ? "#fff" : CMYK.k}
                  stroke={CMYK.k}
                  strokeWidth={1.6}
                />
              );
            })}
            {!narrow && lastStarter && (
              <text
                x={sx(stX(lastStarter.hours)) - 8}
                y={sy(lastStarter.elo) - 10}
                textAnchor="end"
                className="viz-label"
                style={{ fontWeight: 600, paintOrder: "stroke", stroke: "#fff", strokeWidth: 4 }}
              >
                Starter recipe ({fmtT(lastStarter.hours)})
              </text>
            )}
          </g>

          {recs.filter((it) => itemElo(it) >= yMin).map((it, i) => {
            const k = itemModel(it);
            const col = modelStyle(MODEL_BY_KEY[k].label).color;
            return (
              <circle
                key={`rec-${i}`}
                cx={sx(itemX(it))}
                cy={sy(itemElo(it))}
                r={4.2}
                fill={col}
                stroke={CMYK.k}
                strokeWidth={1.3}
              />
            );
          })}

          {labels.map((lb) => (
            <text
              key={lb.text}
              x={lb.x}
              y={lb.y}
              textAnchor={lb.anchor}
              className="viz-axis"
              style={{ fill: "var(--ink)", fontWeight: 500, paintOrder: "stroke", stroke: "#fff", strokeWidth: 4, strokeLinejoin: "round" }}
            >
              {lb.text}
            </text>
          ))}
        </svg>

        {hit && (
          <div
            className="viz-tip"
            style={hit.cy < 110 ? { left: tipLeft, top: hit.cy + 14, transform: "none" } : { left: tipLeft, top: hit.cy - 10 }}
            role="status"
          >
            <strong>{hit.title}</strong>
            {hit.lines.map((l) => (
              <span key={l}>{l}</span>
            ))}
          </div>
        )}
      </div>

      <div className="fr-legend">
        {DATA.models.map((mo) => {
          const st = modelStyle(mo.label);
          const on = hl === null || hl === mo.key;
          return (
            <button
              key={mo.key}
              type="button"
              aria-pressed={on}
              onClick={() => setHl((cur) => (cur === mo.key ? null : mo.key))}
              title="Click to highlight this model"
            >
              <i style={{ background: st.color }} />
              {mo.label}
            </button>
          );
        })}
        {hl && (
          <button type="button" className="fr-all" onClick={() => setHl(null)}>
            Show all
          </button>
        )}
      </div>

      {clippedCount > 0 && (
        <p className="ar-tipline">
          {clippedCount} weaker {train ? "networks and finals" : "networks"} below Elo {fmtE(yMin)} sit on the bottom edge as triangles. Switch to Full range to place them.
        </p>
      )}
    </div>
  );
}
