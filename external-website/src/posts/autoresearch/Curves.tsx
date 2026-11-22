import { useState } from "react";
import { LineChart } from "../shared/text";
import type { Series } from "../shared/text";
import { emptyPokemon, emptyPolytopia, frontier } from "./data";
import { CMYK, modelStyle } from "./colors";

const signed = (v: number) => (v > 0 ? `+${Math.round(v)}` : `${Math.round(v)}`);
const REF = "#c8c8c8";
const gpu = (h: number) => (h < 1 ? `${Math.round(h * 60)} min` : `${+h.toFixed(2)} GPU-h`);

type Curve = { model: string; end: number; best: number[][]; final?: number };

// Best research network so far against research GPU-hours, as a step line per model, with a linear/log x toggle.
function ResearchCurves({ models, refs, yDomain, height, yTicksLabel, defaultLog = false }: {
  models: readonly Curve[];
  refs: { name: string; y: number }[];
  yDomain: [number, number];
  height: number;
  yTicksLabel: string;
  defaultLog?: boolean;
}) {
  const [log, setLog] = useState(defaultLog);
  const maxH = Math.ceil(Math.max(...models.map((m) => m.end)));
  const fx = (h: number) => (log ? Math.log10(h) : h);
  const first = Math.min(0.04, 0.8 * Math.min(...models.map((m) => m.best[0][0])));
  const xDomain: [number, number] = log ? [Math.log10(first), Math.log10(maxH + 0.2)] : [0, maxH];
  const logTicks = [0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, maxH].filter((h) => h >= first);
  const xTicks = log ? logTicks.map(Math.log10) : Array.from({ length: maxH + 1 }, (_, i) => i);
  const xFormat = (v: number) => {
    const h = log ? 10 ** v : v;
    return h < 1 && h > 0 ? `${Math.round(h * 60)} min` : `${+h.toFixed(1)} h`;
  };
  const series: Series[] = [
    ...refs.map((r) => ({
      name: r.name, color: REF, dashed: true, noDots: true,
      points: [{ x: xDomain[0], y: r.y, label: r.name }, { x: xDomain[1], y: r.y, label: `${r.name} end` }],
    })),
    ...models.map((m) => {
      const pts = m.best.filter(([, y]) => y >= yDomain[0]).map(([h, y]) => ({
        x: fx(h), y, label: `${m.model}, new best after ${gpu(h)}`, note: `${signed(y)} Elo`,
      }));
      const last = m.best[m.best.length - 1];
      pts.push({ x: fx(m.end), y: last[1], label: `${m.model}, end of budget (${gpu(m.end)})`, note: m.final === undefined ? `best ${signed(last[1])} Elo` : `best ${signed(last[1])}; fresh 15-min final ${signed(m.final)}` });
      return { name: m.model, ...modelStyle(m.model), points: pts };
    }),
  ];
  return (
    <div>
      <div className="ar-tabs" role="tablist" aria-label="Time axis">
        <button role="tab" aria-selected={!log} onClick={() => setLog(false)}>Linear time</button>
        <button role="tab" aria-selected={log} onClick={() => setLog(true)}>Log time</button>
      </div>
      <LineChart series={series} stepped xDomain={xDomain} yDomain={yDomain} xTicks={xTicks} xFormat={xFormat}
        yFormat={(v) => signed(v)} height={height} yLabel={yTicksLabel} xLabel="research GPU-hours spent" />
    </div>
  );
}

export function EmptyPolytopiaCurves() {
  return (
    <ResearchCurves
      models={emptyPolytopia.models}
      refs={[
        { name: `PPO starter, 200M samples (${signed(emptyPolytopia.refs.starter200m)})`, y: emptyPolytopia.refs.starter200m },
        { name: `Untrained network (${signed(emptyPolytopia.refs.random)})`, y: emptyPolytopia.refs.random },
      ]}
      yDomain={[-1700, 400]}
      height={320}
      defaultLog
      yTicksLabel="Elo (15-min PPO starter = 0)"
    />
  );
}

export function EmptyPokemonCurves() {
  return (
    <ResearchCurves
      models={emptyPokemon.models}
      refs={[{ name: "15-min PPO starter (0)", y: 0 }]}
      yDomain={[-60, 120]}
      height={280}
      yTicksLabel="Elo (15-min PPO starter = 0)"
    />
  );
}

// Polytopia's compute frontier: the unmodified starter trained 30 min to 14.5 h (log time), a fit to the runs of 4 h
// or less extended to 16 h, and two researchers' recipes at 2 h and 8 h.
const L2 = Math.log2;
function fit(pts: number[][]) {
  const xs = pts.map(([h]) => L2(h)), ys = pts.map(([, y]) => y);
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length, my = ys.reduce((a, b) => a + b, 0) / ys.length;
  const b = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  return { a: my - b * mx, b };
}
export const frontierFit = fit(frontier.starter);
const HOURS: Record<number, string> = { [-1]: "30 min", 0: "1 h", 1: "2 h", 2: "4 h", 3: "8 h", 4: "16 h" };
const dots = (pts: number[][], who: string) => pts.map(([h, y], i) => ({
  x: L2(h), y, label: `${who}, ${h < 1 ? `${h * 60} min` : `${h} h`} (#${i + 1})`, note: `${signed(y)} Elo`,
}));

export function FrontierChart() {
  const { a, b } = frontierFit;
  const series: Series[] = [
    { name: "Starter, snapshots inside two 16 h runs", color: "#d2d2d2", dotsOnly: true, points: dots(frontier.snapshots, "Starter snapshot") },
    { name: "Starter, fresh runs of 30 min to 4 h", color: CMYK.k, dotsOnly: true, points: dots(frontier.starter, "Starter run") },
    {
      name: `Fit to the runs of 4 h or less: +${Math.round(b)} Elo per doubling`, color: "#9a9a9a", dashed: true, noDots: true,
      points: [-1, 4].map((x) => ({ x, y: a + b * x, label: `fit ${x}` })),
    },
    { name: "Claude Opus 5.5's recipe", color: modelStyle("opus").color, dotsOnly: true, points: dots(frontier.opus, "Opus's recipe") },
    { name: "DeepSeek V4.1 Flash's recipe", color: modelStyle("deepseek").color, dotsOnly: true, points: dots(frontier.flash, "Flash's recipe") },
  ];
  return (
    <LineChart series={series} xDomain={[-1.3, 4.2]} yDomain={[-100, 450]} xTicks={[-1, 0, 1, 2, 3, 4]}
      xFormat={(v) => HOURS[v] ?? ""} yFormat={(v) => signed(v)} height={320}
      yLabel="Elo (PPO starter at 200M samples = 0)" xLabel="training time (log scale)" />
  );
}
