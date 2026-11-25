import { useRef, useState } from "react";
import { useAgentView } from "../../view";
import { LineChart, Table } from "../shared/text";
import type { Series } from "../shared/text";
import raw from "./budget.json";
import setupFigure from "./setup-figure.svg?raw";
import { DIAMOND, drawBorder, drawCity, drawForest, drawGround, drawMountain, drawRing, drawUnit, HH, HW, Pen, TEAMS } from "./polytopia/mockArt";
import { CMYK, modelStyle } from "./colors";
import { useWidth } from "./Frontier13";

// Polytopia research-budget campaign (tools/build_budget.py): 42 sessions (7 models x 6 budgets, the last four models
// added on 2026-11-19) that each improved the PPO starter under a
// dollar budget, their recipes retrained for 1 h (two seeds) and for 8 h (one seed), the starter trained the same
// way as the yardstick, and two forecasting tests built on the results.

type CurvePoint = { hours: number; elo: number; starterHours: number | null };
type Recipe = {
  key: string; model: string; cell: string; budgetUsd: number; spentUsd: number; researchHours: number; broken: boolean;
  multiplier1h: number | null; seedMultipliers: number[]; finals1h: number[]; curve: CurvePoint[]; early: { hours: number; elo: number }[];
  long: { hours: number; elo: number; lo: number; hi: number }[];   // past 8 h, from the recipe's own 64 h run (GPT-6.2 Astra t2 only)
  lo8: number | null; hi8: number | null;
};
const DATA = raw as unknown as {
  usdPerGpuHour: number;
  games: { budget: number; curves: number };
  starter: { hours: number; elo: number; n: number }[];
  starterEarly: { hours: number; elo: number; n: number }[];   // 10, 15 and 20 minutes, from second runs
  zero: { elo: number; lo: number; hi: number; games: number };
  recipes: Recipe[];
  longAsOf: number | null;   // when the long runs' ratings were taken (Unix seconds)
  polyForecast: {
    truth: Record<string, { elo: number; mean1h: number }>;
    best: string;
    models: { key: string; model: string; train_further: string; ranking: string[]; p50: Record<string, number>; mae_p50: number }[];
    baselines: { name: string; train_further: string; p50: Record<string, number>; mae_p50: number }[];
  };
  pokeForecast: {
    models: { key: string; model: string; sets: number; pairwise: number; top1: string; rho35: number; mae: number }[];
    baselines: { name: string; pairwise: number | null; top1: string | null; rho35: number | null; mae: number | null }[];
  };
  pokeSkill: { rule: number; items: number; finals: number; runs: number; width: [number, number]; pairs: [string, string]; models: { model: string; skill: number; skillCi: [number, number]; lead: number; ci: [number, number]; leadFinals: number; ciFinals: [number, number] }[] };
};

const signed = (v: number) => (v > 0 ? `+${Math.round(v)}` : `${Math.round(v)}`);
const hrs = (h: number) => (h < 1 ? `${Math.round(h * 60)} min` : `${+h.toFixed(1)} h`);
const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2; };

export const budgetGames = DATA.games;
export const BEST = DATA.recipes.find((r) => r.key === "astra62-t2")!;
const withCurves = DATA.recipes.filter((r) => r.curve.length);
const starter8 = DATA.starter.find((p) => p.hours === 8)!.elo;
// Recipes whose 8 h interval lies wholly above / below the starter's 8 h point.
export const at8 = {
  starter: starter8,
  n: withCurves.length,
  above: withCurves.filter((r) => r.lo8! > starter8).length,
  below: withCurves.filter((r) => r.curve[r.curve.length - 1].elo < starter8).length,
};

// ---------------------------------------------------------------------------------------------------------------
// The opening figure: a research agent and a compute budget go in, the budget is spent on training runs, and one
// small policy network comes out and plays an opponent. A fixed schematic (setup-figure.svg), styled by .ar-fig1; its board slot is filled below.

// The board in the figure is drawn by the replay viewer's art (polytopia/mockArt.ts): a 5 x 5 corner of a game, blue
// (the trained policy) pressing on orange's capital. Tile (x, y) sits at ((x - y) * HW, (x + y) * HH).
const BOARD_N = 5;
const BOARD_TEAM = [TEAMS.neutral, TEAMS.blue, TEAMS.orange];
const BOARD_ROWS = ["~~..^", "~^...", "T....", ".....", "...T~"];   // ~ water, ^ mountain, T forest
const BOARD_OWNER = (x: number, y: number) => (x <= 2 && y >= 2 ? 1 : x >= 2 && y <= 2 ? 2 : 0);
const BOARD_CITIES: Record<string, [owner: number, level: number]> = { "1,3": [1, 3], "3,1": [2, 2] };
const BOARD_UNITS: Record<string, [owner: number, kind: number]> = { "2,3": [1, 2], "2,2": [1, 2], "1,2": [1, 3], "3,2": [2, 2], "3,0": [2, 9] };
const boardArt = (() => {
  const cells = Array.from({ length: BOARD_N * BOARD_N }, (_, t) => [t % BOARD_N, Math.floor(t / BOARD_N)] as const)
    .sort((p, q) => p[0] + p[1] - q[0] - q[1] || p[1] - q[1]);
  const put = (x: number, y: number, html: string) => (html ? `<g transform="translate(${(x - y) * HW} ${(x + y) * HH})">${html}</g>` : "");
  const pen = (draw: (g: Pen) => void) => { const g = new Pen(); draw(g); return g.out.join(""); };
  const ground = cells.map(([x, y]) => put(x, y, pen((g) => drawGround(g, BOARD_ROWS[y][x] === "~"))));
  const surface = cells.map(([x, y]) => {
    const owner = BOARD_OWNER(x, y);
    if (!owner) return "";
    let edges = 0;
    [[0, -1], [1, 0], [0, 1], [-1, 0]].forEach(([dx, dy], j) => {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= BOARD_N || ny >= BOARD_N || BOARD_OWNER(nx, ny) !== owner) edges |= 1 << j;
    });
    const main = BOARD_TEAM[owner].main;
    return put(x, y, pen((g) => { g.poly(DIAMOND).fill({ color: main, alpha: 0.22 }); drawBorder(g, edges, main); }));
  });
  const pieces = cells.map(([x, y]) => {
    const city = BOARD_CITIES[`${x},${y}`], unit = BOARD_UNITS[`${x},${y}`], terrain = BOARD_ROWS[y][x];
    return put(x, y, pen((g) => {
      if (terrain === "^") drawMountain(g);
      if (terrain === "T") drawForest(g);
      if (city) drawCity(g, city[1], BOARD_TEAM[city[0]], true);
      if (unit) { drawRing(g, BOARD_TEAM[unit[0]]); drawUnit(g, unit[1], BOARD_TEAM[unit[0]]); }
    }));
  });
  return [...ground, ...surface, ...pieces].join("");
})();
const setupFigureHtml = setupFigure.replace("<!--board-->", boardArt);

export function BudgetCartoon() {
  if (useAgentView()) {
    return (
      <ul>
        <li><strong>Inputs:</strong> a research agent (for example Gemini 4 Neon, deciding things like “This run went flat. Try a higher learning rate.”) and a compute budget, a fixed number of dollars of GPU time.</li>
        <li><strong>Research:</strong> every training run draws down the budget. In the drawing the agent pays for four short test runs, then puts the rest into one long hero run of the best idea.</li>
        <li><strong>Output:</strong> one trained policy, a small neural network, which plays the game against an opponent policy. We grade the policy, not the agent.</li>
      </ul>
    );
  }
  return <div className="ar-fig1" dangerouslySetInnerHTML={{ __html: setupFigureHtml }} />;
}

// ---------------------------------------------------------------------------------------------------------------
// Elo against training time (log scale) for every recipe of the campaign and the starter.

const X0 = Math.log2(0.42), X1 = Math.log2(15.5), Y0 = -170, Y1 = 400;

export function RecipeCurves() {
  const agent = useAgentView();
  const [only, setOnly] = useState<string | null>(null);
  const [tip, setTip] = useState<string | null>(null);
  const yard = DATA.starter.filter((p) => p.hours <= 14);   // fresh finals and two-seed snapshot means only
  if (agent) {
    return (
      <Table
        head={["Recipe", "Research spend", "1 h", "2 h", "4 h", "8 h"]}
        rows={[
          ["Unmodified starter", "$0", ...[1, 2, 4, 8].map((h) => signed(yard.find((p) => p.hours === h)!.elo))],
          ...withCurves.map((r) => [`${r.model}, ${r.cell}`, `$${r.spentUsd.toFixed(2)}`, ...[1, 2, 4, 8].map((h) => signed(r.curve.find((p) => p.hours === h)!.elo))]),
        ]}
      />
    );
  }
  const W = 720, H = 380, m = { l: 46, r: 14, t: 16, b: 36 };
  const sx = (h: number) => m.l + ((Math.log2(h) - X0) / (X1 - X0)) * (W - m.l - m.r);
  const sy = (e: number) => H - m.b - ((e - Y0) / (Y1 - Y0)) * (H - m.t - m.b);
  const line = (pts: { hours: number; elo: number }[]) => pts.map((p, i) => `${i ? "L" : "M"}${sx(p.hours).toFixed(1)},${sy(p.elo).toFixed(1)}`).join("");
  const models = [...new Set(withCurves.map((r) => r.model))];
  // "Same Elo, less training": horizontal links from the best recipe's curve to the starter's.
  const links = BEST.curve.filter((p) => [2, 6].includes(p.hours) && p.starterHours);
  return (
    <div className="viz">
      <div className="ar-tabs ar-tabs-wrap" role="tablist" aria-label="Which recipes to show">
        <button role="tab" aria-selected={only === null} onClick={() => setOnly(null)}>All 17 recipes</button>
        {models.map((mo) => <button key={mo} role="tab" aria-selected={only === mo} onClick={() => setOnly(mo)}>{mo}</button>)}
      </div>
      <div className="viz-legend">
        <span><i style={{ background: CMYK.k }} />Unmodified starter</span>
        <span><i style={{ background: CMYK.m }} />{BEST.model}, {BEST.cell} (best recipe)</span>
        <span><i style={{ background: "#b9b9b9" }} />Every other researched recipe</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Elo against training time for each recipe and the starter">
        {[-100, 0, 100, 200, 300, 400].map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className="viz-grid" />
            <text x={m.l - 8} y={sy(t)} className="viz-axis" textAnchor="end" dominantBaseline="middle">{signed(t)}</text>
          </g>
        ))}
        {[0.5, 1, 2, 4, 8, 14].map((h) => <text key={h} x={sx(h)} y={H - m.b + 18} className="viz-axis" textAnchor="middle">{hrs(h)}</text>)}
        <text x={W - m.r} y={H - 2} className="viz-axis" textAnchor="end">training time on one L40S (log scale)</text>
        <text x={m.l} y={m.t - 4} className="viz-axis">Elo (starter at 200M samples = 0)</text>
        {withCurves.filter((r) => r !== BEST && (!only || r.model === only)).map((r) => (
          <path key={r.key} d={line(r.curve)} fill="none" stroke={only ? modelStyle(r.model).color : "#b9b9b9"} strokeWidth={only ? 1.75 : 1.25}
            onMouseEnter={() => setTip(r.key)} onMouseLeave={() => setTip(null)}>
            <title>{`${r.model}, ${r.cell}: ${signed(r.curve[r.curve.length - 1].elo)} at 8 h`}</title>
          </path>
        ))}
        <path d={line(yard)} fill="none" stroke={CMYK.k} strokeWidth={2.5} strokeLinejoin="round" />
        {yard.map((p) => <circle key={p.hours} cx={sx(p.hours)} cy={sy(p.elo)} r={3} fill={CMYK.k}><title>{`Starter, ${hrs(p.hours)}: ${signed(p.elo)}`}</title></circle>)}
        {(!only || only === BEST.model) && (
          <>
            {links.map((p) => (
              <g key={p.hours}>
                <line x1={sx(p.hours)} x2={sx(p.starterHours!)} y1={sy(p.elo)} y2={sy(p.elo)} stroke={CMYK.m} strokeWidth={1.5} strokeDasharray="4 3" />
                <text x={(sx(p.hours) + sx(p.starterHours!)) / 2} y={sy(p.elo) - 6} className="viz-axis" textAnchor="middle" style={{ fill: CMYK.mDeep, fontWeight: 600, paintOrder: "stroke", stroke: "#fff", strokeWidth: 4 }}>
                  {hrs(p.hours)} instead of {hrs(p.starterHours!)}
                </text>
              </g>
            ))}
            <path d={line(BEST.curve)} fill="none" stroke={CMYK.m} strokeWidth={2.5} strokeLinejoin="round" />
            {BEST.curve.map((p) => <circle key={p.hours} cx={sx(p.hours)} cy={sy(p.elo)} r={3} fill={CMYK.m}><title>{`${BEST.model}, ${hrs(p.hours)}: ${signed(p.elo)}`}</title></circle>)}
          </>
        )}
      </svg>
      <p className="ar-tipline">{tip ? (() => { const r = withCurves.find((x) => x.key === tip)!; return `${r.model}, ${r.cell}: ${signed(r.curve[1].elo)} at 1 h, ${signed(r.curve[r.curve.length - 1].elo)} at 8 h`; })() : "Hover a grey line to see whose recipe it is."}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Elo against total cost: what a recipe's research session spent on GPU plus its training, in dollars on a linear
// axis. A recipe's line starts at its research spend with an untrained network; the starter's starts at $0.

const RATE = DATA.usdPerGpuHour;
const ZERO = DATA.zero;
const winPct = (deltaElo: number) => 100 / (1 + 10 ** (-deltaElo / 400));
const fmtWin = (deltaElo: number) => {
  const p = winPct(deltaElo);
  return p < 0.5 ? "<1%" : p > 99.5 ? ">99%" : `${Math.round(p)}%`;
};
const elo = (v: number) => { const r = Math.round(v); return r > 0 ? `+${r}` : r < 0 ? `−${-r}` : "0"; };
const usd = (v: number) => `$${v < 10 ? v.toFixed(2) : v.toFixed(1)}`;
const costOf = (r: Recipe, hours: number) => r.spentUsd + hours * RATE;
// The starter's Elo after `h` hours of training, log-linear between its measured points.
function starterAt(h: number) {
  const s = DATA.starter;
  if (h <= s[0].hours) return s[0].elo;
  for (let i = 1; i < s.length; i++) {
    const a = s[i - 1], b = s[i];
    if (h <= b.hours) return a.elo + (Math.log(h / a.hours) / Math.log(b.hours / a.hours)) * (b.elo - a.elo);
  }
  return s[s.length - 1].elo;
}
type CostPoint = { r: Recipe; hours: number; elo: number; cost: number; final: boolean };
const costPoints: CostPoint[] = withCurves.flatMap((r) => [
  ...r.curve.map((p) => ({ r, hours: p.hours, elo: p.elo, cost: costOf(r, p.hours), final: false })),
  ...r.finals1h.map((e) => ({ r, hours: 1, elo: e, cost: costOf(r, 1), final: true })),
]);
// The frontier: every recipe point that beats all cheaper ones.
const costFrontier = (() => {
  let best = -Infinity;
  return [...costPoints].sort((a, b) => a.cost - b.cost || b.elo - a.elo).filter((q) => (q.elo > best ? ((best = q.elo), true) : false));
})();
const COST_MAX = Math.ceil(Math.max(...costPoints.map((q) => q.cost)) / 5) * 5;
// Each recipe after 8 h of training against the starter trained for the same total money.
const sameMoney = withCurves.map((r) => {
  const last = r.curve[r.curve.length - 1], total = costOf(r, last.hours), starterElo = starterAt(total / RATE);
  return { r, elo: last.elo, total, starterElo, margin: last.elo - starterElo };
}).sort((a, b) => b.margin - a.margin);
// TOP: the recipe furthest ahead of the starter bought with the same money at 8 h (highlighted at rest in Figure 7).
// BEST stays GPT-6.2 Astra's $8.70 · 2 h recipe, the one with its own run past 8 h.
export const TOP = sameMoney[0].r;
const ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"];
const lastStarter = DATA.starter[DATA.starter.length - 1];
export const payback = {
  n: sameMoney.length, bestMargin: Math.round(sameMoney[0].margin), bestKey: sameMoney[0].r.key,
  topElo: Math.round(sameMoney[0].elo), topWinPct: Math.round(winPct(sameMoney[0].margin)),
  topTotalUsd: Math.round(sameMoney[0].total), ahead: sameMoney.filter((x) => x.margin > 10).length,
  starterLongHours: lastStarter.hours, starterLongElo: lastStarter.elo,
  starterLongCostUsd: Math.round(lastStarter.hours * RATE),
  bestRank: ORDINAL[sameMoney.findIndex((x) => x.r === BEST)] ?? `#${sameMoney.findIndex((x) => x.r === BEST) + 1}`,
};

// The best recipe's own run past 8 h, each point against the starter trained for the same total money.
const longPts = BEST.long.map((p) => {
  const cost = costOf(BEST, p.hours), s = starterAt(cost / RATE);
  return { ...p, cost, starterElo: s, margin: p.elo - s };
});
const starterLate = DATA.starter.filter((p) => p.hours >= 16);
export const longRun = {
  hours: longPts.length ? longPts[longPts.length - 1].hours : 8,
  starterHours: DATA.starter[DATA.starter.length - 1].hours,
  minMargin: Math.round(Math.min(...longPts.map((p) => p.margin))),
  maxMargin: Math.round(Math.max(...longPts.map((p) => p.margin))),
  minWinPct: Math.round(winPct(Math.min(...longPts.map((p) => p.margin)))),
  maxWinPct: Math.round(winPct(Math.max(...longPts.map((p) => p.margin)))),
  lo: Math.min(...longPts.map((p) => p.elo)),
  hi: Math.max(...longPts.map((p) => p.elo)),
  starterLo: Math.min(...starterLate.map((p) => p.elo)),
  starterHi: Math.max(...starterLate.map((p) => p.elo)),
  asOf: DATA.longAsOf ? new Date(DATA.longAsOf * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "America/New_York" }) : null,
};

type CostHit = { cx: number; cy: number; key: string; title: string; lines: string[] };

// `only` pins one view and drops the view tabs.
export function CostCurves({ only }: { only?: "zero" | "trained" | "long" } = {}) {
  const agent = useAgentView();
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref), narrow = W < 560;
  const [view, setView] = useState<"zero" | "trained" | "long">(only ?? "zero");
  const fromZero = view === "zero";
  const [hit, setHit] = useState<CostHit | null>(null);
  if (agent) {
    return (
      <>
        <Table
          head={["Recipe", "Research spend", "Total cost at 8 h", "Win % vs same-money starter", "Elo after 8 h"]}
          align={["l", "r", "r", "r", "r"]}
          rows={sameMoney.map((x) => [`${x.r.model}, ${x.r.cell}`, usd(x.r.spentUsd), usd(x.total), fmtWin(x.margin), elo(x.elo)])}
        />
        {longPts.length > 0 && (
          <Table
            head={[`${BEST.model}, ${BEST.cell}: trained for`, "Total cost", "Win % vs same-money starter", "Elo (95% interval)"]}
            align={["l", "r", "r", "r"]}
            rows={longPts.map((p) => [hrs(p.hours), usd(p.cost), fmtWin(p.margin), `${elo(p.elo)} (${elo(p.lo)} to ${elo(p.hi)})`])}
          />
        )}
      </>
    );
  }
  const tabs = only ? null : (
    <div className="ar-tabs ar-tabs-wrap" role="tablist" aria-label="View">
      <button role="tab" aria-selected={view === "zero"} onClick={() => setView("zero")}>From zero training</button>
      <button role="tab" aria-selected={view === "trained"} onClick={() => setView("trained")}>Trained networks only</button>
      {longPts.length > 0 && <button role="tab" aria-selected={view === "long"} onClick={() => setView("long")}>{BEST.model}, long run</button>}
    </div>
  );
  if (view === "long") return <div className="viz ar-cost" ref={ref}>{tabs}<LongRunPlot W={W} /></div>;
  const H = Math.round(fromZero ? Math.max(420, Math.min(640, W * 0.64)) : Math.max(340, Math.min(520, W * 0.55)));
  const m = { l: 46, r: narrow ? 12 : 92, t: 20, b: 40 };
  const step = fromZero ? 100 : 50;
  const lowest = Math.min(...costPoints.map((q) => q.elo), ...(fromZero ? [ZERO.elo] : []));
  const highest = Math.max(...costPoints.map((q) => q.elo));
  const Y0 = Math.floor((lowest - 30) / step) * step, Y1 = Math.max(400, Math.ceil((highest + 20) / step) * step);
  const sx = (c: number) => m.l + (c / COST_MAX) * (W - m.l - m.r);
  const sy = (e: number) => m.t + (1 - (e - Y0) / (Y1 - Y0)) * (H - m.t - m.b);
  const yTicks: number[] = [];
  for (let y = Y0; y <= Y1; y += step) yTicks.push(y);
  const xTicks: number[] = [];
  for (let x = 0; x <= COST_MAX; x += narrow ? 10 : 5) xTicks.push(x);
  const models = [...new Set(withCurves.map((r) => r.model))];
  const yard = DATA.starter.filter((p) => p.hours * RATE <= COST_MAX);
  const yardEnd = yard[yard.length - 1];
  const on = hit?.key ?? null;
  const rank = (r: Recipe) => (r.key === on ? 2 : r === TOP ? 1 : 0);
  const frontier = new Set(costFrontier);
  const lastFrontier = costFrontier[costFrontier.length - 1];

  const hits: CostHit[] = [];
  const recipeHit = (q: CostPoint): CostHit => {
    const s = starterAt(q.cost / RATE);
    return {
      cx: sx(q.cost), cy: sy(q.elo), key: q.r.key,
      title: `${q.r.model}, ${q.r.cell}${q.final ? " (fresh 1 h final)" : ""}`,
      lines: [
        `${hrs(q.hours)} trained: wins ${fmtWin(q.elo - s)} vs same-money starter (${elo(q.elo - s)} Elo)${frontier.has(q) ? ", best at this cost" : ""}`,
        `research ${usd(q.r.spentUsd)} + training ${usd(q.hours * RATE)} = ${usd(q.cost)}`,
        `rating ${elo(q.elo)} vs starter ${elo(s)}`,
      ],
    };
  };
  for (const q of costPoints) hits.push(recipeHit(q));
  // The first 20 minutes (second runs of each recipe and the starter) are drawn only in the view from zero.
  const early = (r: Recipe) => (fromZero ? r.early : []);
  const yardEarly = fromZero ? DATA.starterEarly : [];
  for (const r of withCurves) for (const p of early(r)) hits.push(recipeHit({ r, hours: p.hours, elo: p.elo, cost: costOf(r, p.hours), final: false }));
  for (const p of yardEarly) hits.push({ cx: sx(p.hours * RATE), cy: sy(p.elo), key: "starter", title: "Unmodified starter",
    lines: [`${hrs(p.hours)} of training (mean of ${p.n} runs): Elo ${elo(p.elo)}`, `cost ${usd(p.hours * RATE)}, no research`] });
  for (const p of yard) hits.push({ cx: sx(p.hours * RATE), cy: sy(p.elo), key: "starter", title: "Unmodified starter",
    lines: [`${hrs(p.hours)} of training (${p.n > 1 ? `mean of ${p.n} runs` : "one run"}): Elo ${elo(p.elo)}`, `cost ${usd(p.hours * RATE)}, no research`] });
  if (fromZero) {
    for (const r of withCurves) hits.push({ cx: sx(r.spentUsd), cy: sy(ZERO.elo), key: r.key, title: `${r.model}, ${r.cell}`,
      lines: [`untrained: Elo ${elo(ZERO.elo)}, the random player's rating`, `research ${usd(r.spentUsd)}, no training yet`] });
    hits.push({ cx: sx(0), cy: sy(ZERO.elo), key: "starter", title: "Unmodified starter", lines: [`untrained: Elo ${elo(ZERO.elo)}, the random player's rating`, "cost $0.00"] });
  }
  const move = (ev: React.PointerEvent<SVGSVGElement>) => {
    const box = ev.currentTarget.getBoundingClientRect(), k = W / box.width;
    const px = (ev.clientX - box.left) * k, py = (ev.clientY - box.top) * k;
    let near: CostHit | null = null, d = 14 * 14;
    for (const h of hits) { const dd = (h.cx - px) ** 2 + (h.cy - py) ** 2; if (dd < d) { d = dd; near = h; } }
    setHit(near);
  };
  const path = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");
  const tipLeft = hit ? Math.max(0, hit.cx + 12 + 240 > W ? hit.cx - 252 : hit.cx + 12) : 0;

  return (
    <div className="viz ar-cost" ref={ref}>
      {tabs}
      <div className="viz-legend">
        {models.map((mo) => { const st = modelStyle(mo); return <span key={mo}><i style={{ background: st.color }} />{mo}'s recipes</span>; })}
        <span><i style={{ background: CMYK.k }} />Unmodified starter</span>
        <span><i className="ring" />Best recipe Elo yet for that total cost</span>
        <span><i className="sq" />Fresh 1 h finals, two seeds</span>
        {fromZero && <span><i className="ring open" />Untrained network</span>}
      </div>
      <div className="ar-cost-plot">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Elo against total cost, research plus training, for every recipe and the unmodified starter"
        onPointerMove={move} onPointerDown={move} onPointerLeave={() => setHit(null)} style={{ touchAction: "pan-y" }}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className="viz-grid" />
            <text x={m.l - 8} y={sy(t)} className="viz-axis" textAnchor="end" dominantBaseline="middle">{elo(t)}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} className="viz-grid" />
            <text x={sx(t)} y={H - m.b + 16} className="viz-axis" textAnchor="middle">${t}</text>
          </g>
        ))}
        <text x={W - m.r} y={H - 4} className="viz-axis" textAnchor="end">
          {narrow ? "total cost: research + training" : `total cost: research GPU spend + training, one L40S at $${RATE.toFixed(2)} per hour`}
        </text>
        <text x={m.l} y={m.t - 8} className="viz-axis">strength (Elo; 200M starter = 0)</text>
        <g>
          <line x1={m.l} x2={W - m.r} y1={sy(lastStarter.elo)} y2={sy(lastStarter.elo)} className="viz-ref" />
          {!narrow && <text x={W - m.r + 6} y={sy(lastStarter.elo)} className="fr-ref" dominantBaseline="middle">{`${lastStarter.hours} h starter ($${payback.starterLongCostUsd})`}</text>}
        </g>
        {/* The best recipe is in full colour and the others sit softly behind it. */}
        {[...withCurves].sort((a, b) => rank(a) - rank(b)).map((r) => {
          const st = modelStyle(r.model), lit = on === r.key, top = r === TOP;
          const stroke = { fill: "none", stroke: st.color, strokeWidth: lit ? 2.6 : top ? 2.4 : 1.4, opacity: top ? 1 : 0.25 };
          const first = r.curve[0];
          return (
            <g key={r.key}>
              {fromZero && <path d={path([[r.spentUsd, ZERO.elo], ...early(r).map((p) => [costOf(r, p.hours), p.elo] as [number, number]), [costOf(r, first.hours), first.elo]])} strokeLinejoin="round" {...stroke} />}
              {early(r).map((p) => <circle key={p.hours} cx={sx(costOf(r, p.hours))} cy={sy(p.elo)} r={lit ? 3.4 : 2} fill={st.color} opacity={top ? 1 : 0.35} />)}
              {fromZero && <circle cx={sx(r.spentUsd)} cy={sy(ZERO.elo)} r={lit ? 4 : 3} fill="#fff" stroke={st.color} strokeWidth={1.4} opacity={top ? 1 : 0.35} />}
              <path d={path(r.curve.map((p) => [costOf(r, p.hours), p.elo]))} strokeLinejoin="round" {...stroke} />
              {r.curve.map((p) => <circle key={p.hours} cx={sx(costOf(r, p.hours))} cy={sy(p.elo)} r={lit ? 4 : 2.8} fill={st.color} stroke="#fff" strokeWidth={1} opacity={top ? 1 : 0.35} />)}
              {r.finals1h.map((e, i) => { const s = lit ? 5 : 4; return <rect key={i} x={sx(costOf(r, 1)) - s} y={sy(e) - s} width={2 * s} height={2 * s} rx={1.2} fill={st.color} stroke={CMYK.k} strokeWidth={1} opacity={top ? 0.9 : 0.35} />; })}
            </g>
          );
        })}
        <g>
          <path d={path([...(fromZero ? [[0, ZERO.elo] as [number, number]] : []), ...[...yardEarly, ...yard].map((p) => [p.hours * RATE, p.elo] as [number, number])])}
            fill="none" stroke={CMYK.k} strokeWidth={on === "starter" ? 3.4 : 2.5} strokeLinejoin="round" />
          {fromZero && <circle cx={sx(0)} cy={sy(ZERO.elo)} r={4.4} fill="#fff" stroke={CMYK.k} strokeWidth={2} />}
          {[...yardEarly, ...yard].map((p) => { const cx = sx(p.hours * RATE), cy = sy(p.elo), q = p.hours < 0.45 ? 3.4 : 4.6; return <path key={p.hours} d={`M${cx},${cy - q}L${cx + q},${cy}L${cx},${cy + q}L${cx - q},${cy}Z`} fill={CMYK.k} stroke="#fff" strokeWidth={1.1} />; })}
          {!narrow && <text x={sx(yardEnd.hours * RATE) + 9} y={sy(yardEnd.elo)} className="viz-label" dominantBaseline="middle" style={{ fontWeight: 600 }}>Starter, {hrs(yardEnd.hours)}</text>}
        </g>
        {fromZero && <text x={sx(0) + 9} y={sy(ZERO.elo) + 18} className="viz-axis">untrained = random play, {elo(ZERO.elo)}</text>}
        {costFrontier.map((q, i) => <circle key={i} cx={sx(q.cost)} cy={sy(q.elo)} r={3.9} fill={modelStyle(q.r.model).color} stroke={CMYK.k} strokeWidth={1.3} opacity={q.r.key === TOP.key ? 1 : 0.45} />)}
        {!narrow && (
          <text x={sx(lastFrontier.cost) - 8} y={sy(lastFrontier.elo) - 10} className="viz-label" textAnchor="end" style={{ fontWeight: 600, paintOrder: "stroke", stroke: "#fff", strokeWidth: 4, strokeLinejoin: "round" }}>
            {lastFrontier.r.model}, {lastFrontier.r.cell}
          </text>
        )}
      </svg>
      {hit && (
        <div className="viz-tip" style={hit.cy < 110 ? { left: tipLeft, top: hit.cy + 14, transform: "none" } : { left: tipLeft, top: hit.cy - 10 }} role="status">
          <strong>{hit.title}</strong>
          {hit.lines.map((l) => <span key={l}>{l}</span>)}
        </div>
      )}
      </div>
    </div>
  );
}

// The best recipe's whole run, 30 minutes to its latest snapshot, against the starter's own long run, both on total cost.
function LongRunPlot({ W }: { W: number }) {
  const [hit, setHit] = useState<CostHit | null>(null);
  const narrow = W < 560;
  const H = Math.round(Math.max(360, Math.min(520, W * 0.56)));
  const m = { l: 46, r: narrow ? 12 : 104, t: 20, b: 40 };
  const st = modelStyle(BEST.model);
  const run = [...BEST.curve.map((p) => ({ hours: p.hours, elo: p.elo, lo: null as number | null, hi: null as number | null })), ...BEST.long];
  const yard = DATA.starter;
  const XMAX = Math.ceil((yard[yard.length - 1].hours * RATE) / 20) * 20;
  const Y0 = -100, Y1 = 450;
  const sx = (c: number) => m.l + (c / XMAX) * (W - m.l - m.r);
  const sy = (e: number) => m.t + (1 - (e - Y0) / (Y1 - Y0)) * (H - m.t - m.b);
  const path = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");
  const yTicks: number[] = [];
  for (let y = Y0; y <= Y1; y += 50) yTicks.push(y);
  const xTicks: number[] = [];
  for (let x = 0; x <= XMAX; x += narrow ? 40 : 20) xTicks.push(x);
  const hits: CostHit[] = [
    ...run.map((p) => {
      const cost = costOf(BEST, p.hours), s = starterAt(cost / RATE);
      return { cx: sx(cost), cy: sy(p.elo), key: BEST.key, title: `${BEST.model}, ${BEST.cell}`, lines: [
        `${hrs(p.hours)} trained: wins ${fmtWin(p.elo - s)} vs same-money starter (${elo(p.elo - s)} Elo)`,
        `research ${usd(BEST.spentUsd)} + training ${usd(p.hours * RATE)} = ${usd(cost)}`,
        `rating ${elo(p.elo)}${p.lo !== null ? ` (95% ${elo(p.lo)} to ${elo(p.hi!)})` : ""} vs starter ${elo(s)}`,
      ] };
    }),
    ...yard.map((p) => ({ cx: sx(p.hours * RATE), cy: sy(p.elo), key: "starter", title: "Unmodified starter",
      lines: [`${hrs(p.hours)} of training (${p.n > 1 ? `mean of ${p.n} runs` : "one run"}): Elo ${elo(p.elo)}`, `cost ${usd(p.hours * RATE)}, no research`] })),
  ];
  const move = (ev: React.PointerEvent<SVGSVGElement>) => {
    const box = ev.currentTarget.getBoundingClientRect(), k = W / box.width;
    const px = (ev.clientX - box.left) * k, py = (ev.clientY - box.top) * k;
    let near: CostHit | null = null, d = 14 * 14;
    for (const h of hits) { const dd = (h.cx - px) ** 2 + (h.cy - py) ** 2; if (dd < d) { d = dd; near = h; } }
    setHit(near);
  };
  const tipLeft = hit ? Math.max(0, hit.cx + 12 + 240 > W ? hit.cx - 252 : hit.cx + 12) : 0;
  const x8 = sx(costOf(BEST, 8)), lastRun = run[run.length - 1], lastYard = yard[yard.length - 1];
  const band = BEST.long.map((p) => [costOf(BEST, p.hours), p.lo, p.hi] as const);
  return (
    <>
      <div className="viz-legend">
        <span><i style={{ background: st.color }} />{BEST.model}, {BEST.cell}: one run, still training</span>
        <span><i style={{ background: CMYK.k }} />Unmodified starter</span>
        <span><i style={{ background: st.color, opacity: 0.18 }} />95% interval past 8 h</span>
      </div>
      <div className="ar-cost-plot">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Elo against total cost for ${BEST.model}'s best recipe trained to ${hrs(lastRun.hours)} and the unmodified starter trained to ${hrs(lastYard.hours)}`}
          onPointerMove={move} onPointerDown={move} onPointerLeave={() => setHit(null)} style={{ touchAction: "pan-y" }}>
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className="viz-grid" />
              <text x={m.l - 8} y={sy(t)} className="viz-axis" textAnchor="end" dominantBaseline="middle">{elo(t)}</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} className="viz-grid" />
              <text x={sx(t)} y={H - m.b + 16} className="viz-axis" textAnchor="middle">${t}</text>
            </g>
          ))}
          <text x={W - m.r} y={H - 4} className="viz-axis" textAnchor="end">{narrow ? "total cost: research + training" : `total cost: research GPU spend + training, one L40S at $${RATE.toFixed(2)} per hour`}</text>
          <text x={m.l} y={m.t - 8} className="viz-axis">strength (Elo; 200M starter = 0)</text>
          <line x1={x8} x2={x8} y1={m.t} y2={H - m.b} stroke={CMYK.k} strokeDasharray="2 3" opacity={0.5} />
          <text x={x8 + 5} y={H - m.b - 8} className="viz-axis">Main chart ends at 8 h</text>
          <path d={`${band.map(([c, lo], i) => `${i ? "L" : "M"}${sx(c).toFixed(1)},${sy(lo).toFixed(1)}`).join("")}${[...band].reverse().map(([c, , hi]) => `L${sx(c).toFixed(1)},${sy(hi).toFixed(1)}`).join("")}Z`} fill={st.color} opacity={0.14} />
          <path d={path(yard.map((p) => [p.hours * RATE, p.elo]))} fill="none" stroke={CMYK.k} strokeWidth={hit?.key === "starter" ? 3.2 : 2.2} strokeLinejoin="round" />
          {yard.map((p) => { const cx = sx(p.hours * RATE), cy = sy(p.elo), q = 3.6; return <path key={p.hours} d={`M${cx},${cy - q}L${cx + q},${cy}L${cx},${cy + q}L${cx - q},${cy}Z`} fill={CMYK.k} stroke="#fff" strokeWidth={0.9} />; })}
          <path d={path(run.map((p) => [costOf(BEST, p.hours), p.elo]))} fill="none" stroke={st.color} strokeWidth={hit?.key === BEST.key ? 3.2 : 2.4} strokeLinejoin="round" />
          {run.map((p) => <circle key={p.hours} cx={sx(costOf(BEST, p.hours))} cy={sy(p.elo)} r={3} fill={st.color} stroke="#fff" strokeWidth={1} />)}
          {!narrow && <text x={sx(costOf(BEST, lastRun.hours)) + 9} y={sy(lastRun.elo) - 12} className="viz-label" style={{ fontWeight: 600, paintOrder: "stroke", stroke: "#fff", strokeWidth: 4 }}>{BEST.model}, {hrs(lastRun.hours)}</text>}
          {!narrow && <text x={sx(lastYard.hours * RATE) + 9} y={sy(lastYard.elo)} className="viz-label" dominantBaseline="middle" style={{ fontWeight: 600 }}>Starter, {hrs(lastYard.hours)}</text>}
        </svg>
        {hit && (
          <div className="viz-tip" style={hit.cy < 110 ? { left: tipLeft, top: hit.cy + 14, transform: "none" } : { left: tipLeft, top: hit.cy - 10 }} role="status">
            <strong>{hit.title}</strong>
            {hit.lines.map((l) => <span key={l}>{l}</span>)}
          </div>
        )}
      </div>
    </>
  );
}

// What the best recipe saves at each training length, in starter hours of equal Elo.
export function PaybackTable() {
  const researchGpuH = BEST.spentUsd / DATA.usdPerGpuHour;
  const rows = BEST.curve.filter((p) => [1, 2, 4, 6, 7].includes(p.hours) && p.starterHours).map((p) => {
    const saved = p.starterHours! - p.hours;
    return [hrs(p.hours), signed(p.elo), hrs(p.starterHours!), `${Math.round((saved / p.starterHours!) * 100)}%`, `${saved.toFixed(1)} h`, saved > researchGpuH ? "yes" : "no"];
  });
  const last = BEST.curve[BEST.curve.length - 1];
  rows.push([hrs(last.hours), signed(last.elo), "more than 14 h", "over 43%", "over 6 h", "yes"]);
  return (
    <Table
      head={["Trained for", "Elo", "Starter needs", "GPU time saved", "Hours saved", `Covers the ${researchGpuH.toFixed(1)} GPU-h of research?`]}
      align={["l", "r", "r", "r", "r", "r"]}
      rows={rows}
    />
  );
}

// Compute multiplier after a 1 h final (two-seed mean) against the research budget, one dot per session.
const BUDGETS = [0.01, 2.9, 8.7, 23];
const BUDGET_LABEL = ["zero-shot", "$2.90", "$8.70", "$23"];
export function BudgetMultipliers() {
  const models = [...new Set(DATA.recipes.map((r) => r.model))];
  // Models side by side within each budget, spread over +-0.33 of a budget step.
  const step = models.length > 1 ? 0.66 / (models.length - 1) : 0;
  const ys = DATA.recipes.flatMap((r) => (r.multiplier1h === null ? [] : [r.multiplier1h]));
  const yDomain: [number, number] = [Math.min(0.6, Math.floor(Math.min(...ys) * 10) / 10), Math.max(2.1, Math.ceil(Math.max(...ys) * 10) / 10)];
  const series: Series[] = models.map((mo, k) => ({
    name: mo, color: modelStyle(mo).color, dotsOnly: true,
    points: DATA.recipes.filter((r) => r.model === mo && r.multiplier1h !== null).map((r, i) => ({
      x: BUDGETS.indexOf(r.budgetUsd) + (k - (models.length - 1) / 2) * step + (r.cell.includes("limit") ? (i % 2 ? 0.025 : -0.025) : 0),
      y: r.multiplier1h!,
      label: `${r.model}, ${r.cell}`,
      note: `${r.multiplier1h!.toFixed(2)}× the starter's compute (seeds ${r.seedMultipliers.map((s) => `${s}×`).join(", ")}); spent $${r.spentUsd.toFixed(2)} in ${r.researchHours} h`,
    })),
  }));
  return (
    <LineChart series={series} xDomain={[-0.5, 3.5]} yDomain={yDomain} xTicks={[0, 1, 2, 3]} xFormat={(v) => BUDGET_LABEL[v] ?? ""}
      yFormat={(v) => `${v}×`} height={280} refLine={{ y: 1, label: "the unmodified starter" }} yLabel="starter compute of equal Elo, 1 h final"
      xLabel="research budget" />
  );
}

// ---------------------------------------------------------------------------------------------------------------
// Forecasting tests.

const NAMES: Record<string, string> = { A: "Claude Opus 5.5, $8.70 (4 h limit)", B: "GPT-6.2 Astra, $23", C: "Claude Opus 5.5, $8.70", D: "GPT-6.2 Sol, zero-shot" };
export const polyForecastN = DATA.polyForecast.models.length;

export function PolyForecast() {
  const f = DATA.polyForecast;
  const order = Object.keys(f.truth).sort((a, b) => f.truth[b].elo - f.truth[a].elo);
  return (
    <Table
      head={["Recipe (hidden from forecasters)", "1 h Elo", "Median forecast, 8 h", "Measured, 8 h", "Chosen to train further", "Ranked last"]}
      align={["l", "r", "r", "r", "r", "r"]}
      rows={order.map((k) => [
        `${k}: ${NAMES[k]}`,
        signed(f.truth[k].mean1h),
        signed(median(f.models.map((mo) => mo.p50[k]))),
        k === f.best ? <strong>{signed(f.truth[k].elo)}</strong> : signed(f.truth[k].elo),
        `${f.models.filter((mo) => mo.train_further === k).length} of ${f.models.length}`,
        `${f.models.filter((mo) => mo.ranking[mo.ranking.length - 1] === k).length} of ${f.models.length}`,
      ])}
    />
  );
}

export function PokeForecast() {
  const f = DATA.pokeForecast;
  const full = f.models.filter((mo) => mo.sets === 7).sort((a, b) => b.pairwise - a.pairwise);
  const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
  const base = f.baselines.find((b) => b.name === "best practice score")!;
  const last = f.baselines.find((b) => b.name === "last practice score")!;
  const rows = [
    ...full.map((mo) => ({ who: mo.model, pair: mo.pairwise, top: mo.top1, rho: mo.rho35, mae: mo.mae.toFixed(1), base: false })),
    { who: "Rule: rank by best practice score", pair: base.pairwise!, top: base.top1!, rho: base.rho35!, mae: "—", base: true },
    { who: "Rule: rank by last practice score", pair: last.pairwise!, top: last.top1!, rho: last.rho35!, mae: "—", base: true },
  ].sort((a, b) => b.pair - a.pair);
  return (
    <Table
      head={["Forecaster", "Pairs ordered correctly", "Best session found", "ρ over 35", "Mean error (lift points)"]}
      align={["l", "r", "r", "r", "r"]}
      rows={rows.map((r) => [r.base ? <em>{r.who}</em> : r.who, pct(r.pair), r.top, r.rho.toFixed(2), r.mae])}
    />
  );
}
export const pokeForecast = (() => {
  const full = DATA.pokeForecast.models.filter((mo) => mo.sets === 7);
  const base = DATA.pokeForecast.baselines.find((b) => b.name === "best practice score")!.pairwise!;
  return { n: full.length, beat: full.filter((mo) => mo.pairwise > base).length, base, best: Math.max(...full.map((mo) => mo.pairwise)), worst: Math.min(...full.map((mo) => mo.pairwise)) };
})();

// Pokémon forecasts scored as relative skill (ordering, spacing and p10-p90 ranges once the overall level is aligned
// with the truth) over 35 final recipes plus 146 training runs, shown as each forecaster's lead over a simple rule
// (a line on practice score, and for runs on run length too, fitted to the other sessions). Intervals are 90% paired
// bootstraps that resample whole research sessions.
const f2 = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
const SKILL = [...DATA.pokeSkill.models].sort((a, b) => b.skill - a.skill);
const SKILL_TONE = { above: CMYK.c, same: "#6b6b6b", below: CMYK.m };
const skillTone = (mo: (typeof SKILL)[number]) => (mo.ci[0] > 0 ? "above" : mo.ci[1] < 0 ? "below" : "same");
export const pokeSkill = {
  n: SKILL.length,
  rule: DATA.pokeSkill.rule,
  items: DATA.pokeSkill.items,
  finals: DATA.pokeSkill.finals,
  runs: DATA.pokeSkill.runs,
  aboveBlind: SKILL.filter((mo) => mo.skillCi[0] > 0).length,
  aboveFinals: SKILL.filter((mo) => mo.ciFinals[0] > 0).length,
  above: SKILL.filter((mo) => skillTone(mo) === "above").map((mo) => mo.model),
  below: SKILL.filter((mo) => skillTone(mo) === "below").map((mo) => mo.model),
  ahead: SKILL.filter((mo) => mo.lead > 0).length,
  best: SKILL[0],
};

export function ForecastLead() {
  const agent = useAgentView();
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref);
  const range = (ci: [number, number]) => `${f2(ci[0])} to ${f2(ci[1])}`;
  if (agent) {
    return (
      <Table
        head={["Forecaster", "Relative skill (0 = blind guess, 1 = perfect)", "90% interval", "Lead over the curve fit", "90% interval of the lead"]}
        rows={[
          ...SKILL.map((mo) => [mo.model, f2(mo.skill), range(mo.skillCi), f2(mo.lead), range(mo.ci)]),
          ["The curve fit", f2(DATA.pokeSkill.rule), "", "0", ""],
        ]}
      />
    );
  }
  const narrow = W < 480, HI = 0.55, FIT = DATA.pokeSkill.rule;
  const row = narrow ? 46 : 32;
  const m = { l: narrow ? 10 : 162, r: narrow ? 46 : 50, t: 28, b: 44 };
  const H = m.t + m.b + SKILL.length * row;
  const sx = (v: number) => m.l + (Math.max(0, Math.min(HI, v)) / HI) * (W - m.l - m.r);
  const fmtSkill = (v: number) => v.toFixed(2);
  return (
    <div className="viz ar-fc" ref={ref}>
      <div className="viz-legend ar-fc-legend">
        {pokeSkill.above.length > 0 && <span><i style={{ background: SKILL_TONE.above }} />Beats baseline</span>}
        <span><i style={{ background: SKILL_TONE.same }} />Tied</span>
        {pokeSkill.below.length > 0 && <span><i style={{ background: SKILL_TONE.below }} />Below baseline</span>}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Each model's forecast skill against a simple baseline, with 90% intervals">
        {[0, 0.1, 0.2, 0.3, 0.4, 0.5].map((t) => (
          <g key={t}>
            <line
              x1={sx(t)}
              x2={sx(t)}
              y1={t === 0 ? m.t - 4 : m.t}
              y2={H - m.b}
              className={t === 0 ? undefined : "viz-grid"}
              stroke={t === 0 ? CMYK.k : undefined}
              strokeWidth={t === 0 ? 1.4 : undefined}
            />
            <text x={sx(t)} y={H - m.b + 20} className="ar-fc-tick" textAnchor="middle">
              {t === 0 ? "0" : t.toFixed(1)}
            </text>
          </g>
        ))}
        <line
          x1={sx(FIT)}
          x2={sx(FIT)}
          y1={m.t - 4}
          y2={H - m.b}
          stroke="#444"
          strokeWidth={1.5}
          strokeDasharray="3 4"
          strokeLinecap="round"
        />
        <text x={sx(FIT)} y={m.t - 11} className="ar-fc-ref" textAnchor="middle">
          Baseline ({fmtSkill(FIT)})
        </text>
        <text x={(sx(0) + sx(HI)) / 2} y={H - 4} className="ar-fc-axis-title" textAnchor="middle">
          Forecast skill
        </text>
        {SKILL.map((mo, i) => {
          const y = m.t + i * row + row / 2 + (narrow ? 8 : 0);
          const tone = skillTone(mo);
          const c = SKILL_TONE[tone];
          const isTop = tone === "above";
          return (
            <g key={mo.model}>
              {narrow ? (
                <text
                  x={m.l + 6}
                  y={y - 14}
                  className="ar-fc-model"
                  style={{ fontWeight: isTop ? 650 : 500 }}
                >
                  {mo.model}
                </text>
              ) : (
                <text
                  x={m.l - 14}
                  y={y}
                  className="ar-fc-model"
                  textAnchor="end"
                  dominantBaseline="middle"
                  style={{ fontWeight: isTop ? 650 : 500 }}
                >
                  {mo.model}
                </text>
              )}
              <line
                x1={sx(mo.skillCi[0])}
                x2={sx(mo.skillCi[1])}
                y1={y}
                y2={y}
                stroke={c}
                strokeOpacity={0.38}
                strokeWidth={5}
                strokeLinecap="round"
              />
              <circle cx={sx(mo.skill)} cy={y} r={6} fill={c} stroke="#fff" strokeWidth={1.6} />
              <text
                x={sx(mo.skillCi[1]) + 9}
                y={y}
                className="ar-fc-val"
                dominantBaseline="middle"
                style={{ fontWeight: isTop ? 700 : 600 }}
              >
                {fmtSkill(mo.skill)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
