import { useRef } from "react";
import { useAgentView } from "../../view";
import { Table } from "../shared/text";
import { modelStyle } from "./colors";
import { useWidth } from "./Frontier13";
import raw from "./top-scores.json";
import budgetRaw from "./budget.json";
import generalsRaw from "./generals-tournament.json";
import { starterHoursFor } from "./CostMultiplier";

// Overall model score (Figure 1): the ten models with all four shared tasks (Polytopia empty file, Pokémon 3v3,
// forecasting and the Polytopia research-budget recipes), plus the Generals tournament for all ten.
// Task scores come from top-scores.json
// (reports/top-level-score-2026-11-20/build.py), budget.json and generals-tournament.json.
// Each task is standardized against the frozen scale below
// (50 = the reference models' average, 10 points = 1 SD) and a model's tasks are averaged; the whisker is ±1 standard
// error of that average across its tasks (sample SD of its task scores / √n).

type Task = {
  id: string;
  name: string;
  game: string;
  unit: string;
  sessions: number;
  scores: Record<string, number>;
};
type Row = {
  model: string;
  score: number;
  lo: number;
  hi: number;
  // Each task's standardized score, in SCORE_TASKS order; null where the model didn't run that task.
  each: (number | null)[];
};

const DATA = raw as unknown as { tasks: Task[] };
const TASK_MAP = Object.fromEntries(DATA.tasks.map((t) => [t.id, t]));

// The recipe task uses every researched cell of the Polytopia research-budget campaign ($2.90, $8.70, $23, and $8.70 with
// a 2 h or 4 h limit; zero-shot is left out because it has no research). Each recipe is measured twice, both as a
// training-compute multiplier (starter's training hours to reach the same Elo / the recipe's hours; research spend not
// counted): the 1 h retrain on two seeds (multiplier1h) and the 8 h retrain read off its curve (Figure 2's yardstick).
// A recipe that crashes in its own trainer counts as 0.5× on both (the 1 h yardstick's lowest rung, as in the bench's
// build.py). For each measurement, log multiplier = model + cell is fit by least squares, so the models that only ran
// the $8.70 cell (`fill`: GPT-6 Astra, GPT-6.1 Sol, Gemini 4 Carbon, campaign poly-budget-fill-20261122b, same
// scales) are adjusted for how that cell went for everyone. The recipe score is exp(mean of the 1 h and 8 h model
// effects): the multiplier on a typical cell. It is standardized on a log scale since it is a ratio. Figures 2 and 3
// don't read `fill`.
type BudgetRecipe = {
  model: string;
  cell: string;
  broken: boolean;
  multiplier1h: number | null;
  curve: { hours: number; elo: number }[];
};
const BUDGET = budgetRaw as unknown as {
  recipes: BudgetRecipe[];
  fill?: BudgetRecipe[];
};
const budgetModel = (mo: string) =>
  mo === "Opus 5.5" ? "Claude Opus 5.5" : mo;
const CRASH_MULT = 0.5;
type CellObs = { model: string; cell: string; y: number };
const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;
// Two-way additive fit y = a[model] + b[cell], mean of b = 0, by alternating means (converges to least squares).
function fitCells(obs: CellObs[]): {
  a: Record<string, number>;
  b: Record<string, number>;
} {
  const models = [...new Set(obs.map((o) => o.model))];
  const cells = [...new Set(obs.map((o) => o.cell))];
  const a: Record<string, number> = Object.fromEntries(
    models.map((m) => [m, 0]),
  );
  const b: Record<string, number> = Object.fromEntries(
    cells.map((c) => [c, 0]),
  );
  for (let it = 0; it < 500; it++) {
    for (const c of cells)
      b[c] = mean(obs.filter((o) => o.cell === c).map((o) => o.y - a[o.model]));
    const mb = mean(cells.map((c) => b[c]));
    for (const c of cells) b[c] -= mb;
    for (const m of models)
      a[m] = mean(obs.filter((o) => o.model === m).map((o) => o.y - b[o.cell]));
  }
  return { a, b };
}
const obs1h: CellObs[] = [];
const obs8h: CellObs[] = [];
for (const r of [...BUDGET.recipes, ...(BUDGET.fill ?? [])]) {
  if (r.cell === "zero-shot") continue;
  const model = budgetModel(r.model);
  const end = r.curve[r.curve.length - 1];
  const m1 = r.broken || r.multiplier1h === null ? CRASH_MULT : r.multiplier1h;
  const m8 =
    r.broken || !end ? CRASH_MULT : starterHoursFor(end.elo).hours / end.hours;
  obs1h.push({ model, cell: r.cell, y: Math.log(m1) });
  obs8h.push({ model, cell: r.cell, y: Math.log(m8) });
}

// Required for a place in the ranking.
const RECIPE_TASKS = ["poly_empty", "poke", "forecast", "recipe8"];
// Averaged in when the model has it.
const OPTIONAL_TASKS = ["generals_t"];
const SCORE_TASKS = [...RECIPE_TASKS, ...OPTIONAL_TASKS];
// Value that gets standardized: the recipe multiplier on a log scale (it is a ratio), the others as reported.
const taskValue = (id: string, v: number) =>
  id === "recipe8" ? Math.log(v) : v;

// Frozen scale. Scores are measured against fixed constants from a reference group, so adding a model later doesn't
// move anyone else: 50 = the average of the ten reference models below, 10 points = their standard deviation on that
// task. Two things are frozen:
//   CELL_OFFSETS: how each recipe budget cell went (log multiplier, 1 h and 8 h), from the least-squares fit above over
//     the reference models. A model's recipe effect is the mean over the cells it ran of (its log multiplier - offset).
//   NORMS: each task's mean and population SD across the reference models (recipe task on a log scale; Generals over
//     all ten reference models in its tournament).
// Both are what the live fit gave on 2026-11-23 (Generals: 2026-11-24). Models added later score against the same
// constants and may land outside 35–65. Generals was explicitly re-normalized on 2026-11-24 after Carbon joined
// the 20-network final. Other task constants remain unchanged. Generals Elo is relative (centered at 1000), so adding
// entrants to that tournament re-fits everyone's Elo and needs new constants. To re-norm, compute new constants with
// computeReference(); don't edit these silently.
const REFERENCE_MODELS = [
  "Gemini 4 Neon",
  "Claude Fable 5.2",
  "Gemini 4 Carbon",
  "GPT-6.2 Sol",
  "GPT-6.2 Astra",
  "GPT-6 Astra",
  "GPT-6.1 Sol",
  "Claude Opus 5.5",
  "DeepSeek V4.1 Flash",
  "Kimi K3.1 Pro",
];
const CELL_OFFSETS: Record<"1h" | "8h", Record<string, number>> = {
  "1h": {
    "$2.90": 0.146952658206,
    "$8.70": -0.010457718437,
    $23: -0.009175502062,
    "$8.70, 2 h limit": 0.020613940092,
    "$8.70, 4 h limit": -0.147933377799,
  },
  "8h": {
    "$2.90": 0.043603072642,
    "$8.70": 0.496266981517,
    $23: -0.334910462831,
    "$8.70, 2 h limit": 0.087929094771,
    "$8.70, 4 h limit": -0.292888686099,
  },
};
const NORMS: Record<string, { mean: number; sd: number }> = {
  poly_empty: { mean: 62.48, sd: 11.394981351455 },
  poke: { mean: 67.76, sd: 7.682994207989 },
  forecast: { mean: 0.2868, sd: 0.061315250958 },
  recipe8: { mean: -0.08217861795, sd: 0.239269587071 },
  generals_t: { mean: 1135.000000000000, sd: 221.360177087027 },
};

// Recompute the frozen constants from the data: the cell fit and task norms over the reference models only.
export function computeReference() {
  const ref = new Set(REFERENCE_MODELS);
  const f1 = fitCells(obs1h.filter((o) => ref.has(o.model)));
  const f8 = fitCells(obs8h.filter((o) => ref.has(o.model)));
  const recipe = Object.fromEntries(
    REFERENCE_MODELS.map((m) => [m, Math.exp((f1.a[m] + f8.a[m]) / 2)]),
  );
  const raw = (id: string, m: string) =>
    id === "recipe8" ? recipe[m] : TASK_MAP[id].scores[m];
  const norms = Object.fromEntries(
    SCORE_TASKS.map((id) => {
      const v = REFERENCE_MODELS.filter((m) => raw(id, m) !== undefined).map(
        (m) => taskValue(id, raw(id, m)),
      );
      const mu = mean(v);
      return [
        id,
        { mean: mu, sd: Math.sqrt(mean(v.map((x) => (x - mu) ** 2))) },
      ];
    }),
  );
  return { cells: { "1h": f1.b, "8h": f8.b }, norms };
}

// A model's recipe score against the frozen cell offsets: exp(mean of its 1 h and 8 h effects).
function frozenEffect(
  obs: CellObs[],
  offsets: Record<string, number>,
  model: string,
) {
  const mine = obs.filter(
    (o) => o.model === model && offsets[o.cell] !== undefined,
  );
  return mine.length ? mean(mine.map((o) => o.y - offsets[o.cell])) : undefined;
}
const RECIPE_ALL: Record<string, number> = {};
for (const m of new Set(obs8h.map((o) => o.model))) {
  const e1 = frozenEffect(obs1h, CELL_OFFSETS["1h"], m);
  const e8 = frozenEffect(obs8h, CELL_OFFSETS["8h"], m);
  if (e1 !== undefined && e8 !== undefined)
    RECIPE_ALL[m] = Math.exp((e1 + e8) / 2);
}
TASK_MAP.recipe8 = {
  id: "recipe8",
  name: "Polytopia recipes, five budget cells at 1 h and 8 h",
  game: "Polytopia",
  unit: "× the starter's training compute on a typical cell (research not counted)",
  sessions: 5,
  scores: RECIPE_ALL,
};

// Generals: tournament Elo among the 20 networks of the Generals expansion (empty starter with an editable slow
// simulator, network variant, 3 L40S-hours of research each; 4,864 games per network). Elo rather than the private
// panel points, which bunch the leading networks on the scripted panel.
const GENERALS = generalsRaw as unknown as {
  rows: { model: string; elo: number }[];
};
TASK_MAP.generals_t = {
  id: "generals_t",
  name: "Generals, 20-network tournament",
  game: "Generals",
  unit: "tournament Elo (field centered at 1000)",
  sessions: 1,
  scores: Object.fromEntries(GENERALS.rows.map((r) => [r.model, r.elo])),
};

// Dev-only: warn if the reference data no longer reproduces the frozen constants (e.g. a reference model's run was
// redone). Then decide deliberately whether to re-norm.
if (import.meta.env?.DEV) {
  const now = computeReference();
  const off = (a: number, b: number) => Math.abs(a - b) > 1e-6;
  const drift = [
    ...(["1h", "8h"] as const).flatMap((k) =>
      Object.keys(now.cells[k])
        .filter((c) => off(now.cells[k][c], CELL_OFFSETS[k][c] ?? NaN))
        .map((c) => `cell ${k} ${c}`),
    ),
    ...SCORE_TASKS.filter(
      (id) =>
        off(now.norms[id].mean, NORMS[id].mean) ||
        off(now.norms[id].sd, NORMS[id].sd),
    ).map((id) => `norm ${id}`),
  ];
  if (drift.length)
    console.warn(
      "TopScore: reference data no longer matches the frozen scale:",
      drift,
      now,
    );
}

// One task value on the frozen scale.
const standardize = (id: string, raw: number) => 50 + (10 * (taskValue(id, raw) - NORMS[id].mean)) / NORMS[id].sd;

// Every model with all four required tasks, scored against the frozen scale; optional tasks join its average when
// it ran them.
export const RECIPE: Row[] = Object.keys(TASK_MAP.poly_empty.scores)
  .filter((mo) => RECIPE_TASKS.every((id) => TASK_MAP[id].scores[mo] !== undefined))
  .map((mo) => {
    const each = SCORE_TASKS.map((id) => {
      const v = TASK_MAP[id].scores[mo];
      return v === undefined ? null : standardize(id, v);
    });
    const have = each.filter((v): v is number => v !== null);
    const n = have.length;
    const score = mean(have);
    const se = Math.sqrt(have.reduce((a, b) => a + (b - score) ** 2, 0) / (n - 1) / n);
    return { model: mo, score, lo: score - se, hi: score + se, each };
  })
  .sort((a, b) => b.score - a.score);

const SHORT_TASK: Record<string, string> = {
  poly_empty: "Polytopia empty",
  poke: "Pokémon 3v3",
  forecast: "Forecasting",
  recipe8: "Polytopia recipes",
  generals_t: "Generals",
};
// Per-task cells in display order: the two Polytopia tasks sit side by side under one shared "Polytopia" head, so the
// columns can stay narrow and leave the width to the dot plot. Indexes into SCORE_TASKS / Row.each.
const CELL_ORDER = ["poly_empty", "recipe8", "poke", "forecast", "generals_t"].map(
  (id) => SCORE_TASKS.indexOf(id),
);
// Two-line column heads; the Polytopia pair shares a spanning first line (drawn separately).
const TASK_HEAD: Record<string, [string, string]> = {
  poly_empty: ["", "empty"],
  recipe8: ["", "recipes"],
  poke: ["Pokémon", "3v3"],
  forecast: ["Fore-", "casting"],
  generals_t: ["Generals", "tourney"],
};

function fmtTaskScore(id: string, v: number) {
  if (id === "recipe8") return `${v.toFixed(2)}×`;
  if (id === "forecast") return `+${v.toFixed(2)}`;
  if (id === "generals_t") return `${v.toFixed(0)} Elo`;
  return v.toFixed(1);
}

// Keep featured models in their post color and give light-grey models legible ink weight on white.
function markColor(model: string) {
  const c = modelStyle(model).color;
  return c === "#b5b5b5" || c === "#bdbdbd" || c === "#9a9a9a" ? "#575752" : c;
}

// Per-task cell tint: a quiet sequential ramp, near-white (low) to pale slate (high), over 30–65 on the score scale.
// Kept light so the cells stay secondary to the overall score; the ink stays dark on every step.
const CELL_LO = [247, 247, 248],
  CELL_HI = [184, 196, 212];
const cellT = (v: number) => Math.max(0, Math.min(1, (v - 30) / 35));
function cellFill(v: number) {
  const t = cellT(v);
  return `rgb(${CELL_LO.map((a, i) => Math.round(a + (CELL_HI[i] - a) * t)).join(",")})`;
}

export default function TopScore() {
  const agent = useAgentView();
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref);
  const rows = RECIPE;

  if (agent) {
    return (
      <Table
        head={[
          "#",
          "Model",
          "Overall score (50 = avg of the ten reference models, +10 = +1 SD)",
          "±1 standard error across tasks",
          ...SCORE_TASKS.map((id) => `${SHORT_TASK[id]} (standardized; raw)`),
        ]}
        align={["r", "l", "r", "r", ...SCORE_TASKS.map(() => "r" as const)]}
        rows={rows.map((r, i) => [
          i + 1,
          r.model,
          r.score.toFixed(1),
          `${r.lo.toFixed(1)}–${r.hi.toFixed(1)}`,
          ...SCORE_TASKS.map((id, k) => {
            const v = r.each[k];
            return v === null
              ? "not run"
              : `${v.toFixed(0)} (${fmtTaskScore(id, TASK_MAP[id].scores[r.model])})`;
          }),
        ])}
      />
    );
  }

  // Wide: rank · name · dot plot · score · five per-task cells. Narrow: name above each dot, per-task cells dropped.
  const narrow = W < 560;
  const LO = 35,
    HI = 65;
  const rowH = narrow ? 40 : 30;
  const cellW = 39,
    cellGap = 2,
    cellsGap = 14;
  const cellsW = narrow
    ? 0
    : SCORE_TASKS.length * cellW + (SCORE_TASKS.length - 1) * cellGap;
  const scoreW = 40;
  const nameW = narrow ? 0 : 144;
  const m = {
    l: narrow ? 8 : 26 + nameW,
    r: narrow ? scoreW : scoreW + (cellsW ? cellsW + cellsGap : 0),
    t: 40,
    b: 40,
  };
  const H = m.t + m.b + rows.length * rowH;
  const plotR = W - m.r;
  const sx = (v: number) =>
    m.l +
    6 +
    ((Math.max(LO, Math.min(HI, v)) - LO) / (HI - LO)) * (plotR - m.l - 12);
  const ticks = [35, 40, 45, 50, 55, 60, 65];
  const scoreX = plotR + scoreW - 4;
  const cellX = (k: number) =>
    plotR + scoreW + cellsGap + k * (cellW + cellGap);

  return (
    <div className="viz ar-top" ref={ref}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Overall model scores across five research tasks"
      >
        {/* Column heads */}
        <text
          x={scoreX}
          y={m.t - 10}
          className="viz-axis ts-head ts-score-head"
          textAnchor="end"
        >
          Score
        </text>
        {!narrow && (
          <>
            <text
              x={cellX(1) - cellGap / 2}
              y={m.t - 24}
              className="viz-axis ts-head ts-cell-head"
              textAnchor="middle"
            >
              Polytopia
            </text>
            <line
              x1={cellX(0) + 2}
              x2={cellX(1) + cellW - 2}
              y1={m.t - 19.5}
              y2={m.t - 19.5}
              className="ts-span-rule"
            />
          </>
        )}
        {!narrow &&
          CELL_ORDER.map((j, k) => {
            const id = SCORE_TASKS[j];
            const cx = cellX(k) + cellW / 2;
            return (
              <text
                key={id}
                x={cx}
                y={m.t - 24}
                className="viz-axis ts-head ts-cell-head"
                textAnchor="middle"
              >
                <tspan x={cx}>{TASK_HEAD[id][0]}</tspan>
                <tspan x={cx} dy={14}>
                  {TASK_HEAD[id][1]}
                </tspan>
              </text>
            );
          })}
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={sx(t)}
              x2={sx(t)}
              y1={m.t}
              y2={H - m.b}
              className="viz-grid"
            />
            <text
              x={sx(t)}
              y={H - m.b + 15}
              className="viz-axis"
              textAnchor="middle"
            >
              {t}
            </text>
          </g>
        ))}
        <text
          x={(sx(LO) + sx(HI)) / 2}
          y={H - 6}
          className="viz-axis"
          textAnchor="middle"
        >
          {narrow
            ? "score (50 = average, +10 = +1 SD)"
            : "overall score (50 = average of the ten reference models, +10 = +1 SD)"}
        </text>
        {rows.map((r, i) => {
          const top = m.t + i * rowH;
          const y = top + (narrow ? rowH * 0.7 : rowH / 2);
          const col = markColor(r.model);
          return (
            <g key={r.model}>
              {narrow ? (
                <text
                  x={m.l}
                  y={y - 13}
                  className="viz-label"
                  style={{
                    paintOrder: "stroke",
                    stroke: "#fff",
                    strokeWidth: 3.5,
                    strokeLinejoin: "round",
                  }}
                >
                  <tspan className="viz-axis">{i + 1}</tspan>
                  <tspan dx={6}>{r.model}</tspan>
                </text>
              ) : (
                <>
                  <text
                    x={16}
                    y={y}
                    className="viz-axis"
                    textAnchor="end"
                    dominantBaseline="middle"
                  >
                    {i + 1}
                  </text>
                  <text
                    x={26}
                    y={y}
                    className="viz-label"
                    dominantBaseline="middle"
                    style={{ fontWeight: 600 }}
                  >
                    {r.model}
                  </text>
                </>
              )}
              <line
                x1={sx(r.lo)}
                x2={sx(r.hi)}
                y1={y}
                y2={y}
                stroke={col}
                strokeOpacity={0.32}
                strokeWidth={5}
                strokeLinecap="round"
              />
              <circle
                cx={sx(r.score)}
                cy={y}
                r={5.5}
                fill={col}
                stroke="#fff"
                strokeWidth={1.5}
              />
              <text
                x={scoreX}
                y={y}
                className="viz-value ts-score"
                textAnchor="end"
                dominantBaseline="middle"
              >
                {r.score.toFixed(1)}
              </text>
              {!narrow &&
                CELL_ORDER.map((j, k) => {
                  const v = r.each[j];
                  return (
                    <g key={k}>
                      <rect
                        x={cellX(k)}
                        y={top + 4}
                        width={cellW}
                        height={rowH - 8}
                        rx={2}
                        fill={v === null ? "none" : cellFill(v)}
                      />
                      <text
                        x={cellX(k) + cellW / 2}
                        y={y}
                        className="viz-axis ts-cell-val"
                        textAnchor="middle"
                        dominantBaseline="middle"
                      >
                        {v === null ? "—" : v.toFixed(0)}
                      </text>
                      {v === null && <title>Not in the Generals tournament</title>}
                    </g>
                  );
                })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
