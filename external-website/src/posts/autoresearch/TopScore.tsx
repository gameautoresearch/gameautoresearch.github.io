import { useRef, useState } from "react";
import { useAgentView } from "../../view";
import { Table } from "../shared/text";
import { CMYK, modelStyle } from "./colors";
import { useWidth } from "./Frontier13";
import raw from "./top-scores.json";
import budgetRaw from "./budget.json";
import { starterHoursFor } from "./CostMultiplier";

// Composite model scores across all campaigns (reports/top-level-score-2026-11-20/build.py).
// Default view ("two"): every model ran the same two tasks (Polytopia empty file, Pokémon 3v3); each task is
// standardized across those models and the two are averaged; the whisker spans the two tasks taken alone.
// Other views: each task's scores are standardized within its field, then one least-squares fit estimates
// score = model strength + task offset (with a small ridge prior). Scaled to 50 = average entrant,
// 10 points = 1 SD. Whiskers show the leave-one-task-out range (min to max when dropping any one task).

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
  tasks: string[];
  rank: number;
};

const DATA = raw as unknown as {
  tasks: Task[];
  two: Row[];
  pooled: Row[];
  judgement: Row[];
};
const CORE = DATA.pooled.filter((r) => r.tasks.length >= 3);
const TASK_MAP = Object.fromEntries(DATA.tasks.map((t) => [t.id, t]));

// "Recipe" view: the two shared tasks plus forecasting and the Polytopia research-budget campaign, restricted to
// models with all four. The recipe task is each model's $8.70 session (one fixed budget level, so models with more
// sessions aren't favoured) read off its 8 h training curve: Figure 2's training-compute multiplier (starter's training hours to reach the same Elo / the
// recipe's 8 h; research spend not counted), standardized on a log scale since it is a ratio. Same method as "two" otherwise: each task
// standardized (population SD) across just these models, averaged; the whisker spans the tasks taken alone.
// 50 = average of this smaller field, so not comparable to the other views.
const BUDGET = budgetRaw as unknown as {
  recipes: { model: string; cell: string; curve: { hours: number; elo: number }[] }[];
};
const budgetModel = (mo: string) => (mo === "Opus 5.5" ? "Claude Opus 5.5" : mo);
const RECIPE_CELL = "$8.70";
const RECIPE8: Record<string, number> = {};
for (const r of BUDGET.recipes) {
  if (r.cell !== RECIPE_CELL || !r.curve.length) continue;
  const end = r.curve[r.curve.length - 1];
  RECIPE8[budgetModel(r.model)] = starterHoursFor(end.elo).hours / end.hours;
}
TASK_MAP.recipe8 = {
  id: "recipe8",
  name: "Polytopia, $8.70 recipe at 8 h",
  game: "Polytopia",
  unit: "× the starter's training compute (research not counted)",
  sessions: 1,
  scores: RECIPE8,
};
const RECIPE_TASKS = ["poly_empty", "poke", "forecast", "recipe8"];
const RECIPE: Row[] = (() => {
  const models = Object.keys(TASK_MAP.poly_empty.scores).filter((mo) =>
    RECIPE_TASKS.every((id) => TASK_MAP[id].scores[mo] !== undefined),
  );
  const val = (id: string, mo: string) => (id === "recipe8" ? Math.log(TASK_MAP[id].scores[mo]) : TASK_MAP[id].scores[mo]);
  const z = Object.fromEntries(
    RECIPE_TASKS.map((id) => {
      const v = models.map((mo) => val(id, mo));
      const mu = v.reduce((a, b) => a + b, 0) / v.length;
      const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length);
      return [id, Object.fromEntries(models.map((mo) => [mo, 50 + (10 * (val(id, mo) - mu)) / sd]))];
    }),
  );
  return models
    .map((mo) => {
      const each = RECIPE_TASKS.map((id) => z[id][mo]);
      return {
        model: mo,
        score: each.reduce((a, b) => a + b, 0) / each.length,
        lo: Math.min(...each),
        hi: Math.max(...each),
        tasks: RECIPE_TASKS,
        rank: 0,
      };
    })
    .sort((a, b) => b.score - a.score)
    .map((r, i) => ({ ...r, rank: i + 1 }));
})();

const SHORT_TASK: Record<string, string> = {
  poly_empty: "Polytopia empty",
  poke: "Pokémon 3v3",
  poly_budget: "Polytopia efficiency",
  poly_suite: "Polytopia $35",
  hex: "Hex 9×9",
  generals: "Generals",
  slowsim: "Generals slow sim",
  clash: "Clash",
  forecast: "Forecasting",
  recipe8: "$8.70 recipe, 8 h",
};

function fmtTaskScore(id: string, v: number) {
  if (id === "poly_budget" || id === "recipe8") return `${v.toFixed(2)}×`;
  if (id === "forecast") return `+${v.toFixed(2)}`;
  return v.toFixed(1);
}

// Keep featured models in their post color and give light-grey models legible ink weight on white.
function markColor(model: string, deep: boolean) {
  const c = modelStyle(model).color;
  if (c === "#b5b5b5" || c === "#bdbdbd" || c === "#9a9a9a") {
    return deep ? "#575752" : "#80807a";
  }
  return c;
}

type Mode = "two" | "recipe" | "all" | "core" | "forecast";

export default function TopScore() {
  const agent = useAgentView();
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref);
  const [mode, setMode] = useState<Mode>("two");
  const [tip, setTip] = useState<string | null>(null);

  const recipe = mode === "recipe";
  // "two" and "recipe" share a method: every listed model ran every task, whiskers are each task alone.
  const two = mode === "two" || recipe;
  const rows: Row[] =
    mode === "two"
      ? DATA.two
      : recipe
        ? RECIPE
        : mode === "forecast"
        ? DATA.judgement
        : mode === "core"
          ? CORE
          : DATA.pooled;
  // Solid marks: a model with enough evidence for the view (both tasks in "two", 3+ tasks in the pooled views).
  const isDeep = (r: Row) => two || r.tasks.length >= 3;

  if (agent) {
    return (
      <Table
        head={["#", "Model", "Two-task score (50 = avg)", "Range (each task alone)", "Tasks", "Per-task scores"]}
        align={["r", "l", "r", "r", "r", "l"]}
        rows={DATA.two.map((r, i) => [
          i + 1,
          r.model,
          r.score.toFixed(1),
          r.tasks.length > 1 ? `${r.lo.toFixed(1)}–${r.hi.toFixed(1)}` : "—",
          r.tasks.length,
          r.tasks.map((id) => `${SHORT_TASK[id]}: ${fmtTaskScore(id, TASK_MAP[id].scores[r.model])}`).join("; "),
        ])}
      />
    );
  }

  const narrow = W < 500;
  const LO = 30, HI = 66;
  const rowH = narrow ? 40 : 27;
  const m = { l: narrow ? 8 : 210, r: narrow ? 42 : 48, t: 24, b: 36 };
  const H = m.t + m.b + rows.length * rowH;
  const sx = (v: number) => m.l + ((Math.max(LO, Math.min(HI, v)) - LO) / (HI - LO)) * (W - m.l - m.r);
  const ticks = narrow ? [30, 40, 50, 60] : [30, 35, 40, 45, 50, 55, 60, 65];
  const hovered = tip ? rows.find((r) => r.model === tip) ?? null : null;

  return (
    <div className="viz ar-top" ref={ref} onMouseLeave={() => setTip(null)}>
      <div className="ar-tabs ar-tabs-wrap" role="tablist" aria-label="Composite score view">
        <button role="tab" aria-selected={mode === "two"} onClick={() => { setMode("two"); setTip(null); }}>
          Two tasks, every model ({DATA.two.length})
        </button>
        <button role="tab" aria-selected={recipe} onClick={() => { setMode("recipe"); setTip(null); }}>
          Two tasks + forecasting + $8.70 recipe ({RECIPE.length})
        </button>
        <button role="tab" aria-selected={mode === "all"} onClick={() => { setMode("all"); setTip(null); }}>
          All research, pooled ({DATA.pooled.length})
        </button>
        <button role="tab" aria-selected={mode === "core"} onClick={() => { setMode("core"); setTip(null); }}>
          3+ tasks only ({CORE.length})
        </button>
        <button role="tab" aria-selected={mode === "forecast"} onClick={() => { setMode("forecast"); setTip(null); }}>
          Research + forecasting
        </button>
      </div>
      <div className="viz-legend">
        <span><i className="ts-dot solid" />{recipe ? "Polytopia empty + Pokémon 3v3 + forecasting + $8.70 recipe at 8 h" : two ? "Polytopia empty + Pokémon 3v3" : "3+ tasks"}</span>
        {!two && mode !== "core" && <span><i className="ts-dot hollow" />1–2 tasks</span>}
        <span><i className="ts-whisk" />{two ? "Each task alone" : "Range dropping any one task"}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Synthesized model scores across research tasks">
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={sx(t)}
              x2={sx(t)}
              y1={t === 50 ? m.t - 6 : m.t}
              y2={H - m.b}
              className={t === 50 ? undefined : "viz-grid"}
              stroke={t === 50 ? CMYK.k : undefined}
              strokeDasharray={t === 50 ? "3 3" : undefined}
              strokeOpacity={t === 50 ? 0.5 : undefined}
            />
            <text x={sx(t)} y={H - m.b + 16} className="viz-axis" textAnchor="middle">
              {t}
            </text>
          </g>
        ))}
        <text x={sx(50)} y={m.t - 10} className="viz-axis" textAnchor="middle">
          average (50)
        </text>
        <text x={(sx(LO) + sx(HI)) / 2} y={H - 4} className="viz-axis" textAnchor="middle">
          {recipe ? "four-task index (50 = average of these models, +10 = +1 SD)" : two ? "two-task index (50 = average model, +10 = +1 SD)" : "pooled index (50 = average entrant, +10 = +1 SD)"}
        </text>
        {rows.map((r, i) => {
          const y = m.t + i * rowH + (narrow ? rowH * 0.68 : rowH / 2);
          const deep = isDeep(r);
          const col = markColor(r.model, deep);
          const lit = tip === r.model;
          const dim = tip !== null && !lit;
          const xScore = sx(r.score);
          const xEnd = r.tasks.length > 1 ? sx(Math.max(r.score, r.hi)) : xScore;
          return (
            <g
              key={r.model}
              tabIndex={0}
              className="ar-top-row"
              onMouseEnter={() => setTip(r.model)}
              onFocus={() => setTip(r.model)}
              opacity={dim ? 0.32 : 1}
            >
              <rect x={0} y={m.t + i * rowH} width={W} height={rowH} fill={lit ? "#f6f6f3" : "transparent"} />
              {narrow ? (
                <text x={m.l} y={y - 12} className="viz-label" style={{ fontWeight: deep ? 600 : 400 }}>
                  <tspan className="viz-axis">{i + 1}. </tspan>
                  {r.model}
                  <tspan className="viz-axis"> · {r.tasks.length}t</tspan>
                </text>
              ) : (
                <>
                  <text x={20} y={y} className="viz-axis" textAnchor="end" dominantBaseline="middle">
                    {i + 1}
                  </text>
                  <text
                    x={m.l - 12}
                    y={y}
                    className="viz-label"
                    textAnchor="end"
                    dominantBaseline="middle"
                    style={{ fontWeight: deep ? 600 : 400 }}
                  >
                    {r.model}
                    <tspan className="viz-axis" style={{ fontWeight: 400 }}>
                      {" "}
                      ({r.tasks.length})
                    </tspan>
                  </text>
                </>
              )}
              {r.tasks.length > 1 && (
                <line
                  x1={sx(r.lo)}
                  x2={sx(r.hi)}
                  y1={y}
                  y2={y}
                  stroke={col}
                  strokeOpacity={lit ? 0.55 : 0.3}
                  strokeWidth={4.5}
                  strokeLinecap="round"
                />
              )}
              <circle
                cx={xScore}
                cy={y}
                r={deep ? 5.2 : 4.6}
                fill={deep ? col : "#fff"}
                stroke={deep ? "#fff" : col}
                strokeWidth={deep ? 1.4 : 2}
              />
              <text
                x={xEnd + 8}
                y={y}
                className="viz-axis"
                dominantBaseline="middle"
                style={{
                  fill: deep ? "var(--ink)" : "var(--secondary)",
                  fontWeight: deep ? 600 : 400,
                  paintOrder: "stroke",
                  stroke: lit ? "#f6f6f3" : "#fff",
                  strokeWidth: 4,
                }}
              >
                {r.score.toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="ar-tl-detail" aria-live="polite">
        {hovered ? (
          <>
            <strong>{hovered.model}</strong>: index <b>{hovered.score.toFixed(1)}</b>
            {hovered.tasks.length > 1 && (
              <span>
                {" "}
                ({two ? "each task alone" : "leave-one-task-out"} {hovered.lo.toFixed(1)}–{hovered.hi.toFixed(1)})
              </span>
            )}
            <span>
              {" "}
              ·{" "}
              {hovered.tasks
                .map((id) => `${SHORT_TASK[id]} ${fmtTaskScore(id, TASK_MAP[id].scores[hovered.model])}`)
                .join(" · ")}
            </span>
          </>
        ) : (
          <span>
            Hover a model for its individual task scores. Parentheses show the number of tasks each model ran.
          </span>
        )}
      </div>
    </div>
  );
}
