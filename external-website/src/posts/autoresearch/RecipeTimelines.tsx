import { useRef, useState } from "react";
import { useAgentView } from "../../view";
import { Table } from "../shared/text";
import { useWidth } from "./Frontier13";
import raw from "./recipe-timelines.json";

// Polytopia $8.70 (b3) recipe-research sessions for all seven models in the main budget campaign.
// Each model had the same $8.70 GPU budget (~3 L40S hours), 7-hour wall clock, and up to 2 GPUs at once.
// Derived by tools/build_recipe_timelines.py from each session's ledger.sqlite3, NOTES.md and budget.json.

type GpuRun = { s: number; e: number; min: number; usd: number; slot: 0 | 1; what: string; note: string };
type CpuJob = { s: number; e: number; min: number; usd: number; what: string; note: string };
type EvalJob = { s: number; e: number; min: number };

type Session = {
  key: string;
  model: string;
  elo8: number;
  spentUsd: number;
  wallH: number;
  gpuH: number;
  aborted: number;
  runs: GpuRun[];
  cpu: CpuJob[];
  evals: EvalJob[];
};

const DATA = raw as unknown as {
  budgetUsd: number;
  wallLimitH: number;
  maxGpus: number;
  starterElo8: number;
  sessions: Session[];
};

const CPU_COLOR = "#d4a000";
const EVAL_COLOR = "#b9b8b0";
const NEON_BLUE = "#2f55c4";

function runColor(min: number) {
  if (min < 20) return "#6e97ea";
  if (min < 45) return "#3459ba";
  return "#19337d";
}

function fmtCost(usd: number) {
  return usd < 0.01 ? "<$0.01" : `$${usd.toFixed(2)}`;
}

function fmtMin(min: number) {
  return min < 2 ? `${min.toFixed(1)} min` : `${Math.round(min)} min`;
}

// No-op reference: the unmodified starter recipe retrained for 8 h, with no research at all.
const STARTER = DATA.starterElo8;
// Rows are ordered by final (8 h) Elo; the starter row sits at its rank among them.
const SESSIONS = [...DATA.sessions].sort((a, b) => b.elo8 - a.elo8);
const STARTER_IDX = SESSIONS.filter((s) => s.elo8 > STARTER).length;

type Tip = {
  id: string;
  cx: number;
  topY: number;
  botY: number;
  title: string;
  cost: string;
  what: string;
  note: string;
};

export default function RecipeTimelines() {
  const agent = useAgentView();
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref);
  const [tip, setTip] = useState<Tip | null>(null);

  if (agent) {
    return (
      <Table
        head={["Model", "GPU runs", "Run lengths (min)", "CPU debug", "Practice evals", "Wall time", "GPU spend", "8 h Elo"]}
        align={["l", "r", "l", "r", "r", "r", "r", "r"]}
        rows={[...SESSIONS.slice(0, STARTER_IDX), null, ...SESSIONS.slice(STARTER_IDX)].map((s) => s ? [
          s.model,
          s.runs.length,
          s.runs.map((r) => `${Math.round(r.min)}m`).join(", "),
          s.cpu.length || "—",
          s.evals.length,
          `${s.wallH.toFixed(1)} h`,
          `$${s.spentUsd.toFixed(2)}`,
          s.elo8,
        ] : ["Starter recipe, no research (no-op)", 0, "—", "—", "—", "—", "$0.00", STARTER])}
      />
    );
  }

  const narrow = W < 540;
  // Tight x-axis: longest session is 4.41 h, so 4.5 h uses the full horizontal span cleanly.
  const T = 4.5;
  const ticks = [0, 1, 2, 3, 4];

  const runsW = 44;
  const eloW = 56;
  const rightGap = 14;
  const m = narrow
    ? { l: 8, r: 8, t: 8, b: 44 }
    : { l: 152, r: rightGap + runsW + eloW, t: 32, b: 44 };
  const rowH = narrow ? 68 : 50;
  const starterRowH = narrow ? 44 : 38;
  const H = m.t + m.b + SESSIONS.length * rowH + starterRowH;
  const rowTop = (i: number) => m.t + i * rowH + (i >= STARTER_IDX ? starterRowH : 0);

  const plotR = W - m.r;
  const sx = (h: number) => m.l + (Math.min(h, T) / T) * (plotR - m.l);
  const runsX = plotR + rightGap + runsW;
  const eloX = runsX + eloW;

  const tipW = narrow ? Math.min(276, W - 16) : 324;
  const tipLeft = tip ? Math.max(4, Math.min(W - tipW - 4, tip.cx - tipW / 2)) : 0;
  const flipBelow = tip ? tip.topY < 84 : false;

  return (
    <div className="ar-rt">
      <div className="viz-legend ar-rt-legend">
        <span>
          <i className="ramp" />
          GPU run (short → long)
        </span>
        <span>
          <i style={{ background: CPU_COLOR }} />
          CPU debug
        </span>
        <span>
          <i className="thin" style={{ background: EVAL_COLOR }} />
          Practice eval
        </span>
      </div>

      <div className="viz" ref={ref} onMouseLeave={() => setTip(null)}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label="Polytopia $8.70 recipe research sessions across seven models, showing GPU run schedules, experiment counts, and 8-hour Elo"
          onClick={(e) => {
            if (e.target === e.currentTarget) setTip(null);
          }}
        >
          {/* Vertical hour gridlines */}
          {ticks.map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={m.t - 3} y2={H - m.b} className="viz-grid" />
              <text x={sx(t)} y={H - m.b + 20} className="ar-rt-tick" textAnchor="middle">
                {t}h
              </text>
            </g>
          ))}

          {/* Column headings */}
          {!narrow && (
            <>
              <text x={m.l} y={m.t - 12} className="ar-rt-colhead">
                Timeline (2 GPUs)
              </text>
              <text x={runsX} y={m.t - 12} className="ar-rt-colhead" textAnchor="end">
                Runs
              </text>
              <text x={eloX} y={m.t - 12} className="ar-rt-colhead" textAnchor="end">
                8h Elo
              </text>
              <line x1={0} x2={W} y1={m.t - 3} y2={m.t - 3} stroke="#d8d8d3" strokeWidth={1} />
            </>
          )}
          <text x={(sx(0) + sx(T)) / 2} y={H - 4} className="ar-rt-axis-title" textAnchor="middle">
            Session time
          </text>

          {SESSIONS.map((s, i) => {
            const top = rowTop(i);
            const barTop = narrow ? top + 24 : top + 8;
            const slotH = 11;
            const slotGap = 3;
            const trackH = slotH * 2 + slotGap + 4; // 29px
            const midY = barTop + trackH / 2;
            const evalY = barTop + trackH + 3.5;
            const isNeon = s.key === "neon-b3";

            return (
              <g key={s.key}>
                {i > 0 && i !== STARTER_IDX && (
                  <line x1={0} x2={W} y1={top} y2={top} stroke="#efefec" strokeWidth={1} />
                )}

                {narrow ? (
                  <>
                    <text x={m.l} y={top + 16} className="ar-rt-model" style={{ fontWeight: isNeon ? 700 : 600 }}>
                      {s.model}
                    </text>
                    <text x={W - m.r} y={top + 16} className="ar-rt-inline" textAnchor="end">
                      <tspan className="ar-rt-inline-num">{s.runs.length} runs</tspan>
                      {" · "}
                      <tspan
                        className="ar-rt-inline-num"
                        style={isNeon ? { fill: NEON_BLUE } : undefined}
                      >
                        {s.elo8} Elo
                      </tspan>
                    </text>
                  </>
                ) : (
                  <text
                    x={0}
                    y={midY}
                    className="ar-rt-model"
                    dominantBaseline="middle"
                    style={{ fontWeight: isNeon ? 700 : 600 }}
                  >
                    {s.model}
                  </text>
                )}

                {/* Active session span backdrop + end-of-session tick */}
                <rect
                  x={sx(0)}
                  y={barTop}
                  width={Math.max(2, sx(s.wallH) - sx(0))}
                  height={trackH}
                  rx={2.5}
                  fill="#f3f3ee"
                />
                <line
                  x1={sx(s.wallH)}
                  x2={sx(s.wallH)}
                  y1={barTop}
                  y2={barTop + trackH}
                  stroke="#a8a8a0"
                  strokeWidth={1.3}
                />

                {/* Two parallel GPU slots */}
                {s.runs.map((r, j) => {
                  const id = `${s.key}-g-${j}`;
                  const rx = sx(r.s);
                  const rw = Math.max(3, sx(r.e) - rx);
                  const ry = barTop + 2 + r.slot * (slotH + slotGap);
                  const active = tip?.id === id;
                  const show = () =>
                    setTip({
                      id,
                      cx: rx + rw / 2,
                      topY: ry,
                      botY: ry + slotH,
                      title: `${s.model} · run ${j + 1}/${s.runs.length}`,
                      cost: `${fmtCost(r.usd)} · ${fmtMin(r.min)}`,
                      what: r.what,
                      note: r.note,
                    });
                  return (
                    <rect
                      key={id}
                      x={rx}
                      y={ry}
                      width={rw}
                      height={slotH}
                      rx={2}
                      fill={runColor(r.min)}
                      stroke={active ? "#111" : "#fff"}
                      strokeWidth={active ? 1.5 : 0.75}
                      className="ar-rt-bar"
                      tabIndex={0}
                      onMouseEnter={show}
                      onFocus={show}
                      onBlur={() => setTip((cur) => (cur?.id === id ? null : cur))}
                      onClick={(e) => {
                        e.stopPropagation();
                        show();
                      }}
                    />
                  );
                })}

                {/* Free CPU practice evals micro-rail */}
                {s.evals.map((ev, j) => {
                  const id = `${s.key}-e-${j}`;
                  const ex = sx(ev.s);
                  const ew = Math.max(2.5, sx(ev.e) - ex);
                  const active = tip?.id === id;
                  const show = () =>
                    setTip({
                      id,
                      cx: ex + ew / 2,
                      topY: evalY,
                      botY: evalY + 4,
                      title: `${s.model} · practice eval`,
                      cost: `free CPU · ${fmtMin(ev.min)}`,
                      what: "512-game public benchmark against 4 fixed opponents",
                      note: "Runs on separate CPU worker outside the $8.70 budget",
                    });
                  return (
                    <rect
                      key={id}
                      x={ex}
                      y={evalY}
                      width={ew}
                      height={4}
                      rx={1}
                      fill={active ? "#555" : EVAL_COLOR}
                      className="ar-rt-bar"
                      onMouseEnter={show}
                      onClick={(e) => {
                        e.stopPropagation();
                        show();
                      }}
                    />
                  );
                })}

                {/* Paid CPU debug / audit jobs */}
                {s.cpu.map((c, j) => {
                  const id = `${s.key}-c-${j}`;
                  const cx = sx(c.s);
                  const cw = Math.max(3, sx(c.e) - cx);
                  const active = tip?.id === id;
                  const show = () =>
                    setTip({
                      id,
                      cx: cx + cw / 2,
                      topY: evalY - 1,
                      botY: evalY + 5,
                      title: `${s.model} · CPU debug`,
                      cost: `${fmtCost(c.usd)} · ${fmtMin(c.min)}`,
                      what: c.what,
                      note: c.note,
                    });
                  return (
                    <rect
                      key={id}
                      x={cx}
                      y={evalY - 0.5}
                      width={cw}
                      height={5}
                      rx={1}
                      fill={CPU_COLOR}
                      stroke={active ? "#111" : "none"}
                      strokeWidth={active ? 1 : 0}
                      className="ar-rt-bar"
                      tabIndex={0}
                      onMouseEnter={show}
                      onFocus={show}
                      onBlur={() => setTip((cur) => (cur?.id === id ? null : cur))}
                      onClick={(e) => {
                        e.stopPropagation();
                        show();
                      }}
                    />
                  );
                })}

                {/* Right-hand summary columns (desktop) */}
                {!narrow && (
                  <>
                    <text
                      x={runsX}
                      y={midY}
                      className="ar-rt-runs"
                      textAnchor="end"
                      dominantBaseline="middle"
                    >
                      {s.runs.length}
                    </text>
                    <text
                      x={eloX}
                      y={midY}
                      className={`ar-rt-elo${isNeon ? " is-top" : ""}`}
                      textAnchor="end"
                      dominantBaseline="middle"
                    >
                      {s.elo8}
                    </text>
                  </>
                )}
              </g>
            );
          })}

          {/* No-op reference row: the starter recipe trained 8 h with no experimentation */}
          {(() => {
            const top = m.t + STARTER_IDX * rowH;
            const midY = narrow ? top + 16 : top + starterRowH / 2;
            const note = "Unmodified baseline (no research)";
            return (
              <g className="ar-rt-noop">
                {[top, top + starterRowH].map((y) => (
                  <line key={y} x1={0} x2={W} y1={y} y2={y} stroke="#cfcfc8" strokeWidth={1.1} strokeDasharray="3 3" />
                ))}
                {narrow ? (
                  <>
                    <text x={m.l} y={midY} className="ar-rt-model">Starter recipe</text>
                    <text x={W - m.r} y={midY} className="ar-rt-inline" textAnchor="end">
                      <tspan className="ar-rt-inline-num">{STARTER} Elo</tspan>
                    </text>
                    <text x={m.l} y={top + 34} className="ar-rt-noop-note">{note}</text>
                  </>
                ) : (
                  <>
                    <text x={0} y={midY} className="ar-rt-model" dominantBaseline="middle">Starter recipe</text>
                    <text x={sx(0) + 8} y={midY} className="ar-rt-noop-note" dominantBaseline="middle">{note}</text>
                    <text x={runsX} y={midY} className="ar-rt-runs" textAnchor="end" dominantBaseline="middle">0</text>
                    <text x={eloX} y={midY} className="ar-rt-elo" textAnchor="end" dominantBaseline="middle">{STARTER}</text>
                  </>
                )}
              </g>
            );
          })()}
        </svg>

        {tip && (
          <div
            className="viz-tip ar-rt-tip"
            role="status"
            style={{
              left: tipLeft,
              width: tipW,
              top: flipBelow ? tip.botY + 8 : tip.topY - 8,
              transform: flipBelow ? "none" : "translateY(-100%)",
            }}
          >
            <div className="ar-rt-tip-head">
              <strong>{tip.title}</strong>
              <span className="ar-rt-tip-cost">{tip.cost}</span>
            </div>
            <span className="ar-rt-tip-what">{tip.what}</span>
            <em className="ar-rt-tip-note">{tip.note}</em>
          </div>
        )}
      </div>
    </div>
  );
}
