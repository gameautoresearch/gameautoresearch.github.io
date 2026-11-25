import { useMemo, useRef, useState } from "react";
import { useAgentView } from "../../view";
import { Table } from "../shared/text";
import { CMYK } from "./colors";
import { useWidth } from "./Frontier13";
import raw from "./capability-timeline.json";

type LabKey =
  | "openai"
  | "anthropic"
  | "google"
  | "deepseek"
  | "moonshot"
  | "meta"
  | "tencent"
  | "zai"
  | "thinking";

type ModelRow = {
  key: string;
  model: string;
  short: string;
  lab: LabKey;
  date: string;
  grade: number;
  finalElo: number;
  finalMult: number | null;
  plotMult: number;
  bestElo: number;
  bestMult: number | null;
  bestPlotMult: number;
  note?: string;
};

type WorkingRow = {
  key: string;
  model: string;
  short: string;
  lab: LabKey;
  date: string;
  mult1h: number | null;
  seeds: number[];
  mult8h: number | null;
  extrap8h: boolean;
  broken: boolean;
  note?: string;
};

type ViewMode = "final" | "best";

const DATA = raw as unknown as {
  floorMult: number;
  models: ModelRow[];
  workingStarter: WorkingRow[];
};

const LAB_NAME: Record<LabKey, string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  google: "Google",
  deepseek: "DeepSeek",
  moonshot: "Moonshot",
  meta: "Meta",
  tencent: "Tencent",
  zai: "Z.ai",
  thinking: "Thinking Machines",
};

// Right-margin labels for the late-2026 frontier cluster (stacked collision-free).
const MARGIN_KEYS: Record<ViewMode, Set<string>> = {
  final: new Set([
    "empty:sol62",
    "empty:gem4n",
    "empty:fable",
    "work:gem4n",
    "work:astra62",
  ]),
  best: new Set([
    "empty:sol62",
    "empty:gem4n",
    "empty:astra62",
    "empty:fable",
    "work:gem4n",
  ]),
};

type InlineCfg = {
  dx: number;
  dy: number;
  anchor: "start" | "middle" | "end";
  narrowHide?: boolean;
};

// Minimal inline callouts for key inflection points.
const INLINE_LABELS: Record<ViewMode, Record<string, InlineCfg>> = {
  final: {
    "empty:opus5": { dx: 0, dy: -11, anchor: "middle" },
    "empty:astra": { dx: -10, dy: 4, anchor: "end" },
    "empty:sol": { dx: 10, dy: 4, anchor: "start", narrowHide: true },
    "work:ds41flash": { dx: -9, dy: -9, anchor: "end", narrowHide: true },
  },
  best: {
    "empty:opus5": { dx: 0, dy: -11, anchor: "middle" },
    "empty:astra": { dx: -10, dy: 4, anchor: "end" },
    "empty:sol": { dx: 10, dy: 4, anchor: "start", narrowHide: true },
    "work:ds41flash": { dx: -9, dy: -9, anchor: "end", narrowHide: true },
  },
};

const parseUtc = (iso: string) => {
  const [y, m, d] = iso.split("-").map(Number);
  return Date.UTC(y, m - 1, d);
};

const fmtDate = (iso: string) =>
  new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });

const fmtMult = (x: number | null) =>
  x == null ? `<${DATA.floorMult.toFixed(2)}×` : `${x >= 0.95 ? x.toFixed(1) : x.toFixed(2)}×`;

const fmtBudgetMult = (x: number | null, broken?: boolean) =>
  broken || x == null ? "broken" : `${x.toFixed(2)}×`;

function layoutLabels(
  items: { id: string; targetY: number }[],
  minGap: number,
  minY: number,
  maxY: number,
) {
  const out = items.map((i) => ({ ...i, y: i.targetY })).sort((a, b) => a.targetY - b.targetY);
  for (let pass = 0; pass < 8; pass++) {
    for (let i = 1; i < out.length; i++) {
      const overlap = out[i - 1].y + minGap - out[i].y;
      if (overlap > 0) {
        out[i - 1].y -= overlap / 2;
        out[i].y += overlap / 2;
      }
    }
    if (out.length && out[0].y < minY) {
      const shift = minY - out[0].y;
      for (const item of out) item.y += shift;
    }
    if (out.length && out[out.length - 1].y > maxY) {
      const shift = out[out.length - 1].y - maxY;
      for (const item of out) item.y -= shift;
    }
  }
  return new Map(out.map((i) => [i.id, i.y]));
}

type PlotPoint = {
  id: string;
  kind: "empty" | "working";
  key: string;
  model: string;
  short: string;
  marginShort: string;
  lab: LabKey;
  date: string;
  val: number;
  beatsStarter: boolean;
  broken?: boolean;
  seeds?: number[];
  labelVal: string;
  empty?: ModelRow;
  working?: WorkingRow;
};

export default function CapabilityTimeline() {
  const agent = useAgentView();
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref);
  const narrow = W < 540;
  const [view, setView] = useState<ViewMode>("final");
  const [hoverId, setHoverId] = useState<string | null>(null);

  const points: PlotPoint[] = useMemo(() => {
    const emptyPts: PlotPoint[] = DATA.models.map((row) => {
      const val = view === "final" ? row.plotMult : row.bestPlotMult;
      const mult = view === "final" ? row.finalMult : row.bestMult;
      return {
        id: `empty:${row.key}`,
        kind: "empty",
        key: row.key,
        model: row.model,
        short: row.short,
        marginShort: row.short,
        lab: row.lab,
        date: row.date,
        val,
        beatsStarter: (mult ?? 0) >= 1,
        labelVal: mult != null ? fmtMult(mult) : "",
        empty: row,
      };
    });

    const workPts: PlotPoint[] = DATA.workingStarter.map((row) => ({
      id: `work:${row.key}`,
      kind: "working",
      key: row.key,
      model: row.model,
      short: row.short,
      marginShort: row.key === "gem4n" ? "Neon ($8.70)" : `${row.short} ($8.70)`,
      lab: row.lab,
      date: row.date,
      val: row.mult1h ?? 0,
      beatsStarter: (row.mult1h ?? 0) >= 1,
      broken: row.broken,
      seeds: row.seeds,
      labelVal: fmtBudgetMult(row.mult1h, row.broken),
      working: row,
    }));

    return [...emptyPts, ...workPts].sort(
      (a, b) => a.date.localeCompare(b.date) || a.val - b.val,
    );
  }, [view]);

  // Monotone running-best frontier across all non-broken points.
  const frontier = useMemo(() => {
    const byDate = new Map<number, number>();
    for (const pt of points) {
      if (pt.broken) continue;
      const t = parseUtc(pt.date);
      const prev = byDate.get(t);
      if (prev === undefined || pt.val > prev) byDate.set(t, pt.val);
    }
    const ordered = [...byDate.entries()].sort((a, b) => a[0] - b[0]);
    const out: { t: number; v: number }[] = [];
    let best = -Infinity;
    for (const [t, v] of ordered) {
      if (v > best) {
        best = v;
        out.push({ t, v });
      }
    }
    return out;
  }, [points]);

  if (agent) {
    return (
      <>
        <Table
          head={[
            "Model (Empty starter)",
            "Lab",
            "Released",
            "15-min final mult",
            "Best in-session mult",
            "Ladder score",
            "Note",
          ]}
          align={["l", "l", "l", "r", "r", "r", "l"]}
          rows={[...DATA.models]
            .sort((a, b) => a.date.localeCompare(b.date) || a.plotMult - b.plotMult)
            .map((m) => [
              m.model,
              LAB_NAME[m.lab],
              fmtDate(m.date),
              fmtMult(m.finalMult),
              fmtMult(m.bestMult),
              `${m.grade.toFixed(1)} / 100`,
              m.note ?? "",
            ])}
        />
        <Table
          head={[
            "Model ($8.70 working starter)",
            "Lab",
            "Released",
            "1-h final mult",
            "8-h training mult",
            "Note",
          ]}
          align={["l", "l", "l", "r", "r", "l"]}
          rows={[...DATA.workingStarter]
            .sort((a, b) => a.date.localeCompare(b.date) || (a.mult1h ?? 0) - (b.mult1h ?? 0))
            .map((m) => [
              m.model,
              LAB_NAME[m.lab],
              fmtDate(m.date),
              fmtBudgetMult(m.mult1h, m.broken),
              m.mult8h == null ? "—" : `${m.extrap8h ? "≈" : ""}${m.mult8h.toFixed(2)}×`,
              m.note ?? "",
            ])}
        />
      </>
    );
  }

  const H = narrow ? 304 : 352;
  const m = { l: 38, r: narrow ? 16 : 152, t: 24, b: 52 };
  const plotW = W - m.l - m.r;
  const plotH = H - m.t - m.b;

  const tMin = Date.UTC(2026, 0, 22);
  const tMax = Date.UTC(2026, 10, 25);
  const yMax = view === "final" ? 3.5 : 5.5;

  const sx = (t: number) => m.l + ((t - tMin) / (tMax - tMin)) * plotW;
  const sy = (x: number) => m.t + ((yMax - x) / yMax) * plotH;

  const monthTicks = narrow
    ? [
        { t: Date.UTC(2026, 1, 1), label: "Feb" },
        { t: Date.UTC(2026, 3, 1), label: "Apr" },
        { t: Date.UTC(2026, 5, 1), label: "Jun" },
        { t: Date.UTC(2026, 7, 1), label: "Aug" },
        { t: Date.UTC(2026, 10, 18), label: "Nov 2026" },
      ]
    : [
        { t: Date.UTC(2026, 1, 1), label: "Feb" },
        { t: Date.UTC(2026, 3, 1), label: "Apr" },
        { t: Date.UTC(2026, 5, 1), label: "Jun" },
        { t: Date.UTC(2026, 7, 1), label: "Aug" },
        { t: Date.UTC(2026, 9, 1), label: "Oct" },
        { t: Date.UTC(2026, 10, 18), label: "Nov 2026" },
      ];

  const relTicks = [
    { t: Date.UTC(2026, 1, 18), label: "9 mo ago" },
    { t: Date.UTC(2026, 4, 18), label: "6 mo ago" },
    { t: Date.UTC(2026, 7, 18), label: "3 mo ago" },
    { t: Date.UTC(2026, 10, 18), label: "Now" },
  ];

  const yTicks = view === "final" ? [0, 1, 2, 3] : [0, 1, 2, 3, 4, 5];

  const endT = Date.UTC(2026, 10, 18);
  const stepPath = (() => {
    if (!frontier.length) return { line: "", area: "" };
    const x0 = sx(frontier[0].t);
    const y0 = sy(frontier[0].v);
    let line = `M${x0.toFixed(1)},${y0.toFixed(1)}`;
    for (let i = 1; i < frontier.length; i++) {
      line += `H${sx(frontier[i].t).toFixed(1)}V${sy(frontier[i].v).toFixed(1)}`;
    }
    line += `H${sx(endT).toFixed(1)}`;
    const area = `${line}V${sy(0).toFixed(1)}H${x0.toFixed(1)}Z`;
    return { line, area };
  })();

  const activeMarginKeys = MARGIN_KEYS[view];
  const marginPoints = points
    .filter((pt) => activeMarginKeys.has(pt.id))
    .sort((a, b) => b.val - a.val);
  const marginY = layoutLabels(
    marginPoints.map((pt) => ({ id: pt.id, targetY: sy(pt.val) })),
    16,
    m.t + 14,
    sy(0) - 18,
  );

  const hovered = hoverId ? points.find((p) => p.id === hoverId) ?? null : null;
  const hoverPos = hovered
    ? { x: sx(parseUtc(hovered.date)), y: sy(hovered.val) }
    : null;

  // Draw order: weaker empty circles -> working-starter diamonds -> >=1x empty circles (so everything stays visible).
  const drawOrder = [
    ...points.filter((p) => p.kind === "empty" && !p.beatsStarter),
    ...points.filter((p) => p.kind === "working"),
    ...points.filter((p) => p.kind === "empty" && p.beatsStarter),
  ];

  const yOne = sy(1);
  const yFloor = sy(DATA.floorMult);
  const activeInline = INLINE_LABELS[view];

  return (
    <div className="ar-ct" ref={ref}>
      <div className="ar-ct-head">
        <div className="ar-ct-bar">
          <div className="ar-tabs" role="tablist" aria-label="Empty-starter metric">
            <button
              type="button"
              role="tab"
              aria-selected={view === "final"}
              onClick={() => {
                setView("final");
                setHoverId(null);
              }}
            >
              Fresh final
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={view === "best"}
              onClick={() => {
                setView("best");
                setHoverId(null);
              }}
            >
              Best in-session
            </button>
          </div>

          <div className="ar-ct-legend" aria-label="Series key">
            <span>
              <i className="dot" style={{ background: CMYK.cDeep }} />
              Empty starter ({view === "final" ? "15-min final" : "best net"}, 30)
            </span>
            <span>
              <i className="dia" style={{ background: CMYK.m }} />
              $8.70 working starter (1-h final, 12)
            </span>
          </div>
        </div>
      </div>

      <div className="viz ar-ct-plot">
        <svg
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Compute multiplier by release date from February to November 2026: ${DATA.models.length} empty-starter models (${view === "final" ? "15-minute fresh final" : "best in-session network"}) and ${DATA.workingStarter.length} $8.70 working-starter models (1-hour final)`}
        >
          {/* Whisper-light tint above 1x (beats the unmodified PPO starter) */}
          <rect x={m.l} y={m.t} width={plotW} height={yOne - m.t} fill="#f4fafd" />

          {/* Subtle band below 0.43x where empty-starter models are too weak to place precisely */}
          <rect
            x={m.l}
            y={yFloor}
            width={plotW}
            height={sy(0) - yFloor}
            fill="#f6f6f3"
          />

          {/* Horizontal gridlines & Y-axis */}
          {yTicks.map((v) => {
            const y = sy(v);
            return (
              <g key={v}>
                <line
                  x1={m.l}
                  x2={m.l + plotW}
                  y1={y}
                  y2={y}
                  className={v === 0 ? "viz-zero" : "viz-grid"}
                />
                <text x={m.l - 8} y={y + 4} textAnchor="end" className="viz-axis">
                  {v}×
                </text>
              </g>
            );
          })}
          <text x={m.l} y={m.t - 9} className="viz-axis">
            {view === "final"
              ? "Compute multiplier vs. unmodified PPO starter"
              : "Compute multiplier vs. unmodified PPO starter (best in-session for empty starter)"}
          </text>

          {/* 1x baseline reference */}
          <line
            x1={m.l}
            x2={m.l + plotW}
            y1={yOne}
            y2={yOne}
            stroke={CMYK.cDeep}
            strokeOpacity={0.55}
            strokeWidth={1.15}
            strokeDasharray="5 4"
          />
          <text x={m.l + 6} y={yOne - 6} className="ar-ct-ref ar-ct-ref-main">
            1× — unmodified PPO starter
          </text>

          {/* Subtle <0.43x floor label */}
          <line
            x1={m.l}
            x2={m.l + plotW}
            y1={yFloor}
            y2={yFloor}
            stroke="#dcdcd6"
            strokeWidth={1}
            strokeDasharray="2 3"
          />
          <text x={m.l + 6} y={yFloor - 4} className="ar-ct-ref">
            {narrow ? "<0.43× floor" : "<0.43× — weaker than 5-min starter"}
          </text>

          {/* Running-best frontier step curve */}
          <path d={stepPath.area} fill={CMYK.c} fillOpacity={0.055} />
          <path
            d={stepPath.line}
            fill="none"
            stroke={CMYK.cDeep}
            strokeWidth={1.9}
            strokeOpacity={0.85}
          />

          {/* X-axis ticks + relative-time anchors */}
          {monthTicks.map(({ t, label }) => {
            const x = sx(t);
            return (
              <g key={label}>
                <line x1={x} x2={x} y1={sy(0)} y2={sy(0) + 5} stroke="#cfcfca" />
                <text
                  x={x}
                  y={sy(0) + 18}
                  textAnchor={label.includes("2026") ? "end" : "middle"}
                  className="viz-axis"
                >
                  {label}
                </text>
              </g>
            );
          })}
          {relTicks.map(({ t, label }) => (
            <text
              key={label}
              x={sx(t)}
              y={sy(0) + 34}
              textAnchor={label === "Now" ? "end" : "middle"}
              className="ar-ct-rel"
            >
              {label}
            </text>
          ))}

          {/* Seed spread whiskers for $8.70 working-starter 1-hour finals */}
          {drawOrder.map((pt) => {
            if (pt.kind !== "working" || !pt.seeds || pt.seeds.length < 2) return null;
            const cx = sx(parseUtc(pt.date));
            const yLo = sy(Math.min(...pt.seeds));
            const yHi = sy(Math.max(...pt.seeds));
            const isHovered = hoverId === pt.id;
            return (
              <line
                key={`w-${pt.id}`}
                x1={cx}
                x2={cx}
                y1={yLo}
                y2={yHi}
                stroke={CMYK.mDeep}
                strokeWidth={1.3}
                strokeOpacity={isHovered ? 0.85 : 0.35}
                pointerEvents="none"
              />
            );
          })}

          {/* Model points */}
          {drawOrder.map((pt) => {
            const cx = sx(parseUtc(pt.date));
            const cy = sy(pt.val);
            const isHovered = hoverId === pt.id;

            if (pt.kind === "working") {
              const s = isHovered ? 6.4 : pt.beatsStarter ? 5.2 : 4.4;
              const fill = pt.beatsStarter ? CMYK.m : "#d47aa9";
              const d = `M${cx},${cy - s}L${cx + s},${cy}L${cx},${cy + s}L${cx - s},${cy}Z`;
              return (
                <g
                  key={pt.id}
                  className="ar-ct-pt"
                  tabIndex={0}
                  role="button"
                  aria-label={`${pt.model} ($8.70 working starter, 1-hour final), released ${fmtDate(pt.date)}, compute multiplier ${pt.labelVal}`}
                  onMouseEnter={() => setHoverId(pt.id)}
                  onMouseLeave={() => setHoverId((k) => (k === pt.id ? null : k))}
                  onFocus={() => setHoverId(pt.id)}
                  onBlur={() => setHoverId((k) => (k === pt.id ? null : k))}
                  onClick={() => setHoverId((k) => (k === pt.id ? null : pt.id))}
                >
                  <circle cx={cx} cy={cy} r={13} fill="transparent" />
                  {isHovered && (
                    <path
                      d={`M${cx},${cy - 9.5}L${cx + 9.5},${cy}L${cx},${cy + 9.5}L${cx - 9.5},${cy}Z`}
                      fill="none"
                      stroke={CMYK.mDeep}
                      strokeWidth={1.5}
                      strokeOpacity={0.55}
                    />
                  )}
                  {pt.broken ? (
                    <path
                      d={d}
                      fill="#fff"
                      stroke="#b57a99"
                      strokeWidth={1.6}
                      strokeDasharray="2 1.5"
                    />
                  ) : (
                    <path
                      d={d}
                      fill={fill}
                      fillOpacity={isHovered ? 1 : pt.beatsStarter ? 0.92 : 0.72}
                      stroke="#fff"
                      strokeWidth={1.3}
                    />
                  )}
                </g>
              );
            }

            const fill = pt.beatsStarter ? CMYK.cDeep : "#989892";
            const r = isHovered ? 6.5 : pt.beatsStarter ? 5 : 4;
            return (
              <g
                key={pt.id}
                className="ar-ct-pt"
                tabIndex={0}
                role="button"
                aria-label={`${pt.model} (empty starter), released ${fmtDate(pt.date)}, compute multiplier ${pt.labelVal || fmtMult(null)}`}
                onMouseEnter={() => setHoverId(pt.id)}
                onMouseLeave={() => setHoverId((k) => (k === pt.id ? null : k))}
                onFocus={() => setHoverId(pt.id)}
                onBlur={() => setHoverId((k) => (k === pt.id ? null : k))}
                onClick={() => setHoverId((k) => (k === pt.id ? null : pt.id))}
              >
                <circle cx={cx} cy={cy} r={13} fill="transparent" />
                {isHovered && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={9.5}
                    fill="none"
                    stroke={fill}
                    strokeWidth={1.5}
                    strokeOpacity={0.55}
                  />
                )}
                <circle
                  cx={cx}
                  cy={cy}
                  r={r}
                  fill={fill}
                  fillOpacity={isHovered ? 1 : pt.beatsStarter ? 0.95 : 0.65}
                  stroke="#fff"
                  strokeWidth={1.5}
                />
              </g>
            );
          })}

          {/* Minimal inline labels */}
          {points.map((pt) => {
            const cfg = activeInline[pt.id];
            if (!cfg || (narrow && cfg.narrowHide)) return null;
            const cx = sx(parseUtc(pt.date));
            const cy = sy(pt.val);
            return (
              <text
                key={`lbl-${pt.id}`}
                x={cx + cfg.dx}
                y={cy + cfg.dy}
                textAnchor={cfg.anchor}
                className={pt.kind === "working" ? "ar-ct-lbl ar-ct-lbl-work" : "ar-ct-lbl"}
              >
                {pt.short}
                {pt.labelVal && <tspan className="ar-ct-lbl-val"> {pt.labelVal}</tspan>}
              </text>
            );
          })}

          {/* Right-margin stacked labels for the November frontier cluster */}
          {!narrow &&
            marginPoints.map((pt) => {
              const cx = sx(parseUtc(pt.date));
              const cy = sy(pt.val);
              const ly = marginY.get(pt.id) ?? cy;
              const xLabel = m.l + plotW + 12;
              const isWork = pt.kind === "working";
              return (
                <g key={`mlbl-${pt.id}`}>
                  <path
                    d={`M${(cx + 6).toFixed(1)},${cy.toFixed(1)}L${(xLabel - 4).toFixed(1)},${ly.toFixed(1)}`}
                    fill="none"
                    stroke={isWork ? CMYK.mDeep : CMYK.cDeep}
                    strokeWidth={0.9}
                    strokeOpacity={0.45}
                  />
                  <text
                    x={xLabel}
                    y={ly + 3.5}
                    className={
                      isWork
                        ? "ar-ct-lbl ar-ct-lbl-margin ar-ct-lbl-work"
                        : "ar-ct-lbl ar-ct-lbl-margin"
                    }
                  >
                    {pt.marginShort}{" "}
                    <tspan className="ar-ct-lbl-val">{pt.labelVal}</tspan>
                  </text>
                </g>
              );
            })}
        </svg>

        {/* Floating tooltip on hover / focus / tap */}
        {hovered && hoverPos && (
          <div
            className="viz-tip ar-ct-tip"
            style={{
              left: Math.min(
                Math.max(8, hoverPos.x > W * 0.58 ? hoverPos.x - 246 : hoverPos.x + 14),
                Math.max(8, W - 244),
              ),
              top: hoverPos.y < 120 ? hoverPos.y + 14 : hoverPos.y - 10,
              transform: hoverPos.y < 120 ? "none" : "translateY(-100%)",
            }}
          >
            <strong>{hovered.model}</strong>
            <span className="ar-ct-tip-meta">
              {LAB_NAME[hovered.lab]} · {fmtDate(hovered.date)} ·{" "}
              {hovered.kind === "working" ? "$8.70 working starter" : "Empty starter"}
            </span>
            {hovered.empty && (
              <>
                <span>
                  15-min final: <b>{fmtMult(hovered.empty.finalMult)}</b>
                </span>
                <span>
                  Best in-session: <b>{fmtMult(hovered.empty.bestMult)}</b>
                </span>
                <span>
                  Ladder score: <b>{hovered.empty.grade.toFixed(1)}</b> / 100
                </span>
                {hovered.empty.note && <em>{hovered.empty.note}</em>}
              </>
            )}
            {hovered.working && (
              <>
                <span>
                  1-h final:{" "}
                  <b>{fmtBudgetMult(hovered.working.mult1h, hovered.working.broken)}</b>
                  {hovered.working.seeds.length > 0 &&
                    ` (${hovered.working.seeds.map((s) => `${s.toFixed(2)}×`).join(", ")})`}
                </span>
                {hovered.working.mult8h != null && (
                  <span>
                    8-h retrain (training-only):{" "}
                    <b>
                      {hovered.working.extrap8h ? "≈" : ""}
                      {hovered.working.mult8h >= 9.5
                        ? Math.round(hovered.working.mult8h)
                        : hovered.working.mult8h.toFixed(2)}
                      ×
                    </b>
                  </span>
                )}
                {hovered.working.note && <em>{hovered.working.note}</em>}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
