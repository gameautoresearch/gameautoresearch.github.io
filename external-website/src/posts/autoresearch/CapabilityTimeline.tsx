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

const DATA = raw as unknown as {
  floorMult: number;
  models: ModelRow[];
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
const MARGIN_KEYS = new Set(["sol62", "gem4n", "astra62", "fable"]);

type InlineCfg = {
  dx: number;
  dy: number;
  anchor: "start" | "middle" | "end";
  narrowHide?: boolean;
};

// Minimal inline callouts for key inflection points.
const INLINE_LABELS: Record<string, InlineCfg> = {
  opus5: { dx: 0, dy: -11, anchor: "middle" },
  astra: { dx: -10, dy: 4, anchor: "end" },
  sol: { dx: 10, dy: 4, anchor: "start", narrowHide: true },
  opus: { dx: -8, dy: -9, anchor: "end", narrowHide: true },
  terra: { dx: 0, dy: -11, anchor: "middle", narrowHide: true },
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

function layoutLabels(
  items: { key: string; targetY: number }[],
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
  return new Map(out.map((i) => [i.key, i.y]));
}

export default function CapabilityTimeline() {
  const agent = useAgentView();
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref);
  const narrow = W < 540;
  const [hoverKey, setHoverKey] = useState<string | null>(null);

  // Plotted value: each model's best in-session network.
  const valOf = (row: ModelRow) => row.bestPlotMult;
  const multOf = (row: ModelRow) => row.bestMult;

  const sortedModels = useMemo(
    () => [...DATA.models].sort((a, b) => a.date.localeCompare(b.date) || valOf(a) - valOf(b)),
    [],
  );

  // Monotone running-best frontier across all releases.
  const frontier = useMemo(() => {
    const byDate = new Map<number, number>();
    for (const row of sortedModels) {
      const t = parseUtc(row.date);
      const v = valOf(row);
      const prev = byDate.get(t);
      if (prev === undefined || v > prev) byDate.set(t, v);
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
  }, [sortedModels]);

  if (agent) {
    return (
      <Table
        head={[
          "Model",
          "Lab",
          "Released",
          "15-min final mult",
          "Best in-session mult",
          "Ladder score",
          "Note",
        ]}
        align={["l", "l", "l", "r", "r", "r", "l"]}
        rows={sortedModels.map((m) => [
          m.model,
          LAB_NAME[m.lab],
          fmtDate(m.date),
          fmtMult(m.finalMult),
          fmtMult(m.bestMult),
          `${m.grade.toFixed(1)} / 100`,
          m.note ?? "",
        ])}
      />
    );
  }

  const H = narrow ? 296 : 344;
  const m = { l: 38, r: narrow ? 16 : 132, t: 24, b: 52 };
  const plotW = W - m.l - m.r;
  const plotH = H - m.t - m.b;

  const tMin = Date.UTC(2026, 0, 22);
  const tMax = Date.UTC(2026, 10, 25);
  const yMax = 5.5;

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

  const yTicks = [0, 1, 2, 3, 4, 5];

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

  const marginModels = sortedModels
    .filter((row) => MARGIN_KEYS.has(row.key))
    .sort((a, b) => valOf(b) - valOf(a));
  const marginY = layoutLabels(
    marginModels.map((row) => ({ key: row.key, targetY: sy(valOf(row)) })),
    16,
    m.t + 14,
    sy(0) - 18,
  );

  const hovered = hoverKey ? sortedModels.find((r) => r.key === hoverKey) ?? null : null;
  const hoverPos = hovered
    ? { x: sx(parseUtc(hovered.date)), y: sy(valOf(hovered)) }
    : null;

  // Draw weaker (<1x) models first so >=1x models sit cleanly on top.
  const drawOrder = [
    ...sortedModels.filter((r) => (multOf(r) ?? 0) < 1),
    ...sortedModels.filter((r) => (multOf(r) ?? 0) >= 1),
  ];

  const yOne = sy(1);
  const yFloor = sy(DATA.floorMult);

  return (
    <div className="ar-ct" ref={ref}>
      <div className="viz ar-ct-plot">
        <svg
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Compute multiplier (best in-session network) of ${DATA.models.length} models by release date from February to November 2026`}
        >
          {/* Whisper-light tint above 1x (beats the unmodified PPO starter) */}
          <rect x={m.l} y={m.t} width={plotW} height={yOne - m.t} fill="#f4fafd" />

          {/* Subtle band below 0.43x where models are too weak to place precisely on the starter curve */}
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
            Compute multiplier of best in-session network vs. 15-min starter
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

          {/* Model points */}
          {drawOrder.map((row) => {
            const cx = sx(parseUtc(row.date));
            const cy = sy(valOf(row));
            const mult = multOf(row);
            const beatsStarter = (mult ?? 0) >= 1;
            const isHovered = hoverKey === row.key;
            const fill = beatsStarter ? CMYK.cDeep : "#989892";
            const r = isHovered ? 6.5 : beatsStarter ? 5 : 4;

            return (
              <g
                key={row.key}
                className="ar-ct-pt"
                tabIndex={0}
                role="button"
                aria-label={`${row.model}, released ${fmtDate(row.date)}, compute multiplier ${fmtMult(mult)}`}
                onMouseEnter={() => setHoverKey(row.key)}
                onMouseLeave={() => setHoverKey((k) => (k === row.key ? null : k))}
                onFocus={() => setHoverKey(row.key)}
                onBlur={() => setHoverKey((k) => (k === row.key ? null : k))}
                onClick={() => setHoverKey((k) => (k === row.key ? null : row.key))}
              >
                <circle cx={cx} cy={cy} r={14} fill="transparent" />
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
                  fillOpacity={isHovered ? 1 : beatsStarter ? 0.95 : 0.65}
                  stroke="#fff"
                  strokeWidth={1.5}
                />
              </g>
            );
          })}

          {/* Minimal inline labels */}
          {sortedModels.map((row) => {
            const cfg = INLINE_LABELS[row.key];
            if (!cfg || (narrow && cfg.narrowHide)) return null;
            const cx = sx(parseUtc(row.date));
            const cy = sy(valOf(row));
            const mult = multOf(row);
            const multStr = mult != null ? ` ${fmtMult(mult)}` : "";
            return (
              <text
                key={`lbl-${row.key}`}
                x={cx + cfg.dx}
                y={cy + cfg.dy}
                textAnchor={cfg.anchor}
                className="ar-ct-lbl"
              >
                {row.short}
                {multStr && <tspan className="ar-ct-lbl-val">{multStr}</tspan>}
              </text>
            );
          })}

          {/* Right-margin stacked labels for the November frontier cluster */}
          {!narrow &&
            marginModels.map((row) => {
              const cx = sx(parseUtc(row.date));
              const cy = sy(valOf(row));
              const ly = marginY.get(row.key) ?? cy;
              const xLabel = m.l + plotW + 12;
              return (
                <g key={`mlbl-${row.key}`}>
                  <path
                    d={`M${(cx + 6).toFixed(1)},${cy.toFixed(1)}L${(xLabel - 4).toFixed(1)},${ly.toFixed(1)}`}
                    fill="none"
                    stroke={CMYK.cDeep}
                    strokeWidth={0.9}
                    strokeOpacity={0.4}
                  />
                  <text x={xLabel} y={ly + 3.5} className="ar-ct-lbl ar-ct-lbl-margin">
                    {row.short}{" "}
                    <tspan className="ar-ct-lbl-val">{fmtMult(multOf(row))}</tspan>
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
                Math.max(8, hoverPos.x > W * 0.58 ? hoverPos.x - 234 : hoverPos.x + 14),
                Math.max(8, W - 232),
              ),
              top: hoverPos.y < 120 ? hoverPos.y + 14 : hoverPos.y - 10,
              transform: hoverPos.y < 120 ? "none" : "translateY(-100%)",
            }}
          >
            <strong>{hovered.model}</strong>
            <span className="ar-ct-tip-meta">
              {LAB_NAME[hovered.lab]} · {fmtDate(hovered.date)}
            </span>
            <span>
              Compute multiplier: <b>{fmtMult(hovered.bestMult)}</b>
            </span>
            <span>
              Ladder score: <b>{hovered.grade.toFixed(1)}</b> / 100
            </span>
            {hovered.note && <em>{hovered.note}</em>}
          </div>
        )}
      </div>
    </div>
  );
}
