import { useEffect, useRef, useState } from "react";
import { useAgentView } from "../../view";

// One row per model, one dot per attempt, on a shared axis. Shows spread, not just the mean.
export type StripRow = { label: string; values: readonly number[]; mean?: number; highlight?: boolean; color?: string };
export type StripRef = { value: number; label: string; strong?: boolean };

export default function Strip({
  rows,
  domain,
  unit = "Elo",
  tickStep = 50,
  format = (v) => (v > 0 ? `+${v}` : String(v)),
  refs,
  xLabel,
}: {
  rows: readonly StripRow[];
  domain: [number, number];
  unit?: string;
  tickStep?: number;
  format?: (v: number) => string;
  refs?: readonly StripRef[];
  xLabel?: string;
}) {
  const agent = useAgentView();
  const ref = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(600);
  const [tip, setTip] = useState<{ left: number; top: number; text: string } | null>(null);
  const [seen, setSeen] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setW(Math.max(260, Math.round(el.getBoundingClientRect().width)));
    measure();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    const io = typeof IntersectionObserver !== "undefined"
      ? new IntersectionObserver(([e]) => e.isIntersecting && setSeen(true), { threshold: 0.3 })
      : null;
    if (io) io.observe(el); else setSeen(true);
    return () => { ro?.disconnect(); io?.disconnect(); };
  }, [agent]);

  if (agent) {
    return (
      <div className="agent-table">
        <table>
          <thead><tr><th>Model</th><th>Mean ({unit})</th><th>Attempts ({unit})</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label}>
                <td>{r.label}</td>
                <td>{r.mean !== undefined ? format(Number(r.mean.toFixed(1))) : "—"}</td>
                <td>{r.values.map((v) => format(Number(v.toFixed(1)))).join(", ")}</td>
              </tr>
            ))}
            {refs?.map((rf) => (
              <tr key={rf.label}><td><em>{rf.label}</em></td><td>{format(Number(rf.value.toFixed(1)))}</td><td>reference</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  const narrow = W < 480;
  const hasRefs = Boolean(refs && refs.length > 0);
  const row = narrow ? 40 : 30, m = { l: narrow ? 8 : 130, r: 18, t: hasRefs ? 24 : 6, b: xLabel ? 40 : 28 };
  const H = m.t + m.b + rows.length * row;
  const sx = (v: number) => m.l + ((v - domain[0]) / (domain[1] - domain[0])) * (W - m.l - m.r);
  const ticks: number[] = [];
  for (let t = Math.ceil(domain[0] / tickStep) * tickStep; t <= domain[1]; t += tickStep) ticks.push(t);

  return (
    <div className={`viz ar-strip${seen ? " seen" : ""}`} ref={ref} onMouseLeave={() => setTip(null)}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={xLabel ?? "Score of each attempt by model"}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} className={!hasRefs && t === 0 ? "viz-zero" : "viz-grid"} />
            <text x={sx(t)} y={H - m.b + 16} className="viz-axis" textAnchor="middle">{format(t)}</text>
          </g>
        ))}
        {refs?.map((rf) => {
          const x = sx(rf.value);
          const anchor = x > W - m.r - 90 ? "end" : x < m.l + 90 ? "start" : "middle";
          return (
            <g key={rf.label}>
              <line
                x1={x} x2={x} y1={m.t - 4} y2={H - m.b}
                stroke={rf.strong ? "#111" : "#8a8a85"}
                strokeWidth={rf.strong ? 1.25 : 1}
                strokeDasharray={rf.value === 50 ? undefined : "4 3"}
              />
              <text x={x} y={m.t - 10} className="viz-axis" textAnchor={anchor} style={rf.strong ? { fill: "var(--ink)", fontWeight: 600 } : undefined}>
                {rf.label}
              </text>
            </g>
          );
        })}
        {xLabel && <text x={W - m.r} y={H - 4} className="viz-axis" textAnchor="end">{xLabel}</text>}
        {rows.map((r, i) => {
          const y = m.t + i * row + row / 2 + (narrow ? 7 : 0);
          const lo = Math.min(...r.values), hi = Math.max(...r.values);
          return (
            <g key={r.label}>
              {narrow
                ? <text x={m.l} y={y - 13} className="viz-label">{r.label}</text>
                : <text x={m.l - 12} y={y} className="viz-label" textAnchor="end" dominantBaseline="middle">{r.label}</text>}
              <line x1={sx(lo)} x2={sx(hi)} y1={y} y2={y} stroke={r.highlight ? (r.color ?? "var(--accent)") : "#e2e2dd"} strokeOpacity={r.highlight ? 0.25 : 1} strokeWidth={6} strokeLinecap="round" />
              {r.mean !== undefined && (
                <line x1={sx(r.mean)} x2={sx(r.mean)} y1={y - 8} y2={y + 8} stroke={r.highlight ? (r.color ?? "var(--accent)") : "#52514e"} strokeWidth={2} strokeLinecap="round" />
              )}
              {r.values.map((v, j) => (
                <circle
                  key={j}
                  className="ar-strip-dot"
                  style={{ ["--i" as string]: i * 5 + j }}
                  cx={sx(v)} cy={y} r={5}
                  fill={r.highlight ? (r.color ?? "var(--accent)") : "#fff"}
                  stroke={r.highlight ? "#fff" : "var(--secondary)"}
                  strokeWidth={1.5}
                  tabIndex={0}
                  onMouseEnter={() => setTip({ left: Math.min(sx(v) * (ref.current!.getBoundingClientRect().width / W), W - 180), top: (y - 10) * (ref.current!.getBoundingClientRect().width / W), text: `${r.label}, attempt ${j + 1}: ${format(Number(v.toFixed(1)))}` })}
                  onFocus={() => setTip({ left: Math.min(sx(v), W - 180), top: y - 10, text: `${r.label}, attempt ${j + 1}: ${format(Number(v.toFixed(1)))}` })}
                />
              ))}
            </g>
          );
        })}
      </svg>
      {tip && <div className="viz-tip" style={{ left: tip.left, top: tip.top }} role="status"><span>{tip.text}</span></div>}
    </div>
  );
}
