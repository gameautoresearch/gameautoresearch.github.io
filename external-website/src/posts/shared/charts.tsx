import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";

// Small hand-rolled SVG charts, shared with thekevinz.com's Polytopia post. They share one look:
// recessive grid, 2px lines, >=8px markers, a hover tooltip on every mark.

export type Point = { x: number; y: number; label: string; note?: string };
export type Series = { name: string; color: string; points: Point[]; dashed?: boolean; dotsOnly?: boolean; noDots?: boolean };

type Hover = { left: number; top: number; body: ReactNode } | null;

function Tooltip({ hover }: { hover: Hover }) {
  if (!hover) return null;
  return (
    <div className="viz-tip" style={{ left: hover.left, top: hover.top }} role="status">
      {hover.body}
    </div>
  );
}

// Charts draw at the width they are shown at, so text stays at its CSS size on a phone instead of shrinking with
// a fixed viewBox. Below NARROW the layouts switch to a stacked form.
const NARROW = 480;
function useWidth(ref: RefObject<HTMLElement>, fallback = 600) {
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setWidth(Math.max(260, Math.round(el.getBoundingClientRect().width)) || fallback);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref, fallback]);
  return width;
}

// An axis label, broken into one line per word when the chart is narrow.
function AxisLabel({ x, y, text, narrow }: { x: number; y: number; text: string; narrow: boolean }) {
  const lines = narrow ? text.split(" ") : [text];
  return (
    <text x={x} y={y} className="viz-axis" textAnchor="middle">
      {lines.map((l, i) => <tspan key={i} x={x} dy={i ? 12 : 0}>{l}</tspan>)}
    </text>
  );
}

// Greedy word wrap to lines of at most `max` characters (a longer single word keeps its own line).
function wrapWords(text: string, max: number) {
  const lines: string[] = [];
  for (const word of text.split(" ")) {
    const last = lines[lines.length - 1];
    if (last !== undefined && (last + " " + word).length <= max) lines[lines.length - 1] = last + " " + word;
    else lines.push(word);
  }
  return lines;
}

function niceTicks(min: number, max: number, count = 5) {
  const span = max - min;
  const step = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = (count * step) / span;
  const mult = err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1;
  const s = step * mult;
  const ticks: number[] = [];
  for (let v = Math.ceil(min / s) * s; v <= max + 1e-9; v += s) ticks.push(+v.toFixed(6));
  return ticks;
}

export function LineChart({
  series,
  xDomain,
  yDomain,
  xTicks,
  xFormat = (v) => String(v),
  yFormat = (v) => String(v),
  yLog = false,
  height = 260,
  refLine,
  xLabel,
  yLabel,
  stepped = false,
}: {
  series: Series[];
  xDomain: [number, number];
  yDomain: [number, number];
  xTicks?: number[];
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  yLog?: boolean;
  height?: number;
  refLine?: { y: number; label: string };
  xLabel?: string;
  yLabel?: string;
  stepped?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref), narrow = W < NARROW;
  const H = narrow ? Math.round(height * 0.85) : height, m = { l: 44, r: 12, t: 14, b: narrow ? 44 : 34 };
  const [hover, setHover] = useState<Hover>(null);
  const [active, setActive] = useState<string | null>(null);
  const sx = (v: number) => m.l + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * (W - m.l - m.r);
  const ly = (v: number) => (yLog ? Math.log10(v) : v);
  const sy = (v: number) =>
    H - m.b - ((ly(v) - ly(yDomain[0])) / (ly(yDomain[1]) - ly(yDomain[0]))) * (H - m.t - m.b);
  const yTicks = yLog
    ? Array.from({ length: Math.round(Math.log10(yDomain[1]) - Math.log10(yDomain[0])) + 1 }, (_, i) => yDomain[0] * 10 ** i)
    : niceTicks(yDomain[0], yDomain[1]);
  const xs = xTicks ?? niceTicks(xDomain[0], xDomain[1]);

  const show = (p: Point, s: Series) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    const k = box.width / W;
    setActive(s.name + p.label);
    setHover({
      left: Math.max(0, Math.min(sx(p.x) * k, box.width - 200)),
      top: sy(p.y) * k - 12,
      body: (
        <>
          <strong>{p.label}</strong>
          <span>{s.name}: {yFormat(p.y)}</span>
          {p.note && <em>{p.note}</em>}
        </>
      ),
    });
  };

  const path = (pts: Point[]) =>
    pts
      .map((p, i) => {
        if (i === 0 || !stepped) return `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`;
        return `H${sx(p.x)}V${sy(p.y)}`;
      })
      .join("");

  return (
    <div className="viz" ref={ref} onMouseLeave={() => { setHover(null); setActive(null); }}>
      {series.length > 1 && (
        <div className="viz-legend">
          {series.map((s) => (
            <span key={s.name}><i style={s.dashed ? { color: s.color } : { background: s.color }} className={s.dashed ? "dashed" : ""} />{s.name}</span>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} role="img">
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className="viz-grid" />
            <text x={m.l - 8} y={sy(t)} className="viz-axis" textAnchor="end" dominantBaseline="middle">{yFormat(t)}</text>
          </g>
        ))}
        {xs.map((t) => (
          <AxisLabel key={t} x={sx(t)} y={H - m.b + 18} text={xFormat(t)} narrow={narrow} />
        ))}
        {xLabel && <text x={W - m.r} y={H - 2} className="viz-axis" textAnchor="end">{xLabel}</text>}
        {yLabel && <text x={m.l} y={m.t - 4} className="viz-axis">{yLabel}</text>}
        {refLine && (
          <g>
            <line x1={m.l} x2={W - m.r} y1={sy(refLine.y)} y2={sy(refLine.y)} className="viz-ref" />
            <text x={W - m.r} y={sy(refLine.y) - 6} className="viz-axis" textAnchor="end">{refLine.label}</text>
          </g>
        )}
        {series.filter((s) => !s.dotsOnly).map((s) => (
          <path key={s.name} d={path(s.points)} fill="none" stroke={s.color} strokeWidth={2}
            strokeDasharray={s.dashed ? "5 4" : undefined} strokeLinejoin="round" />
        ))}
        {series.filter((s) => !s.noDots).map((s) =>
          s.points.map((p) => (
            <g key={s.name + p.label} onMouseEnter={() => show(p, s)} onFocus={() => show(p, s)} tabIndex={0}>
              <circle cx={sx(p.x)} cy={sy(p.y)} r={14} fill="transparent" />
              <circle cx={sx(p.x)} cy={sy(p.y)} r={active === s.name + p.label ? 6 : 4.5}
                fill={s.color} stroke="var(--viz-surface)" strokeWidth={2} />
            </g>
          )),
        )}
      </svg>
      <Tooltip hover={hover} />
    </div>
  );
}

export type Bar = { label: string; value: number; lo?: number; hi?: number; tone: "good" | "flat" | "bad"; note: string };

const TONE = { good: "var(--viz-1)", flat: "var(--viz-flat)", bad: "var(--viz-2)" };

const DEFAULT_LEGEND = { good: "Kept", flat: "No effect", bad: "Hurt" };

export function DeltaBars({ bars, domain, unit = "Elo", legend = DEFAULT_LEGEND, labelWidth = 210 }: {
  bars: Bar[];
  domain: [number, number];
  unit?: string;
  legend?: Partial<Record<Bar["tone"], string>>;
  labelWidth?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref), narrow = W < NARROW;
  const row = narrow ? 44 : 34, m = { l: narrow ? 8 : labelWidth, r: narrow ? 8 : 20, t: 8, b: 28 };
  const H = m.t + m.b + bars.length * row;
  const [hover, setHover] = useState<Hover>(null);
  const sx = (v: number) => m.l + ((v - domain[0]) / (domain[1] - domain[0])) * (W - m.l - m.r);
  const ticks = niceTicks(domain[0], domain[1], 6);
  const show = (b: Bar, i: number) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    const k = box.width / W;
    setHover({
      left: Math.max(0, Math.min(sx(Math.max(b.value, 0)) * k, box.width - 220)),
      top: (m.t + i * row) * k - 8,
      body: (
        <>
          <strong>{b.label}</strong>
          <span>{b.value > 0 ? "+" : ""}{b.value} {unit}{b.lo !== undefined ? ` [${b.lo}, ${b.hi}]` : ""}</span>
          <em>{b.note}</em>
        </>
      ),
    });
  };
  return (
    <div className="viz" ref={ref} onMouseLeave={() => setHover(null)}>
      <div className="viz-legend">
        {(Object.keys(TONE) as Bar["tone"][]).filter((t) => legend[t]).map((t) => (
          <span key={t}><i style={{ background: TONE[t] }} />{legend[t]}</span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} className={t === 0 ? "viz-zero" : "viz-grid"} />
            <text x={sx(t)} y={H - m.b + 18} className="viz-axis" textAnchor="middle">{t > 0 ? "+" : ""}{t}</text>
          </g>
        ))}
        {bars.map((b, i) => {
          const y = m.t + i * row + row / 2 + (narrow ? 8 : 0);
          const x0 = sx(Math.min(0, b.value)), x1 = sx(Math.max(0, b.value));
          return (
            <g key={b.label} onMouseEnter={() => show(b, i)} onFocus={() => show(b, i)} tabIndex={0}>
              <rect x={0} y={y - row / 2 - (narrow ? 8 : 0)} width={W} height={row} fill="transparent" />
              {narrow
                ? <text x={m.l} y={y - 14} className="viz-label">{b.label}</text>
                : <text x={m.l - 12} y={y} className="viz-label" textAnchor="end" dominantBaseline="middle">{b.label}</text>}
              <rect x={x0} y={y - (narrow ? 5 : 7)} width={Math.max(x1 - x0, 2)} height={narrow ? 10 : 14} rx={3} fill={TONE[b.tone]} />
              {b.lo !== undefined && b.hi !== undefined && (
                <g className="viz-ci">
                  <line x1={sx(b.lo)} x2={sx(b.hi)} y1={y} y2={y} />
                  <line x1={sx(b.lo)} x2={sx(b.lo)} y1={y - 5} y2={y + 5} />
                  <line x1={sx(b.hi)} x2={sx(b.hi)} y1={y - 5} y2={y + 5} />
                </g>
              )}
            </g>
          );
        })}
      </svg>
      <Tooltip hover={hover} />
    </div>
  );
}

export type Column = { label: string; value: number; display?: string; note: string; highlight?: boolean };

export function Columns({ cols, max, yFormat = (v) => String(v) }: { cols: Column[]; max: number; yFormat?: (v: number) => string }) {
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref), narrow = W < NARROW;
  const H = 220, m = { l: 40, r: 8, t: 22, b: narrow ? 64 : 40 };
  const [hover, setHover] = useState<Hover>(null);
  const bw = (W - m.l - m.r) / cols.length;
  const sy = (v: number) => H - m.b - (v / max) * (H - m.t - m.b);
  const show = (c: Column, i: number) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    const k = box.width / W;
    setHover({
      left: Math.max(0, Math.min((m.l + i * bw) * k, box.width - 220)),
      top: sy(c.value) * k - 10,
      body: (<><strong>{c.label}</strong><span>{c.display ?? yFormat(c.value)}</span><em>{c.note}</em></>),
    });
  };
  return (
    <div className="viz" ref={ref} onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img">
        {niceTicks(0, max, 3).map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className={t === 0 ? "viz-zero" : "viz-grid"} />
            <text x={m.l - 8} y={sy(t)} className="viz-axis" textAnchor="end" dominantBaseline="middle">{yFormat(t)}</text>
          </g>
        ))}
        {cols.map((c, i) => {
          const x = m.l + i * bw + bw * 0.22, w = bw * 0.56;
          return (
            <g key={c.label} onMouseEnter={() => show(c, i)} onFocus={() => show(c, i)} tabIndex={0}>
              <rect x={m.l + i * bw} y={m.t} width={bw} height={H - m.t - m.b} fill="transparent" />
              <rect x={x} y={sy(c.value)} width={w} height={Math.max(sy(0) - sy(c.value), 2)} rx={3}
                fill={c.highlight ? "var(--viz-1)" : "var(--viz-flat)"} />
              <text x={x + w / 2} y={sy(c.value) - 6} className="viz-value" textAnchor="middle">{c.display ?? yFormat(c.value)}</text>
              {c.label.split("\n").flatMap((line) => (narrow ? wrapWords(line, 10) : [line])).map((line, j) => (
                <text key={j} x={x + w / 2} y={H - m.b + 14 + j * 12} className="viz-axis" textAnchor="middle">{line}</text>
              ))}
            </g>
          );
        })}
      </svg>
      <Tooltip hover={hover} />
    </div>
  );
}

export function Figure({ children, caption, n, wide = false }: { children: ReactNode; caption: ReactNode; n: number; wide?: boolean }) {
  return (
    <figure className={wide ? "pt-figure wide" : "pt-figure"}>
      {children}
      <figcaption><span>Figure {n}.</span> {caption}</figcaption>
    </figure>
  );
}
