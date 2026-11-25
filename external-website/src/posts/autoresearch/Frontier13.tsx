import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useAgentView } from "../../view";
import { modelStyle } from "./colors";
import raw from "./polytopia13.json";

// Polytopia from an empty starter, 20 models at 5 GPU-hours: the 13 of campaigns polytopia-empty15-20261118 and
// polytopia-frontier6-20261118 (readout reports/polytopia/frontier6-2026-11-19.md), Inkling 2 Small
// (poly-timeline-20261120), GPT-6.2 Sol (poly-empty-sol62-20261120) and the five recipe-variant sessions of
// poly-empty-recipe-fill-20261122. One pooled Bradley–Terry fit (poly-fill-pooled-20261122, 865 entrants, about
// 2.5 million games) over every network the counted sessions sealed, their fresh 15-minute finals, earlier model
// releases, the PPO starter (15 min = 0) and the ladder references. polytopia13.json (the name is kept) is derived
// from that fit's elo.json: nets are [GPU-hours when sealed, Elo, run name, formally submitted].

type Net = [number, number, string, number];
type Model = {
  key: string; label: string; gpu: number; note: string | null;
  final: { elo: number; ci: number[] }; best: { elo: number; ci: number[]; run: string; gpu: number }; nets: Net[];
};
const DATA = raw as unknown as { games: number; pairs: number; models: Model[]; refs: Record<string, number> };
const MODELS = DATA.models;
const winPct = (elo: number) => 100 / (1 + 10 ** (-elo / 400));
const fmtWin = (elo: number) => {
  const p = winPct(elo);
  return p < 0.5 ? "<1%" : p > 99.5 ? ">99%" : `${Math.round(p)}%`;
};
const REFS = [
  {
    id: "tuned",
    label: "36-min baseline",
    shortLabel: "36-min baseline",
    elo: DATA.refs["starter-200m"],
    align: "left" as const,
  },
  {
    id: "base15",
    label: "15-min baseline",
    shortLabel: "15-min baseline",
    elo: 0,
    align: "right" as const,
  },
];
const signed = (v: number) => (v > 0 ? `+${v}` : v < 0 ? `−${-v}` : "0");
const gpuText = (h: number) => (h < 1 ? `${Math.round(h * 60)} min` : `${h.toFixed(2)} GPU-h`);
const hourText = (h: number) => (h < 1 ? `${Math.round(h * 60)} min` : `${h.toFixed(1)}h`);

// Label every model that reaches >10% win rate and runs for most of the session (>=3h) directly on the right margin
// (12 models), giving a strict 1-to-1 match between colored curves and right-margin labels. The remaining 8 models
// (7 clustered under 6%, plus GPT-6.2 Astra which broke its env at 1.1h) stay quiet grey and light up in color on hover.
const LABELED_KEYS = new Set(
  MODELS.filter((model) => winPct(model.best.elo) >= 10 && model.gpu >= 3.0).map((model) => model.key),
);
const TOP_KEYS = new Set(
  [...MODELS]
    .filter((model) => LABELED_KEYS.has(model.key))
    .sort((a, b) => b.best.elo - a.best.elo)
    .slice(0, 4)
    .map((model) => model.key),
);

function curveColor(label: string) {
  const c = modelStyle(label).color;
  return c === "#bdbdbd" || c === "#b5b5b5" ? "#7d7d77" : c;
}

function labelTextColor(label: string) {
  const c = curveColor(label);
  if (c === "#d4a000") return "#967000";
  if (c === "#7f9ae0") return "#5470bd";
  if (c === "#f08cc0") return "#c75091";
  return c;
}

type NetPoint = {
  x: number;
  y: number;
  elo: number;
  run: string;
  submitted: number;
  isFrontier: boolean;
  bestAtTime: number;
};

function modelPoints(m: Model): NetPoint[] {
  let best = -Infinity;
  return m.nets.map(([x, elo, run, submitted]) => {
    const isFrontier = elo > best;
    if (isFrontier) best = elo;
    return {
      x,
      y: winPct(elo),
      elo,
      run,
      submitted,
      isFrontier,
      bestAtTime: best,
    };
  });
}

function distSqToSegment(px: number, py: number, x1: number, y1: number, x2: number, y2: number) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return { d2: (px - x1) ** 2 + (py - y1) ** 2, qx: x1, qy: y1 };
  const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq));
  const qx = x1 + t * dx;
  const qy = y1 + t * dy;
  return { d2: (px - qx) ** 2 + (py - qy) ** 2, qx, qy };
}

function layoutLabels(items: { key: string; targetY: number }[], minGap: number, minY: number, maxY: number) {
  const out = items.map((item) => ({ ...item, y: item.targetY })).sort((a, b) => a.targetY - b.targetY);
  if (!out.length) return new Map<string, number>();
  // Center the cluster around its natural centroid when spacing forces expansion.
  for (let pass = 0; pass < 8; pass++) {
    for (let i = 1; i < out.length; i++) {
      const overlap = out[i - 1].y + minGap - out[i].y;
      if (overlap > 0) {
        out[i - 1].y -= overlap / 2;
        out[i].y += overlap / 2;
      }
    }
    if (out[0].y < minY) {
      const shift = minY - out[0].y;
      for (const item of out) item.y += shift;
    }
    const last = out[out.length - 1];
    if (last.y > maxY) {
      const shift = last.y - maxY;
      for (const item of out) item.y -= shift;
    }
  }
  return new Map(out.map((item) => [item.key, item.y]));
}

export function useWidth(ref: React.RefObject<HTMLElement>) {
  const [w, setW] = useState(690);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setW(Math.max(280, Math.round(el.getBoundingClientRect().width)));
    measure();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(measure) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, [ref]);
  return w;
}

type HoverState = {
  key: string;
  cx: number;
  cy: number;
  point: NetPoint | null;
  fromDot: boolean;
};

const MODEL_POINTS = new Map(MODELS.map((model) => [model.key, modelPoints(model)]));
const MODEL_STEPS = new Map(
  MODELS.map((model) => [model.key, (MODEL_POINTS.get(model.key) ?? []).filter((p) => p.isFrontier)]),
);
const MODEL_BY_KEY = new Map(MODELS.map((model) => [model.key, model]));

export function ResearchRace() {
  const agent = useAgentView();
  const plotId = useId();
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref);
  const narrow = W < 540;
  const [hover, setHover] = useState<HoverState | null>(null);
  const [pinnedKey, setPinnedKey] = useState<string | null>(null);
  const [seen, setSeen] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || agent || seen) return;
    if (typeof IntersectionObserver === "undefined") {
      setSeen(true);
      return;
    }
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.intersectionRatio < 0.2) return;
      setSeen(true);
      observer.disconnect();
    }, { threshold: 0.2 });
    observer.observe(el);
    return () => observer.disconnect();
  }, [agent, seen]);

  if (agent) {
    return (
      <ul>
        {MODELS.map((x) => (
          <li key={x.key}>
            {x.label}: {x.nets.length} networks, best {fmtWin(x.best.elo)} vs 15-min PPO baseline (
            {signed(x.best.elo)} Elo, {x.best.run} at {gpuText(x.best.gpu)}), fresh 15-min retrain{" "}
            {fmtWin(x.final.elo)} ({signed(x.final.elo)} Elo).
          </li>
        ))}
      </ul>
    );
  }

  const H = narrow ? 340 : 450;
  const m = { l: 48, r: narrow ? 14 : 156, t: 32, b: 50 };
  const plotW = W - m.l - m.r;
  const plotH = H - m.t - m.b;
  const xMax = 5;
  const sx = (h: number) => m.l + (Math.min(xMax, Math.max(0, h)) / xMax) * plotW;
  const sy = (p: number) => m.t + ((100 - Math.min(100, Math.max(0, p))) / 100) * plotH;
  const xTicks = [0, 1, 2, 3, 4, 5];
  const yTicks = [0, 25, 50, 75, 100];

  const stepPath = (model: Model) => {
    const steps = MODEL_STEPS.get(model.key) ?? [];
    if (!steps.length) return "";
    let d = `M${sx(0).toFixed(1)},${sy(0).toFixed(1)}`;
    for (const s of steps) {
      d += `H${sx(s.x).toFixed(1)}V${sy(s.y).toFixed(1)}`;
    }
    d += `H${sx(model.gpu).toFixed(1)}`;
    return d;
  };

  const labeledModels = MODELS.filter((model) => LABELED_KEYS.has(model.key)).sort(
    (a, b) => b.best.elo - a.best.elo,
  );
  const topModels = labeledModels.filter((model) => TOP_KEYS.has(model.key));
  const labelYMap = layoutLabels(
    labeledModels.map((model) => ({ key: model.key, targetY: sy(winPct(model.best.elo)) })),
    18,
    m.t + 6,
    sy(10),
  );

  const hitAt = (px: number, py: number): HoverState | null => {
    if (px < m.l - 8 || px > W - m.r + 8 || py < m.t - 8 || py > H - m.b + 8) {
      return null;
    }

    // When a model is pinned, let the user inspect all of its experiment dots freely without losing focus.
    if (pinnedKey) {
      const pts = MODEL_POINTS.get(pinnedKey) ?? [];
      let nearPt: NetPoint | null = null;
      let nearD2 = 16 * 16;
      for (const p of pts) {
        const d2 = (sx(p.x) - px) ** 2 + (sy(p.y) - py) ** 2;
        if (d2 < nearD2) {
          nearD2 = d2;
          nearPt = p;
        }
      }
      if (nearPt) {
        return { key: pinnedKey, cx: sx(nearPt.x), cy: sy(nearPt.y), point: nearPt, fromDot: true };
      }
    }

    // Check experiment dots of the currently active model first if the cursor is right on one.
    const currentKey = pinnedKey ?? hover?.key ?? null;
    if (currentKey && !pinnedKey) {
      const pts = MODEL_POINTS.get(currentKey) ?? [];
      let nearPt: NetPoint | null = null;
      let nearD2 = 10 * 10;
      for (const p of pts) {
        const d2 = (sx(p.x) - px) ** 2 + (sy(p.y) - py) ** 2;
        if (d2 < nearD2) {
          nearD2 = d2;
          nearPt = p;
        }
      }
      if (nearPt) {
        return { key: currentKey, cx: sx(nearPt.x), cy: sy(nearPt.y), point: nearPt, fromDot: true };
      }
    }

    const candidates = pinnedKey ? [MODEL_BY_KEY.get(pinnedKey)!].filter(Boolean) : MODELS;
    let bestDist = 16 * 16;
    let found: HoverState | null = null;
    for (const model of candidates) {
      const steps = MODEL_STEPS.get(model.key) ?? [];
      if (!steps.length) continue;
      let curX = sx(0);
      let curY = sy(0);
      let curStep: NetPoint | null = null;

      for (const s of steps) {
        const nx = sx(s.x);
        const ny = sy(s.y);
        const hSeg = distSqToSegment(px, py, curX, curY, nx, curY);
        if (hSeg.d2 < bestDist) {
          bestDist = hSeg.d2;
          found = { key: model.key, cx: hSeg.qx, cy: hSeg.qy, point: curStep ?? s, fromDot: false };
        }
        const vSeg = distSqToSegment(px, py, nx, curY, nx, ny);
        if (vSeg.d2 < bestDist) {
          bestDist = vSeg.d2;
          found = { key: model.key, cx: vSeg.qx, cy: vSeg.qy, point: s, fromDot: false };
        }
        curX = nx;
        curY = ny;
        curStep = s;
      }

      const endX = sx(model.gpu);
      const lastSeg = distSqToSegment(px, py, curX, curY, endX, curY);
      if (lastSeg.d2 < bestDist) {
        bestDist = lastSeg.d2;
        found = { key: model.key, cx: lastSeg.qx, cy: lastSeg.qy, point: curStep, fromDot: false };
      }
    }
    return found;
  };

  const toPlotCoords = (ev: React.PointerEvent<SVGSVGElement>) => {
    const box = ev.currentTarget.getBoundingClientRect();
    const k = W / box.width;
    return { px: (ev.clientX - box.left) * k, py: (ev.clientY - box.top) * k };
  };

  const move = (ev: React.PointerEvent<SVGSVGElement>) => {
    const { px, py } = toPlotCoords(ev);
    setHover(hitAt(px, py));
  };

  const clickPlot = (ev: React.PointerEvent<SVGSVGElement>) => {
    const { px, py } = toPlotCoords(ev);
    const found = hitAt(px, py);
    if (ev.pointerType === "touch") {
      setHover(found);
      setPinnedKey(found ? found.key : null);
      return;
    }
    if (!found) {
      setPinnedKey(null);
      setHover(null);
    } else if (!pinnedKey) {
      setPinnedKey(found.key);
      setHover(found);
    }
  };

  const hoverFromLabel = (model: Model) => {
    if (pinnedKey && pinnedKey !== model.key) return;
    const steps = MODEL_STEPS.get(model.key) ?? [];
    const last = steps[steps.length - 1] ?? null;
    setHover({
      key: model.key,
      cx: sx(model.gpu),
      cy: sy(winPct(model.best.elo)),
      point: last,
      fromDot: false,
    });
  };

  const togglePinLabel = (model: Model) => {
    if (pinnedKey === model.key) {
      setPinnedKey(null);
      setHover(null);
    } else {
      setPinnedKey(model.key);
      setHover(null);
    }
  };

  const activeKey = pinnedKey ?? hover?.key ?? null;
  const activeModel = activeKey ? MODEL_BY_KEY.get(activeKey) ?? null : null;
  const isColored = (model: Model) => LABELED_KEYS.has(model.key);
  const drawOrder = [...MODELS].sort((a, b) => {
    const ra = a.key === activeKey ? 3 : TOP_KEYS.has(a.key) ? 2 : isColored(a) ? 1 : 0;
    const rb = b.key === activeKey ? 3 : TOP_KEYS.has(b.key) ? 2 : isColored(b) ? 1 : 0;
    return ra - rb || a.best.elo - b.best.elo;
  });

  const tipW = 190;
  const tipLeft = hover
    ? Math.max(4, Math.min(W - tipW - 4, hover.cx + 14 + tipW > W - 4 ? hover.cx - tipW - 14 : hover.cx + 14))
    : 0;
  const tipBelow = hover ? hover.cy < 116 : false;

  const otherCount = MODELS.length - labeledModels.length;

  return (
    <div className={`viz fr${seen ? " fr-runs-seen" : ""}`} ref={ref}>
      <div className="fr-head">
        <div className="fr-title">Best 15-minute network found over 5 hours of research</div>
        {pinnedKey && activeModel && (
          <button
            type="button"
            className="fr-pin-chip"
            onClick={() => {
              setPinnedKey(null);
              setHover(null);
            }}
          >
            <i style={{ background: curveColor(activeModel.label) }} />
            {activeModel.label} ({activeModel.nets.length} runs) ×
          </button>
        )}
      </div>
      {narrow && (
        <div className="viz-legend fr-compact-key">
          {topModels.map((model) => (
            <span key={model.key}>
              <i style={{ background: curveColor(model.label) }} />
              {model.label}
            </span>
          ))}
          <span className="fr-compact-others">
            <i style={{ background: "#c8c8c2" }} />
            +{MODELS.length - topModels.length} models
          </span>
        </div>
      )}
      <div className="ar-cost-plot">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Best 15-minute network win rate over 5 hours of research, ${MODELS.length} models`}
          onPointerMove={move}
          onPointerDown={clickPlot}
          onPointerLeave={() => setHover(null)}
          style={{ touchAction: "pan-y" }}
        >
          {/* Horizontal grid + Y-axis ticks */}
          {yTicks.map((t) => (
            <g key={t}>
              <line
                x1={m.l}
                x2={W - m.r}
                y1={sy(t)}
                y2={sy(t)}
                className={t === 0 ? "viz-zero" : "viz-grid"}
              />
              <text x={m.l - 8} y={sy(t)} className="fr-axis" textAnchor="end" dominantBaseline="middle">
                {t}%
              </text>
            </g>
          ))}

          {/* Vertical grid + X-axis ticks */}
          {xTicks.map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} className="viz-grid" />
              <text x={sx(t)} y={H - m.b + 19} className="fr-axis" textAnchor="middle">
                {t}h
              </text>
            </g>
          ))}

          {/* Axis titles */}
          <text x={m.l - 8} y={m.t - 14} className="fr-axis-title">
            Win rate vs. 15-min baseline
          </text>
          <text x={m.l + plotW / 2} y={H - 6} className="fr-axis-title" textAnchor="middle">
            Research time (GPU-hours)
          </text>

          {/* PPO baseline reference lines */}
          {REFS.map((r) => {
            const ry = sy(winPct(r.elo));
            const isLeft = r.align === "left";
            return (
              <g key={r.id}>
                <line x1={m.l} x2={W - m.r} y1={ry} y2={ry} className="fr-ref-line" />
                <text
                  x={isLeft ? m.l + 6 : W - m.r - 6}
                  y={isLeft ? ry - 7 : ry + 16}
                  textAnchor={isLeft ? "start" : "end"}
                  className="fr-ref-lbl"
                >
                  {narrow ? r.shortLabel : r.label}
                </text>
              </g>
            );
          })}

          <clipPath id={plotId}>
            <rect x={m.l - 4} y={m.t - 4} width={plotW + 8} height={plotH + 8} />
          </clipPath>
          <g clipPath={`url(#${plotId})`}>
            <g pointerEvents="none" aria-hidden="true">
              {MODELS.map((model) => {
                const colored = isColored(model);
                const color = colored ? curveColor(model.label) : "#989891";
                const opacity = activeKey === model.key ? 0 : activeKey ? 0.08 : colored ? 0.28 : 0.2;
                return (
                  <g key={model.key} opacity={opacity}>
                    {MODEL_POINTS.get(model.key)?.map((p) => (
                      <circle
                        key={p.run}
                        className="fr-run-dot"
                        cx={sx(p.x)}
                        cy={sy(p.y)}
                        r={p.isFrontier ? 2.6 : 2.3}
                        fill={p.isFrontier ? color : "none"}
                        stroke={p.isFrontier ? "none" : color}
                        strokeWidth={1.1}
                        style={{ transitionDelay: `${(Math.min(xMax, Math.max(0, p.x)) / xMax) * 1000}ms` }}
                      />
                    ))}
                  </g>
                );
              })}
            </g>

            {/* Frontier step curves and highlighted experiment runs */}
            {drawOrder.map((model) => {
              const isTop = TOP_KEYS.has(model.key);
              const colored = isColored(model);
              const isFocus = model.key === activeKey;
              const dimmed = activeKey !== null && !isFocus;
              const col = curveColor(model.label);
              const stroke = isFocus || colored ? col : "#c2c2bc";
              const isSol61 = model.key === "sol";
              const strokeWidth = isFocus ? 2.6 : isTop ? 2.0 : colored ? 1.55 : 1.1;
              const baseOpacity = isTop ? 0.95 : isSol61 ? 0.65 : colored ? 0.8 : 0.45;
              const opacity = isFocus ? 1 : dimmed ? (colored ? 0.14 : 0.08) : baseOpacity;
              const points = MODEL_POINTS.get(model.key) ?? [];
              const endX = sx(model.gpu);
              const endY = sy(winPct(model.best.elo));

              return (
                <g key={model.key}>
                  <path
                    d={stepPath(model)}
                    fill="none"
                    stroke={stroke}
                    strokeWidth={strokeWidth}
                    strokeLinejoin="miter"
                    opacity={opacity}
                  />
                  {/* Mark early-stopped colored curves subtly so it's clear where their session ended */}
                  {colored && model.gpu < 4.75 && !isFocus && (
                    <circle
                      cx={endX}
                      cy={endY}
                      r={2.3}
                      fill={col}
                      stroke="#fff"
                      strokeWidth={1}
                      opacity={opacity}
                    />
                  )}
                  {/* When highlighted, show every experiment run: open dots for sub-frontier/failed runs, solid dots on the frontier */}
                  {isFocus &&
                    points.map((p, idx) => {
                      const isHoveredPt = hover?.point === p;
                      if (p.isFrontier) {
                        return (
                          <circle
                            key={idx}
                            cx={sx(p.x)}
                            cy={sy(p.y)}
                            r={isHoveredPt ? 4.4 : 3.2}
                            fill={col}
                            stroke="#fff"
                            strokeWidth={1.3}
                          />
                        );
                      }
                      return (
                        <circle
                          key={idx}
                          cx={sx(p.x)}
                          cy={sy(p.y)}
                          r={isHoveredPt ? 4.0 : 2.8}
                          fill="#fff"
                          stroke={col}
                          strokeWidth={isHoveredPt ? 1.8 : 1.3}
                          opacity={isHoveredPt ? 1 : 0.78}
                        />
                      );
                    })}
                </g>
              );
            })}
          </g>

          {/* Direct right-margin labels for all 12 models above 10% (plus quiet tail note) */}
          {!narrow && (
            <g className="fr-direct-labels">
              {labeledModels.map((model) => {
                const col = curveColor(model.label);
                const textCol = labelTextColor(model.label);
                const isTop = TOP_KEYS.has(model.key);
                const endX = sx(model.gpu);
                const endY = sy(winPct(model.best.elo));
                const ly = labelYMap.get(model.key) ?? endY;
                const isFocus = activeKey === model.key;
                const dimmed = activeKey !== null && !isFocus;
                const rightEdge = W - m.r;
                const labelX = rightEdge + 12;
                const elbowX = rightEdge + 5;
                const stoppedEarly = model.gpu < 4.75;
                return (
                  <g
                    key={model.key}
                    className="fr-direct-item"
                    opacity={dimmed ? 0.2 : 1}
                    onPointerEnter={() => hoverFromLabel(model)}
                    onPointerLeave={() => setHover(null)}
                    onPointerDown={(ev) => {
                      ev.stopPropagation();
                      togglePinLabel(model);
                    }}
                  >
                    {stoppedEarly && (
                      <line
                        x1={endX + 3}
                        x2={rightEdge}
                        y1={endY}
                        y2={endY}
                        stroke={col}
                        strokeWidth={1}
                        strokeDasharray="1.5 3"
                        opacity={isFocus ? 0.8 : 0.4}
                      />
                    )}
                    <path
                      d={`M${(stoppedEarly ? rightEdge : endX + 2).toFixed(1)},${endY.toFixed(1)}L${elbowX.toFixed(1)},${ly.toFixed(1)}L${(labelX - 3).toFixed(1)},${ly.toFixed(1)}`}
                      fill="none"
                      stroke={col}
                      strokeWidth={isFocus ? 1.5 : 1}
                      opacity={isFocus ? 0.95 : 0.65}
                    />
                    <text x={labelX} y={ly} dominantBaseline="middle" className="fr-lbl">
                      <tspan style={{ fill: textCol, fontWeight: isFocus ? 700 : isTop ? 600 : 500 }}>
                        {model.label}
                      </tspan>
                    </text>
                  </g>
                );
              })}

              {/* Quiet right-margin summary for the remaining unlabeled models */}
              <g opacity={activeKey && LABELED_KEYS.has(activeKey) ? 0.2 : 1}>
                <text x={W - m.r + 12} y={sy(2.5) - 6} className="fr-pack-lbl" dominantBaseline="middle">
                  {otherCount} other models
                </text>
                <text x={W - m.r + 12} y={sy(2.5) + 10} className="fr-pack-sub" dominantBaseline="middle">
                  Hover any line
                </text>
              </g>
            </g>
          )}
        </svg>

        {hover && activeModel && (
          <div
            className="viz-tip"
            style={
              tipBelow
                ? { left: tipLeft, top: hover.cy + 14, transform: "none" }
                : { left: tipLeft, top: hover.cy - 12 }
            }
            role="status"
          >
            <strong className="fr-tip-head">
              <i style={{ background: curveColor(activeModel.label) }} />
              {activeModel.label}
            </strong>
            <span>
              {hover.fromDot && hover.point ? (
                <>
                  <b>{fmtWin(hover.point.elo)}</b> win rate at {hourText(hover.point.x)}
                </>
              ) : (
                <>
                  Best: <b>{fmtWin(activeModel.best.elo)}</b> win rate
                </>
              )}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export function EloLeaderboard() {
  const agent = useAgentView();
  const [by, setBy] = useState<"best" | "final">("best");
  const rows = useMemo(() => [...MODELS].sort((a, b) => b[by].elo - a[by].elo), [by]);
  const lo = -1600, hi = 320, pct = (v: number) => ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * 100;
  if (agent) {
    return <ol>{rows.map((x) => <li key={x.key}>{x.label}: best network {fmtWin(x.best.elo)} vs 15-min starter ({signed(x.best.elo)} Elo), fresh 15-min final {fmtWin(x.final.elo)} ({signed(x.final.elo)} Elo)</li>)}</ol>;
  }
  return (
    <div className="fr-board">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Researcher</th>
            <th aria-sort={by === "best" ? "descending" : "none"}>
              <button type="button" onClick={() => setBy("best")}>Best network (win % vs starter) {by === "best" ? "▾" : ""}</button>
            </th>
            <th aria-sort={by === "final" ? "descending" : "none"}>
              <button type="button" onClick={() => setBy("final")}>15-min final {by === "final" ? "▾" : ""}</button>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((x, i) => {
            const st = modelStyle(x.label), v = x[by].elo;
            return (
              <tr key={x.key}>
                <td className="fr-rank">{i + 1}</td>
                <td>
                  <span className="fr-name"><i style={{ background: st.color }} />{x.label}</span>
                  {x.note && <small>{x.note}</small>}
                </td>
                <td className={by === "best" ? "fr-on" : ""} title={`${signed(x.best.elo)} Elo vs 15-min starter`}>
                  <span className="fr-num">{fmtWin(x.best.elo)}</span>
                  {by === "best" && <span className="fr-bar"><b style={{ left: `${pct(Math.min(0, v))}%`, width: `${Math.abs(pct(v) - pct(0))}%`, background: st.color }} /><em style={{ left: `${pct(0)}%` }} /></span>}
                </td>
                <td className={by === "final" ? "fr-on" : ""} title={`${signed(x.final.elo)} Elo vs 15-min starter`}>
                  <span className="fr-num">{fmtWin(x.final.elo)}</span>
                  {by === "final" && <span className="fr-bar"><b style={{ left: `${pct(Math.min(0, v))}%`, width: `${Math.abs(pct(v) - pct(0))}%`, background: st.color }} /><em style={{ left: `${pct(0)}%` }} /></span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export const frontierGames = DATA.games;
