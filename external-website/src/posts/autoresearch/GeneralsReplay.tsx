import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { CMYK } from "./colors";
import { GeneralsSideIO } from "./NetworkIO";

// Every joint action in the public demos was replayed independently to the exact final state. Frame 0 lists
// every cell as [index, kind, owner, army]; later frames list changes. Kinds: 0 land, 1 mountain, 2 city,
// 3 general. Owner -1 is neutral. The whole board is shown (no fog).

type Frame = { tick: number; changes: number[][]; actions: [number, number] | null; scores: { army: number; land: number }[] };
type Recording = {
  id: string; width: number; height: number; actionWidth?: number; players: [string, string]; frames: Frame[];
  outcome: { winner: 0 | 1 | null; ticks: number; terminal: string };
};

const TICKS_PER_SECOND = 6;
const END_HOLD_MS = 1600;
const SPEEDS = [1, 2, 4];
const SEAT_COLORS = [CMYK.c, CMYK.m];
const DIR_ARROWS = ["↑", "↓", "←", "→"] as const;
const DIR_DR = [-1, 1, 0, 0] as const;
const DIR_DC = [0, 0, -1, 1] as const;

type DecodedOrder = {
  player: 0 | 1;
  from: number;
  to: number;
  fromRow: number;
  fromCol: number;
  toRow: number;
  toCol: number;
  arrow: string;
  split: boolean;
};

function decodeOrder(id: number, width: number, height: number, actionWidth: number, player: 0 | 1): DecodedOrder | null {
  if (!id) return null;
  const a = id - 1, cell = Math.floor(a / 8), row = Math.floor(cell / actionWidth), col = cell % actionWidth;
  const dir = Math.floor(a / 2) % 4, split = a % 2 === 1;
  const toRow = row + DIR_DR[dir], toCol = col + DIR_DC[dir];
  if (row < 0 || row >= height || col < 0 || col >= width || toRow < 0 || toRow >= height || toCol < 0 || toCol >= width) {
    return null;
  }
  return {
    player,
    from: row * width + col,
    to: toRow * width + toCol,
    fromRow: row,
    fromCol: col,
    toRow,
    toCol,
    arrow: DIR_ARROWS[dir],
    split,
  };
}

export type GeneralsCustomContext = {
  board: ReactNode;
  width: number;
  height: number;
  cells: Int32Array;
  tick: number;
  maxTicks: number;
  actions: [number, number] | null;
  actionWidth: number;
  scores: { army: number; land: number }[];
  winner: string | null;
  moves: DecodedOrder[];
};

export default function GeneralsReplay({ src, names, start = 0, initialSpeed = 1, controlledPos, onReady, renderSide, renderCustom }: {
  src: string; names?: [string, string]; start?: number; initialSpeed?: number;
  controlledPos?: number;
  onReady?: (frameCount: number) => void;
  renderSide?: (side: 0 | 1, io?: ReactNode) => ReactNode;
  renderCustom?: (ctx: GeneralsCustomContext) => ReactNode;
}) {
  const [rec, setRec] = useState<Recording | null>(null);
  const [pos, setPos] = useState(controlledPos ?? 0);
  const [playing, setPlaying] = useState(
    () =>
      controlledPos === undefined &&
      (typeof window === "undefined" || !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches),
  );
  const [speed, setSpeed] = useState(initialSpeed);
  const hold = useRef(0);

  useEffect(() => {
    let live = true;
    fetch(src).then((r) => r.json()).then((d: Recording) => {
      if (!live) return;
      setRec(d);
      if (controlledPos === undefined) setPos(Math.floor(start * (d.frames.length - 1)));
      onReady?.(d.frames.length);
    });
    return () => { live = false; };
  }, [src]);

  // Board state after every frame, built once.
  const boards = useMemo(() => {
    if (!rec) return [];
    const n = rec.width * rec.height, out: Int32Array[] = [];
    rec.frames.forEach((f, t) => {
      const cells = t ? new Int32Array(out[t - 1]) : new Int32Array(n * 3);
      for (const [i, k, o, a] of f.changes) cells.set([k, o, a], i * 3);
      out.push(cells);
    });
    return out;
  }, [rec]);

  useEffect(() => {
    if (!playing || !rec) return;
    const last = rec.frames.length - 1;
    let before = performance.now(), raf = 0;
    const frame = (now: number) => {
      const dt = Math.max(0, Math.min(100, now - before));
      before = now;
      setPos((v) => {
        if (v >= last) {
          hold.current += dt;
          if (hold.current < END_HOLD_MS) return v;
          hold.current = 0;
          return 0;
        }
        return Math.min(last, v + (dt / 1000) * TICKS_PER_SECOND * speed);
      });
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [playing, rec, speed]);

  if (!rec || !boards.length) {
    if (renderSide) {
      return (
        <div className="gr gr-flanked">
          <div className="sc-row">
            {renderSide(0)}
            <div className="sc-center sc-wait">Loading game…</div>
            {renderSide(1)}
          </div>
          <div className="gr-controls sc-controls-stub" aria-hidden="true" />
        </div>
      );
    }
    return <div className="gr gr-loading">Loading game…</div>;
  }
  const { width: w, height: h } = rec, last = rec.frames.length - 1;
  const activePos = controlledPos ?? pos;
  const t = Math.max(0, Math.min(last, Math.floor(activePos))), cells = boards[t], f = rec.frames[t];
  const label = names ?? rec.players;
  const C = 30, W = w * C, H = h * C;
  const ordersByPlayer: [DecodedOrder | null, DecodedOrder | null] = [
    decodeOrder(f.actions?.[0] ?? 0, w, h, rec.actionWidth ?? 24, 0),
    decodeOrder(f.actions?.[1] ?? 0, w, h, rec.actionWidth ?? 24, 1),
  ];
  const moves = ordersByPlayer.filter(Boolean) as DecodedOrder[];

  const tiles = [];
  for (let i = 0; i < w * h; i++) {
    const k = cells[i * 3], o = cells[i * 3 + 1], a = cells[i * 3 + 2];
    const x = (i % w) * C, y = Math.floor(i / w) * C;
    const color = o >= 0 ? SEAT_COLORS[o] : null;
    const fill = k === 1 ? "#5f5f5f" : color ? color : k === 2 ? "#cfcfcf" : "#f1f1ee";
    tiles.push(
      <g key={i}>
        <rect x={x + 0.5} y={y + 0.5} width={C - 1} height={C - 1} rx={k === 3 ? 7 : 2} fill={fill}
          fillOpacity={color && k === 0 ? 0.78 : 1} />
        {k === 1 && <path d={`M${x + 7},${y + C - 8}L${x + C / 2},${y + 8}L${x + C - 7},${y + C - 8}Z`} fill="#8a8a8a" />}
        {k === 2 && <rect x={x + 5} y={y + 5} width={C - 10} height={C - 10} rx={2} fill="none" stroke={color ? "#fff" : "#9a9a9a"} strokeWidth={1.5} />}
        {k === 3 && <path d={`M${x + 8},${y + 10}l3.5,4l3.5,-5l3.5,5l3.5,-4v9h-14z`} fill="#fff" opacity={0.95} />}
        {a > 0 && k !== 1 && (
          <text x={x + C / 2} y={y + (k === 3 ? C - 4 : C / 2 + 4)} textAnchor="middle"
            fontSize={a >= 1000 ? 8.5 : a >= 100 ? 10 : 11.5} fontWeight={600} fill={color ? "#fff" : "#333"}>{a}</text>
        )}
      </g>,
    );
  }
  const winner = rec.outcome.winner === null ? "Nobody" : label[rec.outcome.winner];

  const sideMeta = (p: 0 | 1) => `${f.scores[p].army.toLocaleString()} army · ${f.scores[p].land} land`;
  const boardSvg = (
    <svg viewBox={`0 0 ${W} ${H}`} className="gr-board" role="img" aria-label={`Generals: ${label[0]} vs ${label[1]}, tick ${t}`}>
      {tiles}
      {moves.map((m, j) => {
        const fx = m.fromCol * C + C / 2, fy = m.fromRow * C + C / 2;
        const tx = m.toCol * C + C / 2, ty = m.toRow * C + C / 2;
        const dx = tx - fx, dy = ty - fy, len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len;
        const sx = fx + ux * 8, sy = fy + uy * 8;
        const ex = tx - ux * 6, ey = ty - uy * 6;
        const px = -uy, py = ux;
        const head = `${ex},${ey} ${ex - ux * 6.5 + px * 4},${ey - uy * 6.5 + py * 4} ${ex - ux * 6.5 - px * 4},${ey - uy * 6.5 - py * 4}`;
        return (
          <g key={j} pointerEvents="none">
            <rect x={m.toCol * C + 1} y={m.toRow * C + 1} width={C - 2} height={C - 2} rx={3}
              fill="none" stroke="#111" strokeWidth={2} />
            <line x1={sx} y1={sy} x2={ex - ux * 4} y2={ey - uy * 4} stroke="#fff" strokeWidth={4.2} strokeLinecap="round" opacity={0.9} />
            <line x1={sx} y1={sy} x2={ex - ux * 4} y2={ey - uy * 4} stroke="#111" strokeWidth={2} strokeLinecap="round"
              strokeDasharray={m.split ? "2.5 2.5" : undefined} />
            <polygon points={head} fill="#111" stroke="#fff" strokeWidth={1.1} strokeLinejoin="round" />
          </g>
        );
      })}
    </svg>
  );

  const sideIO = (p: 0 | 1) => (
    <GeneralsSideIO
      side={p}
      width={w}
      height={h}
      cells={cells}
      tick={t}
      maxTicks={last}
      actions={f.actions}
      actionWidth={rec.actionWidth ?? 24}
      scores={f.scores}
      accents={[SEAT_COLORS[0], SEAT_COLORS[1]]}
    />
  );

  if (renderCustom) {
    return (
      <div className="replay-custom">
        {renderCustom({
          board: boardSvg,
          width: w,
          height: h,
          cells,
          tick: t,
          maxTicks: last,
          actions: f.actions,
          actionWidth: rec.actionWidth ?? 24,
          scores: f.scores,
          winner: t >= last ? winner : null,
          moves,
        })}
      </div>
    );
  }

  return (
    <div className={`gr${renderSide ? " gr-flanked" : ""}`}>
      {renderSide ? (
        <div className="sc-row">
          {renderSide(0, sideIO(0))}
          <div className="sc-center">
            <div className="sc-live-hud">
              {([0, 1] as const).map((p) => (
                <div key={p} className="sc-live-side">
                  <i style={{ background: SEAT_COLORS[p] }} aria-hidden="true" />
                  <div>
                    <strong>{f.scores[p].army.toLocaleString()} army</strong>
                    <span>{f.scores[p].land} land</span>
                  </div>
                </div>
              ))}
            </div>
            {boardSvg}
          </div>
          {renderSide(1, sideIO(1))}
        </div>
      ) : (
        <>
          <div className="gr-hud">
            {[0, 1].map((p) => (
              <div key={p} className="gr-side">
                <i style={{ background: SEAT_COLORS[p] }} />
                <div>
                  <strong>{label[p]}</strong>
                  <span>{sideMeta(p as 0 | 1)}</span>
                </div>
              </div>
            ))}
          </div>
          {boardSvg}
        </>
      )}
      <div className="gr-controls">
        <button type="button" onClick={() => { if (pos >= last) setPos(0); setPlaying(!playing); }}>{playing ? "Pause" : "Play"}</button>
        <input type="range" min={0} max={last} step="any" value={pos} aria-label="Tick"
          onChange={(e) => { setPlaying(false); setPos(+e.target.value); }} />
        <span className="gr-tick">{t >= last ? `${winner} wins` : `Tick ${t} / ${last}`}</span>
        <span className="gr-speed" role="group" aria-label="Playback speed">
          {SPEEDS.map((s) => <button key={s} type="button" aria-pressed={speed === s} onClick={() => setSpeed(s)}>{s}×</button>)}
        </span>
      </div>
    </div>
  );
}
