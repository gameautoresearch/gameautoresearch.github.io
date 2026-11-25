import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { CMYK } from "./colors";
import { HexSideIO, type HexMoveEntry } from "./NetworkIO";

// Recorded 13x13 Hex games between trained final checkpoints from hex13-recipe-20261123.
// Player 0 (cyan) connects top <-> bottom; Player 1 (magenta) connects left <-> right.

type Recording = {
  id: string;
  size: number;
  players: [string, string];
  winner: 0 | 1;
  moves: HexMoveEntry[];
};

const MOVES_PER_SECOND = 3.2;
const END_HOLD_MS = 2000;
const SPEEDS = [1, 2, 4];
const SEAT_COLORS: [string, string] = [CMYK.c, CMYK.m];
const DIRS: readonly [number, number][] = [
  [-1, 0],
  [-1, 1],
  [0, -1],
  [0, 1],
  [1, -1],
  [1, 0],
];

function winningChain(board: Uint8Array, size: number, winner: 0 | 1): Set<number> {
  const target = winner + 1;
  const q: number[] = [];
  const vis = new Uint8Array(size * size);
  const onSecond = (r: number, c: number) => (winner === 0 ? r === size - 1 : c === size - 1);
  // Seed from first edge
  for (let i = 0; i < size; i++) {
    const idx = winner === 0 ? i : i * size;
    if (board[idx] === target) {
      vis[idx] = 1;
      q.push(idx);
    }
  }
  const reached = new Set<number>();
  let touchesSecond = false;
  while (q.length) {
    const cur = q.shift()!;
    reached.add(cur);
    const r = Math.floor(cur / size),
      c = cur % size;
    if (onSecond(r, c)) touchesSecond = true;
    for (const [dr, dc] of DIRS) {
      const rr = r + dr,
        cc = c + dc;
      if (rr < 0 || rr >= size || cc < 0 || cc >= size) continue;
      const ni = rr * size + cc;
      if (!vis[ni] && board[ni] === target) {
        vis[ni] = 1;
        q.push(ni);
      }
    }
  }
  return touchesSecond ? reached : new Set();
}

export default function HexReplay({
  src,
  names,
  start = 0,
  initialSpeed = 2,
  renderSide,
}: {
  src: string;
  names?: [string, string];
  start?: number;
  initialSpeed?: number;
  renderSide?: (side: 0 | 1, io?: ReactNode) => ReactNode;
}) {
  const [rec, setRec] = useState<Recording | null>(null);
  const [pos, setPos] = useState(0);
  const [playing, setPlaying] = useState(
    () => typeof window === "undefined" || !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches,
  );
  const [speed, setSpeed] = useState(initialSpeed);
  const hold = useRef(0);

  useEffect(() => {
    let live = true;
    fetch(src)
      .then((r) => r.json())
      .then((d: Recording) => {
        if (!live) return;
        setRec(d);
        setPos(Math.max(2, Math.floor(start * d.moves.length)));
      });
    return () => {
      live = false;
    };
  }, [src, start]);

  // Precompute board state before each ply (boards[0] = empty, boards[t] = after t moves).
  const boards = useMemo(() => {
    if (!rec) return [];
    const n = rec.size * rec.size;
    const out: Uint8Array[] = [new Uint8Array(n)];
    for (let i = 0; i < rec.moves.length; i++) {
      const next = new Uint8Array(out[i]);
      next[rec.moves[i].cell] = rec.moves[i].p + 1;
      out.push(next);
    }
    return out;
  }, [rec]);

  useEffect(() => {
    if (!playing || !rec) return;
    const last = rec.moves.length;
    let before = performance.now(),
      raf = 0;
    const frame = (now: number) => {
      const dt = Math.max(0, Math.min(100, now - before));
      before = now;
      setPos((v) => {
        if (v >= last) {
          hold.current += dt;
          if (hold.current < END_HOLD_MS) return v;
          hold.current = 0;
          return 2;
        }
        return Math.min(last, v + (dt / 1000) * MOVES_PER_SECOND * speed);
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

  const size = rec.size;
  const last = rec.moves.length;
  const ply = Math.max(0, Math.min(last, Math.floor(pos)));
  const board = boards[ply];
  const lastMove = ply > 0 ? rec.moves[ply - 1] : null;
  const label = names ?? rec.players;
  const winner = label[rec.winner];

  // For each player's I/O column, show their latest decision (or current ply's decision) and the board right before it.
  const latestDecision = (p: 0 | 1) => {
    for (let i = Math.min(last - 1, ply - 1); i >= 0; i--) {
      if (rec.moves[i].p === p && !rec.moves[i].opening) {
        return { move: rec.moves[i], boardBefore: boards[i], active: ply - 1 === i };
      }
    }
    // Fallback to first non-opening move for that side
    for (let i = 0; i < last; i++) {
      if (rec.moves[i].p === p && !rec.moves[i].opening) {
        return { move: rec.moves[i], boardBefore: boards[i], active: false };
      }
    }
    return { move: null, boardBefore: board, active: false };
  };

  let c0 = 0,
    c1 = 0;
  for (let i = 0; i < board.length; i++) {
    if (board[i] === 1) c0++;
    else if (board[i] === 2) c1++;
  }

  const winSet = ply >= last ? winningChain(board, size, rec.winner) : null;

  // Pointy-top hex geometry for 13x13 rhombus
  const R = 10.2;
  const dx = Math.sqrt(3) * R;
  const dy = 1.5 * R;
  const padX = 16;
  const padY = 18;
  const cx = (r: number, c: number) => padX + (c + r * 0.5) * dx;
  const cy = (r: number) => padY + r * dy;
  const W = padX * 2 + (size - 1) * 1.5 * dx;
  const H = padY * 2 + (size - 1) * dy;

  const hexPts = (x: number, y: number, rad = R - 0.65) => {
    const pts: string[] = [];
    for (let k = 0; k < 6; k++) {
      const ang = (Math.PI / 180) * (60 * k - 30);
      pts.push(`${(x + rad * Math.cos(ang)).toFixed(1)},${(y + rad * Math.sin(ang)).toFixed(1)}`);
    }
    return pts.join(" ");
  };

  // Edge rails marking Top/Bottom (Player 0) and Left/Right (Player 1)
  const topRail = `M${(cx(0, 0) - dx * 0.52).toFixed(1)},${(cy(0) - R * 1.05).toFixed(1)} L${(cx(0, size - 1) + dx * 0.52).toFixed(1)},${(cy(0) - R * 1.05).toFixed(1)}`;
  const botRail = `M${(cx(size - 1, 0) - dx * 0.52).toFixed(1)},${(cy(size - 1) + R * 1.05).toFixed(1)} L${(cx(size - 1, size - 1) + dx * 0.52).toFixed(1)},${(cy(size - 1) + R * 1.05).toFixed(1)}`;
  const leftRail = `M${(cx(0, 0) - dx * 0.62).toFixed(1)},${(cy(0) - R * 0.55).toFixed(1)} L${(cx(size - 1, 0) - dx * 0.62).toFixed(1)},${(cy(size - 1) + R * 0.55).toFixed(1)}`;
  const rightRail = `M${(cx(0, size - 1) + dx * 0.62).toFixed(1)},${(cy(0) - R * 0.55).toFixed(1)} L${(cx(size - 1, size - 1) + dx * 0.62).toFixed(1)},${(cy(size - 1) + R * 0.55).toFixed(1)}`;

  const boardSvg = (
    <svg viewBox={`0 0 ${W.toFixed(1)} ${H.toFixed(1)}`} className="gr-board hex-board" role="img" aria-label={`Hex 13×13: ${label[0]} vs ${label[1]}, move ${ply}`}>
      <path d={topRail} stroke={SEAT_COLORS[0]} strokeWidth={4} strokeLinecap="round" fill="none" />
      <path d={botRail} stroke={SEAT_COLORS[0]} strokeWidth={4} strokeLinecap="round" fill="none" />
      <path d={leftRail} stroke={SEAT_COLORS[1]} strokeWidth={4} strokeLinecap="round" fill="none" />
      <path d={rightRail} stroke={SEAT_COLORS[1]} strokeWidth={4} strokeLinecap="round" fill="none" />
      {Array.from({ length: size * size }, (_, idx) => {
        const r = Math.floor(idx / size),
          c = idx % size;
        const x = cx(r, c),
          y = cy(r);
        const v = board[idx];
        const fill = v === 1 ? SEAT_COLORS[0] : v === 2 ? SEAT_COLORS[1] : "#f0f0ec";
        const isLast = lastMove && lastMove.cell === idx;
        const isWin = winSet?.has(idx);
        return (
          <g key={idx}>
            <polygon
              points={hexPts(x, y)}
              fill={fill}
              stroke={v ? "#fff" : "#d8d8d2"}
              strokeWidth={v ? 1.1 : 0.85}
            />
            {isWin && <circle cx={x} cy={y} r={3.2} fill="#fff" opacity={0.92} />}
            {isLast && !isWin && (
              <circle cx={x} cy={y} r={4.2} fill="none" stroke="#fff" strokeWidth={2} />
            )}
          </g>
        );
      })}
    </svg>
  );

  const sideIO = (p: 0 | 1) => {
    const dec = latestDecision(p);
    return (
      <HexSideIO
        side={p}
        size={size}
        boardBefore={dec.boardBefore}
        move={dec.move}
        isActing={dec.active}
        accents={SEAT_COLORS}
      />
    );
  };

  return (
    <div className={`gr${renderSide ? " gr-flanked" : ""}`}>
      {renderSide ? (
        <div className="sc-row">
          {renderSide(0, sideIO(0))}
          <div className="sc-center">
            <div className="sc-live-hud">
              <div className="sc-live-side">
                <i style={{ background: SEAT_COLORS[0] }} aria-hidden="true" />
                <div>
                  <strong>{c0} stones</strong>
                  <span>top ↕ bottom</span>
                </div>
              </div>
              <div className="sc-live-side">
                <i style={{ background: SEAT_COLORS[1] }} aria-hidden="true" />
                <div>
                  <strong>{c1} stones</strong>
                  <span>left ↔ right</span>
                </div>
              </div>
            </div>
            {boardSvg}
          </div>
          {renderSide(1, sideIO(1))}
        </div>
      ) : (
        boardSvg
      )}
      <div className="gr-controls">
        <button
          type="button"
          onClick={() => {
            if (pos >= last) setPos(2);
            setPlaying(!playing);
          }}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          min={0}
          max={last}
          step="any"
          value={pos}
          aria-label="Move"
          onChange={(e) => {
            setPlaying(false);
            setPos(+e.target.value);
          }}
        />
        <span className="gr-tick">{ply >= last ? `${winner} wins` : `Move ${ply} / ${last}`}</span>
        <span className="gr-speed" role="group" aria-label="Playback speed">
          {SPEEDS.map((s) => (
            <button key={s} type="button" aria-pressed={speed === s} onClick={() => setSpeed(s)}>
              {s}×
            </button>
          ))}
        </span>
      </div>
    </div>
  );
}
