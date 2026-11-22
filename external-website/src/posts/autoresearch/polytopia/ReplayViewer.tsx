import { Component, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  art, DIAMOND, drawBorder, drawCity, drawForest, drawGround, drawHealth, drawMountain, drawResource, drawRing, drawRoad,
  drawStructure, drawUnit, HH, HW, STEPS, TEAMS, TH, TW, type Team,
} from "./mockArt";

// A replay of one recorded game, played back one action at a time with the Studio's mock board art. Ported from
// thekevinz.com's Polytopia post. The data (tools/convert_polytopia_replays.py) is the god-view state before the
// first action plus, for every action, what it changed.

type Tile = [terrain: number, owner: number, structure: number, resource: number, road: number, capital: number];
type Unit = [owner: number, kind: number, hp: number, veteran: number];
type City = [owner: number, level: number];
type Action = [
  player: number, turn: number, kind: number, source: number, target: number, label: number,
  units: ([number] | [number, ...Unit])[], tiles: [number, ...Tile][], cities: [number, ...City][],
  score: [number, number], value: number | null,
];
export type Moment = { turn: number; player: number; text: string };
type Replay = {
  size: number;
  moments?: Moment[];
  names: Record<string, string>;
  units: Record<string, string>;
  labels: string[];
  init: { tiles: Tile[]; units: [number, ...Unit][]; cities: [number, ...City][] };
  actions: Action[];
};
type State = { tiles: Tile[]; units: Map<number, Unit>; cities: Map<number, City> };
type Piece = { key: string; x: number; y: number; depth: number; html: string; opacity?: number; scale?: number };

const TEAM: Team[] = [TEAMS.neutral, TEAMS.blue, TEAMS.orange];
const PLAYER = TEAM.map((t) => "#" + t.main.toString(16).padStart(6, "0"));
const SIDE = ["", "Blue", "Orange"];
const TERRAIN: Record<number, string> = { 1: "Shallow water", 2: "Ocean", 3: "Field", 4: "Mountain", 5: "Forest" };
// Neighbours across the four tile edges NE, SE, SW, NW, as [column step, row step].
const EDGE_STEPS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const PAD_TOP = 52;
// Units that shoot from a distance instead of charging in: catapult, archer, pirate, bomber, juggernaut.
const RANGED = new Set([8, 9, 40, 41, 45]);

// Milliseconds per action at 1×, by command kind: 1 move, 2 attack, 3 unit or tile ability, 4 train,
// 5 harvest, 6 build, 7 research, 8 capture, 9 city reward, 10 end turn.
const DURATION: Record<number, number> = { 1: 110, 2: 260, 3: 150, 4: 150, 5: 160, 6: 160, 7: 40, 8: 320, 9: 160, 10: 200 };
const duration = (kind: number) => DURATION[kind] ?? 150;
const SPEEDS = [1, 2, 4];
const END_HOLD_MS = 2600;


const ease = (p: number) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);
// Progress through the window [from, to] of an action's animation.
const phase = (p: number, from: number, to: number) => Math.max(0, Math.min(1, (p - from) / (to - from)));

const unitArt = (u: Unit) =>
  art(`ring:${u[0]}`, (g) => drawRing(g, TEAM[u[0]])) +
  art(`unit:${u[1]}:${u[0]}`, (g) => drawUnit(g, u[1], TEAM[u[0]])) +
  art(`hp:${u[0]}`, (g) => drawHealth(g, TEAM[u[0]])) +
  `<text x="-15" y="-30.5" text-anchor="middle" font-size="10" font-weight="700" fill="#fff">${Math.round(u[2])}</text>`;
const cityArt = (c: City, capital: boolean) => {
  const level = c[0] ? Math.min(c[1], 5) : 0;
  return art(`city:${level}:${c[0]}:${capital}`, (g) => drawCity(g, level, TEAM[c[0]], capital));
};

// Every state of the game, before each action and after the last. Unchanged parts are shared between states.
function buildStates(data: Replay): State[] {
  let state: State = {
    tiles: data.init.tiles,
    units: new Map(data.init.units.map(([t, ...u]) => [t, u as Unit])),
    cities: new Map(data.init.cities.map(([t, ...c]) => [t, c as City])),
  };
  const states = [state];
  for (const a of data.actions) {
    const units = new Map(state.units);
    for (const [t, ...u] of a[6]) {
      if (u.length) units.set(t, u as Unit);
      else units.delete(t);
    }
    let tiles = state.tiles;
    if (a[7].length) {
      tiles = tiles.slice();
      for (const [t, ...tile] of a[7]) tiles[t] = tile as Tile;
    }
    let cities = state.cities;
    if (a[8].length) {
      cities = new Map(cities);
      for (const [t, ...c] of a[8]) cities.set(t, c as City);
    }
    state = { tiles, units, cities };
    states.push(state);
  }
  return states;
}

// A failure inside the replay should cost the figure, never the whole post.
class ReplayBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? <div className="replay replay-loading">The replay couldn't load. Reload the page to try again.</div> : this.props.children;
  }
}

// `still` freezes the board at that action index, paused (tools/og renders the link-preview image this way).
// `start` begins playback that far into the game (0–1), `initialSpeed` sets the starting speed, and `compact` keeps
// only the scoreboard, the board and the play controls (the post's opening showcase uses all three).
type ViewerProps = { src: string; names?: [string, string]; still?: number; start?: number; initialSpeed?: number; compact?: boolean };
export default function ReplayViewer({ src, ...rest }: ViewerProps) {
  return (
    <ReplayBoundary>
      <ReplayBoard key={src} url={src} {...rest} />
    </ReplayBoundary>
  );
}

function ReplayBoard({ url, names, still, start, initialSpeed, compact }: Omit<ViewerProps, "src"> & { url: string }) {
  const [data, setData] = useState<Replay | null>(null);
  const [pos, setPos] = useState(still ?? 0); // action index; the fraction is how far that action's animation has run
  // Plays on its own and loops, unless the reader prefers reduced motion. Play/Pause, the slider, the speed
  // and the moment buttons still take over.
  const [playing, setPlaying] = useState(
    () =>
      still === undefined &&
      (typeof window === "undefined" || !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches),
  );
  const [speed, setSpeed] = useState(initialSpeed ?? 4);
  const [visible, setVisible] = useState(true);
  const [hover, setHover] = useState<number | null>(null);
  const hold = useRef(0);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    fetch(url).then((r) => r.json()).then((d) => {
      if (!live) return;
      setData(d as Replay);
      if (start !== undefined && still === undefined) setPos(Math.floor(start * (d as Replay).actions.length));
    });
    return () => { live = false; };
  }, [url]);

  // Only advance while the board is on screen.
  useEffect(() => {
    const el = root.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting));
    io.observe(el);
    return () => io.disconnect();
  }, [data]);

  useEffect(() => {
    if (!playing || !visible || !data) return;
    const last = data.actions.length;
    let before = performance.now(), raf = 0;
    const frame = (now: number) => {
      // A frame's timestamp can be slightly earlier than the performance.now() taken when playback started.
      const dt = Math.max(0, Math.min(100, now - before));
      before = Math.max(before, now);
      setPos((v) => {
        v = Math.max(0, v);
        if (v >= last) {
          // Rest on the final position, then start over.
          hold.current += dt;
          if (hold.current < END_HOLD_MS) return v;
          hold.current = 0;
          return 0;
        }
        // Spend the frame's time across as many actions as it covers.
        let budget = dt * speed;
        while (v < last) {
          const k = Math.floor(v), ms = duration(data.actions[k][2]), left = (k + 1 - v) * ms;
          if (budget < left) return v + budget / ms;
          budget -= left;
          v = k + 1;
        }
        return last;
      });
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [playing, visible, data, speed]);

  const geom = useMemo(() => {
    if (!data) return null;
    const n = data.size, last = data.actions.length;
    const W = n * TW, H = n * TH + PAD_TOP + 12;
    const at = (t: number) => {
      const x = t % n, y = Math.floor(t / n);
      return { x: W / 2 + (x - y) * HW, y: PAD_TOP + HH + (x + y) * HH };
    };
    const order = Array.from({ length: n * n }, (_, t) => t).sort(
      (a, b) => (a % n) + Math.floor(a / n) - (b % n) - Math.floor(b / n) || a - b,
    );
    const diamond = (t: number) => {
      const { x, y } = at(t);
      return DIAMOND.map((v, k) => (k % 2 ? y + v : x + v)).join(",");
    };
    // Water stays water, so the ground renders once.
    const ground = order
      .map((t) => {
        const water = data.init.tiles[t][0] <= 2;
        const { x, y } = at(t);
        return `<g transform="translate(${x} ${y})">${art("ground:" + water, (g) => drawGround(g, water))}</g>`;
      })
      .join("");
    const MOMENTS = data.moments ?? [];
    const turnStart = MOMENTS.map((m) => data.actions.findIndex((a) => a[1] === m.turn && a[0] === m.player));
    // Scores and orange's own win estimate after each action, for the strip under the board.
    const SW = 600, SH = 92;
    const maxScore = Math.max(...data.actions.map((a) => Math.max(...a[9])));
    const sx = (i: number) => 8 + (i / last) * (SW - 16);
    const sy = (v: number) => SH - 8 - (v / maxScore) * (SH - 16);
    const sampled = data.actions.map((_, i) => i).filter((i) => i % 6 === 0 || i === last - 1);
    const path = (y: (a: Action) => number) =>
      sampled.map((i, j) => `${j ? "L" : "M"}${sx(i + 1).toFixed(1)},${y(data.actions[i]).toFixed(1)}`).join("");
    const strip = {
      SW, SH, sx,
      blue: path((a) => sy(a[9][0])),
      orange: path((a) => sy(a[9][1])),
      win: path((a) => SH - 8 - (((a[10] ?? 0) + 1) / 2) * (SH - 16)),
    };
    return { n, W, H, at, order, diamond, ground, states: buildStates(data), turnStart, strip };
  }, [data]);

  if (!data || !geom) return <div className="replay replay-loading">Loading replay…</div>;

  const { n, W, H, at, order, diamond, ground, states, turnStart, strip } = geom;
  const last = data.actions.length;
  const k = Math.max(0, Math.min(last, Math.floor(pos) || 0));
  const p = k >= last ? 0 : pos - k;
  const action = k < last ? data.actions[k] : null;
  const before = states[k], after = states[Math.min(last, k + 1)];
  const shownIndex = p < 0.5 ? k : Math.min(last, k + 1);
  const shown = states[shownIndex]; // what the HUD and hover describe

  // ---------- tiles: territory and roads underneath, then terrain, resources, buildings, cities ----------
  const surfaceOf = (s: State, t: number) => {
    const tile = s.tiles[t], owner = tile[1], x = t % n, y = Math.floor(t / n);
    let html = "";
    if (owner) {
      let edges = 0;
      EDGE_STEPS.forEach(([dx, dy], j) => {
        const nx = x + dx, ny = y + dy;
        const other = nx < 0 || ny < 0 || nx >= n || ny >= n ? 0 : s.tiles[ny * n + nx][1];
        if (other !== owner) edges |= 1 << j;
      });
      html +=
        art(`tint:${owner}`, (g) => g.poly(DIAMOND).fill({ color: TEAM[owner].main, alpha: 0.22 })) +
        art(`border:${owner}:${edges}`, (g) => drawBorder(g, edges, TEAM[owner].main));
    }
    if (tile[4]) {
      let roads = 0;
      STEPS.forEach(([dc, dr], j) => {
        const nx = x + dc, ny = y + dr;
        if (nx < 0 || ny < 0 || nx >= n || ny >= n) return;
        if (s.tiles[ny * n + nx][4] || s.cities.has(ny * n + nx)) roads |= 1 << j;
      });
      html += art("road:" + roads, (g) => drawRoad(g, roads));
    }
    return html;
  };
  const tileArt = (s: State, t: number) => {
    const tile = s.tiles[t], city = s.cities.get(t);
    let html = "";
    if (tile[0] === 4) html += art("mountain", drawMountain);
    if (tile[0] === 5 && !city) html += art("forest", drawForest);
    if (tile[3]) html += art("resource:" + tile[3], (g) => drawResource(g, tile[3]));
    if (tile[2] > 1 && tile[2] !== 3 && !city) html += art("structure:" + tile[2], (g) => drawStructure(g, tile[2]));
    if (city) html += cityArt(city, tile[5] !== 0);
    return html;
  };

  const surface: Piece[] = [];
  const pieces: Piece[] = [];
  const effects: Piece[] = [];
  // Draw a tile layer from `from` to `to`: once if unchanged, else cross-faded over `q` with a small pop.
  const blend = (list: Piece[], key: string, t: number, depth: number, from: string, to: string, q: number) => {
    const { x, y } = at(t);
    if (from === to) {
      if (from) list.push({ key, x, y, depth, html: from });
      return;
    }
    if (from && q < 1) list.push({ key: key + "a", x, y, depth, html: from, opacity: 1 - q });
    if (to && q > 0) list.push({ key: key + "b", x, y, depth, html: to, opacity: q, scale: 0.85 + 0.15 * Math.sin((Math.PI / 2) * q) });
  };
  // Tiles change as a move arrives or an attack resolves; for everything else, across the action.
  const kind = action?.[2] ?? 0;
  const tileQ = kind === 1 ? phase(p, 0.6, 1) : kind === 2 ? phase(p, 0.5, 0.9) : phase(p, 0.15, 0.85);
  for (const t of order) {
    const { y } = at(t);
    blend(surface, "s" + t, t, y, surfaceOf(before, t), surfaceOf(after, t), tileQ);
    blend(pieces, "g" + t, t, y, tileArt(before, t), tileArt(after, t), tileQ);
  }

  // ---------- units ----------
  const src = action?.[3] ?? -1, dst = action?.[4] ?? -1;
  const handled = new Set<number>();
  const push = (key: string, t: number, u: Unit, dx = 0, dy = 0, opacity?: number, scale?: number) => {
    const { x, y } = at(t);
    pieces.push({ key, x: x + dx, y: y + dy, depth: y + Math.max(0, dy) + 0.2, html: unitArt(u), opacity, scale });
  };
  const toward = (from: number, to: number, f: number) => {
    const a = at(from), b = at(to);
    return [(b.x - a.x) * f, (b.y - a.y) * f] as const;
  };
  if (kind === 1 && before.units.has(src)) {
    // Move: slide from source to target with a hop.
    const u = before.units.get(src)!, e = ease(p);
    const [dx, dy] = toward(src, dst, e);
    push("u" + src, src, u, dx, dy - Math.sin(Math.PI * e) * 12);
    handled.add(src).add(dst);
  } else if (kind === 2 && before.units.has(src) && before.units.has(dst)) {
    // Attack: a charge or a shot, a hit flash and the damage, then any death and the winner moving in.
    const atk = before.units.get(src)!, def = before.units.get(dst)!;
    const atkAfter = after.units.get(src), defAfter = after.units.get(dst);
    const dist = Math.max(Math.abs((src % n) - (dst % n)), Math.abs(Math.floor(src / n) - Math.floor(dst / n)));
    const ranged = RANGED.has(atk[1]) || dist > 1;
    const advances = !atkAfter && defAfter?.[0] === atk[0]; // the attacker ends on the target tile
    const defDies = !defAfter || defAfter[0] !== def[0];
    const lunge = ranged ? 0 : 0.35 * Math.sin(Math.PI * phase(p, 0, 0.7));
    if (advances) {
      const [dx, dy] = toward(src, dst, p < 0.6 ? lunge : 0.35 + 0.65 * ease(phase(p, 0.6, 1)));
      push("u" + src, src, p < 0.6 ? atk : defAfter!, dx, dy);
    } else {
      const [dx, dy] = toward(src, dst, lunge);
      push("u" + src, src, p < 0.6 || !atkAfter ? atk : atkAfter, dx, dy, atkAfter ? undefined : 1 - phase(p, 0.6, 1));
    }
    const hit = phase(p, 0.35, 0.6);
    if (!advances || p < 0.6) {
      const shake = hit > 0 && hit < 1 ? Math.sin(hit * Math.PI * 6) * 3 : 0;
      push("d" + dst, dst, p < 0.6 || defDies ? def : defAfter!, shake, 0, defDies ? 1 - phase(p, 0.55, 0.9) : undefined);
    }
    const { x, y } = at(dst);
    if (ranged && p < 0.45) {
      // The shot: a stone or arrow on an arc.
      const f = phase(p, 0, 0.45), [dx, dy] = toward(src, dst, f), s0 = at(src);
      effects.push({ key: "shot", x: s0.x + dx, y: s0.y + dy - 26 - Math.sin(Math.PI * f) * 30, depth: 0, html: '<circle r="3.2" fill="#3b3b3b"/>' });
    }
    if (hit > 0 && hit < 1)
      effects.push({ key: "flash", x, y: y - 14, depth: 0, opacity: 0.6 * Math.sin(Math.PI * hit), html: '<ellipse rx="22" ry="20" fill="#ff3b30"/>' });
    if (p >= 0.4) {
      const dmg = def[2] - (defDies ? 0 : defAfter![2]), f = phase(p, 0.4, 1);
      effects.push({
        key: "dmg", x, y: y - 46 - 22 * f, depth: 0, opacity: 1 - f * f,
        html: `<text text-anchor="middle" font-size="15" font-weight="800" fill="#e8402f" stroke="#fff" stroke-width="3" paint-order="stroke">-${Math.round(dmg)}</text>`,
      });
    }
    handled.add(src).add(dst);
  }
  for (const t of new Set([...before.units.keys(), ...after.units.keys()])) {
    if (handled.has(t)) continue;
    const u = before.units.get(t), v = after.units.get(t);
    if (u && v && u.join() === v.join()) push("u" + t, t, u);
    else if (u && v) push("u" + t, t, p < 0.5 ? u : v, 0, 0, undefined, p > 0.3 && p < 0.7 ? 1.08 : undefined);
    else if (u) push("u" + t, t, u, 0, 0, 1 - phase(p, 0.3, 0.9));
    else if (v) {
      // A new unit pops up out of its city.
      const q = phase(p, 0.1, 0.8);
      push("u" + t, t, v, 0, 0, Math.min(1, q * 2), 0.4 + 0.6 * q + Math.sin(Math.PI * q) * 0.15);
    }
  }
  if (action && [3, 5, 6, 8, 9].includes(kind)) {
    // A ring that spreads from the tile the action happened on; team-coloured for a capture.
    const t = dst >= 0 ? dst : src;
    if (t >= 0) {
      const { x, y } = at(t), f = phase(p, 0.05, 0.9);
      effects.push({
        key: "ring", x, y, depth: 0, opacity: 1 - f, scale: 0.6 + f * 0.9,
        html: `<ellipse rx="${HW}" ry="${HH}" fill="none" stroke="${kind === 8 ? PLAYER[action[0]] : "#fff"}" stroke-width="4"/>`,
      });
    }
  }
  pieces.sort((a, b) => a.depth - b.depth);
  const draw = (list: Piece[]) =>
    list.map((q) => (
      <g key={q.key} opacity={q.opacity}
        transform={`translate(${q.x.toFixed(1)} ${q.y.toFixed(1)})${q.scale ? ` scale(${q.scale.toFixed(3)})` : ""}`}
        dangerouslySetInnerHTML={{ __html: q.html }} />
    ));

  // ---------- HUD, caption, controls ----------
  const score = data.actions[Math.max(0, shownIndex - 1)][9];
  const count = (values: Iterable<Unit | City>) => {
    const c = [0, 0, 0];
    for (const v of values) c[v[0]]++;
    return c;
  };
  const cityCount = count(shown.cities.values()), unitCount = count(shown.units.values());
  const now = data.actions[Math.min(last - 1, k)];
  const MOMENTS = data.moments ?? [];
  const hasValue = data.actions.some((a) => a[10] !== null);
  const moment = [...MOMENTS].reverse().find((m) => now[1] >= m.turn && now[1] - m.turn < 6);
  const hoverInfo = hover !== null && (() => {
    const tile = shown.tiles[hover], c = shown.cities.get(hover), u = shown.units.get(hover);
    return [
      TERRAIN[tile[0]],
      tile[1] ? `${SIDE[tile[1]]} territory` : null,
      c ? (c[0] ? `${SIDE[c[0]]} city, level ${c[1]}` : "Village") : null,
      u ? `${SIDE[u[0]]} ${data.units[u[1]] ?? "unit"}, ${Math.round(u[2])} HP` : null,
    ].filter(Boolean).join(" · ");
  })();

  return (
    <div className={`replay${compact ? " replay-compact" : ""}`} ref={root}>
      <div className="replay-hud">
        {[1, 2].map((side) => (
          <div key={side} className="replay-side">
            <i style={{ background: PLAYER[side] }} />
            <div>
              <strong>{names?.[side - 1] ?? data.names[side]}</strong>
              <span>{score[side - 1].toLocaleString()} pts · {cityCount[side]} cities · {unitCount[side]} units</span>
            </div>
          </div>
        ))}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="replay-board" onMouseLeave={() => setHover(null)}>
        <g dangerouslySetInnerHTML={{ __html: ground }} />
        <g>{draw(surface)}</g>
        <g>{draw(pieces)}</g>
        <g pointerEvents="none">{draw(effects)}</g>
        {hover !== null && <polygon points={diamond(hover)} fill="#fff" fillOpacity={0.22} stroke="#fff" strokeWidth={2.5} pointerEvents="none" />}
        <g fill="transparent">
          {order.map((t) => <polygon key={t} points={diamond(t)} onMouseEnter={() => setHover(t)} />)}
        </g>
      </svg>

      {!compact && <p className="replay-caption" aria-live="polite">
        {hoverInfo || moment?.text || (MOMENTS.length ? "Press play, drag the slider, or jump to a moment below." : "Press play or drag the slider. Hover a tile to inspect it.")}
      </p>}

      <div className="replay-controls">
        <button type="button" onClick={() => { if (pos >= last) setPos(0); setPlaying(!playing); }}>
          {playing ? "Pause" : "Play"}
        </button>
        <input type="range" min={0} max={last} step="any" value={pos} aria-label="Action"
          onChange={(e) => { setPlaying(false); setPos(+e.target.value); }} />
        <span className="replay-turn">
          Turn {now[1]}
          {action && <> · <span style={{ color: PLAYER[action[0]] }}>{data.labels[action[5]]}</span></>}
        </span>
      </div>

      {!compact && <>
      <svg viewBox={`0 0 ${strip.SW} ${strip.SH}`} className="replay-strip" aria-hidden="true">
        <line x1={8} x2={strip.SW - 8} y1={strip.SH / 2} y2={strip.SH / 2} className="viz-grid" />
        <path d={strip.blue} stroke={PLAYER[1]} strokeWidth={2} fill="none" />
        <path d={strip.orange} stroke={PLAYER[2]} strokeWidth={2} fill="none" />
        {hasValue && <path d={strip.win} stroke="#111" strokeWidth={1.25} strokeDasharray="3 3" fill="none" />}
        <line x1={strip.sx(pos)} x2={strip.sx(pos)} y1={4} y2={strip.SH - 4} stroke="#111" strokeWidth={1} />
      </svg>
      <div className="viz-legend replay-legend">
        <span><i style={{ background: PLAYER[1] }} />Blue score</span>
        <span><i style={{ background: PLAYER[2] }} />Orange score</span>
        {hasValue && <span><i className="dashed" style={{ color: "#111" }} />Orange's own win estimate (midline = 50%)</span>}
      </div>

      <div className="replay-moments">
        {MOMENTS.map((m, j) => (
          <button key={m.turn} type="button" aria-pressed={moment?.turn === m.turn}
            onClick={() => { setPlaying(false); setPos(turnStart[j] < 0 ? last : turnStart[j]); }}>T{m.turn}</button>
        ))}
        <span className="replay-speed" role="group" aria-label="Playback speed">
          {SPEEDS.map((s) => (
            <button key={s} type="button" aria-pressed={speed === s} onClick={() => setSpeed(s)}>{s}×</button>
          ))}
        </span>
      </div>
      <div className="replay-key">
        {[
          { label: "unit (number = HP)", svg: unitArt([1, 2, 10, 0]), box: "-24 -44 44 54" },
          { label: "boat", svg: unitArt([1, 39, 10, 0]), box: "-24 -44 48 54" },
          { label: "Cloak / Dagger", svg: unitArt([2, 38, 10, 0]), box: "-24 -44 44 54" },
          { label: "city (more houses = higher level)", svg: cityArt([2, 3], false), box: "-26 -22 52 40" },
        ].map((item) => (
          <span key={item.label}>
            <svg width="22" height="26" viewBox={item.box} dangerouslySetInnerHTML={{ __html: item.svg }} />
            {item.label}
          </span>
        ))}
      </div>
      </>}
    </div>
  );
}
