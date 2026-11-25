import { useMemo } from "react";
import type { PolyAction, PolyCity, PolyTile, PolyUnit } from "../posts/autoresearch/polytopia/ReplayViewer";
import type { PokeSnapshot } from "../posts/autoresearch/NetworkIO";

export const POLY_BLUE = "#2f95e0";
export const POLY_ORANGE = "#eb6834";
export const GEN_CYAN = "#00a0d6";
export const GEN_MAGENTA = "#e0218a";
export const POKE_CYAN = "#0081ad";
export const POKE_MAGENTA = "#e0218a";

/* -------------------------------------------------------------------------- */
/*                     1. SIMPLIFIED POLYTOPIA POLICY I/O                     */
/* -------------------------------------------------------------------------- */

const CORE_POLY_ACTIONS = [
  "move",
  "attack",
  "train",
  "build",
  "capture",
  "harvest",
  "research",
  "end",
] as const;
type CorePolyAction = (typeof CORE_POLY_ACTIONS)[number];

function mapPolyAction(kind: number, label: string): CorePolyAction {
  const lo = label.toLowerCase();
  if (kind === 1 || lo.startsWith("move")) return "move";
  if (kind === 2 || lo.startsWith("attack")) return "attack";
  if (kind === 4 || lo.startsWith("train")) return "train";
  if (kind === 5 || lo.startsWith("harvest")) return "harvest";
  if (kind === 6 || lo.startsWith("build") || lo.includes("forest")) return "build";
  if (kind === 7 || lo.startsWith("research")) return "research";
  if (kind === 8 || lo.startsWith("capture")) return "capture";
  if (kind === 9 || lo.includes("reward")) return "build";
  if (kind === 10 || lo.includes("end turn")) return "end";
  return "move";
}

export function SimplePolytopiaIO({
  size,
  tiles,
  units,
  cities,
  action,
  labels,
}: {
  size: number;
  tiles: PolyTile[];
  units: Map<number, PolyUnit>;
  cities: Map<number, PolyCity>;
  action: PolyAction | null;
  labels: string[];
}) {
  const viewPlayer = 1;
  const oppPlayer = 2;
  const isActing = action ? action[0] === viewPlayer : false;

  const computed = useMemo(() => {
    const n = size * size;
    const known = new Uint8Array(n);
    for (let t = 0; t < n; t++) {
      const r = Math.floor(t / size), c = t % size;
      const ownTile = tiles[t][1] === viewPlayer;
      const ownUnit = units.get(t)?.[0] === viewPlayer;
      const ownCity = cities.get(t)?.[0] === viewPlayer;
      const rad = ownUnit || ownCity ? 2 : ownTile ? 1 : -1;
      if (rad < 0) continue;
      for (let dr = -rad; dr <= rad; dr++) {
        for (let dc = -rad; dc <= rad; dc++) {
          const rr = r + dr, cc = c + dc;
          if (rr >= 0 && rr < size && cc >= 0 && cc < size) known[rr * size + cc] = 1;
        }
      }
    }

    const terrain = new Float32Array(n);
    const ownTiles = new Uint8Array(n);
    const ownUnits = new Float32Array(n);
    const foeUnits = new Float32Array(n);
    const cityPlane = new Float32Array(n);
    const resources = new Uint8Array(n);
    const roads = new Uint8Array(n);
    const fog = new Uint8Array(n);

    const ownUnitTiles: number[] = [];
    const foeUnitTiles: number[] = [];
    const ownEmptyCities: number[] = [];
    const ownResourceTiles: number[] = [];
    const ownBuildTiles: number[] = [];
    const capturableTiles: number[] = [];

    for (let t = 0; t < n; t++) {
      const [terr, owner, struct, res, road, capital] = tiles[t];
      const vis = known[t] === 1;
      fog[t] = vis ? 0 : 1;
      if (!vis) continue;
      terrain[t] = terr <= 2 ? 0.2 : terr === 3 ? 0.45 : terr === 5 ? 0.72 : 1.0;
      ownTiles[t] = owner === viewPlayer ? 1 : 0;
      const u = units.get(t);
      if (u) {
        if (u[0] === viewPlayer) {
          ownUnits[t] = Math.min(1, Math.max(0.28, u[2] / 15));
          ownUnitTiles.push(t);
          const ci = cities.get(t);
          if (ci && ci[0] !== viewPlayer) capturableTiles.push(t);
        } else {
          foeUnits[t] = Math.min(1, Math.max(0.28, u[2] / 15));
          foeUnitTiles.push(t);
        }
      }
      const ci = cities.get(t);
      if (ci) {
        cityPlane[t] = ci[0] === viewPlayer ? Math.min(1, 0.5 + ci[1] * 0.12) : ci[0] === oppPlayer ? 0.6 : 0.3;
        if (ci[0] === viewPlayer && !u) ownEmptyCities.push(t);
      }
      if (res || struct > 1) resources[t] = 1;
      if (road || capital) roads[t] = 1;
      if (owner === viewPlayer) {
        if (res) ownResourceTiles.push(t);
        if (terr >= 3 && !ci && struct <= 1) ownBuildTiles.push(t);
      }
    }

    const actLabel = action ? labels[action[5]] ?? "" : "";
    const chosenAct: CorePolyAction | null = isActing && action ? mapPolyAction(action[2], actLabel) : null;
    const chosenSrc = isActing && action ? action[3] : -1;
    const chosenDst = isActing && action ? action[4] : -1;

    const legalActs = new Set<CorePolyAction>(["end", "research"]);
    if (ownUnitTiles.length > 0) legalActs.add("move");
    const attackers: number[] = [];
    for (const ut of ownUnitTiles) {
      const r = Math.floor(ut / size), c = ut % size;
      if (foeUnitTiles.some((ft) => Math.max(Math.abs(Math.floor(ft / size) - r), Math.abs((ft % size) - c)) <= 2)) {
        attackers.push(ut);
      }
    }
    if (attackers.length > 0) legalActs.add("attack");
    if (ownEmptyCities.length > 0) legalActs.add("train");
    if (ownResourceTiles.length > 0) legalActs.add("harvest");
    if (ownBuildTiles.length > 0) legalActs.add("build");
    if (capturableTiles.length > 0) legalActs.add("capture");
    if (chosenAct) legalActs.add(chosenAct);

    const displayAct: CorePolyAction = chosenAct ?? "move";
    const legalSrc = new Set<number>();
    if (displayAct === "move") ownUnitTiles.forEach((t) => legalSrc.add(t));
    else if (displayAct === "attack") attackers.forEach((t) => legalSrc.add(t));
    else if (displayAct === "train") ownEmptyCities.forEach((t) => legalSrc.add(t));
    else if (displayAct === "harvest") ownResourceTiles.forEach((t) => legalSrc.add(t));
    else if (displayAct === "build") ownBuildTiles.slice(0, 14).forEach((t) => legalSrc.add(t));
    else if (displayAct === "capture") capturableTiles.forEach((t) => legalSrc.add(t));
    if (chosenSrc >= 0 && chosenSrc < n) legalSrc.add(chosenSrc);

    const isSpatialTarget = displayAct === "move" || displayAct === "attack";
    const legalDst = new Set<number>();
    if (isSpatialTarget) {
      const anchor = chosenSrc >= 0 ? chosenSrc : ownUnitTiles[0] ?? -1;
      if (anchor >= 0) {
        const ar = Math.floor(anchor / size), ac = anchor % size;
        if (displayAct === "move") {
          for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              if (!dr && !dc) continue;
              const rr = ar + dr, cc = ac + dc;
              if (rr < 0 || rr >= size || cc < 0 || cc >= size) continue;
              const ti = rr * size + cc;
              if (tiles[ti][0] > 2 && !units.has(ti)) legalDst.add(ti);
            }
          }
        } else {
          for (const ft of foeUnitTiles) {
            if (Math.max(Math.abs(Math.floor(ft / size) - ar), Math.abs((ft % size) - ac)) <= 2) legalDst.add(ft);
          }
        }
      }
      if (chosenDst >= 0 && chosenDst < n) legalDst.add(chosenDst);
    }

    return {
      terrain, ownTiles, ownUnits, foeUnits, cityPlane, resources, roads, fog,
      chosenAct, chosenSrc, chosenDst, legalActs, legalSrc, legalDst, isSpatialTarget, actLabel,
    };
  }, [size, tiles, units, cities, action, isActing, labels]);

  const planes = [
    { name: "terrain", values: computed.terrain, color: "#5f8c69", continuous: true },
    { name: "own", values: computed.ownTiles, color: POLY_BLUE },
    { name: "units", values: computed.ownUnits, color: POLY_BLUE, continuous: true },
    { name: "foe", values: computed.foeUnits, color: POLY_ORANGE, continuous: true },
    { name: "cities", values: computed.cityPlane, color: "#d4a000", continuous: true },
    { name: "res", values: computed.resources, color: "#3d8b4d" },
    { name: "roads", values: computed.roads, color: "#7b6c5a" },
    { name: "fog", values: computed.fog, color: "#767670" },
  ];

  const paramLabel = computed.actLabel.replace(/^(Train|Build|Research)\s+/i, "") || computed.chosenAct || "";

  return (
    <aside className={`sp-panel${isActing ? "" : " is-waiting"}`}>
      <div className="sp-head">
        <span className="sp-name">
          <i style={{ background: POLY_BLUE }} />
          <strong>GPT-6 Astra</strong>
        </span>
        <span className="sp-tag">Trained policy</span>
      </div>

      <div className="sp-sec">
        <div className="sp-sec-title">Inputs</div>
        <div className="sp-planes">
          {planes.map((pl) => (
            <div key={pl.name} className="sp-plane">
              <svg viewBox={`0 0 ${size} ${size}`} shapeRendering="crispEdges">
                <rect width={size} height={size} fill="#f1f1ec" />
                {Array.from({ length: size * size }, (_, idx) => {
                  const v = pl.values[idx];
                  if (!v) return null;
                  return (
                    <rect
                      key={idx}
                      x={idx % size}
                      y={Math.floor(idx / size)}
                      width={1}
                      height={1}
                      fill={pl.color}
                      fillOpacity={pl.continuous ? Math.max(0.28, v) : 0.92}
                    />
                  );
                })}
              </svg>
              <span>{pl.name}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="sp-sec sp-sec-out">
        <div className="sp-sec-title">
          <span>Output</span>
          {!isActing && <span className="sp-wait-pill">opponent turn</span>}
        </div>

        <div className="sp-actions">
          {CORE_POLY_ACTIONS.map((a) => {
            const chosen = computed.chosenAct === a;
            const legal = computed.legalActs.has(a);
            return (
              <span
                key={a}
                className={`sp-act${chosen ? " is-chosen" : legal ? " is-legal" : " is-masked"}`}
              >
                {a}
              </span>
            );
          })}
        </div>

        <div className="sp-grids">
          <div className="sp-grid-box">
            <span className="sp-grid-lbl">Source</span>
            <svg viewBox={`0 0 ${size} ${size}`} shapeRendering="crispEdges">
              <rect width={size} height={size} fill="#ecece6" />
              {Array.from(computed.legalSrc, (ti) => (
                <rect
                  key={ti}
                  x={ti % size}
                  y={Math.floor(ti / size)}
                  width={1}
                  height={1}
                  fill={ti === computed.chosenSrc ? POLY_BLUE : "#7a7a74"}
                />
              ))}
            </svg>
          </div>

          <div className="sp-grid-box">
            <span className="sp-grid-lbl">{computed.isSpatialTarget ? "Target" : "Action"}</span>
            {computed.isSpatialTarget ? (
              <svg viewBox={`0 0 ${size} ${size}`} shapeRendering="crispEdges">
                <rect width={size} height={size} fill="#ecece6" />
                {computed.chosenSrc >= 0 && (
                  <rect
                    x={computed.chosenSrc % size}
                    y={Math.floor(computed.chosenSrc / size)}
                    width={1}
                    height={1}
                    fill={POLY_BLUE}
                    fillOpacity={0.22}
                  />
                )}
                {Array.from(computed.legalDst, (ti) => (
                  <rect
                    key={ti}
                    x={ti % size}
                    y={Math.floor(ti / size)}
                    width={1}
                    height={1}
                    fill={ti === computed.chosenDst ? POLY_BLUE : "#7a7a74"}
                  />
                ))}
              </svg>
            ) : (
              <div className="sp-param-card">
                <span className={`sp-param-pill${isActing ? " is-chosen" : ""}`}>
                  {isActing ? paramLabel : "—"}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
}

/* -------------------------------------------------------------------------- */
/*                     2. SIMPLIFIED GENERALS POLICY I/O                      */
/* -------------------------------------------------------------------------- */

const GEN_DIRS: readonly [number, number, string][] = [
  [-1, 0, "↑"],
  [1, 0, "↓"],
  [0, -1, "←"],
  [0, 1, "→"],
];

export function SimpleGeneralsIO({
  width,
  height,
  cells,
  actions,
  actionWidth,
}: {
  width: number;
  height: number;
  cells: Int32Array;
  actions: [number, number] | null;
  actionWidth: number;
}) {
  const side = 0; // Cyan = GPT-6.1 Sol

  const data = useMemo(() => {
    const n = width * height;
    const seen = new Uint8Array(n);
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        if (cells[(r * width + c) * 3 + 1] === side) {
          for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              const rr = r + dr, cc = c + dc;
              if (rr >= 0 && rr < height && cc >= 0 && cc < width) seen[rr * width + cc] = 1;
            }
          }
        }
      }
    }

    const armies = new Float32Array(n);
    const owned = new Uint8Array(n);
    const foe = new Uint8Array(n);
    const fog = new Uint8Array(n);
    const cities = new Uint8Array(n);
    const mtns = new Uint8Array(n);
    const gen = new Uint8Array(n);
    const neut = new Uint8Array(n);

    for (let i = 0; i < n; i++) {
      const k = cells[i * 3], o = cells[i * 3 + 1], a = cells[i * 3 + 2];
      const vis = seen[i] === 1;
      armies[i] = vis ? Math.min(1, Math.log1p(a) / 5.5) : 0;
      owned[i] = o === side ? 1 : 0;
      foe[i] = vis && o === 1 ? 1 : 0;
      fog[i] = !vis ? 1 : 0;
      cities[i] = vis && k === 2 ? 1 : 0;
      mtns[i] = vis && k === 1 ? 1 : 0;
      gen[i] = vis && k === 3 ? 1 : 0;
      neut[i] = vis && o < 0 && k !== 1 ? 1 : 0;
    }

    const legalPairs: { id: number; r: number; c: number; tr: number; tc: number; dir: number }[] = [];
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const idx = r * width + c;
        if (cells[idx * 3 + 1] !== side || cells[idx * 3 + 2] <= 1) continue;
        for (let d = 0; d < 4; d++) {
          const [dr, dc] = GEN_DIRS[d];
          const tr = r + dr, tc = c + dc;
          if (tr < 0 || tr >= height || tc < 0 || tc >= width) continue;
          if (cells[(tr * width + tc) * 3] === 1) continue;
          legalPairs.push({ id: 1 + ((r * actionWidth + c) * 4 + d) * 2, r, c, tr, tc, dir: d });
        }
      }
    }

    const chosenId = actions?.[side] ?? 0;
    let chosen: { r: number; c: number; tr: number; tc: number; dir: number; arrow: string; split: boolean } | null = null;
    if (chosenId > 0) {
      const a = chosenId - 1;
      const cell = Math.floor(a / 8);
      const r = Math.floor(cell / actionWidth), c = cell % actionWidth;
      const d = Math.floor(a / 2) % 4, split = a % 2 === 1;
      const [dr, dc, arrow] = GEN_DIRS[d];
      const tr = r + dr, tc = c + dc;
      if (r >= 0 && r < height && c >= 0 && c < width && tr >= 0 && tr < height && tc >= 0 && tc < width) {
        chosen = { r, c, tr, tc, dir: d, arrow, split };
      }
    }

    return { armies, owned, foe, fog, cities, mtns, gen, neut, legalPairs, chosen };
  }, [cells, width, height, actions, actionWidth]);

  const planes = [
    { name: "army", values: data.armies, color: "#111", continuous: true },
    { name: "own", values: data.owned, color: GEN_CYAN },
    { name: "foe", values: data.foe, color: GEN_MAGENTA },
    { name: "fog", values: data.fog, color: "#767670" },
    { name: "cities", values: data.cities, color: "#d4a000" },
    { name: "mtns", values: data.mtns, color: "#5a5a56" },
    { name: "gen", values: data.gen, color: GEN_CYAN },
    { name: "neut", values: data.neut, color: "#9c9c95" },
  ];

  const cellSz = 12;
  const mapW = width * cellSz, mapH = height * cellSz;

  return (
    <aside className="sp-panel">
      <div className="sp-head">
        <span className="sp-name">
          <i style={{ background: GEN_CYAN }} />
          <strong>GPT-6.1 Sol</strong>
        </span>
        <span className="sp-tag">Trained policy</span>
      </div>

      <div className="sp-sec">
        <div className="sp-sec-title">Inputs</div>
        <div className="sp-planes">
          {planes.map((pl) => (
            <div key={pl.name} className="sp-plane">
              <svg viewBox={`0 0 ${width} ${height}`} shapeRendering="crispEdges">
                <rect width={width} height={height} fill="#f1f1ec" />
                {Array.from({ length: width * height }, (_, idx) => {
                  const v = pl.values[idx];
                  if (!v) return null;
                  return (
                    <rect
                      key={idx}
                      x={idx % width}
                      y={Math.floor(idx / width)}
                      width={1}
                      height={1}
                      fill={pl.color}
                      fillOpacity={pl.continuous ? Math.max(0.24, v) : 0.92}
                    />
                  );
                })}
              </svg>
              <span>{pl.name}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="sp-sec sp-sec-out">
        <div className="sp-sec-title">
          <span>Output</span>
          <span className="sp-gen-choice" style={{ color: GEN_CYAN }}>
            {data.chosen
              ? `(${data.chosen.c}, ${data.chosen.r}) ${data.chosen.split ? "½" : ""}${data.chosen.arrow}`
              : "Pass"}
          </span>
        </div>

        <div className="sp-gen-map-wrap">
          <svg viewBox={`0 0 ${mapW} ${mapH}`} className="sp-gen-map">
            <rect width={mapW} height={mapH} rx={4} fill="#f2f2ed" />
            {Array.from({ length: width * height }, (_, idx) => {
              const k = cells[idx * 3], o = cells[idx * 3 + 1];
              if (k !== 1 && o !== side) return null;
              return (
                <rect
                  key={idx}
                  x={(idx % width) * cellSz + 0.6}
                  y={Math.floor(idx / width) * cellSz + 0.6}
                  width={cellSz - 1.2}
                  height={cellSz - 1.2}
                  rx={1.5}
                  fill={k === 1 ? "#d4d4ce" : GEN_CYAN}
                  fillOpacity={k === 1 ? 0.85 : 0.16}
                />
              );
            })}
            {data.legalPairs.map((m) => {
              const isChosen = data.chosen && data.chosen.r === m.r && data.chosen.c === m.c && data.chosen.dir === m.dir;
              if (isChosen) return null;
              const x1 = m.c * cellSz + cellSz / 2, y1 = m.r * cellSz + cellSz / 2;
              const x2 = m.tc * cellSz + cellSz / 2, y2 = m.tr * cellSz + cellSz / 2;
              return (
                <line
                  key={m.id}
                  x1={x1}
                  y1={y1}
                  x2={x1 + (x2 - x1) * 0.65}
                  y2={y1 + (y2 - y1) * 0.65}
                  stroke="#555550"
                  strokeWidth={1.3}
                  strokeLinecap="round"
                  opacity={0.48}
                />
              );
            })}
            {data.chosen && (() => {
              const m = data.chosen;
              const x1 = m.c * cellSz + cellSz / 2, y1 = m.r * cellSz + cellSz / 2;
              const x2 = m.tc * cellSz + cellSz / 2, y2 = m.tr * cellSz + cellSz / 2;
              return (
                <g>
                  <rect
                    x={m.tc * cellSz + 0.8}
                    y={m.tr * cellSz + 0.8}
                    width={cellSz - 1.6}
                    height={cellSz - 1.6}
                    rx={2}
                    fill={GEN_CYAN}
                    fillOpacity={0.28}
                    stroke={GEN_CYAN}
                    strokeWidth={1.6}
                  />
                  <line
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={GEN_CYAN}
                    strokeWidth={2.8}
                    strokeLinecap="round"
                    strokeDasharray={m.split ? "2.5 2.2" : undefined}
                  />
                  <circle cx={x2} cy={y2} r={2.6} fill={GEN_CYAN} />
                </g>
              );
            })()}
          </svg>
        </div>
      </div>
    </aside>
  );
}

/* -------------------------------------------------------------------------- */
/*                     3. SIMPLIFIED POKÉMON POLICY I/O                       */
/* -------------------------------------------------------------------------- */

export function SimplePokemonIO({
  snapshots,
  turn,
  finished,
}: {
  snapshots: PokeSnapshot[];
  turn: number;
  finished: boolean;
}) {
  const side = 0; // Cyan = GPT-6.1 Sol
  const maxTurn = Math.max(1, snapshots.length - 1);
  const snap = snapshots[Math.max(1, Math.min(maxTurn, turn || 1))] ?? snapshots[0];
  if (!snap) return null;

  const ownTeam = snap.teams[side];
  const foeTeam = snap.teams[1 - side];
  const activeOwn = ownTeam.find((m) => m.active) ?? ownTeam[0];
  const choice = snap.choices[side];

  // Clean 4-move list for the active Pokémon
  const knownMoves = activeOwn?.moves ?? [];
  const moveNames = [
    knownMoves[0] ?? "Fire Blast",
    knownMoves[1] ?? "Dark Pulse",
    knownMoves[2] ?? "Sludge Bomb",
    knownMoves[3] ?? "Nasty Plot",
  ];

  const isTeraMove = choice.kind === "tera-move";

  return (
    <aside className="sp-panel">
      <div className="sp-head">
        <span className="sp-name">
          <i style={{ background: POKE_CYAN }} />
          <strong>GPT-6.1 Sol</strong>
        </span>
        <span className="sp-tag">Trained policy</span>
      </div>

      {/* Team state inputs */}
      <div className="sp-sec">
        <div className="sp-sec-title">Inputs</div>
        <div className="sp-poke-teams">
          {ownTeam.map((m, idx) => {
            const visible = m.revealed || idx === 0;
            const hpPct = visible ? Math.round(m.hp * 100) : 100;
            return (
              <div key={`o-${idx}`} className={`sp-mon${m.active ? " is-act" : ""}`}>
                <span className="sp-mon-name">{visible ? m.species : `Bench ${idx}`}</span>
                <span className="sp-mon-bar">
                  <i style={{ width: `${hpPct}%`, background: m.fainted ? "#ccc" : POKE_CYAN }} />
                </span>
                <span className="sp-mon-tag">
                  {m.fainted ? "KO" : m.tera ? "TERA" : m.active ? `${hpPct}%` : ""}
                </span>
              </div>
            );
          })}
          <div className="sp-poke-sep" />
          {foeTeam.map((m, idx) => {
            const visible = m.revealed || (finished && idx <= 2);
            const hpPct = finished ? 0 : visible ? Math.round(m.hp * 100) : 0;
            const ko = finished || m.fainted;
            return (
              <div key={`f-${idx}`} className={`sp-mon${m.active && !finished ? " is-act" : ""}${!visible ? " is-unk" : ""}`}>
                <span className="sp-mon-name">{visible ? m.species : "Hidden"}</span>
                <span className="sp-mon-bar">
                  <i style={{ width: `${hpPct}%`, background: ko ? "#ccc" : POKE_MAGENTA }} />
                </span>
                <span className="sp-mon-tag">
                  {!visible ? "?" : ko ? "KO" : m.tera ? "TERA" : `${hpPct}%`}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Simplified Output */}
      <div className="sp-sec sp-sec-out">
        <div className="sp-sec-title">
          <span>Output</span>
          {isTeraMove && <span className="sp-tera-chip">Tera Fire</span>}
        </div>

        <div className="sp-poke-moves">
          {moveNames.map((mv) => {
            const chosen = (choice.kind === "move" || choice.kind === "tera-move") && choice.label === mv;
            return (
              <div key={mv} className={`sp-move-btn${chosen ? " is-chosen" : ""}`}>
                <span>{mv}</span>
                {chosen && isTeraMove && <em className="sp-move-tera">T</em>}
              </div>
            );
          })}
        </div>

        <div className="sp-poke-switches">
          <span className="sp-grid-lbl">Switch</span>
          <div className="sp-switch-row">
            {ownTeam.slice(1).map((m, idx) => {
              const chosen = choice.kind === "switch" && choice.label === m.species;
              return (
                <span key={idx} className={`sp-sw-pill${chosen ? " is-chosen" : ""}`}>
                  {m.revealed ? m.species : `Bench ${idx + 1}`}
                </span>
              );
            })}
          </div>
        </div>
      </div>
    </aside>
  );
}
