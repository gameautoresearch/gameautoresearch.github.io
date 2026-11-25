import { useMemo } from "react";

// Compact, dense, live-synced per-side visualizations of each trained policy network's actual
// inputs and outputs (chosen, legal not chosen, and masked) for the flanking side columns of <Showcase />.
// Schemas follow each simulator directly:
// - Pokémon: simulators/pokemon/view.cjs & network.py (12 mon slots, 40 global, 14 masked actions)
// - Polytopia: simulators/polytopia/.../observation.py, actions.py & network.py (11×11×151 + 525g; 17 fam → 121 src → 121/128 tgt)
// - Generals: simulators/generals/autoresearch/slow_sim/generals_sim/view.py (16×16×14 fogged; 2,049 masked actions)

/* -------------------------------------------------------------------------- */
/*                                 GENERALS IO                                */
/* -------------------------------------------------------------------------- */

const GEN_DIRS: readonly [number, number, string][] = [
  [-1, 0, "↑"],
  [1, 0, "↓"],
  [0, -1, "←"],
  [0, 1, "→"],
];

interface GeneralsLegalMove {
  id: number;
  r: number;
  c: number;
  tr: number;
  tc: number;
  dir: number;
  arrow: string;
  split: boolean;
}

export function GeneralsSideIO({
  side,
  width,
  height,
  cells,
  tick,
  maxTicks,
  actions,
  actionWidth,
  scores,
  accents,
}: {
  side: 0 | 1;
  width: number;
  height: number;
  cells: Int32Array;
  tick: number;
  maxTicks: number;
  actions: [number, number] | null;
  actionWidth: number;
  scores: { army: number; land: number }[];
  accents: [string, string];
}) {
  const accent = accents[side];
  const oppAccent = accents[1 - side];

  const data = useMemo(() => {
    const n = width * height;
    const seen = new Uint8Array(n);
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        if (cells[(r * width + c) * 3 + 1] === side) {
          for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              const rr = r + dr,
                cc = c + dc;
              if (rr >= 0 && rr < height && cc >= 0 && cc < width) {
                seen[rr * width + cc] = 1;
              }
            }
          }
        }
      }
    }

    const armies = new Float32Array(n);
    const owned = new Uint8Array(n);
    const opponent = new Uint8Array(n);
    const fog = new Uint8Array(n);
    const fogStruct = new Uint8Array(n);
    const castles = new Uint8Array(n);
    const mountains = new Uint8Array(n);
    const generals = new Uint8Array(n);
    const neutral = new Uint8Array(n);

    for (let i = 0; i < n; i++) {
      const k = cells[i * 3],
        o = cells[i * 3 + 1],
        a = cells[i * 3 + 2];
      const vis = seen[i] === 1;
      const isStruct = k === 1 || k === 2;
      armies[i] = vis ? Math.min(1, Math.log1p(a) / 6.0) : 0;
      owned[i] = o === side ? 1 : 0;
      opponent[i] = vis && o === 1 - side ? 1 : 0;
      fog[i] = !vis && !isStruct ? 1 : 0;
      fogStruct[i] = !vis && isStruct ? 1 : 0;
      castles[i] = vis && k === 2 ? 1 : 0;
      mountains[i] = vis && k === 1 ? 1 : 0;
      generals[i] = vis && k === 3 ? 1 : 0;
      neutral[i] = vis && o < 0 && k !== 1 ? 1 : 0;
    }

    const legalMoves: GeneralsLegalMove[] = [];
    const legalIds = new Set<number>([0]);
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const idx = r * width + c;
        if (cells[idx * 3 + 1] !== side || cells[idx * 3 + 2] <= 1) continue;
        for (let d = 0; d < 4; d++) {
          const [dr, dc, arrow] = GEN_DIRS[d];
          const tr = r + dr,
            tc = c + dc;
          if (tr < 0 || tr >= height || tc < 0 || tc >= width) continue;
          if (cells[(tr * width + tc) * 3] === 1) continue;
          const base = 1 + ((r * actionWidth + c) * 4 + d) * 2;
          legalIds.add(base);
          legalIds.add(base + 1);
          legalMoves.push({ id: base, r, c, tr, tc, dir: d, arrow, split: false });
          legalMoves.push({ id: base + 1, r, c, tr, tc, dir: d, arrow, split: true });
        }
      }
    }

    const chosenId = actions?.[side] ?? 0;
    let chosenMove: GeneralsLegalMove | null = null;
    if (chosenId > 0) {
      const a = chosenId - 1;
      const cell = Math.floor(a / 8);
      const r = Math.floor(cell / actionWidth),
        c = cell % actionWidth;
      const d = Math.floor(a / 2) % 4,
        split = a % 2 === 1;
      const [dr, dc, arrow] = GEN_DIRS[d];
      const tr = r + dr,
        tc = c + dc;
      if (r >= 0 && r < height && c >= 0 && c < width && tr >= 0 && tr < height && tc >= 0 && tc < width) {
        chosenMove = { id: chosenId, r, c, tr, tc, dir: d, arrow, split };
        legalIds.add(chosenId);
      }
    }

    return {
      armies,
      owned,
      opponent,
      fog,
      fogStruct,
      castles,
      mountains,
      generals,
      neutral,
      legalMoves,
      legalIds,
      chosenId,
      chosenMove,
    };
  }, [cells, width, height, side, actions, actionWidth]);

  const planes: { name: string; values: Float32Array | Uint8Array; color: string; continuous?: boolean }[] = [
    { name: "army", values: data.armies, color: "#111", continuous: true },
    { name: "own", values: data.owned, color: accent },
    { name: "foe", values: data.opponent, color: oppAccent },
    { name: "fog", values: data.fog, color: "#767670" },
    { name: "f_st", values: data.fogStruct, color: "#555550" },
    { name: "city", values: data.castles, color: "#d4a000" },
    { name: "mtn", values: data.mountains, color: "#5a5a56" },
    { name: "gen", values: data.generals, color: accent },
    { name: "neut", values: data.neutral, color: "#9c9c95" },
  ];

  const scalars = [
    { k: "L", v: Math.min(1, scores[side].land / (width * height)), opp: false },
    { k: "A", v: Math.min(1, Math.log1p(scores[side].army) / 8), opp: false },
    { k: "fL", v: Math.min(1, scores[1 - side].land / (width * height)), opp: true },
    { k: "fA", v: Math.min(1, Math.log1p(scores[1 - side].army) / 8), opp: true },
    { k: "t", v: maxTicks ? Math.min(1, tick / maxTicks) : 0, opp: false },
  ];

  const totalActions = 1 + actionWidth * actionWidth * 8;
  const legalCount = data.legalIds.size;
  const unchosenCount = Math.max(0, legalCount - 1);

  const dirPairs = useMemo(() => {
    const seen = new Set<string>();
    const out: GeneralsLegalMove[] = [];
    for (const m of data.legalMoves) {
      if (m.split) continue;
      const k = `${m.r},${m.c},${m.dir}`;
      if (!seen.has(k)) {
        seen.add(k);
        out.push(m);
      }
    }
    return out;
  }, [data.legalMoves]);

  const sampleUnchosen = useMemo(() => {
    const out: string[] = [];
    if (data.chosenId !== 0) out.push("Pass");
    for (const m of data.legalMoves) {
      if (m.id === data.chosenId) continue;
      if (out.length >= 4) break;
      out.push(`(${m.c},${m.r})${m.arrow}${m.split ? "½" : ""}`);
    }
    return out;
  }, [data.legalMoves, data.chosenId]);

  const barcodeSlots = 110;
  const barcodeBins = useMemo(() => {
    const bins = new Uint8Array(barcodeSlots);
    for (const id of data.legalIds) {
      const b = Math.min(barcodeSlots - 1, Math.floor((id / totalActions) * barcodeSlots));
      if (bins[b] < 1) bins[b] = 1;
    }
    const cb = Math.min(barcodeSlots - 1, Math.floor((data.chosenId / totalActions) * barcodeSlots));
    bins[cb] = 2;
    return bins;
  }, [data.legalIds, data.chosenId, totalActions]);

  const cellSz = 7;
  const mapW = width * cellSz,
    mapH = height * cellSz;

  return (
    <div className="sio">
      {/* INPUTS */}
      <div className="sio-sec">
        <div className="sio-hd">
          <span>In</span>
          <small>{width}×{height}×14 fog</small>
        </div>
        <div className="sio-planes-5">
          {planes.map((pl) => (
            <div key={pl.name} className="sio-plane">
              <svg viewBox={`0 0 ${width} ${height}`} shapeRendering="crispEdges" aria-hidden="true">
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
                      fillOpacity={pl.continuous ? Math.max(0.2, v) : 0.92}
                    />
                  );
                })}
              </svg>
              <span>{pl.name}</span>
            </div>
          ))}
          <div className="sio-scalars" title="5 global scalars: own land, own army, foe land, foe army, tick">
            {scalars.map((s) => (
              <div key={s.k} className="sio-sc-r">
                <span>{s.k}</span>
                <b>
                  <i style={{ width: `${Math.round(s.v * 100)}%`, background: s.opp ? oppAccent : accent }} />
                </b>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* OUTPUTS */}
      <div className="sio-sec sio-sec-out">
        <div className="sio-hd">
          <span>Out</span>
          <small>1 ch · {unchosenCount} not ch</small>
        </div>
        <div className="sio-gen-out">
          <div className="sio-gen-map" title="All legal directional moves (grey) vs chosen move (colored)">
            <svg viewBox={`0 0 ${mapW} ${mapH}`} aria-label="Legal and chosen move vectors">
              <rect width={mapW} height={mapH} rx={2} fill="#f3f3ee" />
              {Array.from({ length: width * height }, (_, idx) => {
                const k = cells[idx * 3],
                  o = cells[idx * 3 + 1];
                if (k !== 1 && o !== side) return null;
                return (
                  <rect
                    key={idx}
                    x={(idx % width) * cellSz + 0.4}
                    y={Math.floor(idx / width) * cellSz + 0.4}
                    width={cellSz - 0.8}
                    height={cellSz - 0.8}
                    rx={0.8}
                    fill={k === 1 ? "#d4d4ce" : accent}
                    fillOpacity={k === 1 ? 0.85 : 0.14}
                  />
                );
              })}
              {dirPairs.map((m) => {
                const isChosen =
                  data.chosenMove &&
                  data.chosenMove.r === m.r &&
                  data.chosenMove.c === m.c &&
                  data.chosenMove.dir === m.dir;
                if (isChosen) return null;
                const x1 = m.c * cellSz + cellSz / 2,
                  y1 = m.r * cellSz + cellSz / 2;
                const x2 = m.tc * cellSz + cellSz / 2,
                  y2 = m.tr * cellSz + cellSz / 2;
                return (
                  <line
                    key={m.id}
                    x1={x1}
                    y1={y1}
                    x2={x1 + (x2 - x1) * 0.65}
                    y2={y1 + (y2 - y1) * 0.65}
                    stroke="#4a4a46"
                    strokeWidth={0.95}
                    strokeLinecap="round"
                    opacity={0.55}
                  />
                );
              })}
              {data.chosenMove && (() => {
                const m = data.chosenMove;
                const x1 = m.c * cellSz + cellSz / 2,
                  y1 = m.r * cellSz + cellSz / 2;
                const x2 = m.tc * cellSz + cellSz / 2,
                  y2 = m.tr * cellSz + cellSz / 2;
                return (
                  <g>
                    <rect
                      x={m.tc * cellSz + 0.4}
                      y={m.tr * cellSz + 0.4}
                      width={cellSz - 0.8}
                      height={cellSz - 0.8}
                      rx={1}
                      fill={accent}
                      fillOpacity={0.28}
                      stroke={accent}
                      strokeWidth={1}
                    />
                    <line
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke={accent}
                      strokeWidth={1.8}
                      strokeLinecap="round"
                      strokeDasharray={m.split ? "1.6 1.4" : undefined}
                    />
                    <circle cx={x2} cy={y2} r={1.6} fill={accent} />
                  </g>
                );
              })()}
            </svg>
          </div>
          <div className="sio-gen-col">
            <div className="sio-chosen" style={{ borderColor: accent, background: `${accent}14` }}>
              <strong style={{ color: accent }}>
                {data.chosenMove
                  ? `(${data.chosenMove.c},${data.chosenMove.r}) ${data.chosenMove.split ? "½" : ""}${data.chosenMove.arrow}`
                  : "Pass"}
              </strong>
              <small>#{data.chosenId}</small>
            </div>
            <div className="sio-chips">
              {sampleUnchosen.map((txt) => (
                <span key={txt} className="sio-chip">{txt}</span>
              ))}
              {unchosenCount > sampleUnchosen.length && (
                <span className="sio-chip is-more">+{unchosenCount - sampleUnchosen.length}</span>
              )}
            </div>
          </div>
        </div>
        <div className="sio-bar" title={`Full ${totalActions}-logit masked action vector`}>
          <svg viewBox={`0 0 ${barcodeSlots} 9`} preserveAspectRatio="none" aria-hidden="true">
            <rect width={barcodeSlots} height={9} rx={1.2} fill="#ecece6" />
            {Array.from(barcodeBins, (v, idx) => {
              if (v === 0) return null;
              return (
                <rect
                  key={idx}
                  x={idx}
                  y={v === 2 ? 0 : 2}
                  width={v === 2 ? 1.8 : 0.85}
                  height={v === 2 ? 9 : 5}
                  fill={v === 2 ? accent : "#6b6b66"}
                />
              );
            })}
          </svg>
          <div className="sio-bar-lbl">
            <span>0</span>
            <span>{totalActions.toLocaleString()} masked logits</span>
            <span>{totalActions - 1}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*                                 POLYTOPIA IO                               */
/* -------------------------------------------------------------------------- */

const POLY_FAMILIES = [
  "move",
  "attack",
  "summon",
  "build",
  "research",
  "harvest",
  "forest",
  "reward",
  "capture",
  "recover",
  "promote",
  "disband",
  "heal",
  "destroy",
  "diplo",
  "end",
  "resign",
] as const;
type PolyFamily = (typeof POLY_FAMILIES)[number];

const POLY_UNIT_PARAMS = ["Warrior", "Archer", "Rider", "Defender", "Catapult", "Sword"];
const POLY_TECH_PARAMS = ["Climbing", "Forestry", "Mining", "Archery", "Riding", "Math"];
const POLY_BUILD_PARAMS = ["Lumber", "Farm", "Mine", "Sawmill", "Road", "Temple"];
const POLY_REWARD_PARAMS = ["Workshop", "Explorer", "Wall", "Stars", "Pop+3", "Giant"];
const POLY_FOREST_PARAMS = ["Clear", "Burn", "Grow"];

function mapPolyFamily(kind: number, label: string): PolyFamily {
  const lo = label.toLowerCase();
  if (kind === 1 || lo.startsWith("move")) return "move";
  if (kind === 2 || lo.startsWith("attack")) return "attack";
  if (kind === 4 || lo.startsWith("train")) return "summon";
  if (kind === 5 || lo.startsWith("harvest")) return "harvest";
  if (kind === 6 || lo.startsWith("build")) return "build";
  if (kind === 7 || lo.startsWith("research")) return "research";
  if (kind === 8 || lo.startsWith("capture")) return "capture";
  if (kind === 9 || lo.includes("reward")) return "reward";
  if (kind === 10 || lo.includes("end turn")) return "end";
  if (lo.includes("recover")) return "recover";
  if (lo.includes("promote")) return "promote";
  if (lo.includes("forest")) return "forest";
  if (lo.includes("disband")) return "disband";
  if (lo.includes("destroy")) return "destroy";
  if (lo.includes("heal")) return "heal";
  return "move";
}

export function PolytopiaSideIO({
  side,
  size,
  tiles,
  units,
  cities,
  action,
  turn,
  score,
  accents,
  labels,
}: {
  side: 0 | 1;
  size: number;
  tiles: [number, number, number, number, number, number][];
  units: Map<number, [number, number, number, number]>;
  cities: Map<number, [number, number]>;
  action: [number, number, number, number, number, number, ...unknown[]] | null;
  turn: number;
  score: [number, number];
  accents: [string, string];
  labels: string[];
}) {
  const viewPlayer = side + 1;
  const oppPlayer = 3 - viewPlayer;
  const accent = accents[side];
  const oppAccent = accents[1 - side];
  const isActing = action ? action[0] === viewPlayer : false;

  const computed = useMemo(() => {
    const n = size * size;
    const known = new Uint8Array(n);
    for (let t = 0; t < n; t++) {
      const r = Math.floor(t / size),
        c = t % size;
      const ownTile = tiles[t][1] === viewPlayer;
      const ownUnit = units.get(t)?.[0] === viewPlayer;
      const ownCity = cities.get(t)?.[0] === viewPlayer;
      const rad = ownUnit || ownCity ? 2 : ownTile ? 1 : -1;
      if (rad < 0) continue;
      for (let dr = -rad; dr <= rad; dr++) {
        for (let dc = -rad; dc <= rad; dc++) {
          const rr = r + dr,
            cc = c + dc;
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
    const ownForestTiles: number[] = [];
    const ownBuildTiles: number[] = [];
    const damagedOwnUnits: number[] = [];
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
          ownUnits[t] = Math.min(1, Math.max(0.25, u[2] / 15));
          ownUnitTiles.push(t);
          if (u[2] < 10) damagedOwnUnits.push(t);
          const ci = cities.get(t);
          if (ci && ci[0] !== viewPlayer) capturableTiles.push(t);
        } else {
          foeUnits[t] = Math.min(1, Math.max(0.25, u[2] / 15));
          foeUnitTiles.push(t);
        }
      }
      const ci = cities.get(t);
      if (ci) {
        cityPlane[t] = ci[0] === viewPlayer ? Math.min(1, 0.45 + ci[1] * 0.12) : ci[0] === oppPlayer ? 0.6 : 0.28;
        if (ci[0] === viewPlayer && !u) ownEmptyCities.push(t);
      }
      if (res || struct > 1) resources[t] = 1;
      if (road || capital) roads[t] = 1;
      if (owner === viewPlayer) {
        if (res) ownResourceTiles.push(t);
        if (terr === 5 && !ci) ownForestTiles.push(t);
        if (terr >= 3 && !ci && struct <= 1) ownBuildTiles.push(t);
      }
    }

    const actLabel = action ? labels[action[5]] ?? "" : "";
    const chosenFam: PolyFamily | null = isActing && action ? mapPolyFamily(action[2], actLabel) : null;
    const chosenSrc = isActing && action ? action[3] : -1;
    const chosenDst = isActing && action ? action[4] : -1;

    const legalFams = new Set<PolyFamily>(["end", "resign", "research"]);
    if (ownUnitTiles.length > 0) legalFams.add("move");
    const attackers: number[] = [];
    for (const ut of ownUnitTiles) {
      const r = Math.floor(ut / size),
        c = ut % size;
      if (foeUnitTiles.some((ft) => Math.max(Math.abs(Math.floor(ft / size) - r), Math.abs((ft % size) - c)) <= 2)) {
        attackers.push(ut);
      }
    }
    if (attackers.length > 0) legalFams.add("attack");
    if (ownEmptyCities.length > 0) legalFams.add("summon");
    if (ownResourceTiles.length > 0) legalFams.add("harvest");
    if (ownBuildTiles.length > 0) legalFams.add("build");
    if (ownForestTiles.length > 0) legalFams.add("forest");
    if (damagedOwnUnits.length > 0) legalFams.add("recover");
    if (capturableTiles.length > 0) legalFams.add("capture");
    if (chosenFam) legalFams.add(chosenFam);

    const displayFam: PolyFamily = chosenFam ?? "move";
    const legalSrc = new Set<number>();
    if (displayFam === "move") ownUnitTiles.forEach((t) => legalSrc.add(t));
    else if (displayFam === "attack") attackers.forEach((t) => legalSrc.add(t));
    else if (displayFam === "summon") ownEmptyCities.forEach((t) => legalSrc.add(t));
    else if (displayFam === "harvest") ownResourceTiles.forEach((t) => legalSrc.add(t));
    else if (displayFam === "forest") ownForestTiles.forEach((t) => legalSrc.add(t));
    else if (displayFam === "build") ownBuildTiles.slice(0, 14).forEach((t) => legalSrc.add(t));
    else if (displayFam === "recover") damagedOwnUnits.forEach((t) => legalSrc.add(t));
    else if (displayFam === "capture") capturableTiles.forEach((t) => legalSrc.add(t));
    else if (displayFam === "reward") {
      for (const [t, c] of cities) if (c[0] === viewPlayer) legalSrc.add(t);
    }
    if (chosenSrc >= 0 && chosenSrc < n) legalSrc.add(chosenSrc);

    const isSpatialTarget = displayFam === "move" || displayFam === "attack";
    const legalDst = new Set<number>();
    if (isSpatialTarget) {
      const anchor = chosenSrc >= 0 ? chosenSrc : ownUnitTiles[0] ?? -1;
      if (anchor >= 0) {
        const ar = Math.floor(anchor / size),
          ac = anchor % size;
        if (displayFam === "move") {
          for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              if (!dr && !dc) continue;
              const rr = ar + dr,
                cc = ac + dc;
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

    let paramCandidates: string[] = [];
    let chosenParam = actLabel;
    if (!isSpatialTarget) {
      if (displayFam === "summon") {
        chosenParam = actLabel.replace(/^Train\s+/i, "") || "Warrior";
        paramCandidates = POLY_UNIT_PARAMS;
      } else if (displayFam === "research") {
        chosenParam = actLabel.replace(/^Research\s+/i, "") || "Archery";
        paramCandidates = POLY_TECH_PARAMS;
      } else if (displayFam === "build") {
        chosenParam = actLabel.replace(/^Build\s+/i, "") || "Lumber";
        paramCandidates = POLY_BUILD_PARAMS;
      } else if (displayFam === "reward") {
        paramCandidates = POLY_REWARD_PARAMS;
      } else if (displayFam === "forest") {
        paramCandidates = POLY_FOREST_PARAMS;
      } else {
        paramCandidates = [actLabel || displayFam];
      }
      if (chosenParam && !paramCandidates.some((x) => x.toLowerCase() === chosenParam.toLowerCase())) {
        paramCandidates = [chosenParam, ...paramCandidates.slice(0, 5)];
      }
    }

    return {
      terrain,
      ownTiles,
      ownUnits,
      foeUnits,
      cityPlane,
      resources,
      roads,
      fog,
      chosenFam,
      chosenSrc,
      chosenDst,
      legalFams,
      legalSrc,
      legalDst,
      isSpatialTarget,
      paramCandidates,
      chosenParam,
    };
  }, [size, tiles, units, cities, action, viewPlayer, oppPlayer, isActing, labels]);

  const planes: { name: string; values: Float32Array | Uint8Array; color: string; continuous?: boolean }[] = [
    { name: "terr", values: computed.terrain, color: "#5f8c69", continuous: true },
    { name: "own", values: computed.ownTiles, color: accent },
    { name: "unit", values: computed.ownUnits, color: accent, continuous: true },
    { name: "foe", values: computed.foeUnits, color: oppAccent, continuous: true },
    { name: "city", values: computed.cityPlane, color: "#d4a000", continuous: true },
    { name: "res", values: computed.resources, color: "#3d8b4d" },
    { name: "road", values: computed.roads, color: "#7b6c5a" },
    { name: "fog", values: computed.fog, color: "#767670" },
  ];

  const famLegal = computed.legalFams.size;
  const famCh = computed.chosenFam ? 1 : 0;

  return (
    <div className={`sio${isActing ? " is-acting" : " is-waiting"}`}>
      {/* INPUTS */}
      <div className="sio-sec">
        <div className="sio-hd">
          <span>In</span>
          <small>11×11×151 · T{turn} · {score[side].toLocaleString()}</small>
        </div>
        <div className="sio-planes-4">
          {planes.map((pl) => (
            <div key={pl.name} className="sio-plane">
              <svg viewBox={`0 0 ${size} ${size}`} shapeRendering="crispEdges" aria-hidden="true">
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
                      fillOpacity={pl.continuous ? Math.max(0.25, v) : 0.9}
                    />
                  );
                })}
              </svg>
              <span>{pl.name}</span>
            </div>
          ))}
        </div>
      </div>

      {/* OUTPUTS */}
      <div className="sio-sec sio-sec-out">
        <div className="sio-hd">
          <span>Out · 3 heads</span>
          <small>{isActing ? `${famCh} ch · ${famLegal - famCh} not ch` : `wait · ${famLegal} legal`}</small>
        </div>
        <div className="sio-fams">
          {POLY_FAMILIES.map((f) => {
            const chosen = computed.chosenFam === f;
            const legal = computed.legalFams.has(f);
            return (
              <span
                key={f}
                className={`sio-fam${chosen ? " is-chosen" : legal ? " is-legal" : " is-masked"}`}
                style={chosen ? { background: accent, borderColor: accent, color: "#fff" } : undefined}
              >
                {f}
              </span>
            );
          })}
        </div>
        <div className="sio-h23">
          <div className="sio-hg">
            <span className="sio-k">Src {computed.legalSrc.size}/121</span>
            <svg viewBox={`0 0 ${size} ${size}`} shapeRendering="crispEdges" aria-label="Source tile mask">
              <rect width={size} height={size} fill="#ecece6" />
              {Array.from(computed.legalSrc, (ti) => (
                <rect
                  key={ti}
                  x={ti % size}
                  y={Math.floor(ti / size)}
                  width={1}
                  height={1}
                  fill={ti === computed.chosenSrc ? accent : "#62625c"}
                />
              ))}
            </svg>
          </div>
          <div className="sio-hg">
            <span className="sio-k">{computed.isSpatialTarget ? `Tgt ${computed.legalDst.size}/121` : "Param /128"}</span>
            {computed.isSpatialTarget ? (
              <svg viewBox={`0 0 ${size} ${size}`} shapeRendering="crispEdges" aria-label="Target tile mask">
                <rect width={size} height={size} fill="#ecece6" />
                {computed.chosenSrc >= 0 && (
                  <rect
                    x={computed.chosenSrc % size}
                    y={Math.floor(computed.chosenSrc / size)}
                    width={1}
                    height={1}
                    fill={accent}
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
                    fill={ti === computed.chosenDst ? accent : "#62625c"}
                  />
                ))}
              </svg>
            ) : (
              <div className="sio-chips">
                {computed.paramCandidates.slice(0, 6).map((p) => {
                  const chosen = isActing && p.toLowerCase() === computed.chosenParam.toLowerCase();
                  return (
                    <span
                      key={p}
                      className={`sio-chip${chosen ? " is-chosen" : ""}`}
                      style={chosen ? { background: accent, borderColor: accent, color: "#fff" } : undefined}
                    >
                      {p}
                    </span>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*                                 POKÉMON IO                                 */
/* -------------------------------------------------------------------------- */

interface PokeMonState {
  ident: string;
  species: string;
  level: number;
  hp: number;
  fainted: boolean;
  active: boolean;
  status: string;
  boosts: Record<string, number>;
  tera: string;
  confused: boolean;
  revealed: boolean;
  moves: string[];
}

interface PokeChoice {
  kind: "move" | "tera-move" | "switch" | "none";
  label: string;
  teraType?: string;
}

export interface PokeSnapshot {
  turn: number;
  teams: [PokeMonState[], PokeMonState[]];
  teraUsed: [boolean, boolean];
  weather: string;
  terrain: string;
  choices: [PokeChoice, PokeChoice];
}

function parseHpFraction(cond: string): { hp: number; fainted: boolean; status: string } {
  if (!cond) return { hp: 1, fainted: false, status: "" };
  if (cond.includes("fnt") || cond.startsWith("0 ")) return { hp: 0, fainted: true, status: "fnt" };
  const m = cond.match(/^(\d+)(?:\/(\d+))?(?:\s+([a-z]+))?/);
  if (!m) return { hp: 1, fainted: false, status: "" };
  const cur = Number(m[1]),
    max = Number(m[2] || 100);
  return { hp: max > 0 ? Math.max(0, Math.min(1, cur / max)) : 1, fainted: cur <= 0, status: m[3] || "" };
}

function shortSpecies(s: string): string {
  return s.split("-")[0];
}

export function parsePokemonTimeline(log: string): PokeSnapshot[] {
  const lines = log.split("\n");
  const knownSpecies: [string[], string[]] = [[], []];
  const knownLevels: [Map<string, number>, Map<string, number>] = [new Map(), new Map()];
  const knownMoves: [Map<string, string[]>, Map<string, string[]>] = [new Map(), new Map()];
  const identOf = (raw: string) => raw.replace(/^(p[12])[a-z]: /, "$1: ").trim();

  for (const line of lines) {
    const p = line.split("|");
    const cmd = p[1];
    if ((cmd === "switch" || cmd === "drag" || cmd === "replace") && p[2]) {
      const s = p[2].startsWith("p2") ? 1 : 0;
      const id = identOf(p[2]);
      const details = p[3] || "";
      const lvl = Number((details.match(/L(\d+)/) || [0, 100])[1]);
      if (!knownSpecies[s].includes(id)) knownSpecies[s].push(id);
      knownLevels[s].set(id, lvl);
      if (!knownMoves[s].has(id)) knownMoves[s].set(id, []);
    } else if (cmd === "move" && p[2] && p[3]) {
      const s = p[2].startsWith("p2") ? 1 : 0;
      const id = identOf(p[2]);
      const mv = p[3];
      const arr = knownMoves[s].get(id) ?? [];
      if (!arr.includes(mv) && arr.length < 4) arr.push(mv);
      knownMoves[s].set(id, arr);
    }
  }

  const makeInitialTeam = (s: 0 | 1): PokeMonState[] => {
    const out: PokeMonState[] = [];
    for (let i = 0; i < 3; i++) {
      const id = knownSpecies[s][i];
      if (id) {
        out.push({
          ident: id,
          species: shortSpecies(id.slice(4)),
          level: knownLevels[s].get(id) ?? 100,
          hp: 1,
          fainted: false,
          active: false,
          status: "",
          boosts: {},
          tera: "",
          confused: false,
          revealed: false,
          moves: [...(knownMoves[s].get(id) ?? [])],
        });
      } else {
        out.push({
          ident: `p${s + 1}: bench-${i + 1}`,
          species: `Slot ${i + 1}`,
          level: 100,
          hp: 1,
          fainted: false,
          active: false,
          status: "",
          boosts: {},
          tera: "",
          confused: false,
          revealed: false,
          moves: [],
        });
      }
    }
    return out;
  };

  const teams: [PokeMonState[], PokeMonState[]] = [makeInitialTeam(0), makeInitialTeam(1)];
  const teraUsed: [boolean, boolean] = [false, false];
  let weather = "";
  let terrain = "";
  let curTurn = 0;

  const cloneTeams = (): [PokeMonState[], PokeMonState[]] => [
    teams[0].map((m) => ({ ...m, boosts: { ...m.boosts }, moves: [...m.moves] })),
    teams[1].map((m) => ({ ...m, boosts: { ...m.boosts }, moves: [...m.moves] })),
  ];

  const findMon = (raw: string): [0 | 1, PokeMonState | undefined] => {
    const s: 0 | 1 = raw.startsWith("p2") ? 1 : 0;
    const id = identOf(raw);
    return [s, teams[s].find((m) => m.ident === id)];
  };

  const snapshots: PokeSnapshot[] = [];
  let turnStartSnap: PokeSnapshot | null = null;
  let teraThisTurn: [string, string] = ["", ""];
  let choicesThisTurn: [PokeChoice, PokeChoice] = [
    { kind: "none", label: "" },
    { kind: "none", label: "" },
  ];

  const flushTurn = () => {
    if (!turnStartSnap) return;
    turnStartSnap.choices = [{ ...choicesThisTurn[0] }, { ...choicesThisTurn[1] }];
    snapshots[turnStartSnap.turn] = turnStartSnap;
  };

  for (const line of lines) {
    const p = line.split("|");
    const cmd = p[1];
    if (cmd === "turn") {
      flushTurn();
      curTurn = Number(p[2]);
      teraThisTurn = ["", ""];
      choicesThisTurn = [
        { kind: "none", label: "" },
        { kind: "none", label: "" },
      ];
      turnStartSnap = {
        turn: curTurn,
        teams: cloneTeams(),
        teraUsed: [teraUsed[0], teraUsed[1]],
        weather,
        terrain,
        choices: [
          { kind: "none", label: "" },
          { kind: "none", label: "" },
        ],
      };
      continue;
    }

    if (cmd === "switch" || cmd === "drag" || cmd === "replace") {
      const [s, mon] = findMon(p[2] || "");
      for (const m of teams[s]) m.active = false;
      if (mon) {
        mon.active = true;
        mon.revealed = true;
        mon.boosts = {};
        mon.confused = false;
        if (p[4]) {
          const parsed = parseHpFraction(p[4]);
          mon.hp = parsed.hp;
          mon.fainted = parsed.fainted;
          if (parsed.status) mon.status = parsed.status;
        }
        if (curTurn > 0 && choicesThisTurn[s].kind === "none") {
          choicesThisTurn[s] = { kind: "switch", label: mon.species };
        }
      }
    } else if (cmd === "-terastallize") {
      const [s, mon] = findMon(p[2] || "");
      if (mon) {
        mon.tera = p[3] || "Tera";
        teraUsed[s] = true;
        teraThisTurn[s] = mon.tera;
      }
    } else if (cmd === "move") {
      const [s] = findMon(p[2] || "");
      const mv = p[3] || "";
      if (curTurn > 0 && choicesThisTurn[s].kind === "none") {
        choicesThisTurn[s] = teraThisTurn[s]
          ? { kind: "tera-move", label: mv, teraType: teraThisTurn[s] }
          : { kind: "move", label: mv };
      }
    } else if (cmd === "-damage" || cmd === "-heal" || cmd === "-sethp") {
      const [, mon] = findMon(p[2] || "");
      if (mon && p[3]) {
        const parsed = parseHpFraction(p[3]);
        mon.hp = parsed.hp;
        mon.fainted = parsed.fainted;
        if (parsed.status && parsed.status !== "fnt") mon.status = parsed.status;
      }
    } else if (cmd === "faint") {
      const [, mon] = findMon(p[2] || "");
      if (mon) {
        mon.hp = 0;
        mon.fainted = true;
        mon.status = "fnt";
      }
    } else if (cmd === "-status") {
      const [, mon] = findMon(p[2] || "");
      if (mon) mon.status = p[3] || "";
    } else if (cmd === "-curestatus") {
      const [, mon] = findMon(p[2] || "");
      if (mon) mon.status = "";
    } else if (cmd === "-boost" || cmd === "-unboost") {
      const [, mon] = findMon(p[2] || "");
      if (mon && p[3]) {
        const delta = Number(p[4] || 1) * (cmd === "-boost" ? 1 : -1);
        mon.boosts[p[3]] = (mon.boosts[p[3]] || 0) + delta;
      }
    } else if (cmd === "-start" && p[3]?.toLowerCase().includes("confusion")) {
      const [, mon] = findMon(p[2] || "");
      if (mon) mon.confused = true;
    } else if (cmd === "-end" && p[3]?.toLowerCase().includes("confusion")) {
      const [, mon] = findMon(p[2] || "");
      if (mon) mon.confused = false;
    } else if (cmd === "-weather") {
      weather = p[2] === "none" ? "" : p[2] || "";
    } else if (cmd === "-fieldstart" && p[2]?.includes("Terrain")) {
      terrain = p[2].replace(/^move:\s*/i, "");
    } else if (cmd === "-fieldend") {
      terrain = "";
    }
  }
  flushTurn();

  if (!snapshots[0] && snapshots[1]) snapshots[0] = snapshots[1];
  return snapshots;
}

export function PokemonSideIO({
  side,
  snapshots,
  turn,
  accents,
}: {
  side: 0 | 1;
  snapshots: PokeSnapshot[];
  turn: number;
  accents: [string, string];
}) {
  const accent = accents[side];
  const oppAccent = accents[1 - side];
  const maxTurn = Math.max(1, snapshots.length - 1);
  const snap = snapshots[Math.max(1, Math.min(maxTurn, turn || 1))] ?? snapshots[0];
  if (!snap) return null;

  const ownTeam = snap.teams[side];
  const foeTeam = snap.teams[1 - side];
  const activeOwn = ownTeam.find((m) => m.active) ?? ownTeam[0];
  const teraAvail = !snap.teraUsed[side];
  const choice = snap.choices[side];

  const moves = Array.from({ length: 4 }, (_, i) => activeOwn?.moves[i] ?? `Move ${i + 1}`);
  const isSwitchOnly = choice.kind === "switch";

  const slots: { idx: number; label: string; state: "chosen" | "legal" | "masked" }[] = [];
  for (let i = 0; i < 4; i++) {
    const mv = moves[i];
    const chosen = choice.kind === "move" && choice.label === mv;
    slots.push({ idx: i, label: mv, state: chosen ? "chosen" : isSwitchOnly ? "masked" : "legal" });
  }
  for (let i = 0; i < 4; i++) {
    const mv = moves[i];
    const chosen = choice.kind === "tera-move" && choice.label === mv;
    slots.push({ idx: 4 + i, label: `T+${mv}`, state: chosen ? "chosen" : !isSwitchOnly && teraAvail ? "legal" : "masked" });
  }
  for (let i = 0; i < 6; i++) {
    const m = ownTeam[i];
    if (!m) {
      slots.push({ idx: 8 + i, label: "pad", state: "masked" });
      continue;
    }
    const chosen = choice.kind === "switch" && choice.label === m.species;
    const legal = !m.active && !m.fainted;
    slots.push({ idx: 8 + i, label: m.species, state: chosen ? "chosen" : legal ? "legal" : "masked" });
  }

  const chosenCount = slots.filter((s) => s.state === "chosen").length;
  const unchosenCount = slots.filter((s) => s.state === "legal").length;
  const foeSeen = foeTeam.filter((m) => m.revealed).length;

  const badgeOf = (m: PokeMonState) => {
    if (m.fainted) return "FNT";
    if (m.tera) return `T:${m.tera.slice(0, 3)}`;
    if (m.status) return m.status.toUpperCase();
    const b = Object.entries(m.boosts).find(([, v]) => v !== 0);
    if (b) return `${b[1] > 0 ? "+" : ""}${b[1]}${b[0].slice(0, 3)}`;
    if (m.confused) return "CNF";
    if (m.active) return "ACT";
    return `L${m.level}`;
  };

  const renderRow = (m: PokeMonState, isOwn: boolean, idx: number) => {
    const visible = isOwn || m.revealed;
    const barColor = isOwn ? accent : oppAccent;
    const tag = visible ? badgeOf(m) : "tok=0";
    return (
      <div key={`${isOwn ? "o" : "f"}-${idx}`} className={`sio-mon${m.active && visible ? " is-act" : ""}${!visible ? " is-unk" : ""}`}>
        <span className="sio-mon-s" style={{ color: barColor }}>{isOwn ? `O${idx + 1}` : `F${idx + 1}`}</span>
        <span className="sio-mon-n">{visible ? m.species : "? hidden"}</span>
        <span className="sio-mon-hp">
          <i style={{ width: visible ? `${Math.round(m.hp * 100)}%` : "0%", background: m.fainted ? "#ccc" : barColor }} />
        </span>
        <em className={`sio-mon-b${visible && m.active ? " is-act" : ""}`}>{tag}</em>
      </div>
    );
  };

  return (
    <div className="sio">
      {/* INPUTS */}
      <div className="sio-sec">
        <div className="sio-hd">
          <span>In</span>
          <small>6 mons · {teraAvail ? "tera ok" : "no tera"} · {foeSeen}/3</small>
        </div>
        <div className="sio-mons">
          {ownTeam.map((m, idx) => renderRow(m, true, idx))}
          <div className="sio-sep" />
          {foeTeam.map((m, idx) => renderRow(m, false, idx))}
        </div>
      </div>

      {/* OUTPUTS */}
      <div className="sio-sec sio-sec-out">
        <div className="sio-hd">
          <span>Out · 14 logits</span>
          <small>{chosenCount} ch · {unchosenCount} not ch</small>
        </div>
        <div className="sio-pout">
          <div className="sio-sg">
            <span className="sio-k">0–3 Moves</span>
            <div className="sio-sgrid">
              {slots.slice(0, 4).map((s) => (
                <span
                  key={s.idx}
                  className={`sio-slot is-${s.state}`}
                  style={s.state === "chosen" ? { background: accent, borderColor: accent, color: "#fff" } : undefined}
                  title={`#${s.idx}: ${s.label} (${s.state})`}
                >
                  <i>{s.idx}</i>
                  <span>{s.label}</span>
                </span>
              ))}
            </div>
          </div>
          <div className="sio-sg">
            <span className="sio-k">4–7 Tera + Move</span>
            <div className="sio-sgrid">
              {slots.slice(4, 8).map((s) => (
                <span
                  key={s.idx}
                  className={`sio-slot is-${s.state}`}
                  style={s.state === "chosen" ? { background: accent, borderColor: accent, color: "#fff" } : undefined}
                  title={`#${s.idx}: ${s.label} (${s.state})`}
                >
                  <i>{s.idx}</i>
                  <span>{s.label}</span>
                </span>
              ))}
            </div>
          </div>
          <div className="sio-sg">
            <span className="sio-k">8–13 Switches</span>
            <div className="sio-sgrid is-3">
              {slots.slice(8, 14).map((s) => (
                <span
                  key={s.idx}
                  className={`sio-slot is-${s.state}`}
                  style={s.state === "chosen" ? { background: accent, borderColor: accent, color: "#fff" } : undefined}
                  title={`#${s.idx}: ${s.label} (${s.state})`}
                >
                  <i>{s.idx}</i>
                  <span>{s.label}</span>
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
