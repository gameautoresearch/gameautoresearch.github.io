// Adapted from the 2D editorial prototype. The game board is illustrative.
import type { PolyArt } from "./art";

type UnitMotion = {
  kind: number;
  color: string;
  skin: string;
  spawnAt?: number;
  koAt?: number;
  hops: { r0: number; c0: number; r1: number; c1: number; t0: number; t1: number }[];
  lunges?: { t0: number; t1: number; tr: number; tc: number }[];
  hpFn: (time: number) => number;
  flip?: boolean;
  scale?: number;
};

function el<K extends keyof SVGElementTagNameMap>(
  tag: K, attrs: Record<string, string | number>, parent: SVGGElement,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, String(value));
  parent.appendChild(node);
  return node;
}

const clamp = (v: number, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const invLerp = (a: number, b: number, v: number) => clamp((v - a) / (b - a));
const easeOutCubic = (t: number) => 1 - Math.pow(1 - clamp(t), 3);
const easeInOutCubic = (t: number) => {
  const x = clamp(t);
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
};
const springPop = (t: number) => {
  const x = clamp(t);
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return 1 - Math.exp(-7.5 * x) * Math.cos(10.5 * x);
};
function drawImpactBurst(parent: SVGGElement, cx: number, cy: number, r: number, color: string, op = 1) {
  if (op <= 0.02) return;
  const pts = [];
  for (let i = 0; i < 14; i++) {
    const a = (i / 14) * Math.PI * 2;
    const rad = i % 2 === 0 ? r : r * 0.44;
    pts.push(`${i === 0 ? "M" : "L"}${(cx + Math.cos(a) * rad).toFixed(1)} ${(cy + Math.sin(a) * rad).toFixed(1)}`);
  }
  el("path", {
    d: pts.join(" ") + " Z",
    fill: "#FFFFFF", stroke: color, "stroke-width": 2.4, "stroke-linejoin": "round", opacity: op.toFixed(3),
  }, parent);
  el("circle", { cx: cx.toFixed(1), cy: cy.toFixed(1), r: (r * 0.3).toFixed(1), fill: "#FFD43B", opacity: op.toFixed(3) }, parent);
}

// Exact Polytopia sprite math (from simulators/polytopia/web/src/board/art/real.ts)
const POLY_N = 4;
const POLY_TW = 128;
const POLY_TH = 76.8;
const POLY_HW = POLY_TW / 2;
const POLY_HH = POLY_TH / 2;
const POLY_U = POLY_TW / 0.96574;
const POLY_CY = 0.0778;
const POLY_TERRAIN_Y = -0.556;
const POLY_UNIT_SCALE = 1.15;

const POLY_BORDERS: readonly [string, number, number][] = [
  ["BorderYGFX", 0.2569, 0.2595],
  ["BorderXGFX", 0.2569, -0.0302],
  ["BorderYGFX", -0.2421, -0.0303],
  ["BorderXGFX", -0.2421, 0.2596],
];

function drawPolySprite(polyData: PolyArt, parent: SVGGElement, tileX: number, tileY: number, name: string, ux: number, uy: number, { sx = 1, sy = 1, rot = 0, tintColor = null as string | null, opacity = 1, extraScale = 1 } = {}) {
  const sp = polyData.sprites[name];
  if (!sp) return;
  const href = (tintColor && polyData.tinted[`${name}:${tintColor}`]) || polyData.urls[name];
  if (!href) return;

  const px = tileX + ux * POLY_U;
  const py = tileY + (POLY_CY - uy) * POLY_U;
  const w = sp.w * sp.unit * POLY_U * Math.abs(sx) * extraScale;
  const h = sp.h * sp.unit * POLY_U * Math.abs(sy) * extraScale;
  const ox = -sp.ax * w;
  const oy = -sp.ay * h;

  const flipX = sx < 0 ? -1 : 1;
  const flipY = sy < 0 ? -1 : 1;
  let tf = `translate(${px.toFixed(2)} ${py.toFixed(2)})`;
  if (rot) tf += ` rotate(${(-rot).toFixed(2)})`;
  if (flipX !== 1 || flipY !== 1) tf += ` scale(${flipX} ${flipY})`;

  el("image", {
    href, x: ox.toFixed(2), y: oy.toFixed(2), width: w.toFixed(2), height: h.toFixed(2),
    transform: tf, opacity: opacity.toFixed(3), preserveAspectRatio: "none",
  }, parent);
}

function drawPolyUnit(polyData: PolyArt, parent: SVGGElement, tileX: number, tileY: number, kind: number, tintColor: string, tribeSkin: string, { scale = 1, hopY = 0, dx = 0, dy = 0, flip = false, hp = null as number | null, opacity = 1 } = {}) {
  const parts = polyData.units[String(kind)] || polyData.units["2"] || [];
  const ug = el("g", { transform: `translate(${dx.toFixed(1)} ${(dy - hopY).toFixed(2)})`, opacity: opacity.toFixed(3) }, parent);
  const k = POLY_UNIT_SCALE * scale;
  for (const [rawName, ux, uy, sx, sy, rot, , takesTint] of parts) {
    let name = rawName;
    if (tribeSkin && name.endsWith("_bardur")) {
      const candidate = name.slice(0, -"bardur".length) + tribeSkin;
      if (polyData.sprites[candidate]) name = candidate;
    }
    const fx = flip ? -1 : 1;
    drawPolySprite(polyData, ug, tileX, tileY, name, ux * k * fx, uy * k, {
      sx: sx * k * fx, sy: sy * k, rot: flip ? -rot : rot, tintColor: takesTint ? tintColor : null,
    });
  }
  // Authentic Polytopia outlined HP number at upper-left of unit!
  if (hp !== null && hp > 0) {
    const hy = kind === 12 ? tileY - 56 : tileY - 36;
    const hx = tileX - 24;
    const hpEl = el("text", {
      x: hx.toFixed(1), y: hy.toFixed(1), "text-anchor": "middle",
      fill: "#FFFFFF", stroke: "#18222B", "stroke-width": 4.2, "paint-order": "stroke",
      "font-size": kind === 12 ? 21 : 18, "font-weight": 850, class: "tabular",
    }, ug);
    hpEl.textContent = String(hp);
  }
}

function drawPolyCity(polyData: PolyArt, parent: SVGGElement, tileX: number, tileY: number, tribeSkin: string, { level = 2, capital = false, wall = false, popScale = 1, ownerColor = null as string | null, showBar = false } = {}) {
  const cg = el("g", {
    transform: `translate(${tileX.toFixed(1)} ${(tileY + 10).toFixed(1)}) scale(${popScale.toFixed(3)}) translate(${(-tileX).toFixed(1)} ${(-(tileY + 10)).toFixed(1)})`,
  }, parent);
  const grid = [[0, 0.15], [-0.12, 0.06], [0.12, 0.06], [0, -0.02]];
  const slots = level === 1 ? [grid[3]] : level === 2 ? [grid[1], grid[2]] : grid.slice(0, Math.min(4, level));
  const kinds = [Math.min(level, 3), 2, 1, 1];
  slots.forEach(([hx, hy], idx) => {
    const spName = polyData.sprites[`House_${kinds[idx]}_${tribeSkin}`]
      ? `House_${kinds[idx]}_${tribeSkin}`
      : `House_${kinds[idx]}_bardur`;
    drawPolySprite(polyData, cg, tileX, tileY, spName, hx, hy);
  });
  if (capital) {
    const capName = polyData.sprites[`House_7_${tribeSkin}`] ? `House_7_${tribeSkin}` : "House_7_bardur";
    drawPolySprite(polyData, cg, tileX, tileY, capName, 0, level >= 3 ? 0.07 : 0.10);
  }
  if (wall) {
    drawPolySprite(polyData, cg, tileX, tileY, "CityWallGFX", 0, 0);
  }
  // Crisp miniature Polytopia city-level pip bar
  if (showBar && ownerColor) {
    const bw = 42;
    el("rect", { x: (tileX - bw / 2).toFixed(1), y: (tileY + 13).toFixed(1), width: bw, height: 8, rx: 4, fill: "rgba(17,24,32,0.78)" }, parent);
    const pips = Math.min(4, level);
    for (let p = 0; p < 4; p++) {
      el("rect", {
        x: (tileX - bw / 2 + 3 + p * 9.5).toFixed(1), y: (tileY + 15).toFixed(1),
        width: 7.5, height: 4, rx: 1.5, fill: p < pips ? ownerColor : "#4A5560",
      }, parent);
    }
  }
}

export function drawPolytopiaBoard(parent: SVGGElement, t: number, polyData: PolyArt, opponentWins: boolean, shadowId: string) {
  const g = el("g", { "data-board-art": "polytopia" }, parent);
  const boardCx = 1564;
  const COL = {
    cyan: opponentWins ? "#e0218a" : "#00a0d6",
    magenta: opponentWins ? "#00a0d6" : "#e0218a",
    gold: "#e8a800",
    bad: "#d64545",
  };
  // Real 4x4 Polytopia board
  const topCenterY = 532;
  const tilePos = (r: number, c: number): [number, number] => [boardCx + (c - r) * POLY_HW, topCenterY + (c + r) * POLY_HH];

  // Rapid gameplay event timestamps across 0.0s -> 8.8s!
  const T_HARVEST = 0.45;
  const T_LUMBER = 0.72;
  const T_FARM = 2.05;
  const T_CAP_V1 = 2.45;   // Cyan Rider captures (1, 2)
  const T_ARROW1 = 4.05;   // Cyan Archer shoots Magenta Warrior at (2, 1)
  const T_MELEE1 = 4.55;   // Cyan Warrior KOs Magenta Warrior at (2, 1)
  const T_CAP_V2 = 4.95;   // Cyan Warrior captures (2, 1)
  const T_MINE = 5.35;     // Cyan builds Mine on (0, 3) -> Lv4 Capital + GIANT SPAWN!
  const T_GIANT_SPAWN = 5.45;
  const T_ARROW2 = 6.10;   // Cyan Archer shoots Magenta Defender at (2, 2)
  const T_KNIGHT_SPAWN = 6.45;
  const T_GIANT_SMASH = 7.15; // Cyan Giant smashes Magenta Defender at (2, 2) -> claims center!
  const T_KNIGHT_HIT = 7.75;  // Cyan Knight flanks & strikes Magenta Swordsman at (3, 3)
  const T_FINAL_KO = 8.20;    // Cyan takes Magenta capital corner!

  const v1Cap = t >= T_CAP_V1;
  const v2Cap = t >= T_CAP_V2;
  const centerCap = t >= T_GIANT_SMASH;
  const finalCap = t >= T_FINAL_KO;

  const ownerOf = (r: number, c: number) => {
    if (r < 0 || r >= POLY_N || c < 0 || c >= POLY_N) return 0;
    if (r <= 1 && c <= 1) return 1;
    if (v1Cap && r <= 1 && c >= 2) return 1;
    if (v2Cap && r >= 2 && c <= 1) return 1;
    if (centerCap && r === 2 && c === 2) return 1;
    if (centerCap && r === 2 && c === 3 && t >= 7.55) return 1;
    if (finalCap && r === 3 && c === 2) return 1;
    if (r >= 2 && c >= 2) return 2;
    return 0;
  };

  // Subtle island drop shadow
  const [topX, topY] = tilePos(0, 0);
  const [leftX, leftY] = tilePos(POLY_N - 1, 0);
  const [rightX, rightY] = tilePos(0, POLY_N - 1);
  const [botX, botY] = tilePos(POLY_N - 1, POLY_N - 1);
  el("polygon", {
    points: `${topX},${topY + 34} ${rightX + POLY_HW - 8},${rightY + 58} ${botX},${botY + POLY_HH + 72} ${leftX - POLY_HW + 8},${leftY + 58}`,
    fill: "#1B241E", opacity: 0.11, filter: `url(#${shadowId})`,
  }, g);

  const groundLayer = el("g", {}, g);
  const borderLayer = el("g", {}, g);
  const pieceLayer = el("g", {}, g);
  const fxLayer = el("g", {}, g);

  const tileOrder: [number, number][] = [];
  for (let r = 0; r < POLY_N; r++) for (let c = 0; c < POLY_N; c++) tileOrder.push([r, c]);
  tileOrder.sort((a, b) => (a[0] + a[1]) - (b[0] + b[1]) || a[1] - b[1]);

  const edgeDirs = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  for (const [r, c] of tileOrder) {
    const [tx, ty] = tilePos(r, c);
    const isNorth = r + c <= 1 || (r === 0 && c === 2);
    drawPolySprite(polyData, groundLayer, tx, ty, isNorth ? "ground_bardur" : "ground_imperius", 0, POLY_TERRAIN_Y);

    const o = ownerOf(r, c);
    if (o > 0) {
      const bCol = o === 1 ? COL.cyan : COL.magenta;
      el("path", {
        d: `M${tx} ${ty - POLY_HH} L${tx + POLY_HW} ${ty} L${tx} ${ty + POLY_HH} L${tx - POLY_HW} ${ty} Z`,
        fill: bCol, opacity: o === 1 ? 0.14 : 0.13,
      }, borderLayer);
      edgeDirs.forEach(([dr, dc], bit) => {
        if (ownerOf(r + dr, c + dc) !== o) {
          const [bName, bx, by] = POLY_BORDERS[bit];
          drawPolySprite(polyData, borderLayer, tx, ty, bName, bx, by, { tintColor: bCol, opacity: 0.92 });
        }
      });
    }
  }

  // Subtle Polytopia move-target ring pulses right before key hops!
  const moveTargets = [
    { r: 1, c: 2, t0: 2.05, t1: 2.45, col: COL.cyan },
    { r: 2, c: 1, t0: 4.65, t1: 5.00, col: COL.cyan },
    { r: 2, c: 2, t0: 6.75, t1: 7.20, col: COL.cyan },
    { r: 3, c: 2, t0: 7.35, t1: 7.75, col: COL.cyan },
  ];
  for (const mt of moveTargets) {
    if (t >= mt.t0 && t <= mt.t1) {
      const [mx, my] = tilePos(mt.r, mt.c);
      const pulse = Math.sin(invLerp(mt.t0, mt.t1, t) * Math.PI);
      el("ellipse", {
        cx: mx, cy: my + 2, rx: (22 + pulse * 6).toFixed(1), ry: (13 + pulse * 3.5).toFixed(1),
        fill: "none", stroke: mt.col, "stroke-width": 3, opacity: (pulse * 0.85).toFixed(3),
      }, borderLayer);
    }
  }

  // Painter-sorted terrain, improvements, cities, and lively combat units!
  const drawItems: [number, () => void][] = [];

  // Terrain & live building upgrades
  const addSpriteItem = (r: number, c: number, depthOff: number, name: string, ux: number, uy: number, popAt = -1) => {
    const [tx, ty] = tilePos(r, c);
    if (popAt >= 0 && t < popAt) return;
    const sc = popAt >= 0 ? 0.4 + 0.6 * springPop(clamp((t - popAt) / 0.35)) : 1;
    drawItems.push([r + c + depthOff, () => {
      drawPolySprite(polyData, pieceLayer, tx, ty, name, ux, uy, { extraScale: sc });
    }]);
  };

  // Forests, Mountain, Resources -> Live Improvements (Lumber Hut, Farm, Mine)!
  if (t < T_LUMBER) addSpriteItem(0, 2, 0.05, "Forest_bardur", 0.0013, POLY_TERRAIN_Y + 0.7069);
  else {
    addSpriteItem(0, 2, 0.04, "Forest_bardur", 0.0013, POLY_TERRAIN_Y + 0.7069);
    addSpriteItem(0, 2, 0.08, "Lumber Hut", 0.043, 0.057, T_LUMBER);
  }
  addSpriteItem(0, 3, 0.05, "mountain_imperius", 0.0017, POLY_TERRAIN_Y + 0.0821);
  if (t >= T_MINE) addSpriteItem(0, 3, 0.09, "Mine", -0.05, 0.02, T_MINE);

  addSpriteItem(3, 0, 0.05, "Forest_imperius", 0.0013, POLY_TERRAIN_Y + 0.7069);
  addSpriteItem(2, 3, 0.05, "Forest_imperius", 0.0013, POLY_TERRAIN_Y + 0.7069);

  // Fruit on (1, 1) harvested early; Crop on (2, 0) upgraded into real Farm!
  if (t < T_HARVEST) addSpriteItem(1, 1, 0.06, "ResourceGFX_fruit_imperius", 0, 0.02);
  if (t < T_FARM) addSpriteItem(2, 0, 0.06, "ResourceGFX_crop", -0.013, 0.074);
  else addSpriteItem(2, 0, 0.07, "Farm", 0, 0.04, T_FARM);

  // Capitals (Cyan Capital upgrades from Lv.2 -> Lv.3 -> Lv.4 with CityWall!)
  const cyanCapLv = t >= T_GIANT_SPAWN ? 4 : (t >= T_FARM ? 3 : 2);
  const capPop = (t >= T_GIANT_SPAWN && t < T_GIANT_SPAWN + 0.4)
    ? 1 + 0.14 * Math.sin(((t - T_GIANT_SPAWN) / 0.4) * Math.PI)
    : 1;
  drawItems.push([0.12, () => {
    const [tx, ty] = tilePos(0, 0);
    drawPolyCity(polyData, pieceLayer, tx, ty, "imperius", {
      level: cyanCapLv, capital: true, wall: t >= T_GIANT_SPAWN, popScale: capPop, ownerColor: COL.cyan, showBar: true,
    });
  }]);
  drawItems.push([6.12, () => {
    const [tx, ty] = tilePos(3, 3);
    drawPolyCity(polyData, pieceLayer, tx, ty, "bardur", {
      level: 2, capital: true, popScale: 1, ownerColor: COL.magenta, showBar: true,
    });
  }]);

  // Neutral Tribe Villages -> Real Imperius Cities on capture!
  const addVillage = (r: number, c: number, capTime: number) => {
    const [tx, ty] = tilePos(r, c);
    const captured = t >= capTime;
    const capAge = Math.max(0, t - capTime);
    drawItems.push([r + c + 0.10, () => {
      if (captured && capAge < 2.2) {
        for (let ri = 0; ri < 2; ri++) {
          const rf = ((capAge * 0.95 + ri * 0.45) % 1);
          const rFade = (1 - rf) * (1 - easeInOutCubic(invLerp(1.3, 2.2, capAge)));
          if (rFade > 0.02) {
            el("ellipse", {
              cx: tx, cy: ty + 4, rx: (18 + rf * 44).toFixed(1), ry: (10 + rf * 26).toFixed(1),
              fill: "none", stroke: COL.cyan, "stroke-width": (3.0 * (1 - rf * 0.45)).toFixed(2), opacity: (rFade * 0.88).toFixed(3),
            }, pieceLayer);
          }
        }
      }
      if (!captured) {
        drawPolySprite(polyData, pieceLayer, tx, ty, "Tribe", -0.008, 0.037);
      } else {
        const pop = springPop(clamp(capAge / 0.35));
        drawPolyCity(polyData, pieceLayer, tx, ty, "imperius", { level: 2, capital: false, popScale: 0.55 + 0.45 * pop, ownerColor: COL.cyan, showBar: true });
      }
    }]);
  };
  addVillage(1, 2, T_CAP_V1);
  addVillage(2, 1, T_CAP_V2);

  // Helper: lively unit with multi-hop movement, melee lunges, HP transitions, spawn/KO effects!
  const addUnit = ({ kind, color, skin, spawnAt = -1, koAt = 999, hops, lunges = [], hpFn, flip = false, scale = 1.04 }: UnitMotion) => {
    if (t < spawnAt) return;
    if (t >= koAt + 0.35) return;

    let r = hops[0].r0, c = hops[0].c0, hopY = 0;
    for (const h of hops) {
      if (t < h.t0) break;
      if (t >= h.t0 && t <= h.t1) {
        const f = easeInOutCubic((t - h.t0) / (h.t1 - h.t0));
        r = lerp(h.r0, h.r1, f);
        c = lerp(h.c0, h.c1, f);
        hopY = Math.sin(f * Math.PI) * (kind === 12 ? 22 : 30);
        break;
      } else {
        r = h.r1;
        c = h.c1;
      }
    }

    // Melee lunge offset toward target tile
    let ldx = 0, ldy = 0;
    for (const lg of lunges) {
      if (t >= lg.t0 && t <= lg.t1) {
        const lf = Math.sin(invLerp(lg.t0, lg.t1, t) * Math.PI);
        const [p0x, p0y] = tilePos(r, c);
        const [p1x, p1y] = tilePos(lg.tr, lg.tc);
        ldx += (p1x - p0x) * 0.42 * lf;
        ldy += (p1y - p0y) * 0.42 * lf;
      }
    }

    const [px, py] = tilePos(r, c);
    const spawnSc = spawnAt >= 0 ? springPop(clamp((t - spawnAt) / 0.30)) : 1;
    const koProg = t >= koAt ? clamp((t - koAt) / 0.35) : 0;
    const effScale = scale * spawnSc * (1 - koProg * 0.45);
    const opacity = 1 - koProg;
    const hp = hpFn ? hpFn(t) : 10;

    drawItems.push([r + c + 0.30, () => {
      if (opacity > 0.03) {
        const idle = hopY > 0.2 ? 0 : Math.abs(Math.sin(t * 5.2 + r * 2 + c)) * 2.6;
        drawPolyUnit(polyData, pieceLayer, px, py, kind, color, skin, {
          scale: effScale, hopY: hopY + idle, dx: ldx, dy: ldy, flip, hp: koProg > 0 ? null : hp, opacity,
        });
      }
      // Crisp white KO dust-poof rings on defeat!
      if (koProg > 0 && koProg < 1) {
        const popOp = 1 - koProg;
        for (let d = 0; d < 5; d++) {
          const ang = (d / 5) * Math.PI * 2 + 0.4;
          const dist = 8 + koProg * 28;
          el("circle", {
            cx: (px + Math.cos(ang) * dist).toFixed(1),
            cy: (py - 10 + Math.sin(ang) * dist * 0.65).toFixed(1),
            r: (9 * (1 - koProg * 0.6)).toFixed(1),
            fill: "#FFFFFF", stroke: "#C9D0CC", "stroke-width": 1.6, opacity: popOp.toFixed(3),
          }, fxLayer);
        }
      }
    }]);
  };

  // === RAPID-FIRE LIVE POLYTOPIA UNITS ===
  // 1. Cyan Rider (Unit 3): rushes north-east village (1, 2), captures at 2.45s, scouts flank (1, 3)!
  addUnit({
    kind: 3, color: COL.cyan, skin: "imperius",
    hops: [
      { r0: 0, c0: 1, r1: 1, c1: 1, t0: 0.78, t1: 1.18 },
      { r0: 1, c0: 1, r1: 1, c1: 2, t0: 2.08, t1: 2.45 },
      { r0: 1, c0: 2, r1: 1, c1: 3, t0: 5.70, t1: 6.10 },
    ],
    hpFn: () => 10,
  });

  // 2. Cyan Warrior (Unit 2): takes (2, 0), strikes Magenta Warrior, captures (2, 1), pushes (3, 1)!
  addUnit({
    kind: 2, color: COL.cyan, skin: "imperius",
    hops: [
      { r0: 1, c0: 0, r1: 2, c1: 0, t0: 1.55, t1: 1.95 },
      { r0: 2, c0: 0, r1: 2, c1: 1, t0: 4.68, t1: 4.95 },
      { r0: 2, c0: 1, r1: 3, c1: 1, t0: 7.45, t1: 7.80 },
    ],
    lunges: [
      { t0: 4.42, t1: 4.68, tr: 2, tc: 1 },
      { t0: 8.05, t1: 8.32, tr: 3, tc: 2 },
    ],
    hpFn: (tt) => (tt >= 4.55 ? 8 : 10),
  });

  // 3. Cyan Archer (Unit 9): spawns at (0, 0) at 3.25s, hops to (1, 1), shifts to (1, 0), fires 2 volleys!
  addUnit({
    kind: 9, color: COL.cyan, skin: "imperius", spawnAt: 3.25,
    hops: [
      { r0: 0, c0: 0, r1: 1, c1: 1, t0: 3.55, t1: 3.92 },
      { r0: 1, c0: 1, r1: 1, c1: 0, t0: 5.45, t1: 5.80 },
    ],
    hpFn: () => 10,
  });

  // 4. TOWERING CYAN GIANT (Unit 12, HP 40!): spawns on Lv.4 Capital at 5.45s, stomps center (2, 2)!
  addUnit({
    kind: 12, color: COL.cyan, skin: "imperius", spawnAt: T_GIANT_SPAWN, scale: 1.12,
    hops: [
      { r0: 0, c0: 0, r1: 1, c1: 1, t0: 5.75, t1: 6.18 },
      { r0: 1, c0: 1, r1: 2, c1: 2, t0: 6.82, t1: 7.22 },
    ],
    lunges: [
      { t0: 7.02, t1: 7.26, tr: 2, tc: 2 },
      { t0: 8.08, t1: 8.34, tr: 3, tc: 3 },
    ],
    hpFn: (tt) => (tt >= 7.15 ? 36 : 40),
  });

  // 5. CYAN KNIGHT (Unit 4!): spawns at captured city (1, 2) at 6.45s, rapid chain-charge (2, 3) -> (3, 2)!
  addUnit({
    kind: 4, color: COL.cyan, skin: "imperius", spawnAt: T_KNIGHT_SPAWN,
    hops: [
      { r0: 1, c0: 2, r1: 2, c1: 3, t0: 6.95, t1: 7.30 },
      { r0: 2, c0: 3, r1: 3, c1: 2, t0: 7.45, t1: 7.82 },
    ],
    lunges: [{ t0: 7.72, t1: 7.98, tr: 3, tc: 3 }],
    hpFn: () => 10,
  });

  // 6. Magenta Warrior (Unit 2): rushes (2, 1) at 1.25s, gets arrowed then KO'd by Cyan Warrior at 4.55s!
  addUnit({
    kind: 2, color: COL.magenta, skin: "bardur", flip: true, koAt: T_MELEE1,
    hops: [{ r0: 2, c0: 2, r1: 2, c1: 1, t0: 1.20, t1: 1.60 }],
    hpFn: (tt) => (tt >= T_ARROW1 + 0.15 ? 5 : 10),
  });

  // 7. Magenta Defender (Unit 5, HP 15): holds center (2, 2), arrowed -> smashed by Giant at 7.15s!
  addUnit({
    kind: 5, color: COL.magenta, skin: "bardur", flip: true, koAt: T_GIANT_SMASH,
    hops: [{ r0: 3, c0: 2, r1: 2, c1: 2, t0: 2.75, t1: 3.15 }],
    hpFn: (tt) => (tt >= T_ARROW2 + 0.15 ? 11 : 15),
  });

  // 8. Magenta Swordsman (Unit 11, HP 15): spawns at (3, 3) at 6.55s, battles Knight & Giant -> KO'd at 8.20s!
  addUnit({
    kind: 11, color: COL.magenta, skin: "bardur", flip: true, spawnAt: 6.55, koAt: T_FINAL_KO,
    hops: [{ r0: 3, c0: 3, r1: 3, c1: 3, t0: 6.55, t1: 6.56 }],
    lunges: [{ t0: 6.72, t1: 6.98, tr: 2, tc: 2 }],
    hpFn: (tt) => (tt >= T_KNIGHT_HIT + 0.08 ? 8 : 15),
  });

  // Sort and execute all board items
  drawItems.sort((a, b) => a[0] - b[0]);
  for (const [, fn] of drawItems) fn();

  // === LIVELY COMBAT FX: CURVED ARCHER ARROWS, IMPACT BURSTS, FLOATING +★ & -DMG ===
  const drawArrowShot = (r0: number, c0: number, r1: number, c1: number, t0: number, t1: number) => {
    if (t < t0 || t > t1) return;
    const f = clamp((t - t0) / (t1 - t0));
    const [x0, y0] = tilePos(r0, c0);
    const [x1, y1] = tilePos(r1, c1);
    const sx = x0, sy = y0 - 24, ex = x1, ey = y1 - 20;
    const mx = (sx + ex) / 2, my = (sy + ey) / 2 - 58;
    // Trail
    const inv = 1 - f;
    const ax = inv * inv * sx + 2 * inv * f * mx + f * f * ex;
    const ay = inv * inv * sy + 2 * inv * f * my + f * f * ey;
    const ddx = 2 * inv * (mx - sx) + 2 * f * (ex - mx);
    const ddy = 2 * inv * (my - sy) + 2 * f * (ey - my);
    const ang = (Math.atan2(ddy, ddx) * 180) / Math.PI;
    // Faint flight arc
    el("path", { d: `M${sx.toFixed(1)} ${sy.toFixed(1)} Q${mx.toFixed(1)} ${my.toFixed(1)} ${ex.toFixed(1)} ${ey.toFixed(1)}`, fill: "none", stroke: COL.cyan, "stroke-width": 2, "stroke-dasharray": "4 5", opacity: 0.45 }, fxLayer);
    const ag = el("g", { transform: `translate(${ax.toFixed(1)} ${ay.toFixed(1)}) rotate(${ang.toFixed(1)})` }, fxLayer);
    el("line", { x1: -14, y1: 0, x2: 10, y2: 0, stroke: "#4E3822", "stroke-width": 3, "stroke-linecap": "round" }, ag);
    el("path", { d: "M6 -4.5 L15 0 L6 4.5 Z", fill: COL.cyan }, ag);
    el("path", { d: "M-16 -3.5 L-11 0 L-16 3.5", fill: "none", stroke: "#FFFFFF", "stroke-width": 2 }, ag);
  };
  drawArrowShot(1, 1, 2, 1, T_ARROW1 - 0.25, T_ARROW1 + 0.16);
  drawArrowShot(1, 0, 2, 2, T_ARROW2 - 0.25, T_ARROW2 + 0.16);

  // Impact bursts + floating popups (+★ economy & -dmg combat!)
  const popups = [
    { r: 1, c: 1, t0: T_HARVEST, dur: 0.85, label: "+1★", col: COL.gold },
    { r: 0, c: 2, t0: T_LUMBER, dur: 0.85, label: "+2★", col: COL.gold },
    { r: 2, c: 0, t0: T_FARM, dur: 0.85, label: "+2★", col: COL.gold },
    { r: 2, c: 1, t0: T_ARROW1 + 0.14, dur: 0.72, label: "-5", col: COL.bad, burst: COL.cyan },
    { r: 2, c: 1, t0: T_MELEE1, dur: 0.72, label: "-6", col: COL.bad, burst: COL.cyan },
    { r: 0, c: 3, t0: T_MINE, dur: 0.85, label: "+2★", col: COL.gold },
    { r: 0, c: 0, t0: T_GIANT_SPAWN, dur: 0.90, label: "GIANT!", col: COL.cyan },
    { r: 2, c: 2, t0: T_ARROW2 + 0.14, dur: 0.72, label: "-4", col: COL.bad, burst: COL.cyan },
    { r: 2, c: 2, t0: T_GIANT_SMASH, dur: 0.78, label: "-11", col: COL.bad, burst: COL.cyan },
    { r: 3, c: 3, t0: T_KNIGHT_HIT, dur: 0.72, label: "-7", col: COL.bad, burst: COL.cyan },
    { r: 3, c: 3, t0: T_FINAL_KO, dur: 0.82, label: "-8", col: COL.bad, burst: COL.cyan },
  ];
  for (const p of popups) {
    if (t < p.t0 || t > p.t0 + p.dur) continue;
    const f = clamp((t - p.t0) / p.dur);
    const [px, py] = tilePos(p.r, p.c);
    if (p.burst && f < 0.45) {
      const bf = f / 0.45;
      drawImpactBurst(fxLayer, px, py - 16, 14 + bf * 16, p.burst, 1 - bf * 0.8);
    }
    const op = f < 0.2 ? f / 0.2 : 1 - (f - 0.2) / 0.8;
    const fy = py - 28 - easeOutCubic(f) * 34;
    const label = el("text", {
      x: px.toFixed(1), y: fy.toFixed(1), "text-anchor": "middle",
      fill: p.col, stroke: "#FFFFFF", "stroke-width": 4.2, "paint-order": "stroke",
      "font-size": 26, "font-weight": 850, opacity: op.toFixed(3),
    }, fxLayer);
    label.textContent = p.label;
  }
}
