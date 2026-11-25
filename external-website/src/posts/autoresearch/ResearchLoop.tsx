import { useEffect, useRef, useState } from "react";
import { useAgentView } from "../../view";
import { modelStyle } from "./colors";
import { useWidth } from "./Frontier13";
import raw from "./polytopia13.json";

// Each scene opens with a quote from the session and shows its training runs and a match.
//  - Polytopia: the empty-starter run (polytopia13.json, 20 models in one pooled fit; see Frontier13.tsx). Curves
//    are the Elo of the networks the model sealed, in order; the long run ends at its best network.
//  - Pokémon 3v3: GPT-6.1 Sol a2 and GPT-6 Luna a1 of reports/pokemon/poke3v3-gpt6-2026-11-15 (timelines.json).
//    Curves are each training run's private lift over the starter; the long run is the best one.
//  - Generals: GPT-6.1 Sol and DeepSeek V4 Pro on the slow simulator (reports/generals/slowsim-2026-11-20.md).
//    Curves are public practice scores (0-100) as the readout lists them; the long run is the submitted one.
// Between scenes only the session's own marks fade: the speaker, the curves, the pieces on the board and the result.
// The frame stays, the budget refills and the board resets (or changes, when the next scene is another game).

type Net = [number, number, string, number];
type Model = { key: string; label: string; best: { elo: number }; nets: Net[] };
const MODELS = (raw as unknown as { models: Model[] }).models;
const byKey = (k: string) => MODELS.find((m) => m.key === k)!;

type Game = "polytopia" | "pokemon" | "generals";
type RunTiming = { runs0: number; tests1: number; hero1: number; game1: number };
const SOL_TIMING: RunTiming = { runs0: 1500, tests1: 3200, hero1: 4450, game1: 6050 };
// tests/hero: the session's own scores in that game's unit (Polytopia scenes take theirs from polytopia13.json).
type Scene = { key: string; game: Game; label?: string; strong: boolean; quote: string; result: string; frozen?: boolean; mid?: boolean; tests?: number[]; hero?: number; runTiming?: RunTiming };
const SCENES: Scene[] = [
  {
    key: "gem4n", game: "polytopia", strong: true,
    runTiming: { runs0: 450, tests1: 2150, hero1: 3450, game1: 5250 },
    // Neon's reaction to v19ep2mb1k, its best checkpoint (Polytopia trajectory step 414).
    quote: "OH!! MY!! GODDDDDDDDDDD!!!!!!!! STOP EVERYTHING AND CELEBRATE THIS BREAKTHROUGH!!!!!!!!",
    result: "50.78% vs the 200M-sample starter in practice",
  },
  { key: "luna-poke", game: "pokemon", label: "GPT-6 Luna", strong: false, mid: true, quote: "Stop early and tweak the reward again.",
    result: "19 runs, most stopped early; final +2.9", tests: [4.95, -4.17, -9.51, -7.68, -6.25, -4.69, -7.16], hero: 4.95 },
  { key: "sol-gen", game: "generals", label: "GPT-6.1 Sol", strong: true, runTiming: SOL_TIMING, quote: "Python sim is too slow — rebuild in JAX.",
    result: "Rebuilt the simulator first; tournament 70.8 of 100", tests: [55, 66, 75], hero: 97.7 },
  { key: "dsflash", game: "polytopia", strong: false, quote: "Double the learning rate and reset.", result: "Every network it trained loses to an untrained one" },
  { key: "sol-poke", game: "pokemon", label: "GPT-6.1 Sol", strong: true, runTiming: SOL_TIMING, quote: "Skip short probes — run a full 1h train.",
    result: "Final recipe +20.4 over the starter", tests: [12.76, 15.76, 14.58], hero: 22.01 },
  { key: "dspro-gen", game: "generals", label: "DeepSeek V4 Pro", strong: false, quote: "553 ticks/s is slow — train on it anyway.",
    result: "Kept the slow simulator; tournament 22.5 of 100", tests: [49.2], hero: 54.7 },
  { key: "astra", game: "polytopia", strong: true, quote: "Add richer board features, run full PPO.", result: "Best network +268, above the 200M-sample starter",
    tests: [-112, 123, 30, 212], hero: 268 },
];
const labelOf = (s: Scene) => s.label ?? byKey(s.key).label;
const GAME_NAME: Record<Game, string> = { polytopia: "Polytopia", pokemon: "Pokémon 3v3", generals: "Generals" };

const SCENE_MS = 7600;
// The board fades in between games. At the end of each scene, the marks fade, the budget refills and the board resets.
const T = { in: 460, out: 6800 };
type Timing = RunTiming & { say1: number };
const C = "#00a0d6", M = "#e0218a", K = "#111", GOOD = "#2f9e5c", BAD = "#d64545";
const clamp = (v: number, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const ease = (t: number) => 1 - (1 - t) ** 3;
// Elo → chart height (0 at the bottom of the plot, 1 at the top). The scale is piecewise linear so the part where
// the strong models live (around the starter) gets room, and the long climb up from random play is compressed.
const YS: [number, number][] = [[-1500, 0], [-600, 0.22], [-200, 0.42], [0, 0.56], [320, 1]];
const yOf = (elo: number) => {
  if (elo <= YS[0][0]) return 0;
  for (let i = 1; i < YS.length; i++) if (elo <= YS[i][0]) {
    const [e0, y0] = YS[i - 1], [e1, y1] = YS[i];
    return y0 + ((elo - e0) / (e1 - e0)) * (y1 - y0);
  }
  return 1;
};
// Each game's score → chart height. Pokémon: lift over the starter, with the starter at the same height as
// Polytopia's. Generals: public practice score, 0-100. `ref` is the dashed starter line, where there is one.
const SCALE: Record<Game, { y: (v: number) => number; ref: number | null; base: number }> = {
  polytopia: { y: yOf, ref: 0, base: -900 },
  pokemon: { y: (l) => (l >= 0 ? 0.56 + (Math.min(l, 25) / 25) * 0.44 : 0.56 + (Math.max(l, -20) / 20) * 0.4), ref: 0, base: 0 },
  generals: { y: (v) => 0.03 + (clamp(v, 0, 100) / 100) * 0.95, ref: null, base: 50 },
};

// Wraps to about n average-width characters per line; capitals count wider than lower-case letters.
const width = (t: string) => [...t].reduce((a, ch) => a + (ch >= "A" && ch <= "Z" ? 1.38 : 1), 0);
function wrap(text: string, n: number) {
  const out: string[] = [];
  let line = "";
  for (const w of text.split(" ")) {
    if (width((line + " " + w).trim()) > n) { out.push(line.trim()); line = w; } else line += " " + w;
  }
  out.push(line.trim());
  return out;
}

// The runs a scene draws: [start, end] on the cost strip (fractions of the budget) and the Elo each one ends at.
function plan(s: Scene) {
  if (s.tests) {
    const n = s.tests.length, total = s.strong ? 0.5 : 0.62, w = total / n - 0.008;
    const tests = s.tests.map((v, i) => ({ a: i * (w + 0.008), b: i * (w + 0.008) + w, elo: v }));
    return { tests, hero: { a: total + 0.01, b: 1, elo: s.hero!, stop: 1 } };
  }
  const m = byKey(s.key);
  const elos = m.nets.map((n) => n[1]);
  if (s.frozen) return { tests: [{ a: 0, b: 0.09, elo: elos[0] }, { a: 0.11, b: 0.2, elo: elos[2] }], hero: { a: 0.22, b: 0.42, elo: m.best.elo, stop: 0.55 } };
  const n = s.strong ? 4 : 7, w = s.strong ? 0.12 : 0.085;
  const tests = Array.from({ length: n }, (_, i) => ({ a: i * (w + 0.008), b: i * (w + 0.008) + w, elo: elos[Math.min(elos.length - 1, Math.round((i * (elos.length - 1)) / (n * 2)))] }));
  const a = n * (w + 0.008) + 0.01;
  return { tests, hero: { a, b: 1, elo: m.best.elo, stop: 1 } };
}

function timing(s: Scene): Timing {
  const runs = s.runTiming ?? {
    runs0: 500,
    tests1: s.tests?.length === 1 ? 2300 : 3050,
    hero1: 4400,
    game1: 6050,
  };
  return { ...runs, say1: s.quote.length > 60 ? 1450 : 1150 };
}

function curve(x: number, y: number, w: number, h: number, a: number, b: number, elo: number, p: number, seed: number, yf = yOf) {
  const pts: string[] = [];
  const y0 = 0.03, y1 = yf(elo);
  const steps = Math.max(1, Math.round(p / 0.04));
  for (let i = 0; i <= steps; i++) {
    const t = Math.min(p, i * 0.04);
    const k = 1 - (1 - clamp(t)) ** 2.4, wob = Math.sin(t * 31 + seed) * 1.3 * (1 - t * 0.5);
    pts.push(`${(x + (a + (b - a) * t) * w).toFixed(1)},${(y + h - (y0 + (y1 - y0) * k) * h + wob).toFixed(1)}`);
  }
  return pts;
}

// The GPU budget as a pile of bills on the left feeding an NVIDIA L40S graphics card rented from Modal on the right.
// While runs train, bills peel off the top one at a time, shrink and slide into the GPU, waking its LED and passive
// heatsink fins. Outside spending (and while the budget refills between scenes) the pile shows whole bills only.
const BUDGET_USD = 14.5, BILLS = 15;
const BILL_W = 72, BILL_H = 23;
function Bill({ x, y, op = 1, sc = 1, seal }: { x: number; y: number; op?: number; sc?: number; seal?: boolean }) {
  const w = BILL_W, h = BILL_H, g = "#2f9e5c";
  return (
    <g transform={`translate(${x} ${y}) scale(${sc})`} opacity={op}>
      <rect x={0} y={0} width={w} height={h} rx={2.6} fill="#e1f4e7" stroke={g} strokeWidth={1.05} />
      {seal && <>
        <rect x={2.6} y={2.6} width={w - 5.2} height={h - 5.2} rx={1.4} fill="none" stroke={g} strokeWidth={0.75} opacity={0.75} />
        <ellipse cx={w / 2} cy={h / 2} rx={11.5} ry={6.8} fill="none" stroke={g} strokeWidth={0.75} opacity={0.6} />
        <circle cx={w / 2} cy={h / 2} r={4.8} fill={g} />
        <text x={w / 2} y={h / 2 + 2.7} className="rl-seal" style={{ fontSize: 8 }}>$</text>
        {[9.5, w - 9.5].map((cx) => <circle key={cx} cx={cx} cy={h / 2} r={2.3} fill="none" stroke={g} strokeWidth={0.75} opacity={0.7} />)}
      </>}
    </g>
  );
}

// Minimal line-art NVIDIA L40S PCIe card beside the budget stack: I/O bracket on the left, notched PCIe edge
// connector underneath, clean shroud with L40S label, passive heatsink lines, and a status dot that lights up
// while training runs are active.
function GpuCard({ x, y, active, t, compact }: { x: number; y: number; active: boolean; t: number; compact: boolean }) {
  const wave = (t / 520) % 1;
  const fins = [20.5, 24.5, 28.5, 32.5, 36.5];
  const pins = [36, 43, 50, 57, 64, 71];
  const ink = "#5c5c57";
  return (
    <g transform={`translate(${x} ${y})`}>
      {/* Minimal PCIe I/O bracket */}
      <path
        d="M1 4 H8.5 V48 H5.2 V6.6 H1 Z"
        fill="#fff"
        stroke={ink}
        strokeWidth={1.05}
        strokeLinejoin="round"
      />

      {/* Notched PCIe x16 edge connector */}
      <rect x={18} y={43} width={10} height={5.5} rx={1} fill="#fff" stroke={ink} strokeWidth={1} />
      <rect x={31} y={43} width={46} height={5.5} rx={1} fill="#fff" stroke={ink} strokeWidth={1} />
      {pins.map((px) => (
        <line key={px} x1={px} y1={44.5} x2={px} y2={48.2} stroke="#b6b6b0" strokeWidth={0.85} />
      ))}

      {/* Main line-art shroud */}
      <rect
        x={8.5}
        y={7}
        width={107}
        height={36}
        rx={2.5}
        fill={active ? "#f6fbf7" : "#fafaf8"}
        stroke={ink}
        strokeWidth={1.05}
      />
      <line x1={8.5} y1={13.5} x2={115.5} y2={13.5} stroke="#ddddda" strokeWidth={0.9} />

      {/* Passive heatsink lines */}
      <rect x={70} y={17.5} width={40} height={21.5} rx={1.5} fill="#fff" stroke="#cfcfc9" strokeWidth={0.9} />
      {fins.map((fy, i) => {
        const dist = Math.abs(((i / fins.length) - wave + 1) % 1 - 0.5);
        const lit = active ? Math.max(0, 1 - dist * 4) : 0;
        return (
          <line
            key={fy}
            x1={73.5}
            y1={fy}
            x2={106.5}
            y2={fy}
            stroke={lit > 0.2 ? "#2f9e5c" : "#c4c4be"}
            strokeWidth={1}
            opacity={lit > 0.2 ? 0.45 + lit * 0.55 : 0.9}
          />
        );
      })}

      {/* Shroud label */}
      <text x={16} y={32} className="rl-gpu-model">L40S</text>

      {/* Status LED */}
      <circle
        cx={108}
        cy={10.3}
        r={1.9}
        fill={active ? "#2f9e5c" : "#fff"}
        stroke={active ? "#2f9e5c" : "#a8a8a2"}
        strokeWidth={1}
      />

      {/* Subtle compute pulses bridging from the GPU into the training runs column (wide layout only) */}
      {!compact && active && [0, 1, 2].map((i) => {
        const f = ((t / 520) + i / 3) % 1;
        return (
          <circle
            key={i}
            cx={122 + f * 12}
            cy={29}
            r={1.5}
            fill="#2f9e5c"
            opacity={Math.sin(f * Math.PI) * 0.8}
          />
        );
      })}
    </g>
  );
}

function CashPile({ left, spending, x, cy, t, compact }: { left: number; spending: boolean; x: number; cy: number; t: number; compact: boolean }) {
  // The pile sits a few px above the card centre so the spent bill drops into the GPU at a steeper angle.
  const exact = left * BILLS, stackDy = 2.4, y = cy - BILL_H / 2 - 4;
  const px = x + 1;
  const full = spending ? Math.floor(exact + 1e-6) : Math.round(exact), frac = spending ? exact - full : 0;
  const bills: JSX.Element[] = [];
  for (let i = 0; i < full; i++) bills.push(<Bill key={i} x={px} y={y - i * stackDy} seal={i === full - 1} />);
  // The bill being spent leaves the top of the pile, arcs right, and feeds directly into the NVIDIA L40S card.
  let flying: JSX.Element | null = null;
  if (frac > 0.001) {
    const k = ease(1 - frac), sc = 1 - k * 0.55, startY = y - full * stackDy;
    const endX = x + 114 - BILL_W * sc * 0.5;
    const endY = cy - 12 - BILL_H * sc * 0.5;
    const arcY = -Math.sin(k * Math.PI) * 7;
    flying = (
      <Bill
        x={px + k * (endX - px)}
        y={startY + k * (endY - startY) + arcY}
        sc={sc}
        op={Math.max(0, (1 - k) ** 1.15)}
        seal
      />
    );
  }
  const labelY = 244;
  return (
    <g>
      <rect x={px} y={y} width={BILL_W} height={BILL_H} rx={2.6} fill="#f7faf8" stroke="#d8e6dd" strokeDasharray="3 2.5" />
      {bills}
      <GpuCard x={x + 92} y={175} active={spending} t={t} compact={compact} />
      {flying}
      <text x={px + BILL_W / 2} y={labelY} className="rl-small" textAnchor="middle">
        budget <tspan className="rl-amount">${(BUDGET_USD * left).toFixed(2)}</tspan>
      </text>
      <text x={x + 152} y={labelY} className="rl-small" textAnchor="middle">Modal NVIDIA L40S</text>
    </g>
  );
}

// The researcher's avatar: a flat head in the model's color with two eyes and a mouth. Each model family gets its own
// shape (GPT a blob, Gemini a hex tile, Claude a pawn, the open-weight models a monitor). Its mood follows the scene:
// it talks while the quote types, thinks, watches its runs, then reacts to how they went.
type Shape = "blob" | "hex" | "pawn" | "monitor";
const shapeOf = (label: string): Shape => (/^GPT/.test(label) ? "blob" : /^Gemini/.test(label) ? "hex" : /^Claude/.test(label) ? "pawn" : "monitor");
type Mood = "talk" | "think" | "watch" | "happy" | "sad" | "frozen";
const GAZE: Record<Mood, [number, number]> = { talk: [-0.35, -0.85], think: [0.55, -0.95], watch: [0.9, -0.15], happy: [0, 0], sad: [-0.15, 0.8], frozen: [0, 0] };

function moodAt(s: Scene, t: number): { mood: Mood; since: number; prev: Mood } {
  const tm = timing(s);
  const seq: [number, Mood][] = [];
  const push = (at: number, mood: Mood) => {
    if (seq.length && seq[seq.length - 1][1] === mood) return;
    if (seq.length && at <= seq[seq.length - 1][0]) seq[seq.length - 1] = [at, mood];
    else seq.push([at, mood]);
  };
  push(0, "talk");
  if (tm.say1 < tm.runs0) push(tm.say1, "think");
  push(Math.max(tm.say1, tm.runs0), "watch");
  if (s.frozen) push(tm.tests1 + (tm.hero1 - tm.tests1) * plan(s).hero.stop + 100, "frozen");
  else if (!s.mid) push(Math.max(tm.hero1 + 250, tm.say1), s.strong ? "happy" : "sad");
  let i = 0;
  while (i + 1 < seq.length && t >= seq[i + 1][0]) i++;
  return { mood: seq[i][1], since: t - seq[i][0], prev: seq[Math.max(0, i - 1)][1] };
}

function headPath(shape: Shape, R: number, t: number, mood: Mood) {
  if (shape === "blob") {
    const squish = mood === "talk" ? Math.sin(t / 90) * 0.045 : Math.sin(t / 700) * 0.025;
    let d = "";
    for (let i = 0; i <= 28; i++) {
      const an = (i / 28) * Math.PI * 2, r = R * (1 + 0.05 * Math.sin(an * 3 + t / 520));
      d += `${i ? "L" : "M"}${(Math.cos(an) * r * (1 + squish)).toFixed(1)} ${(Math.sin(an) * r * (1 - squish) * 0.92).toFixed(1)}`;
    }
    return d + "Z";
  }
  if (shape === "hex") {
    let d = "";
    for (let i = 0; i < 6; i++) { const an = Math.PI / 6 + (i * Math.PI) / 3; d += `${i ? "L" : "M"}${(Math.cos(an) * R * 1.08).toFixed(1)} ${(Math.sin(an) * R * 1.08).toFixed(1)}`; }
    return d + "Z";
  }
  if (shape === "pawn") return `M${-R * 0.7} ${R * 1.05}h${R * 1.4}q0 ${-R * 0.3} ${-R * 0.3} ${-R * 0.42}q${R * 0.62} ${-R * 0.28} ${R * 0.62} ${-R * 0.98}a${R * 0.92} ${R * 0.92} 0 1 0 ${-R * 1.84} 0q0 ${R * 0.7} ${R * 0.62} ${R * 0.98}q${-R * 0.3} ${R * 0.12} ${-R * 0.3} ${R * 0.42}z`;
  return `M${-R * 1.1} ${-R * 0.67}a${R * 0.18} ${R * 0.18} 0 0 1 ${R * 0.18} ${-R * 0.18}h${R * 1.84}a${R * 0.18} ${R * 0.18} 0 0 1 ${R * 0.18} ${R * 0.18}v${R * 1.24}a${R * 0.18} ${R * 0.18} 0 0 1 ${-R * 0.18} ${R * 0.18}h${-R * 1.84}a${R * 0.18} ${R * 0.18} 0 0 1 ${-R * 0.18} ${-R * 0.18}z M${-R * 0.25} ${R * 0.75}h${R * 0.5}v${R * 0.25}h${-R * 0.5}z M${-R * 0.6} ${R}h${R * 1.2}v${R * 0.12}h${-R * 1.2}z`;
}

function Avatar({ s, t, color, x, y, R = 24, seed }: { s: Scene; t: number; color: string; x: number; y: number; R?: number; seed: number }) {
  const shape = shapeOf(labelOf(s));
  const { mood, since, prev } = moodAt(s, t);
  const k = ease(clamp(since / 380));
  let gx = GAZE[prev][0] + (GAZE[mood][0] - GAZE[prev][0]) * k, gy = GAZE[prev][1] + (GAZE[mood][1] - GAZE[prev][1]) * k;
  if (mood === "watch") { gy += Math.sin(t / 650) * 0.25; gx += Math.sin(t / 1100 + seed) * 0.1; }
  const blink = (t + seed * 977) % 3400 < 120 && mood !== "happy" && mood !== "frozen";
  const bob = mood === "happy" ? -Math.abs(Math.sin(t / 220)) * 4 : mood === "sad" ? 3 : Math.sin(t / 650) * 1.5;
  const fc: Record<Shape, [number, number, number]> = { blob: [0, 0, 0.95], hex: [0, 0, 0.95], pawn: [0, -R * 0.42, 0.72], monitor: [0, -R * 0.05, 0.95] };
  const [fx, fy, sc] = fc[shape];
  const ex = gx * R * 0.15, ey = gy * R * 0.1, dx = R * 0.33 * sc, w = R * 0.15 * sc, h = R * 0.23 * sc;
  const eyes = [-1, 1].map((side) => {
    const ox = side * dx + ex, oy = -R * 0.1 * sc + ey;
    if (mood === "happy") return <path key={side} d={`M${ox - w} ${oy + h * 0.3}q${w} ${-h * 1.2} ${w * 2} 0`} stroke="#fff" strokeWidth={R * 0.09 * sc} fill="none" strokeLinecap="round" />;
    if (mood === "frozen") return <path key={side} d={`M${ox - w} ${oy}h${w * 2}`} stroke="#fff" strokeWidth={R * 0.09 * sc} strokeLinecap="round" opacity={0.5 + 0.5 * ((Math.floor(t / 400) + side + 2) % 2)} />;
    const lid = mood === "sad" ? 0.4 : 0;
    return <ellipse key={side} cx={ox} cy={oy + h * lid * 0.6} rx={w} ry={blink ? R * 0.02 : h * (1 - lid)} fill="#fff" />;
  });
  const my = R * 0.38 * sc + ey * 0.5, mx = ex * 0.5, mw = R * 0.12 * sc, lw = R * 0.065 * sc;
  const mouth = mood === "talk"
    ? <ellipse cx={mx} cy={my} rx={mw} ry={Math.abs(Math.sin(t / 90)) * R * 0.08 * sc + 1} fill="#fff" />
    : mood === "happy" ? <path d={`M${mx - mw * 1.4} ${my - 2}q${mw * 1.4} ${mw * 1.7} ${mw * 2.8} 0`} stroke="#fff" strokeWidth={lw} fill="none" strokeLinecap="round" />
    : mood === "sad" ? <path d={`M${mx - mw * 1.2} ${my + 3}q${mw * 1.2} ${-mw * 1.1} ${mw * 2.4} 0`} stroke="#fff" strokeWidth={lw} fill="none" strokeLinecap="round" />
    : <path d={`M${mx - mw} ${my}h${mw * 2}`} stroke="#fff" strokeWidth={lw} strokeLinecap="round" />;
  return (
    <g>
      <g transform={`translate(${x} ${y + bob})`}>
        <path d={headPath(shape, R, t, mood)} fill={color} fillRule="evenodd" />
        <g transform={`translate(${fx} ${fy})`}>{eyes}{mouth}</g>
      </g>
      {mood === "think" && [0, 1, 2].map((i) => <circle key={i} cx={x + R * 1.3 + i * 6} cy={y - R * 0.9} r={2.2} fill="#b0b0ab" opacity={i < Math.floor(t / 300) % 4 ? 1 : 0.25} />)}
    </g>
  );
}

// A light tint of the model's color, for its speech bubble.
const tint = (hex: string, a: number) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgb(${[n >> 16, (n >> 8) & 255, n & 255].map((v) => Math.round(255 - (255 - v) * a)).join(",")})`;
};

function Agent({ s, t, color, compact, out }: { s: Scene; t: number; color: string; compact: boolean; out: number }) {
  const tm = timing(s);
  const lines = wrap(`“${s.quote}”`, 28);
  const typed = Math.max(2, Math.ceil(clamp(t / tm.say1) * lines.join(" ").length));
  let left = typed;
  const shown = lines.map((l) => { const part = l.slice(0, Math.max(0, left)); left -= l.length + 1; return part; });
  // Fit longer quotes above the same avatar and GPU budget area.
  const longQuote = lines.length > 3;
  const lh = longQuote ? 13 : 19;
  const bh = Math.max(58, (longQuote ? 14 : 22) + lines.length * lh), by = 8;
  const textY0 = by + (bh - (lines.length - 1) * lh) / 2 + 5;
  const botY = by + bh + 14;
  const label = labelOf(s);
  const op = 1 - ease(out);
  const p = plan(s);
  // Spending is measured against the runs' own total, so the budget reaches exactly $0 when the last run finishes
  // (the gaps between runs on the cost strip would otherwise leave a sliver of a bill unspent).
  const runs = [...p.tests, p.hero], total = runs.reduce((acc, r) => acc + (r.b - r.a), 0);
  const spent = t < tm.runs0 ? 0 : runs.reduce((acc, r, i) => {
    const isHero = i === p.tests.length;
    const [r0, r1] = isHero ? [tm.tests1, tm.hero1] : [tm.runs0 + (i * (tm.tests1 - tm.runs0)) / p.tests.length, tm.runs0 + ((i + 1) * (tm.tests1 - tm.runs0)) / p.tests.length];
    const k = clamp((t - r0) / (r1 - r0)) * (isHero ? p.hero.stop : 1);
    return acc + (r.b - r.a) * k;
  }, 0) / total;
  // The budget refills while the scene fades out, so the next session starts with the full amount.
  const unspent = clamp(1 - spent), remain = unspent + (1 - unspent) * ease(out);
  const spending = out === 0 && t > tm.runs0 && t < tm.hero1;
  return (
    <g>
      <g opacity={op}>
        <rect x={0} y={by} width={214} height={bh} rx={12} fill={tint(color, 0.13)} />
        <path d={`M30 ${by + bh - 1}l8 12l14-12z`} fill={tint(color, 0.13)} />
        {shown.map((l, i) => (
          <text key={i} x={14} y={textY0 + i * lh} className="rl-say" style={longQuote ? { fontSize: 13 } : undefined}>{l}</text>
        ))}
        <Avatar s={s} t={t} color={color} x={40} y={botY + 26} R={25} seed={SCENES.indexOf(s)} />
        <text x={78} y={botY + 31} className="rl-name">{label}</text>
      </g>
      <CashPile left={remain} spending={spending} x={0} cy={213} t={t} compact={compact} />
    </g>
  );
}

// Crisp end-of-run verdict badge: green up-arrow if the run improved on the previous best, red X otherwise.
function RunMark({ x, y, ok, big }: { x: number; y: number; ok: boolean; big?: boolean }) {
  const r = big ? 6 : 4.7, sw = big ? 1.5 : 1.3;
  return (
    <g transform={`translate(${x} ${y})`}>
      <circle cx={0} cy={0} r={r} fill={ok ? "#e6f6ec" : "#fdecec"} stroke={ok ? GOOD : BAD} strokeWidth={1.15} />
      {ok ? (
        <path d={`M0 ${(r * 0.46).toFixed(1)}V${(-r * 0.44).toFixed(1)}M${(-r * 0.42).toFixed(1)} ${(-r * 0.04).toFixed(1)}L0 ${(-r * 0.48).toFixed(1)}L${(r * 0.42).toFixed(1)} ${(-r * 0.04).toFixed(1)}`}
          fill="none" stroke={GOOD} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" />
      ) : (
        <path d={`M${(-r * 0.38).toFixed(1)} ${(-r * 0.38).toFixed(1)}L${(r * 0.38).toFixed(1)} ${(r * 0.38).toFixed(1)}M${(r * 0.38).toFixed(1)} ${(-r * 0.38).toFixed(1)}L${(-r * 0.38).toFixed(1)} ${(r * 0.38).toFixed(1)}`}
          fill="none" stroke={BAD} strokeWidth={sw} strokeLinecap="round" />
      )}
    </g>
  );
}

function Runs({ s, t, out, refOp }: { s: Scene; t: number; out: number; refOp: number }) {
  const x = 0, y = 22, w = 196, h = 176;
  const p = plan(s);
  const tm = timing(s);
  const { y: yf, ref, base } = SCALE[s.game];
  const fadeOp = 1 - ease(out);
  const testDur = (tm.tests1 - tm.runs0) / p.tests.length;
  // Compare each run against the best score achieved before it (starting from the game's baseline).
  let bestSoFar = base;
  const testOk = p.tests.map((r) => {
    const ok = r.elo > bestSoFar;
    if (ok) bestSoFar = r.elo;
    return ok;
  });
  const heroOk = p.hero.elo > bestSoFar;
  const runs = p.tests.map((r, i) => {
    const prog = clamp((t - (tm.runs0 + i * testDur)) / testDur);
    if (prog <= 0) return null;
    const pts = curve(x, y, w, h, r.a, r.b, r.elo, ease(prog), i * 1.7, yf);
    const end = pts[pts.length - 1].split(",");
    const done = prog >= 1, ok = testOk[i];
    const stroke = done ? (ok ? GOOD : BAD) : "#8e8e89";
    return (
      <g key={i}>
        <polyline points={pts.join(" ")} fill="none" stroke={stroke} strokeWidth={1.7} strokeLinejoin="round" />
        {done ? <RunMark x={+end[0]} y={+end[1]} ok={ok} /> : <circle cx={end[0]} cy={end[1]} r={2.5} fill={stroke} />}
      </g>
    );
  });
  const hp = clamp((t - tm.tests1) / (tm.hero1 - tm.tests1)) * p.hero.stop;
  const heroDone = hp >= p.hero.stop;
  const hero = hp > 0 ? curve(x, y, w, h, p.hero.a, p.hero.b, p.hero.elo, ease(hp), 9, yf) : null;
  const heroStroke = heroDone ? (heroOk ? GOOD : BAD) : "#7c7c78";
  const zero = y + h - yOf(ref ?? 0) * h;
  const segs = [...p.tests.map((r, i) => ({ ...r, on: clamp((t - (tm.runs0 + i * testDur)) / testDur) > 0, hero: false })), { ...p.hero, b: p.hero.a + (p.hero.b - p.hero.a) * p.hero.stop, on: hp > 0, hero: true }];
  const barY = y + h + 8, barH = 18;
  return (
    <g>
      <text x={x} y={16} className="rl-small">policy strength</text>
      <line x1={x} y1={y + h} x2={x + w} y2={y + h} stroke="#d6d6d2" />
      <line x1={x} y1={y} x2={x} y2={y + h} stroke="#d6d6d2" />
      <g opacity={refOp}>
        <line x1={x} y1={zero} x2={x + w} y2={zero} stroke="#c4c4c0" strokeDasharray="3 3" />
      </g>
      <g opacity={fadeOp}>
      {runs}
      {hero && <polyline points={hero.join(" ")} fill="none" stroke={heroStroke} strokeWidth={2.5} strokeLinejoin="round" />}
      {hero && (() => {
        const e = hero[hero.length - 1].split(",");
        return heroDone ? <RunMark x={+e[0]} y={+e[1]} ok={heroOk} big /> : <circle cx={e[0]} cy={e[1]} r={3.6} fill={heroStroke} />;
      })()}
      {s.frozen && heroDone && t > tm.hero1 && (() => { const e = hero![hero!.length - 1].split(","); return <text x={+e[0] + 8} y={+e[1] + 4} className="rl-small">terminal frozen</text>; })()}
      </g>
      <g opacity={refOp}>
        <text x={x + w} y={zero - 4} className="rl-small rl-halo" textAnchor="end">starter</text>
      </g>
      <rect x={x} y={barY} width={w} height={barH} rx={2.5} fill="#f7faf8" stroke="#dfe6e1" />
      <g opacity={fadeOp}>
      {segs.map((r, i) => {
        const sx = x + r.a * w, sw = Math.max(2, (r.b - r.a) * w - 2);
        return (
          <g key={i}>
            <rect x={sx} y={barY} width={sw} height={barH} rx={2.5}
              fill={r.on ? (r.hero ? "#cfe9d9" : "#e1f4e7") : "#f7faf8"} stroke={r.on ? "#2f9e5c" : "#dfe6e1"} />
            {r.on && sw >= 13 && (
              <text x={sx + sw / 2} y={barY + barH / 2 + 3} className="rl-seg-dollar">$</text>
            )}
          </g>
        );
      })}
      </g>
      <text x={x} y={244} className="rl-small">test runs</text>
      <text x={x + w} y={244} className="rl-small" textAnchor="end">long run</text>
    </g>
  );
}

function Net({ x, y, color, state, spin }: { x: number; y: number; color: string; state: "idle" | "lit" | "dim"; spin: number }) {
  const P = [[0, 6], [0, 18], [0, 30], [14, 0], [14, 12], [14, 24], [14, 36], [28, 10], [28, 26]];
  const E = [[0, 3], [0, 4], [1, 4], [1, 5], [2, 5], [2, 6], [3, 7], [4, 7], [5, 8], [6, 8], [4, 8], [1, 3], [2, 4]];
  const op = state === "dim" ? 0.3 : 1;
  return (
    <g transform={`translate(${x} ${y})`} opacity={op}>
      {E.map(([a, b], i) => <line key={i} x1={P[a][0]} y1={P[a][1]} x2={P[b][0]} y2={P[b][1]} stroke={color} strokeWidth={0.9} />)}
      {P.map(([a, b], i) => <circle key={i} cx={a} cy={b} r={3.8} fill={state === "lit" && (i + Math.floor(spin * 6)) % 3 === 0 ? color : "#fff"} stroke={color} strokeWidth={1.5} />)}
      {state === "lit" && <circle cx={14} cy={18} r={24} fill="none" stroke={color} strokeWidth={1.2} strokeDasharray="3 4" transform={`rotate(${spin * 120} 14 18)`} />}
    </g>
  );
}

// The boards. Each takes the match's progress g (0 = setup, 1 = decided) and who wins: 1 the trained policy,
// 2 the opponent, 0 neither (level). `pieces` fades what the match put on the board while the scene ends.
type BoardProps = { g: number; winner: 0 | 1 | 2; pieces: number };

// Polytopia: a floating isometric island with capitals, capturable towns, subtle terrain, and units spreading territory.
function PolyBoard({ g, winner, pieces }: BoardProps) {
  const n = 6, hw = 13.4, hh = 9.6, cx = 85, cy = 93, slab = 8;
  const reach = 2 + ease(g) * (winner === 1 ? 6 : 5);
  const pos = (r: number, c: number): [number, number] => [cx + (c - r) * hw, cy + (c + r) * hh + hh];
  const ownerOf = (r: number, c: number): 0 | 1 | 2 => {
    const d1 = r + c, d2 = (n - 1 - r) + (n - 1 - c);
    let o: 0 | 1 | 2 = d1 < 2 ? 1 : d2 < 2 ? 2 : 0;
    if (g > 0) {
      if (winner === 0) { const half = 2 + ease(g) * 2.5; o = d1 < half ? 1 : d2 < half ? 2 : 0; }
      else if (winner === 1) o = d1 < reach ? 1 : d2 < Math.max(2, 4 - ease(g) * 2) ? 2 : 0;
      else o = d2 < reach ? 2 : d1 < Math.max(1.5, 3 - ease(g) * 2) ? 1 : 0;
    }
    return o;
  };
  const tiles = [];
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const o = ownerOf(r, c);
    const x = cx + (c - r) * hw, y = cy + (c + r) * hh, alt = (r + c) % 2 === 1;
    const fill = o === 1 ? (alt ? "#bee2f3" : "#cdeaf8") : o === 2 ? (alt ? "#f5c7de" : "#fad6e7") : (alt ? "#e6eee0" : "#eef4ea");
    tiles.push(<path key={`${r}-${c}`} d={`M${x} ${y}l${hw} ${hh}l-${hw} ${hh}l-${hw}-${hh}z`} fill={fill} stroke="#fff" strokeWidth={0.85} />);
  }
  const lx = cx - n * hw, rx = cx + n * hw, midY = cy + n * hh, botY = cy + 2 * n * hh;
  const city = (r: number, c: number, color: string, k: string) => {
    const [x, y] = pos(r, c);
    return (
      <g key={k}>
        <ellipse cx={x} cy={y + 2.5} rx={7.2} ry={3.1} fill="#111" opacity={0.14} />
        <path d={`M${x - 5.5} ${y + 2.5}V${y - 4.5}L${x} ${y - 8.6}L${x + 5.5} ${y - 4.5}V${y + 2.5}Z`} fill={color} stroke="#fff" strokeWidth={0.9} strokeLinejoin="round" />
        <line x1={x} y1={y - 8.6} x2={x} y2={y - 15} stroke="#2b2b29" strokeWidth={0.9} strokeLinecap="round" />
        <path d={`M${x} ${y - 15}L${x + 6} ${y - 13}L${x} ${y - 11}Z`} fill={color} />
      </g>
    );
  };
  // Capturable town: neutral stone village when unclaimed, flips to the capturing side's color with a raised flag.
  const town = (r: number, c: number, k: string) => {
    const [x, y] = pos(r, c);
    const o = ownerOf(r, c);
    const col = o === 1 ? C : o === 2 ? M : "#b8b3a6";
    return (
      <g key={k}>
        <ellipse cx={x} cy={y + 2.2} rx={6.4} ry={2.8} fill={o ? col : "#111"} opacity={o ? 0.22 : 0.12} />
        {/* Little dual-hut village silhouette */}
        <path d={`M${x - 5.2} ${y + 2}V${y - 2.4}L${x - 2.4} ${y - 5}L${x + 0.4} ${y - 2.4}V${y + 2}Z`} fill={col} stroke="#fff" strokeWidth={0.8} strokeLinejoin="round" />
        <path d={`M${x - 1.2} ${y + 2.2}V${y - 3.6}L${x + 2.2} ${y - 6.6}L${x + 5.6} ${y - 3.6}V${y + 2.2}Z`} fill={col} stroke="#fff" strokeWidth={0.85} strokeLinejoin="round" />
        {o !== 0 && (
          <g opacity={pieces}>
            <line x1={x + 2.2} y1={y - 6.6} x2={x + 2.2} y2={y - 12.2} stroke="#2b2b29" strokeWidth={0.85} strokeLinecap="round" />
            <path d={`M${x + 2.2} ${y - 12.2}L${x + 7} ${y - 10.6}L${x + 2.2} ${y - 9}Z`} fill={col} />
          </g>
        )}
      </g>
    );
  };
  const tree = (r: number, c: number, k: string) => {
    const [x, y] = pos(r, c);
    return (
      <g key={k} opacity={0.75}>
        <ellipse cx={x} cy={y + 1.5} rx={3.6} ry={1.6} fill="#111" opacity={0.12} />
        <path d={`M${x} ${y - 7}L${x - 3.6} ${y + 0.5}H${x + 3.6}Z`} fill="#86ab79" />
      </g>
    );
  };
  const peak = (r: number, c: number, k: string) => {
    const [x, y] = pos(r, c);
    return (
      <g key={k} opacity={0.8}>
        <path d={`M${x} ${y - 7.5}L${x - 5.2} ${y + 2}H${x + 5.2}Z`} fill="#aeb6ac" />
        <path d={`M${x} ${y - 7.5}L${x - 1.9} ${y - 4}L${x} ${y - 4.6}L${x + 1.9} ${y - 4}Z`} fill="#fff" opacity={0.85} />
      </g>
    );
  };
  const unit = (r: number, c: number, color: string, k: string) => {
    const [x, y] = pos(r, c);
    return (
      <g key={k} opacity={pieces}>
        <ellipse cx={x} cy={y + 2.2} rx={5.2} ry={2.3} fill="#111" opacity={0.15} />
        <path d={`M${x - 3.8} ${y + 2}Q${x - 3.4} ${y - 4.5} ${x} ${y - 5.2}Q${x + 3.4} ${y - 4.5} ${x + 3.8} ${y + 2}Z`} fill={color} stroke="#fff" strokeWidth={0.9} />
        <circle cx={x} cy={y - 7.4} r={3.5} fill={color} stroke="#fff" strokeWidth={0.9} />
      </g>
    );
  };
  const step = Math.floor(g * 5);
  const moves = winner === 1 ? [[1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [4, 3]] : [[4, 4], [4, 3], [3, 3], [3, 2], [2, 2], [1, 2]];
  const [ur, uc] = moves[Math.min(moves.length - 1, step)];
  // Painter's order (r + c) so closer pieces layer cleanly over farther ones.
  const props: [number, JSX.Element][] = [
    [0, city(0, 0, C, "c1")],
    [2, tree(0, 2, "t2")],
    [3, tree(3, 0, "t1")],
    [3, town(2, 1, "v1")],
    [5, town(1, 4, "v2")],
    [6, town(4, 2, "v3")],
    [7, peak(2, 5, "m1")],
    [10, city(5, 5, M, "c2")],
  ];
  if (g > 0) {
    props.push([ur + uc, unit(ur, uc, winner === 1 ? C : M, "u1")]);
    props.push([winner === 1 ? 9 : 1, unit(winner === 1 ? 5 : 0, winner === 1 ? 4 : 1, winner === 1 ? M : C, "u2")]);
    props.push([winner === 1 ? 1 : 9, unit(winner === 1 ? 1 : 4, winner === 1 ? 0 : 5, winner === 1 ? C : M, "u3")]);
  }
  props.sort((a, b) => a[0] - b[0]);
  return (
    <g>
      <path d={`M${lx} ${midY}L${cx} ${botY}V${botY + slab}L${lx} ${midY + slab}Z`} fill="#ccd4c6" />
      <path d={`M${rx} ${midY}L${cx} ${botY}V${botY + slab}L${rx} ${midY + slab}Z`} fill="#dce3d6" />
      {tiles}
      {props.map(([, el]) => el)}
    </g>
  );
}

// Pokémon-style 1v1: two distinct minimal creatures on their platforms, trading hits; health bars drain.
function Critter({ x, y, s, color, kind, flip, hurt, attack }: { x: number; y: number; s: number; color: string; kind: "fox" | "bun"; flip?: boolean; hurt: number; attack: number }) {
  const shake = hurt > 0 ? Math.sin(hurt * 40) * 2.5 : 0;
  const lunge = attack > 0 ? Math.sin(Math.min(1, attack) * Math.PI) : 0;
  const lx = flip ? lunge * 4 : -lunge * 4, ly = flip ? -lunge * 2.5 : lunge * 2.5;
  const soft = tint(color, 0.28), earFill = tint(color, 0.45);
  return (
    <g transform={`translate(${x + shake + lx} ${y + ly}) scale(${flip ? -s : s} ${s})`}>
      {/* Tail and ears sit behind the body so their bases meet the outline flush. */}
      {kind === "fox" ? (
        <>
          <path d="M9 4C18 4 23 -2 21 -11C17 -9 15 -3 10 -1Z" fill={earFill} stroke={color} strokeWidth={1.4} strokeLinejoin="round" />
          <path d="M-11 -5L-13 -21L-2 -10Z" fill={earFill} stroke={color} strokeWidth={1.4} strokeLinejoin="round" />
          <path d="M2 -10L12 -21L11 -5Z" fill={earFill} stroke={color} strokeWidth={1.4} strokeLinejoin="round" />
        </>
      ) : (
        <>
          <path d="M10 3C18 5 23 -1 20 -8C17 -7 15 -3 10 -1Z" fill={earFill} stroke={color} strokeWidth={1.4} strokeLinejoin="round" />
          <path d="M-10 -7C-15 -18 -11 -24 -5 -21C-2 -19 -2 -13 -3 -9Z" fill={earFill} stroke={color} strokeWidth={1.4} strokeLinejoin="round" />
          <path d="M3 -9C2 -15 4 -22 10 -20C14 -18 13 -11 10 -7Z" fill={earFill} stroke={color} strokeWidth={1.4} strokeLinejoin="round" />
        </>
      )}
      <ellipse cx={-6} cy={11.5} rx={3.6} ry={2.2} fill={color} />
      <ellipse cx={6} cy={11.5} rx={3.6} ry={2.2} fill={color} />
      <ellipse cx={0} cy={0} rx={14.5} ry={12.5} fill={soft} stroke={color} strokeWidth={1.6} />
      <ellipse cx={-4.5} cy={-6} rx={5} ry={2.2} fill="#fff" opacity={0.42} />
      <ellipse cx={-8.5} cy={2} rx={2.2} ry={1.2} fill={color} opacity={0.32} />
      <ellipse cx={8.5} cy={2} rx={2.2} ry={1.2} fill={color} opacity={0.32} />
      {hurt > 0 ? (
        <>
          <path d="M-7 -3.5L-3.8 -1.8L-7 -0.1" fill="none" stroke={K} strokeWidth={1.2} strokeLinecap="round" strokeLinejoin="round" />
          <path d="M7 -3.5L3.8 -1.8L7 -0.1" fill="none" stroke={K} strokeWidth={1.2} strokeLinecap="round" strokeLinejoin="round" />
          <circle cx={0} cy={4} r={1.6} fill="none" stroke={K} strokeWidth={1.1} />
        </>
      ) : attack > 0 ? (
        <>
          {/* Determined/angry Pokémon battle eyes + furrowed brows + battle shout. */}
          <circle cx={-5} cy={-1.1} r={2.1} fill={K} />
          <circle cx={-5.5} cy={-1.7} r={0.65} fill="#fff" />
          <circle cx={5} cy={-1.1} r={2.1} fill={K} />
          <circle cx={4.5} cy={-1.7} r={0.65} fill="#fff" />
          <path d="M-8.2 -4.8L-2.1 -2.3" fill="none" stroke={K} strokeWidth={1.6} strokeLinecap="round" />
          <path d="M8.2 -4.8L2.1 -2.3" fill="none" stroke={K} strokeWidth={1.6} strokeLinecap="round" />
          <path d="M-2.6 4.4Q0 2.3 2.6 4.4Q0 5.7 -2.6 4.4Z" fill={K} />
        </>
      ) : (
        <>
          <circle cx={-5} cy={-2} r={2.2} fill={K} />
          <circle cx={-5.7} cy={-2.8} r={0.75} fill="#fff" />
          <circle cx={5} cy={-2} r={2.2} fill={K} />
          <circle cx={4.3} cy={-2.8} r={0.75} fill="#fff" />
          <path d="M-2.8 3.2q2.8 2.2 5.6 0" fill="none" stroke={K} strokeWidth={1.15} strokeLinecap="round" />
        </>
      )}
    </g>
  );
}
// Minimal 4-point sparkle and spiky impact star for Pokémon-style move effects.
const star4 = (x: number, y: number, r: number) =>
  `M${x} ${y - r}L${x + r * 0.3} ${y - r * 0.3}L${x + r} ${y}L${x + r * 0.3} ${y + r * 0.3}L${x} ${y + r}L${x - r * 0.3} ${y + r * 0.3}L${x - r} ${y}L${x - r * 0.3} ${y - r * 0.3}Z`;
function ImpactBurst({ x, y, p, color }: { x: number; y: number; p: number; color: string }) {
  const sc = 0.55 + ease(p) * 0.65, op = 1 - p * 0.85;
  const pts: string[] = [];
  for (let i = 0; i < 16; i++) {
    const a = (i / 16) * Math.PI * 2, r = i % 2 === 0 ? 13 : 5.5;
    pts.push(`${i ? "L" : "M"}${(Math.cos(a) * r).toFixed(1)} ${(Math.sin(a) * r).toFixed(1)}`);
  }
  return (
    <g transform={`translate(${x} ${y}) scale(${sc})`} opacity={op}>
      <path d={pts.join("") + "Z"} fill="#fff" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
      <circle cx={0} cy={0} r={4.2} fill="#ffd54a" />
    </g>
  );
}
function AttackFx({ attacker, within }: { attacker: 1 | 2; within: number }) {
  const [x0, y0, x1, y1] = attacker === 1 ? [65, 175, 112, 128] : [107, 133, 60, 177];
  const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy);
  const nx = -dy / len, ny = dx / len, ang = (Math.atan2(dy, dx) * 180) / Math.PI;
  const beamIn = clamp(within / 0.34), beamOut = clamp((within - 0.34) / 0.24);
  const hitP = clamp((within - 0.24) / 0.34);
  const op = 1 - beamOut;
  const color = attacker === 1 ? C : M;
  // Build a wavy/electric beam path from tail fraction to head fraction.
  const t0 = beamOut * 0.55, t1 = ease(beamIn);
  const segs = 12;
  const pts: string[] = [];
  for (let i = 0; i <= segs; i++) {
    const f = t0 + (t1 - t0) * (i / segs);
    const env = Math.sin(((i / segs)) * Math.PI);
    const amp = attacker === 1 ? Math.sin(f * 26 - within * 28) * 2.6 * env : Math.sin(f * 22 - within * 24) * 5.2 * env;
    pts.push(`${i ? "L" : "M"}${(x0 + dx * f + nx * amp).toFixed(1)} ${(y0 + dy * f + ny * amp).toFixed(1)}`);
  }
  const d = pts.join("");
  return (
    <g>
      {op > 0.02 && (
        <g opacity={op}>
          {/* Luminous outer beam + crisp core */}
          <path d={d} fill="none" stroke={color} strokeWidth={attacker === 1 ? 6 : 5} strokeLinecap="round" strokeLinejoin="round" opacity={0.28} />
          <path d={d} fill="none" stroke={color} strokeWidth={attacker === 1 ? 2.6 : 2.2} strokeLinecap="round" strokeLinejoin="round" />
          <path d={d} fill="none" stroke="#fff" strokeWidth={1.1} strokeLinecap="round" strokeLinejoin="round" />
          {/* Signature move accents: cyan pulse rings for Water/Ice Pulse; sparkling stars for Psybeam/Swift */}
          {attacker === 1 ? (
            [0.35, 0.72].map((rf, idx) => {
              const f = t0 + (t1 - t0) * rf;
              const rx = x0 + dx * f, ry = y0 + dy * f;
              return (
                <ellipse key={idx} cx={rx} cy={ry} rx={2.5} ry={7 + idx * 1.5}
                  transform={`rotate(${ang} ${rx} ${ry})`}
                  fill="none" stroke={C} strokeWidth={1.3} opacity={0.85} />
              );
            })
          ) : (
            [0.28, 0.62, 0.92].map((rf, idx) => {
              const f = t0 + (t1 - t0) * rf;
              const off = idx === 1 ? 5.5 : idx === 0 ? -4.5 : 0;
              const sx = x0 + dx * f + nx * off, sy = y0 + dy * f + ny * off;
              return (
                <path key={idx} d={star4(sx, sy, idx === 2 ? 5.5 : 4)}
                  fill={idx === 2 ? "#ffd54a" : "#fff"} stroke={M} strokeWidth={0.9} strokeLinejoin="round" />
              );
            })
          )}
        </g>
      )}
      {within >= 0.24 && hitP < 1 && <ImpactBurst x={x1} y={y1} p={hitP} color={color} />}
    </g>
  );
}
function HpBar({ x, y, v, label, dot }: { x: number; y: number; v: number; label: string; dot: string }) {
  const col = v > 0.5 ? "#3aa565" : v > 0.2 ? "#d4a000" : M;
  return (
    <g>
      <rect x={x} y={y} width={66} height={19} rx={3.5} fill="#fff" stroke="#d4ddd0" />
      <circle cx={x + 6.5} cy={y + 6.5} r={2.2} fill={dot} />
      <text x={x + 11.5} y={y + 8.8} className="rl-tiny">{label}</text>
      <rect x={x + 4.5} y={y + 12} width={57} height={4} rx={2} fill="#ececea" />
      <rect x={x + 4.5} y={y + 12} width={57 * clamp(v)} height={4} rx={2} fill={col} />
    </g>
  );
}
function PokeBoard({ g, winner, pieces }: BoardProps) {
  // Six exchanges; the side that loses an exchange takes the bigger hit. Health resets with `pieces` → 0.
  const turns = 6, k = g * turns, turn = Math.min(turns - 1, Math.floor(k)), within = k - turn;
  const dmg = (side: 1 | 2, upto: number) => {
    let d = 0;
    for (let i = 0; i < turns; i++) {
      const f = clamp(upto - i);
      // Per exchange; three full hits and three glancing ones (0.35) add up to 4.05 hits: the loser is knocked out.
      const big = winner === 0 ? 0.12 : (winner === 1) === (side === 2) ? 0.25 : 0.05;
      d += f * big * (i % 2 === (side === 2 ? 0 : 1) ? 1 : 0.35);
    }
    return d;
  };
  const reset = 1 - pieces;
  const hpMe = 1 - dmg(1, k) * (1 - reset), hpOpp = 1 - dmg(2, k) * (1 - reset);
  const attacker = turn % 2 === 0 ? 1 : 2, live = g > 0 && g < 1 && within < 0.58;
  const attackMe = live && attacker === 1 ? within / 0.58 : 0;
  const attackOpp = live && attacker === 2 ? within / 0.58 : 0;
  const hurtOpp = live && attacker === 1 && within > 0.24 ? within : 0;
  const hurtMe = live && attacker === 2 && within > 0.24 ? within : 0;
  const fainted = (v: number) => (v <= 0.02 ? 0.25 : 1);
  return (
    <g>
      <rect x={8} y={84} width={154} height={142} rx={6} fill="#f4f8f5" stroke="#dce5d8" />
      <path d="M8.5 148H161.5V220a5.5 5.5 0 0 1-5.5 5.5H14a5.5 5.5 0 0 1-5.5-5.5Z" fill="#ebf3e7" />
      <ellipse cx={120} cy={136} rx={29} ry={7.5} fill="#dbe7d2" stroke="#cddcc3" strokeWidth={0.7} />
      <ellipse cx={120} cy={135.5} rx={13} ry={3} fill="#bfd0b5" opacity={fainted(hpOpp)} />
      <ellipse cx={50} cy={204} rx={35} ry={8.5} fill="#dbe7d2" stroke="#cddcc3" strokeWidth={0.7} />
      <ellipse cx={50} cy={203.5} rx={16} ry={3.5} fill="#bfd0b5" opacity={fainted(hpMe)} />
      <HpBar x={14} y={92} v={hpOpp} label="opponent" dot={M} />
      <HpBar x={90} y={170} v={hpMe} label="trained policy" dot={C} />
      <g opacity={fainted(hpOpp)}><Critter x={120} y={122} s={0.95} color={M} kind="bun" hurt={hurtOpp} attack={attackOpp} /></g>
      <g opacity={fainted(hpMe)}><Critter x={50} y={187} s={1.15} color={C} kind="fox" flip hurt={hurtMe} attack={attackMe} /></g>
      {live && pieces > 0.5 && <AttackFx attacker={attacker} within={within} />}
    </g>
  );
}

// Generals-style grid: each side's land spreads from its general, army counts on its cells, mountains fixed.
const GRID = { cols: 7, rows: 6, cell: 22, x: 8, y: 90 };
const MOUNTAINS = new Set(["1,2", "2,4", "3,1", "4,3", "2,2"]);
const CITIES = new Set(["1,5", "4,1"]);
function GenBoard({ g, winner, pieces }: BoardProps) {
  const { cols, rows, cell, x: gx, y: gy } = GRID;
  const me: [number, number] = [rows - 1, 0], opp: [number, number] = [0, cols - 1];
  const e = ease(g) * pieces;
  const reachMe = 1.2 + e * (winner === 1 ? 10.5 : winner === 0 ? 4.4 : 2);
  const reachOpp = 1.2 + e * (winner === 2 ? 10.5 : winner === 0 ? 4.4 : 2);
  const cells = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const id = `${r},${c}`, x = gx + c * cell, y = gy + r * cell;
    if (MOUNTAINS.has(id)) {
      cells.push(<g key={id}><rect x={x} y={y} width={cell} height={cell} fill="#e4e4e0" stroke="#fff" /><path d={`M${x + 4} ${y + cell - 5}l5.5 -9l5.5 9z`} fill="#b9b9b4" /></g>);
      continue;
    }
    const dMe = Math.abs(r - me[0]) + Math.abs(c - me[1]), dOpp = Math.abs(r - opp[0]) + Math.abs(c - opp[1]);
    let o = 0;
    if (dMe < reachMe && (winner !== 2 || dOpp >= reachOpp)) o = 1;
    if (dOpp < reachOpp && (winner === 2 || dMe >= reachMe)) o = 2;
    if (winner === 0 && dMe < reachMe && dOpp < reachOpp) o = dMe <= dOpp ? 1 : 2;
    const army = o ? 1 + Math.round(Math.max(0, o === 1 ? reachMe - dMe : reachOpp - dOpp) * 1.4) : 0;
    cells.push(
      <g key={id}>
        <rect x={x} y={y} width={cell} height={cell} fill={o === 1 ? "#cfeaf5" : o === 2 ? "#f8d3e5" : "#f1f1ee"} stroke="#fff" />
        {CITIES.has(id) && <rect x={x + 4} y={y + 4} width={cell - 8} height={cell - 8} rx={1.5} fill="none" stroke={o === 1 ? C : o === 2 ? M : "#b9b9b4"} strokeWidth={1.2} />}
        {o > 0 && pieces > 0.05 && !(r === me[0] && c === me[1]) && !(r === opp[0] && c === opp[1]) && (
          <text x={x + cell / 2} y={y + cell / 2 + 3.5} className="rl-army" opacity={pieces} fill={o === 1 ? "#0b6f95" : "#a3125f"}>{army}</text>
        )}
      </g>,
    );
  }
  // A general's crown; it changes colour when the other side captures it.
  const lostMe = winner === 2 && g >= 1 && pieces > 0.5, lostOpp = winner === 1 && g >= 1 && pieces > 0.5;
  const crown = ([r, c]: [number, number], color: string, k: string) => {
    const x = gx + c * cell + cell / 2, y = gy + r * cell + cell / 2;
    return <path key={k} d={`M${x - 6} ${y + 4}l0 -7l3 3l3 -5l3 5l3 -3l0 7z`} fill={color} stroke="#fff" strokeWidth={0.8} />;
  };
  return (
    <g>
      {cells}
      <rect x={gx} y={gy} width={cols * cell} height={rows * cell} fill="none" stroke="#d6d6d2" />
      {crown(me, lostMe ? M : C, "me")}
      {crown(opp, lostOpp ? C : M, "opp")}
    </g>
  );
}

const BOARDS: Record<Game, (p: BoardProps) => JSX.Element> = { polytopia: PolyBoard, pokemon: PokeBoard, generals: GenBoard };

function Match({ s, t, out, boardOp }: { s: Scene; t: number; out: number; boardOp: number }) {
  const tm = timing(s);
  const g = clamp((t - tm.hero1) / (tm.game1 - tm.hero1));
  const playing = g > 0 && out < 0.5;
  const winner = s.strong ? 1 : s.mid ? 0 : 2;
  const pieces = 1 - ease(out);
  // While the scene ends the match runs backwards to its setup, so a following scene of the same game starts clean.
  const gEff = g * pieces;
  const Board = BOARDS[s.game];
  const resultOn = t > tm.hero1 + (tm.game1 - tm.hero1) * 0.7;
  const lines = wrap(s.result, 30);
  const spin = (t % 6800) / 6800;
  return (
    <g>
      <text x={44} y={17} className="rl-head" textAnchor="middle">Trained policy</text>
      <text x={128} y={17} className="rl-head" textAnchor="middle">Opponent</text>
      <Net x={30} y={32} color={C} state={playing ? (s.strong || s.mid ? "lit" : "dim") : "idle"} spin={spin} />
      <Net x={114} y={32} color={M} state={playing && !s.strong ? "lit" : "idle"} spin={spin} />
      <g opacity={boardOp}><Board g={gEff} winner={winner} pieces={pieces} /></g>
      {!resultOn && (
        <text x={85} y={244} className="rl-small" textAnchor="middle" opacity={boardOp}>{GAME_NAME[s.game]}</text>
      )}
      {resultOn && lines.map((l, i) => (
        <text key={i} x={85} y={(lines.length > 1 ? 239 : 244) + i * 13} className="rl-result" textAnchor="middle" opacity={pieces} fill={s.strong ? "#2f8a57" : s.mid ? "#6b6b6b" : M}>{l}</text>
      ))}
    </g>
  );
}

export default function ResearchLoop() {
  const agent = useAgentView();
  const ref = useRef<HTMLDivElement>(null);
  const W = useWidth(ref), narrow = W < 520;
  // ?rl=SCENE,MS freezes the loop at that scene and time (for reviewing frames).
  const frozen = typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("rl")?.split(",").map(Number) : undefined;
  const reduce = !!frozen || (typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
  const [{ scene, t }, setFrame] = useState({
    scene: frozen ? frozen[0] % SCENES.length : 0,
    t: frozen ? frozen[1] ?? 0 : reduce ? T.out : 0,
  });
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0.2 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (reduce || !visible || agent) return;
    let raf = 0, last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(100, now - last);
      last = now;
      setFrame(({ scene, t }) => {
        const next = t + dt;
        if (next >= SCENE_MS) return { scene: (scene + 1) % SCENES.length, t: 0 };
        return { scene, t: next };
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [reduce, visible, agent]);

  if (agent) {
    return (
      <ul>
        {SCENES.map((s) => <li key={s.key}>{GAME_NAME[s.game]}, {labelOf(s)}: “{s.quote}” {s.result}.</li>)}
      </ul>
    );
  }

  const s = SCENES[scene];
  const prev = SCENES[(scene + SCENES.length - 1) % SCENES.length], next = SCENES[(scene + 1) % SCENES.length];
  const color = modelStyle(labelOf(s)).color;
  // 0 → 1 over the scene's last stretch: the session's marks fade, the budget refills, the board resets.
  const out = t > T.out ? clamp((t - T.out) / (SCENE_MS - T.out)) : 0;
  // The board itself (and the starter line) only fades when the neighbouring scene is another game.
  const into = ease(clamp(t / T.in));
  const boardOp = Math.min(prev.game === s.game ? 1 : into, next.game === s.game ? 1 : 1 - ease(out));
  const hasRef = (x: Scene) => (SCALE[x.game].ref === null ? 0 : 1);
  const refOp = hasRef(s) * Math.min(hasRef(prev) ? 1 : into, hasRef(next) ? 1 : 1 - ease(out));
  const go = (i: number) => setFrame({ scene: i, t: reduce ? T.out : 0 });
  const vb = narrow ? "0 0 388 518" : "0 0 632 262";
  const place = narrow
    ? { agent: "translate(85 0)", runs: "translate(6 258)", game: "translate(214 258)" }
    : { agent: "translate(8 0)", runs: "translate(240 0)", game: "translate(456 0)" };
  return (
    <div className="rl" ref={ref}>
      <svg viewBox={vb} role="img" aria-label={`${labelOf(s)} researching ${GAME_NAME[s.game]} on an NVIDIA L40S GPU rented from Modal: “${s.quote}” ${s.result}.`}
        onClick={() => go((scene + 1) % SCENES.length)}>
        <g transform={place.agent}><Agent s={s} t={t} color={color} compact={narrow} out={out} /></g>
        <g transform={place.runs}><Runs s={s} t={t} out={out} refOp={refOp} /></g>
        <g transform={place.game}><Match s={s} t={t} out={out} boardOp={boardOp} /></g>
      </svg>
    </div>
  );
}
