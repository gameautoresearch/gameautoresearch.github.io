import { lazy, Suspense, useEffect, useRef, useState, type ReactNode } from "react";
import { useAgentView } from "../../view";
import ReplayViewer from "./polytopia/ReplayViewer";
import GeneralsReplay from "./GeneralsReplay";
import { usePokemonReplay } from "./pokemonReplay";
import { CMYK } from "./colors";

// The post's opening showcase: recorded games between trained RL policies in three games.
// Game tabs above, live replay stage in the center, compact policy "basketball cards" and matchup switcher below.
// Tours on its own until the reader interacts.
// Sources for card stats (all NVIDIA L40S):
// - Pokémon 3v3: Nov 14–15 recipes/configs & cost_effectiveness.json (fresh ~55 min final retrains at $2.903/h).
// - Polytopia: Nov 15 recipe suite finals (2 h L40S at $2.903/h), Nov 14 Astra `refined` (shaped + refine leases),
//   and the 200M-sample starter baseline (~0.7 h on L40S).
// - Generals: Nov 20 slow-simulator campaign (slowsim-2026-11-20.md & trajectory ledger, cumulative training GPU time).

const BattleReplay = lazy(() => import("../../showdown/react/BattleReplay"));
const ROTATE_MS = 4000;

type Game = "pokemon" | "polytopia" | "generals";
type CardStat = { label: string; value: string };
type PolicyCard = {
  name: string;
  tag: string;
  stats: [CardStat, CardStat, CardStat];
  arch: string;
  train: string;
};
type Item = {
  game: Game;
  id: string;
  players: [string, string];
  cards: [PolicyCard, PolicyCard];
  start: number;
};

const SIDE_COLORS: Record<Game, [string, string]> = {
  pokemon: [CMYK.cDeep, CMYK.m],
  polytopia: ["#2f95e0", "#eb6834"],
  generals: [CMYK.c, CMYK.m],
};

const CARDS = {
  pokeSolA1: {
    name: "GPT-6.1 Sol",
    tag: "a1 · L40S",
    stats: [
      { label: "Params", value: "338K" },
      { label: "GPU", value: "0.93 h" },
      { label: "Cost", value: "$2.70" },
    ],
    arch: "96w MLP · damage, STAB & Tera prior",
    train: "PPO (2 ep, γ=.997) + HP shaping",
  },
  pokeAstraA1: {
    name: "GPT-6 Astra",
    tag: "a1 · L40S",
    stats: [
      { label: "Params", value: "477K" },
      { label: "GPU", value: "0.93 h" },
      { label: "Cost", value: "$2.70" },
    ],
    arch: "128w MLP · stat-stage combat head",
    train: "PPO (3 ep, γ=.995) + HP shaping",
  },
  pokeInklingB2: {
    name: "Inkling 2 Small",
    tag: "b2 · L40S",
    stats: [
      { label: "Params", value: "526K" },
      { label: "GPU", value: "0.93 h" },
      { label: "Cost", value: "$2.70" },
    ],
    arch: "2× wider starter MLP (128w, 16d emb)",
    train: "Self-play PPO, 2× entropy (0.02)",
  },
  pokeDeepSeekA2: {
    name: "DeepSeek V4 Pro",
    tag: "a2 · L40S",
    stats: [
      { label: "Params", value: "814K" },
      { label: "GPU", value: "0.93 h" },
      { label: "Cost", value: "$2.70" },
    ],
    arch: "2-layer SiLU MLPs (160w, 20d emb)",
    train: "PPO + 0.5× team-HP shaping",
  },
  pokeSolA2: {
    name: "GPT-6.1 Sol",
    tag: "a2 · L40S",
    stats: [
      { label: "Params", value: "421K" },
      { label: "GPU", value: "0.92 h" },
      { label: "Cost", value: "$2.67" },
    ],
    arch: "128w MLP · 18×18 type-chart prior",
    train: "PPO + lagged & heuristic opp mix",
  },
  pokeGlmA3: {
    name: "GLM-5.3",
    tag: "a3 · L40S",
    stats: [
      { label: "Params", value: "903K" },
      { label: "GPU", value: "0.93 h" },
      { label: "Cost", value: "$2.70" },
    ],
    arch: "320w side-aware cross-attention",
    train: "Plain self-play PPO, unshaped",
  },
  polyOpus: {
    name: "Claude Opus 5.5",
    tag: "recipe · L40S",
    stats: [
      { label: "Params", value: "192K" },
      { label: "GPU", value: "2.0 h" },
      { label: "Cost", value: "$5.72" },
    ],
    arch: "2×32 ResNet + autoregressive heads",
    train: "2-ep PPO, LR decay + score bonus",
  },
  polyLuna: {
    name: "GPT-6 Luna",
    tag: "starter · L40S",
    stats: [
      { label: "Params", value: "192K" },
      { label: "GPU", value: "2.0 h" },
      { label: "Cost", value: "$5.71" },
    ],
    arch: "2×32 ResNet + autoregressive heads",
    train: "Unmodified starter PPO (LR 3e-4)",
  },
  polyAstra: {
    name: "GPT-6 Astra",
    tag: "refined · L40S",
    stats: [
      { label: "Params", value: "390K" },
      { label: "GPU", value: "2.7 h" },
      { label: "Cost", value: "$10.36" },
    ],
    arch: "3×48 context-v3 ResNet (global→tile)",
    train: "Score shaping + low-LR refine",
  },
  polyStarter: {
    name: "Starter",
    tag: "200M · L40S",
    stats: [
      { label: "Params", value: "192K" },
      { label: "GPU", value: "~0.7 h" },
      { label: "Cost", value: "~$2.00" },
    ],
    arch: "2×32 ResNet + autoregressive heads",
    train: "Plain self-play PPO (200M samples)",
  },
  polySol: {
    name: "GPT-6.1 Sol",
    tag: "recipe · L40S",
    stats: [
      { label: "Params", value: "192K" },
      { label: "GPU", value: "2.0 h" },
      { label: "Cost", value: "$5.71" },
    ],
    arch: "2×32 ResNet + autoregressive heads",
    train: "Self-play PPO, γ=1.0 + dev reward",
  },
  genOpus: {
    name: "Claude Opus 5.5",
    tag: "main7 · L40S",
    stats: [
      { label: "Speed", value: "~14.6k/s" },
      { label: "GPU", value: "2.7 h" },
      { label: "Cost", value: "$7.73" },
    ],
    arch: "Rebuilt batched JAX sim + jitted PPO",
    train: "7 chained runs, anchored self-play",
  },
  genSol: {
    name: "GPT-6.1 Sol",
    tag: "final1 · L40S",
    stats: [
      { label: "Speed", value: "~5.4k/s" },
      { label: "GPU", value: "3.3 h" },
      { label: "Cost", value: "$11.23" },
    ],
    arch: "Rebuilt batched JAX sim (37M ticks)",
    train: "PPO + shaping, pool & symmetries",
  },
  genDeepSeek: {
    name: "DeepSeek V4 Pro",
    tag: "train6 · L40S",
    stats: [
      { label: "Speed", value: "~565/s" },
      { label: "GPU", value: "2.3 h" },
      { label: "Cost", value: "$6.57" },
    ],
    arch: "8×96 CNN on supplied Python sim",
    train: "PPO + castle rewards (~5M ticks)",
  },
  genKimi: {
    name: "Kimi K3.1 Pro",
    tag: "ppo7 · L40S",
    stats: [
      { label: "Speed", value: "~630/s" },
      { label: "GPU", value: "2.4 h" },
      { label: "Cost", value: "$7.11" },
    ],
    arch: "5×64 ResNet on supplied Python sim",
    train: "PPO + reward tuning (~5M ticks)",
  },
} satisfies Record<string, PolicyCard>;

const ITEMS: Item[] = [
  {
    game: "pokemon",
    id: "sol61-a1-vs-astra6-a1",
    players: ["GPT-6.1 Sol", "GPT-6 Astra"],
    cards: [CARDS.pokeSolA1, CARDS.pokeAstraA1],
    start: 2,
  },
  {
    game: "polytopia",
    id: "polytopia-v3a-opus-vs-luna",
    players: ["Claude Opus 5.5", "GPT-6 Luna"],
    cards: [CARDS.polyOpus, CARDS.polyLuna],
    start: 0.55,
  },
  {
    game: "generals",
    id: "generals-20261120-opus-vs-sol",
    players: ["Claude Opus 5.5", "GPT-6.1 Sol"],
    cards: [CARDS.genOpus, CARDS.genSol],
    start: 0.76358,
  },
  {
    game: "pokemon",
    id: "inkling2s-b2-vs-deepseek-v4pro-a2",
    players: ["Inkling 2 Small", "DeepSeek V4 Pro"],
    cards: [CARDS.pokeInklingB2, CARDS.pokeDeepSeekA2],
    start: 6,
  },
  {
    game: "polytopia",
    id: "polytopia-v3a-astra-vs-starter",
    players: ["GPT-6 Astra", "Starter"],
    cards: [CARDS.polyAstra, CARDS.polyStarter],
    start: 0.5,
  },
  {
    game: "generals",
    id: "generals-20261120-sol-vs-deepseek",
    players: ["GPT-6.1 Sol", "DeepSeek V4 Pro"],
    cards: [CARDS.genSol, CARDS.genDeepSeek],
    start: 0.81033,
  },
  {
    game: "pokemon",
    id: "sol61-a2-vs-glm53-a3",
    players: ["GPT-6.1 Sol", "GLM-5.3"],
    cards: [CARDS.pokeSolA2, CARDS.pokeGlmA3],
    start: 1,
  },
  {
    game: "polytopia",
    id: "polytopia-v3a-opus-vs-sol",
    players: ["Claude Opus 5.5", "GPT-6.1 Sol"],
    cards: [CARDS.polyOpus, CARDS.polySol],
    start: 0.5,
  },
  {
    game: "generals",
    id: "generals-20261120-opus-vs-kimi",
    players: ["Claude Opus 5.5", "Kimi K3.1 Pro"],
    cards: [CARDS.genOpus, CARDS.genKimi],
    start: 0.73141,
  },
];
const GAMES: { id: Game; label: string }[] = [
  { id: "pokemon", label: "Pokémon" },
  { id: "polytopia", label: "Polytopia" },
  { id: "generals", label: "Generals" },
];

function SideCard({ card, accent, live }: { card: PolicyCard; accent: string; live: string }) {
  return (
    <div className="sc-card" style={{ borderTopColor: accent }}>
      <div className="sc-card-head">
        <span className="sc-card-name">
          <i style={{ background: accent }} aria-hidden="true" />
          <strong>{card.name}</strong>
        </span>
        <span className="sc-card-live">{live}</span>
      </div>
      <div className="sc-card-stats">
        {card.stats.map((s) => (
          <div key={s.label}>
            <span className="sc-stat-k">{s.label}</span>
            <strong className="sc-stat-v">{s.value}</strong>
          </div>
        ))}
      </div>
      <dl className="sc-card-notes">
        <div>
          <dt>Arch</dt>
          <dd>{card.arch}</dd>
        </div>
        <div>
          <dt>Train</dt>
          <dd>{card.train}</dd>
        </div>
      </dl>
    </div>
  );
}

function Pokemon({ item, renderSide }: { item: Item; renderSide: (side: 0 | 1, live: string) => ReactNode }) {
  const file = usePokemonReplay(item.id);
  const wait = (
    <div className="psr psr-flanked">
      <div className="sc-row">
        {renderSide(0, "3/3 Pokémon left")}
        <div className="sc-center sc-wait">{file === "error" ? "This battle could not load." : "Loading battle…"}</div>
        {renderSide(1, "3/3 Pokémon left")}
      </div>
      <div className="psr-bar sc-controls-stub" aria-hidden="true" />
    </div>
  );
  return (
    <Suspense fallback={wait}>
      {file && file !== "error" && file.id === item.id
        ? <BattleReplay key={item.id} log={file.log} startTurn={item.start} initialSpeed="fast" cropSidebars renderSide={renderSide} />
        : wait}
    </Suspense>
  );
}

function Stage({ item }: { item: Item }) {
  const accents = SIDE_COLORS[item.game];
  const renderSide = (side: 0 | 1, live: string) => (
    <SideCard card={item.cards[side]} accent={accents[side]} live={live} />
  );
  if (item.game === "pokemon") return <Pokemon item={item} renderSide={renderSide} />;
  if (item.game === "polytopia") {
    return (
      <ReplayViewer
        key={item.id}
        src={`${import.meta.env.BASE_URL}polytopia/${item.id}.json`}
        names={item.players}
        start={item.start}
        initialSpeed={4}
        compact
        renderSide={renderSide}
      />
    );
  }
  return (
    <GeneralsReplay
      key={item.id}
      src={`${import.meta.env.BASE_URL}generals/${item.id}.json`}
      names={item.players}
      start={item.start}
      initialSpeed={2}
      renderSide={renderSide}
    />
  );
}

export default function Showcase() {
  const agent = useAgentView();
  const [i, setI] = useState(0);
  const [touring, setTouring] = useState(
    () => typeof window === "undefined" || !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches,
  );
  const [visible, setVisible] = useState(true);
  const root = useRef<HTMLDivElement>(null);
  const item = ITEMS[i];

  useEffect(() => {
    const el = root.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0.3 });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  useEffect(() => {
    if (!touring || !visible) return;
    const t = window.setTimeout(() => setI((v) => (v + 1) % ITEMS.length), ROTATE_MS);
    return () => window.clearTimeout(t);
  }, [touring, visible, i]);

  if (agent) {
    const describe = (c: PolicyCard) =>
      `${c.name} (${c.stats.map((s) => `${s.label}: ${s.value}`).join(", ")}; ${c.arch}; ${c.train})`;
    return (
      <ul>
        {ITEMS.map((g) => (
          <li key={g.id}>
            {GAMES.find((x) => x.id === g.game)!.label}: {describe(g.cards[0])} vs {describe(g.cards[1])}.
          </li>
        ))}
      </ul>
    );
  }

  const takeOver = () => setTouring(false);
  const sameGame = ITEMS.map((g, j) => ({ g, j })).filter(({ g }) => g.game === item.game);

  return (
    <div className="sc" ref={root} onPointerDownCapture={takeOver} onKeyDownCapture={takeOver}>
      <div className="sc-head">
        <div className="sc-games" role="tablist" aria-label="Game">
          {GAMES.map((g) => (
            <button
              key={g.id}
              type="button"
              role="tab"
              aria-selected={item.game === g.id}
              onClick={() => setI(ITEMS.findIndex((x) => x.game === g.id))}
            >
              {g.label}
            </button>
          ))}
        </div>
        {touring ? (
          <div
            className="sc-progress"
            key={`${i}-${visible}`}
            style={{ animationDuration: `${ROTATE_MS}ms`, animationPlayState: visible ? "running" : "paused" }}
          />
        ) : (
          <div className="sc-progress is-idle" aria-hidden="true" />
        )}
      </div>

      <div className="sc-stage">
        <Stage item={item} />
      </div>

      <div className="sc-matchups" role="tablist" aria-label="Matchup">
        {sameGame.map(({ g, j }) => (
          <button key={g.id} type="button" role="tab" aria-selected={i === j} onClick={() => setI(j)}>
            {g.players[0]} vs {g.players[1]}
          </button>
        ))}
      </div>
    </div>
  );
}
