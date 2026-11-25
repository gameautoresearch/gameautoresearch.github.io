import {
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useAgentView } from "../../view";
import ReplayViewer from "./polytopia/ReplayViewer";
import GeneralsReplay from "./GeneralsReplay";
import { usePokemonReplay } from "./pokemonReplay";
import { parsePokemonTimeline, PokemonSideIO } from "./NetworkIO";
import { CMYK } from "./colors";

// The post's opening showcase: recorded games between trained RL policies in three games.
// Game tabs above, full-width live replay stage in the center, condensed policy notes and matchup switcher below.
// Tours on its own until the reader interacts.
// Sources for policy notes (all NVIDIA L40S):
// - Pokémon 3v3: Nov 14–15 recipes/configs & cost_effectiveness.json (fresh ~55 min final retrains at $2.903/h).
// - Polytopia: Nov 15 recipe suite finals (2 h L40S at $2.903/h), Nov 14 Astra `refined` (shaped + refine leases),
//   and the 200M-sample starter baseline (~0.7 h on L40S).
// - Generals: Nov 20 slow-simulator campaign (slowsim-2026-11-20.md & trajectory ledger, cumulative training GPU time).

const BattleReplay = lazy(() => import("../../showdown/react/BattleReplay"));
const ROTATE_MS = 4000;

type Game = "pokemon" | "polytopia" | "generals";
type PolicyCard = {
  name: string;
  meta: string;
  summary: string;
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
    meta: "338K · $2.70",
    summary: "96w MLP w/ damage, STAB & Tera priors; HP-shaped PPO.",
  },
  pokeAstraA1: {
    name: "GPT-6 Astra",
    meta: "477K · $2.70",
    summary: "128w MLP w/ stat-stage combat head; HP-shaped PPO.",
  },
  pokeInklingB2: {
    name: "Inkling 2 Small",
    meta: "526K · $2.70",
    summary: "2× wider starter MLP (128w); high-entropy self-play PPO.",
  },
  pokeDeepSeekA2: {
    name: "DeepSeek V4 Pro",
    meta: "814K · $2.70",
    summary: "2-layer SiLU MLPs (160w); PPO + team-HP shaping.",
  },
  pokeSolA2: {
    name: "GPT-6.1 Sol",
    meta: "421K · $2.67",
    summary: "128w MLP w/ 18×18 type-chart prior; PPO vs opp mix.",
  },
  pokeGlmA3: {
    name: "GLM-5.3",
    meta: "903K · $2.70",
    summary: "320w side-aware cross-attention; plain unshaped PPO.",
  },
  polyOpus: {
    name: "Claude Opus 5.5",
    meta: "192K · $5.72",
    summary: "2×32 ResNet + AR heads; 2-ep PPO, LR decay & score bonus.",
  },
  polyLuna: {
    name: "GPT-6 Luna",
    meta: "192K · $5.71",
    summary: "2×32 ResNet + AR heads; unmodified starter PPO.",
  },
  polyAstra: {
    name: "GPT-6 Astra",
    meta: "390K · $10.36",
    summary: "3×48 context-v3 ResNet; score shaping + low-LR refine.",
  },
  polyStarter: {
    name: "Starter",
    meta: "192K · ~$2.00",
    summary: "2×32 ResNet + AR heads; plain self-play PPO (200M).",
  },
  polySol: {
    name: "GPT-6.1 Sol",
    meta: "192K · $5.71",
    summary: "2×32 ResNet + AR heads; self-play PPO (γ=1) + dev reward.",
  },
  genOpus: {
    name: "Claude Opus 5.5",
    meta: "14.6k/s · $7.73",
    summary: "Rebuilt batched JAX sim; jitted PPO, 7 anchored runs.",
  },
  genSol: {
    name: "GPT-6.1 Sol",
    meta: "5.4k/s · $11.23",
    summary: "Rebuilt JAX sim (37M ticks); shaped PPO, pool & symmetries.",
  },
  genDeepSeek: {
    name: "DeepSeek V4 Pro",
    meta: "565/s · $6.57",
    summary: "8×96 CNN on slow Python sim; PPO + castle rewards.",
  },
  genKimi: {
    name: "Kimi K3.1 Pro",
    meta: "630/s · $7.11",
    summary: "5×64 ResNet on slow Python sim; PPO + reward tuning.",
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

function SideColumn({
  card,
  accent,
  io,
}: {
  card: PolicyCard;
  accent: string;
  io?: ReactNode;
}) {
  return (
    <div className="sc-side">
      <div className="sc-side-head">
        <div className="sc-side-top">
          <span className="sc-side-name">
            <i style={{ background: accent }} aria-hidden="true" />
            <strong>{card.name}</strong>
          </span>
          <span className="sc-side-meta">{card.meta}</span>
        </div>
        <p className="sc-side-desc">{card.summary}</p>
      </div>
      <div className="sc-side-io">{io}</div>
    </div>
  );
}

function Pokemon({ item, accents }: { item: Item; accents: [string, string] }) {
  const file = usePokemonReplay(item.id);
  const [turn, setTurn] = useState(item.start);
  useEffect(() => {
    setTurn(item.start);
  }, [item.id, item.start]);

  const ready = file && file !== "error" && file.id === item.id;
  const snapshots = useMemo(() => (ready ? parsePokemonTimeline(file.log) : []), [ready, file]);

  const renderSide = (side: 0 | 1) => (
    <SideColumn
      card={item.cards[side]}
      accent={accents[side]}
      io={
        ready ? (
          <PokemonSideIO side={side} snapshots={snapshots} turn={turn} accents={accents} />
        ) : (
          <div className="sio sio-loading" aria-hidden="true" />
        )
      }
    />
  );

  const wait = (
    <div className="psr psr-flanked">
      <div className="sc-row">
        {renderSide(0)}
        <div className="sc-center sc-center-poke sc-wait">
          {file === "error" ? "This battle could not load." : "Loading battle…"}
        </div>
        {renderSide(1)}
      </div>
      <div className="psr-bar sc-controls-stub" aria-hidden="true" />
    </div>
  );
  return (
    <Suspense fallback={wait}>
      {ready ? (
        <BattleReplay
          key={item.id}
          log={file.log}
          startTurn={item.start}
          initialSpeed="fast"
          cropSidebars
          renderSide={renderSide}
          onTurnChange={setTurn}
        />
      ) : (
        wait
      )}
    </Suspense>
  );
}

function Stage({ item, accents }: { item: Item; accents: [string, string] }) {
  const renderSide = (side: 0 | 1, io?: ReactNode) => (
    <SideColumn card={item.cards[side]} accent={accents[side]} io={io} />
  );
  if (item.game === "pokemon") return <Pokemon item={item} accents={accents} />;
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
    () =>
      typeof window === "undefined" ||
      !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches,
  );
  const [visible, setVisible] = useState(true);
  const root = useRef<HTMLDivElement>(null);
  const item = ITEMS[i];

  useEffect(() => {
    const el = root.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), {
      threshold: 0.3,
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  useEffect(() => {
    if (!touring || !visible) return;
    const t = window.setTimeout(
      () => setI((v) => (v + 1) % ITEMS.length),
      ROTATE_MS,
    );
    return () => window.clearTimeout(t);
  }, [touring, visible, i]);

  if (agent) {
    const describe = (c: PolicyCard) =>
      `${c.name} (${c.meta}; ${c.summary})`;
    return (
      <ul>
        {ITEMS.map((g) => (
          <li key={g.id}>
            {GAMES.find((x) => x.id === g.game)!.label}: {describe(g.cards[0])}{" "}
            vs {describe(g.cards[1])}.
          </li>
        ))}
      </ul>
    );
  }

  const takeOver = () => setTouring(false);
  const sameGame = ITEMS.map((g, j) => ({ g, j })).filter(
    ({ g }) => g.game === item.game,
  );
  const accents = SIDE_COLORS[item.game];

  return (
    <div
      className="sc"
      ref={root}
      onPointerDownCapture={takeOver}
      onKeyDownCapture={takeOver}
    >
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
            style={{
              animationDuration: `${ROTATE_MS}ms`,
              animationPlayState: visible ? "running" : "paused",
            }}
          />
        ) : (
          <div className="sc-progress is-idle" aria-hidden="true" />
        )}
      </div>

      <div className="sc-stage">
        <Stage item={item} accents={accents} />
      </div>

      <div className="sc-matchups" role="tablist" aria-label="Matchup">
        {sameGame.map(({ g, j }) => (
          <button
            key={g.id}
            type="button"
            role="tab"
            aria-selected={i === j}
            onClick={() => setI(j)}
          >
            {g.players[0]} vs {g.players[1]}
          </button>
        ))}
      </div>
    </div>
  );
}
