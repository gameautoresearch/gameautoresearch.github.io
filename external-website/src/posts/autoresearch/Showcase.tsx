import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { useAgentView } from "../../view";
import ReplayViewer from "./polytopia/ReplayViewer";
import GeneralsReplay from "./GeneralsReplay";
import { usePokemonReplay } from "./pokemonReplay";

// The post's opening showcase: recorded games between trained RL policies in three games.
// Game tabs above, live replay stage in the center, matchup switcher below.
// Tours on its own until the reader interacts.
// The November 20 Generals finalists play fresh demo maps, with playback starting just before city captures.

const BattleReplay = lazy(() => import("../../showdown/react/BattleReplay"));
const ROTATE_MS = 4000;

type Game = "pokemon" | "polytopia" | "generals";
type Item = {
  game: Game;
  id: string;
  players: [string, string];
  start: number;
};

const ITEMS: Item[] = [
  {
    game: "pokemon",
    id: "sol61-a1-vs-astra6-a1",
    players: ["GPT-6.1 Sol", "GPT-6 Astra"],
    start: 2,
  },
  {
    game: "polytopia",
    id: "polytopia-v3a-opus-vs-luna",
    players: ["Claude Opus 5.5", "GPT-6 Luna"],
    start: 0.55,
  },
  {
    game: "generals",
    id: "generals-20261120-opus-vs-sol",
    players: ["Claude Opus 5.5", "GPT-6.1 Sol"],
    start: 0.76358,
  },
  {
    game: "pokemon",
    id: "inkling2s-b2-vs-deepseek-v4pro-a2",
    players: ["Inkling 2 Small", "DeepSeek V4 Pro"],
    start: 6,
  },
  {
    game: "polytopia",
    id: "polytopia-v3a-astra-vs-starter",
    players: ["GPT-6 Astra", "Starter"],
    start: 0.5,
  },
  {
    game: "generals",
    id: "generals-20261120-sol-vs-deepseek",
    players: ["GPT-6.1 Sol", "DeepSeek V4 Pro"],
    start: 0.81033,
  },
  {
    game: "pokemon",
    id: "sol61-a2-vs-glm53-a3",
    players: ["GPT-6.1 Sol", "GLM-5.3"],
    start: 1,
  },
  {
    game: "polytopia",
    id: "polytopia-v3a-opus-vs-sol",
    players: ["Claude Opus 5.5", "GPT-6.1 Sol"],
    start: 0.5,
  },
  {
    game: "generals",
    id: "generals-20261120-opus-vs-kimi",
    players: ["Claude Opus 5.5", "Kimi K3.1 Pro"],
    start: 0.73141,
  },
];
const GAMES: { id: Game; label: string }[] = [
  { id: "pokemon", label: "Pokémon" },
  { id: "polytopia", label: "Polytopia" },
  { id: "generals", label: "Generals" },
];

function Pokemon({ item }: { item: Item }) {
  const file = usePokemonReplay(item.id);
  const wait = <div className="sc-wait">{file === "error" ? "This battle could not load." : "Loading battle…"}</div>;
  return (
    <Suspense fallback={wait}>
      {file && file !== "error" && file.id === item.id
        ? <BattleReplay key={item.id} log={file.log} startTurn={item.start} initialSpeed="fast" />
        : wait}
    </Suspense>
  );
}

function Stage({ item }: { item: Item }) {
  if (item.game === "pokemon") return <Pokemon item={item} />;
  if (item.game === "polytopia") return <ReplayViewer key={item.id} src={`${import.meta.env.BASE_URL}polytopia/${item.id}.json`} names={item.players} start={item.start} initialSpeed={4} compact />;
  return <GeneralsReplay key={item.id} src={`${import.meta.env.BASE_URL}generals/${item.id}.json`} names={item.players} start={item.start} initialSpeed={2} />;
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
    return (
      <ul>
        {ITEMS.map((g) => (
          <li key={g.id}>
            {GAMES.find((x) => x.id === g.game)!.label}: {g.players[0]} vs {g.players[1]}.
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
