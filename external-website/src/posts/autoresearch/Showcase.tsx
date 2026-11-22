import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { useAgentView } from "../../view";
import ReplayViewer from "./polytopia/ReplayViewer";
import GeneralsReplay from "./GeneralsReplay";
import { usePokemonReplay } from "./pokemonReplay";

// The post's opening showcase: recorded games between trained RL policies in three games.
// Compact two-column layout: game/matchup selector + trained-policy badge on the left,
// live replay viewer on the right. Tours on its own until the reader interacts.
// The November 20 Generals finalists play fresh demo maps, with playback starting just before city captures.

const BattleReplay = lazy(() => import("../../showdown/react/BattleReplay"));
const ROTATE_MS = 4000;

type Game = "pokemon" | "polytopia" | "generals";
type Item = {
  game: Game;
  id: string;
  players: [string, string];
  trained: [string, string];
  result: string;
  start: number;
};

const ITEMS: Item[] = [
  {
    game: "pokemon",
    id: "sol61-a1-vs-astra6-a1",
    players: ["GPT-6.1 Sol", "GPT-6 Astra"],
    trained: ["1 h trained policy", "1 h trained policy"],
    result: "Astra wins · 7 turns",
    start: 2,
  },
  {
    game: "polytopia",
    id: "polytopia-v3a-opus-vs-luna",
    players: ["Claude Opus 5.5", "GPT-6 Luna"],
    trained: ["2 h trained policy", "2 h starter policy"],
    result: "Opus wins · 27 rounds",
    start: 0.55,
  },
  {
    game: "generals",
    id: "generals-20261120-opus-vs-sol",
    players: ["Claude Opus 5.5", "GPT-6.1 Sol"],
    trained: ["Nov 20 finalist", "Nov 20 finalist"],
    result: "Opus wins · 3 cities captured",
    start: 0.76358,
  },
  {
    game: "pokemon",
    id: "inkling2s-b2-vs-deepseek-v4pro-a2",
    players: ["Inkling 2 Small", "DeepSeek V4 Pro"],
    trained: ["1 h trained policy", "1 h trained policy"],
    result: "Inkling wins · 16 turns",
    start: 6,
  },
  {
    game: "polytopia",
    id: "polytopia-v3a-astra-vs-starter",
    players: ["GPT-6 Astra", "Starter"],
    trained: ["~2 h trained policy", "36 min starter policy"],
    result: "Astra wins on score",
    start: 0.5,
  },
  {
    game: "generals",
    id: "generals-20261120-sol-vs-deepseek",
    players: ["GPT-6.1 Sol", "DeepSeek V4 Pro"],
    trained: ["Nov 20 finalist", "Nov 20 finalist"],
    result: "Sol wins · 4 cities captured",
    start: 0.81033,
  },
  {
    game: "pokemon",
    id: "sol61-a2-vs-glm53-a3",
    players: ["GPT-6.1 Sol", "GLM-5.3"],
    trained: ["1 h trained policy", "1 h trained policy"],
    result: "Sol wins · 4 turns",
    start: 1,
  },
  {
    game: "polytopia",
    id: "polytopia-v3a-opus-vs-sol",
    players: ["Claude Opus 5.5", "GPT-6.1 Sol"],
    trained: ["2 h trained policy", "2 h trained policy"],
    result: "Sol wins on score",
    start: 0.5,
  },
  {
    game: "generals",
    id: "generals-20261120-opus-vs-kimi",
    players: ["Claude Opus 5.5", "Kimi K3.1 Pro"],
    trained: ["Nov 20 finalist", "Nov 20 finalist"],
    result: "Opus wins · 3 cities captured",
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
  if (item.game === "polytopia") return <ReplayViewer key={item.id} src={`${import.meta.env.BASE_URL}polytopia/${item.id}.json`} start={item.start} initialSpeed={4} compact />;
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
            {GAMES.find((x) => x.id === g.game)!.label}: {g.players[0]} ({g.trained[0]}) vs {g.players[1]} ({g.trained[1]}). {g.result}.
          </li>
        ))}
      </ul>
    );
  }

  const takeOver = () => setTouring(false);
  const sameGame = ITEMS.map((g, j) => ({ g, j })).filter(({ g }) => g.game === item.game);

  return (
    <div className="sc" ref={root} onPointerDownCapture={takeOver} onKeyDownCapture={takeOver}>
      <div className="sc-grid">
        <div className="sc-nav">
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

          <div className="sc-matchups" role="tablist" aria-label="Matchup">
            {sameGame.map(({ g, j }) => (
              <button key={g.id} type="button" role="tab" aria-selected={i === j} onClick={() => setI(j)}>
                <strong>{g.players[0]} vs {g.players[1]}</strong>
                <span>{g.result}</span>
              </button>
            ))}
          </div>

          <div className="sc-policies">
            <span className="sc-pol-head">Trained RL policies</span>
            {[0, 1].map((p) => (
              <div key={p} className="sc-pol-row">
                <strong>{item.players[p]}</strong>
                <span>{item.trained[p]}</span>
              </div>
            ))}
            <p className="sc-note">Neural networks play directly — no LLM in the loop.</p>
          </div>
        </div>

        <div className="sc-main">
          {touring && (
            <div
              className="sc-progress"
              key={`${i}-${visible}`}
              style={{ animationDuration: `${ROTATE_MS}ms`, animationPlayState: visible ? "running" : "paused" }}
            />
          )}
          <div className="sc-stage">
            <Stage item={item} />
          </div>
        </div>
      </div>
    </div>
  );
}
