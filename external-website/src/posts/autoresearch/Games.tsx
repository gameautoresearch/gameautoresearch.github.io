import { lazy, Suspense, useState } from "react";
import { useAgentView } from "../../view";
import ReplayViewer from "./polytopia/ReplayViewer";
import pokemonReplays from "./pokemon-replays.json";
import { replayUrl, usePokemonReplay } from "./pokemonReplay";
import polytopiaRaw from "./polytopia-replays.json";

// Recorded games between the networks the research sessions produced. Pokémon battles play in <BattleReplay>, Pokémon
// Showdown's battle engine with the site's controls (src/showdown/; logs in public/pokemon/*.json, built by
// tools/build_pokemon_replays.py); Polytopia games play in the board
// viewer from thekevinz.com (public/polytopia/*.json, built by tools/convert_polytopia_replays.py).

type PolyGame = { id: string; title: string; players: string[]; winner: number; reason: string; scores: number[]; rounds: number; actions: number };
const polytopiaReplays = polytopiaRaw as PolyGame[];

const short = (name: string) => name.replace(/^GPT-6(\.1)? /, "").replace(" · ", " ");

// The replay component and the battle engine load only when the battles section renders.
const BattleReplay = lazy(() => import("../../showdown/react/BattleReplay"));

export function PokemonTheater() {
  const agent = useAgentView();
  const [i, setI] = useState(0);
  const r = pokemonReplays[i];
  const file = usePokemonReplay(r.id);
  if (agent) {
    return (
      <ul>
        {pokemonReplays.map((g) => (
          <li key={g.id}>{g.players.join(" vs ")}: {g.winner} wins in {g.turns} turns (<a href={replayUrl(g.id)}>replay</a>)</li>
        ))}
      </ul>
    );
  }
  const placeholder = <div className="ar-replay-wait">{file === "error" ? "This battle could not load." : null}</div>;
  return (
    <div className="ar-theater">
      <div className="ar-tabs ar-tabs-wrap" role="tablist" aria-label="Battle">
        {pokemonReplays.map((g, j) => (
          <button key={g.id} role="tab" aria-selected={i === j} onClick={() => setI(j)}>
            {short(g.players[0])} vs {short(g.players[1])}
          </button>
        ))}
      </div>
      <p className="ar-theater-meta">
        <strong>{r.winner}</strong> wins in {r.turns} turns. Sound is off until you turn it on.{" "}
        <a href={replayUrl(r.id)} target="_blank" rel="noreferrer">Open full size ↗</a>
      </p>
      <Suspense fallback={placeholder}>
        {file && file !== "error" && file.id === r.id ? <BattleReplay key={r.id} log={file.log} /> : placeholder}
      </Suspense>
    </div>
  );
}

const OUTCOME: Record<string, string> = { elimination: "by capturing every city", round_cap_score: "on score at the round cap" };

export function PolytopiaTheater() {
  const agent = useAgentView();
  const [i, setI] = useState(0);
  if (polytopiaReplays.length === 0) return null;
  if (agent) {
    return (
      <ul>
        {polytopiaReplays.map((r) => (
          <li key={r.id}>{r.players.join(" (blue) vs ")} (orange): {r.winner ? r.players[r.winner - 1] : "Nobody"} wins {OUTCOME[r.reason] ?? ""}, {r.scores.join("–")}, {r.rounds} rounds.</li>
        ))}
      </ul>
    );
  }
  const r = polytopiaReplays[i];
  return (
    <div className="ar-theater">
      {polytopiaReplays.length > 1 && (
        <div className="ar-tabs ar-tabs-wrap" role="tablist" aria-label="Game">
          {polytopiaReplays.map((g, j) => (
            <button key={g.id} role="tab" aria-selected={i === j} onClick={() => setI(j)}>{g.players.map((n) => short(n.split(" (")[0]).replace("Claude ", "")).join(" vs ")}</button>
          ))}
        </div>
      )}
      <p className="ar-theater-meta">
        <strong>{r.winner ? r.players[r.winner - 1] : "Nobody"}</strong> wins {OUTCOME[r.reason] ?? ""}, {r.scores.join("–")} after {r.rounds} rounds ({r.actions.toLocaleString()} actions).
      </p>
      <ReplayViewer src={`${import.meta.env.BASE_URL}polytopia/${r.id}.json`} />
    </div>
  );
}
