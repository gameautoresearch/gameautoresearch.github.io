// Full-size view of one recorded Pokémon battle: /replays/pokemon/<id>.
import { lazy, Suspense, useEffect } from "react";
import { usePokemonReplay } from "./posts/autoresearch/pokemonReplay";

const BattleReplay = lazy(() => import("./showdown/react/BattleReplay"));

export default function ReplayPage({ id }: { id: string }) {
  const file = usePokemonReplay(id);
  const ready = file && file !== "error" ? file : null;
  useEffect(() => { if (ready) document.title = `${ready.title} · battle replay`; }, [ready]);
  return (
    <div className="page-shell">
      <div className="site-layout">
        <main className="profile replay-page">
          <p className="replay-page-back"><a href={`${import.meta.env.BASE_URL}#pokemon`}>← Back to the post</a></p>
          {file === "error" && <p>No recorded battle is called “{id}”.</p>}
          {ready && (
            <>
              <h1>{ready.title}</h1>
              <p className="replay-page-meta">{ready.winner} wins in {ready.turns} turns. Sound is off until you turn it on.</p>
              <Suspense fallback={null}><BattleReplay log={ready.log} /></Suspense>
              <p className="replay-page-credit">
                Played by <a href="https://github.com/smogon/pokemon-showdown-client">Pokémon Showdown</a>'s battle engine.
                Sprites and sounds load from play.pokemonshowdown.com. See the replay viewer’s{" "}
                <a href="https://github.com/gameautoresearch/gameautoresearch.github.io/tree/main/external-website/src/showdown">source and license notices</a>. Keys: Space play/pause, ← → turns, M sound,
                S switch sides.
              </p>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
