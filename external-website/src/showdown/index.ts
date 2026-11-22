// Pokémon Showdown battle replays for the site.
//
//   engine/   Showdown's battle engine as TypeScript modules (MIT; tools/vendor_showdown.mjs, MANIFEST.json), plus the
//             site's stand-ins for the two client modules it imports (client-main.ts, client-connection.ts)
//   load.ts   loads the engine and its data on first use
//   react/    <BattleReplay>, the useShowdownBattle hook, the turn scrubber and summarizeReplay
//
// Sprites, backgrounds, effects and sounds load from play.pokemonshowdown.com; the site hosts no game art.
export { default as BattleReplay, type BattleReplayProps } from "./react/BattleReplay";
export { useShowdownBattle, type PlaybackState, type Speed } from "./react/useShowdownBattle";
export { Scrubber, type ScrubberMark } from "./react/Scrubber";
export { summarizeReplay, type ReplaySummary, type Knockout, type SideID } from "./react/replay";
export { loadShowdown, type ShowdownEngine } from "./load";
