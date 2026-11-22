// Loads Pokémon Showdown's battle engine (src/showdown/engine, brought in by tools/vendor_showdown.mjs) on demand.
//
// The engine is written for Showdown's own build, which flattens everything into globals, so before its modules run
// this puts in place what it reads from window: jQuery, the Caja HTML sanitizer and MD5 (classic scripts), and the
// game data (Pokédex, moves, abilities, items, battle text, sprite sizes). It all loads once, in its own chunks, the
// first time a battle is shown, so pages without one never download it.
import type { Battle } from "./engine/battle";

export interface ShowdownEngine {
  Battle: typeof Battle;
}

let engine: Promise<ShowdownEngine> | undefined;

// A classic <script> runs in global scope, as these two files expect (top-level declarations become globals).
function runGlobalScript(source: string) {
  const el = document.createElement("script");
  el.textContent = source;
  document.head.appendChild(el);
  el.remove();
}

export function loadShowdown(): Promise<ShowdownEngine> {
  engine ??= (async () => {
    const [jquery, sanitizer, misc, pokedex, moves, abilities, items, text, aliases, typechart, sprites, spritesBW] =
      await Promise.all([
        import("jquery"),
        import("./vendor/html-sanitizer.js?raw"),
        import("./vendor/battle-log-misc.js?raw"),
        import("./data/pokedex.json"),
        import("./data/moves.json"),
        import("./data/abilities.json"),
        import("./data/items.json"),
        import("./data/text.json"),
        import("./data/aliases.json"),
        import("./data/typechart.json"),
        import("./data/pokedex-mini.json"),
        import("./data/pokedex-mini-bw.json"),
      ]);
    const w = window as any;
    w.$ = w.jQuery = jquery.default;
    runGlobalScript(sanitizer.default);
    runGlobalScript(misc.default);
    Object.assign(w, {
      BattlePokedex: pokedex.default, BattleMovedex: moves.default, BattleAbilities: abilities.default,
      BattleItems: items.default, BattleText: text.default, BattleAliases: aliases.default,
      BattleTypeChart: typechart.default, BattlePokemonSprites: sprites.default, BattlePokemonSpritesBW: spritesBW.default,
    });
    // battle-animations.ts prefixes its move-effect images (fx/*.png) with Dex.fxPrefix at load time, but only if a
    // global Dex exists then, as it does in Showdown's own concatenated build. Without one the images resolve
    // against this site's root and 404. Dex itself is initialized before that step runs (battle-dex.ts finishes
    // evaluating inside the import cycle first), so a placeholder is enough; afterwards the global is the real Dex.
    w.Dex ??= {};
    const { Battle } = await import("./engine");
    w.Dex = (await import("./engine/battle-dex")).Dex;
    return { Battle };
  })();
  engine.catch(() => { engine = undefined; }); // a failed load (e.g. offline) can be retried
  return engine;
}
