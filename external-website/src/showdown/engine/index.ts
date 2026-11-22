// Site entry point to Showdown's battle engine: import the engine from here, not from battle.ts.
//
// battle-animations.ts and battle-animations-moves.ts import each other, and the move table reads BattleOtherAnims
// while it is being defined. Showdown's build concatenates the files in a fixed order; as ES modules, the move table
// has to be evaluated first so that it, not battle-animations.ts, is the module waiting on the cycle.
import "./battle-animations-moves";

export { Battle, Pokemon, Side } from "./battle";
