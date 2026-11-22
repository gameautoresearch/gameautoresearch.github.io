// Site stand-in for Pokémon Showdown's client-main.ts, which the battle engine imports for two things: Config (where
// sprites, sounds and effects load from, plus a few server settings) and PS (the chat client's preferences). Replays
// have no chat client, so PS is absent and the engine falls back to its own defaults.

export interface PSConfig {
  routes: { root: string, client: string, dex: string, replays: string, users: string };
  server: { id: string, host: string, port: number, afd?: boolean };
  defaultserver: { id: string, host: string, port: number };
  customcolors: { [userid: string]: string };
  whitelist: string[];
  testclient?: boolean;
  translationCachebuster?: string;
}

const server = { id: "showdown", host: "sim3.psim.us", port: 443 };

// routes.client is where game art loads from: it stays on Showdown's server, the site hosts none.
export const Config: PSConfig = {
  routes: { root: "pokemonshowdown.com", client: "play.pokemonshowdown.com", dex: "dex.pokemonshowdown.com",
    replays: "replay.pokemonshowdown.com", users: "pokemonshowdown.com/users" },
  server, defaultserver: server, customcolors: {}, whitelist: [],
};
(window as any).Config = Config;

// The shape the engine reads; the value is undefined at runtime (the engine checks `typeof PS === 'object'`).
export interface PSStandIn {
  prefs: { [pref: string]: any, mute: boolean, effectvolume: number, musicvolume: number,
    subscribeAndRun(listener: (key: string | null) => void): void };
  user: { userid: string };
}
export const PS = undefined as unknown as PSStandIn;
