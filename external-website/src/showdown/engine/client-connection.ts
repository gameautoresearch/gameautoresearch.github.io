// Site stand-in for Pokémon Showdown's client-connection.ts. The battle log imports Net only to fetch chat name
// colours from Showdown's server when a chat client is present; replays make no network requests of their own.
export function Net(_uri: string): { get(): Promise<string> } {
  return { get: () => Promise.reject(new Error("no network in the replay viewer")) };
}
