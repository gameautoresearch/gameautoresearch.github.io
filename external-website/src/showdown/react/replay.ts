// What a Showdown protocol log says about the battle, read without running the engine: players, turn count, winner
// and every knockout (for the scrubber's marks).

export type SideID = "p1" | "p2";

export interface Knockout {
  turn: number;
  side: SideID;
  pokemon: string;
}

export interface ReplaySummary {
  players: Record<SideID, string>;
  turns: number;
  winner: string | null;
  knockouts: Knockout[];
}

export function summarizeReplay(log: string): ReplaySummary {
  const players: Record<SideID, string> = { p1: "Player 1", p2: "Player 2" };
  const knockouts: Knockout[] = [];
  let turn = 0, winner: string | null = null;
  for (const line of log.split("\n")) {
    const p = line.split("|");
    if (p[1] === "player" && (p[2] === "p1" || p[2] === "p2") && p[3]) players[p[2]] = p[3];
    else if (p[1] === "turn") turn = Number(p[2]) || turn;
    else if (p[1] === "win") winner = p[2] ?? null;
    else if (p[1] === "faint" && p[2]) {
      const side = p[2].slice(0, 2);
      if (side === "p1" || side === "p2") knockouts.push({ turn, side, pokemon: p[2].replace(/^p\d[a-z]?: /, "") });
    }
  }
  return { players, turns: turn, winner, knockouts };
}
