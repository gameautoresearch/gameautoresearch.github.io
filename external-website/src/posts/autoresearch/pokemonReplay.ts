// The post's recorded Pokémon battles: public/pokemon/<id>.json, written by tools/build_pokemon_replays.py.
import { useEffect, useState } from "react";

export interface PokemonReplayFile {
  id: string;
  title: string;
  players: string[];
  winner: string;
  turns: number;
  log: string;
}

export const replayUrl = (id: string) => `${import.meta.env.BASE_URL}replays/pokemon/${id}/`;

/** The battle file, or null while loading, or "error". */
export function usePokemonReplay(id: string): PokemonReplayFile | null | "error" {
  const [file, setFile] = useState<PokemonReplayFile | null | "error">(null);
  useEffect(() => {
    let current = true;
    setFile(null);
    fetch(`${import.meta.env.BASE_URL}pokemon/${id}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data: PokemonReplayFile) => { if (current) setFile(data); })
      .catch(() => { if (current) setFile("error"); });
    return () => { current = false; };
  }, [id]);
  return file;
}
