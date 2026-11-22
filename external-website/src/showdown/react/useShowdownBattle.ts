// React hook around Pokémon Showdown's Battle: give it a protocol log and two elements (the field and the log), get
// back the playback state and the controls. The engine loads on first use (see ../load.ts).
import { useCallback, useEffect, useRef, useState } from "react";
import type { Battle } from "../engine/battle";
import { loadShowdown } from "../load";

export type Speed = "slow" | "normal" | "fast";

// Showdown's own replay timings: message fade and message hold, in ms.
const SPEEDS: Record<Speed, [fade: number, hold: number]> = { slow: [500, 1000], normal: [300, 1], fast: [50, 1] };

export interface PlaybackState {
  status: "idle" | "loading" | "ready" | "error";
  /** Current turn (0 before turn 1). While seeking, the turn being sought. */
  turn: number;
  playing: boolean;
  /** At the end of the log, after the result. */
  finished: boolean;
  started: boolean;
}

const INITIAL: PlaybackState = { status: "idle", turn: 0, playing: false, finished: false, started: false };

function read(b: Battle, status: PlaybackState["status"]): PlaybackState {
  const finished = b.ended && b.atQueueEnd && b.seeking === null;
  const turn = b.seeking !== null && b.seeking !== Infinity ? b.seeking : b.turn;
  return { status, turn: Math.max(0, turn), playing: !b.paused && !finished, finished, started: b.started };
}

const same = (a: PlaybackState, b: PlaybackState) =>
  a.status === b.status && a.turn === b.turn && a.playing === b.playing && a.finished === b.finished && a.started === b.started;

export function useShowdownBattle(log: string, { enabled = true }: { enabled?: boolean } = {}) {
  const frameRef = useRef<HTMLDivElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const battleRef = useRef<Battle | null>(null);
  const [state, setState] = useState<PlaybackState>(INITIAL);
  const [speed, setSpeedState] = useState<Speed>("normal");
  const [muted, setMutedState] = useState(true);
  const speedRef = useRef(speed);
  const mutedRef = useRef(muted);

  const refresh = useCallback(() => {
    const b = battleRef.current;
    if (!b) return;
    const next = read(b, "ready");
    setState((prev) => (same(prev, next) ? prev : next));
  }, []);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false, timer = 0;
    setState({ ...INITIAL, status: "loading" });
    loadShowdown().then(({ Battle }) => {
      if (cancelled || !frameRef.current || !logRef.current) return;
      const b = new Battle({ $frame: $(frameRef.current), $logFrame: $(logRef.current), log: log.split("\n"),
        isReplay: true, paused: true, autoresize: false });
      battleRef.current = b;
      b.setMute(mutedRef.current);
      applySpeed(b, speedRef.current);
      b.subscribe(refresh);
      // The engine reports play/pause/end, not every turn, so the turn count is also read on a short timer.
      timer = window.setInterval(refresh, 150);
      refresh();
    }, () => { if (!cancelled) setState({ ...INITIAL, status: "error" }); });
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      const b = battleRef.current;
      battleRef.current = null;
      if (b) { b.pause(); b.subscribe(null); b.destroy(); }
    };
  }, [log, enabled, refresh]);

  const act = useCallback((fn: (b: Battle) => void) => {
    const b = battleRef.current;
    if (b) { fn(b); refresh(); }
  }, [refresh]);

  const play = useCallback(() => act((b) => {
    if (b.ended && b.atQueueEnd && b.seeking === null) b.reset();
    b.play();
  }), [act]);
  const pause = useCallback(() => act((b) => b.pause()), [act]);
  /** Jump to the start of a turn; 0 is the start of the battle, anything past the last turn is the end. */
  const seekTurn = useCallback((turn: number) => act((b) => {
    if (turn <= 0) { const wasPlaying = !b.paused; b.reset(); if (wasPlaying) b.play(); }
    else b.seekTurn(turn, turn <= b.turn);
  }), [act]);
  const step = useCallback((delta: number) => act((b) => {
    if (delta > 0 && b.ended && b.atQueueEnd) return;
    b.seekBy(delta);
  }), [act]);
  const switchSides = useCallback(() => act((b) => b.switchViewpoint()), [act]);
  const setSpeed = useCallback((s: Speed) => {
    speedRef.current = s;
    setSpeedState(s);
    if (battleRef.current) applySpeed(battleRef.current, s);
  }, []);
  const setMuted = useCallback((m: boolean) => {
    mutedRef.current = m;
    setMutedState(m);
    battleRef.current?.setMute(m);
  }, []);

  return { frameRef, logRef, state, speed, muted, play, pause, seekTurn, step, switchSides, setSpeed, setMuted };
}

function applySpeed(b: Battle, s: Speed) {
  [b.messageFadeTime, b.messageShownTime] = SPEEDS[s];
  (b.scene as { updateAcceleration?: () => void }).updateAcceleration?.();
}
