// <BattleReplay log={...} />: a Pokémon Showdown battle, played by Showdown's own engine, with the site's controls.
// Reusable anywhere in the app: pass a protocol log. The engine loads when the replay comes near the screen, plays
// (muted) while it is on screen, fills its container's width and pauses when it leaves, unless the viewer paused it.
//
// Keys, while the replay has focus: Space or K play/pause, ← → (or J L) turns, M sound, S switch sides.
import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { Scrubber } from "./Scrubber";
import { summarizeReplay } from "./replay";
import { useShowdownBattle, type Speed } from "./useShowdownBattle";
import "../styles/showdown.css";
import "./replay.css";

export interface BattleReplayProps {
  /** Showdown protocol log, one message per line. */
  log: string;
  /** Play while on screen (default true). */
  autoplay?: boolean;
  /** Turn to start from once the engine is ready (default: the beginning). */
  startTurn?: number;
  /** Starting playback speed (default normal). */
  initialSpeed?: Speed;
  /** Crop Showdown's 100px trainer sidebars and focus on the 440×360 center arena. */
  cropSidebars?: boolean;
  /** Optional left/right side cards rendered flanking the arena (0 = p1, 1 = p2). */
  renderSide?: (side: 0 | 1) => ReactNode;
  /** Optional callback fired whenever the displayed turn changes. */
  onTurnChange?: (turn: number) => void;
  className?: string;
}

const SPEED_ORDER: Speed[] = ["normal", "fast", "slow"];
const SPEED_LABEL: Record<Speed, string> = { normal: "1×", fast: "2×", slow: "½×" };

const Icon = {
  play: <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4.5v15a1 1 0 0 0 1.5.86l12-7.5a1 1 0 0 0 0-1.72l-12-7.5A1 1 0 0 0 7 4.5Z" fill="currentColor" /></svg>,
  pause: <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5.5" y="4" width="4.5" height="16" rx="1.2" fill="currentColor" /><rect x="14" y="4" width="4.5" height="16" rx="1.2" fill="currentColor" /></svg>,
  replay: <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3M4.5 4v4.5H9" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /></svg>,
  soundOff: <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" fill="currentColor" /><path d="m16 9.5 5 5m0-5-5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>,
  soundOn: <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" fill="currentColor" /><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>,
};

export default function BattleReplay({ log, autoplay = true, startTurn, initialSpeed, cropSidebars = false, renderSide, onTurnChange, className }: BattleReplayProps) {
  const summary = useMemo(() => summarizeReplay(log), [log]);
  const end = summary.turns + 1; // the scrubber's last stop: the end of the battle, after the final turn
  const stage = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(false);
  const [visible, setVisible] = useState(false);
  const [scale, setScale] = useState(1);
  const userPaused = useRef(false);
  const b = useShowdownBattle(log, { enabled: near });
  const { state } = b;
  const arenaWidth = cropSidebars ? 440 : 640;

  // Load when within a screen of the viewport; play while at least half the field is visible.
  useEffect(() => {
    const el = stage.current;
    if (!el || typeof IntersectionObserver === "undefined") { setNear(true); setVisible(true); return; }
    const nearObs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setNear(true); }, { rootMargin: "100% 0px" });
    const visObs = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0.5 });
    nearObs.observe(el); visObs.observe(el);
    return () => { nearObs.disconnect(); visObs.disconnect(); };
  }, []);
  useEffect(() => { userPaused.current = false; }, [log]);
  // Before autoplay starts: jump to the starting turn and set the starting speed.
  useEffect(() => {
    if (state.status !== "ready") return;
    if (initialSpeed) b.setSpeed(initialSpeed);
    if (startTurn) b.seekTurn(startTurn);
  }, [state.status]);
  useEffect(() => {
    if (state.status !== "ready" || !autoplay) return;
    if (visible && !state.playing && !state.finished && !userPaused.current) b.play();
    else if (!visible && state.playing) b.pause();
  }, [visible, state.status, autoplay]);

  // The field is laid out at Showdown's 640 × 360 (or cropped 440 × 360 center arena) and scaled to fill width.
  useEffect(() => {
    const el = stage.current;
    if (!el) return;
    const fit = () => setScale(el.clientWidth / arenaWidth);
    fit();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(fit) : null;
    ro?.observe(el);
    return () => ro?.disconnect();
  }, [arenaWidth]);

  const toggle = () => {
    if (state.playing) { userPaused.current = true; b.pause(); }
    else { userPaused.current = false; b.play(); }
  };
  const position = state.finished ? end : Math.min(state.turn, end);
  useEffect(() => {
    onTurnChange?.(position);
  }, [position, onTurnChange]);
  const turnText = position >= end ? "End" : `${position}/${summary.turns}`;

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const onButton = (e.target as HTMLElement).tagName === "BUTTON" || (e.target as HTMLElement).tagName === "SUMMARY";
    switch (e.key) {
      case " ": if (onButton) return; e.preventDefault(); toggle(); break;
      case "k": case "K": toggle(); break;
      case "ArrowLeft": case "j": case "J": e.preventDefault(); b.step(-1); break;
      case "ArrowRight": case "l": case "L": e.preventDefault(); b.step(1); break;
      case "m": case "M": b.setMuted(!b.muted); break;
      case "s": case "S": b.switchSides(); break;
    }
  };

  const marks = summary.knockouts.map((k) => ({
    at: k.turn, className: k.side,
    title: `Turn ${k.turn}: ${k.pokemon} (${summary.players[k.side]}) fainted`,
  }));
  const playLabel = state.finished ? "Watch again" : state.playing ? "Pause" : "Play";
  const stageEl = (
    <div className="psr-stage" ref={stage} style={{ height: Math.round(360 * scale) }} onClick={toggle}>
      <div
        className="psr-scale ps-battle"
        style={{ transform: cropSidebars ? `scale(${scale}) translateX(-100px)` : `scale(${scale})` }}
      >
        <div className="battle" ref={b.frameRef} />
      </div>
      {state.status !== "ready" && (
        <div className="psr-note">{state.status === "error" ? "The battle viewer could not load." : "Loading battle…"}</div>
      )}
    </div>
  );

  return (
    <div className={`psr${cropSidebars ? " psr-crop" : ""}${renderSide ? " psr-flanked" : ""} ${className ?? ""}`} onKeyDown={onKeyDown}>
      {renderSide ? (
        <div className="sc-row">
          {renderSide(0)}
          <div className="sc-center sc-center-poke">{stageEl}</div>
          {renderSide(1)}
        </div>
      ) : (
        stageEl
      )}
      <div className="psr-bar">
        <button type="button" className="psr-ic psr-play" aria-label={playLabel} title={`${playLabel} (Space)`}
          onClick={toggle} disabled={state.status !== "ready"}>
          {state.finished ? Icon.replay : state.playing ? Icon.pause : Icon.play}
        </button>
        <Scrubber value={position} max={end} onSeek={b.seekTurn} marks={marks} label="Turn" valueText={turnText} />
        <span className="psr-turn" aria-hidden="true">{turnText}</span>
        <button type="button" className="psr-txt" title="Speed" aria-label={`Speed ${SPEED_LABEL[b.speed]}`}
          onClick={() => b.setSpeed(SPEED_ORDER[(SPEED_ORDER.indexOf(b.speed) + 1) % SPEED_ORDER.length])}>
          {SPEED_LABEL[b.speed]}
        </button>
        <button type="button" className="psr-ic" aria-pressed={!b.muted} aria-label={b.muted ? "Sound off" : "Sound on"}
          title="Sound (M)" onClick={() => b.setMuted(!b.muted)}>
          {b.muted ? Icon.soundOff : Icon.soundOn}
        </button>
      </div>
      <details className="psr-log">
        <summary>Battle log</summary>
        <div className="ps-battle"><div className="battle-log" ref={b.logRef} /></div>
      </details>
    </div>
  );
}
