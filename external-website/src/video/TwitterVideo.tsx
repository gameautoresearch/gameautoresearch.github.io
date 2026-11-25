import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import ResearchLoop from "../posts/autoresearch/ResearchLoop";
import ReplayViewer, { type ReplayCustomContext } from "../posts/autoresearch/polytopia/ReplayViewer";
import GeneralsReplay, { type GeneralsCustomContext } from "../posts/autoresearch/GeneralsReplay";
import { parsePokemonTimeline } from "../posts/autoresearch/NetworkIO";
import { useShowdownBattle } from "../showdown/react/useShowdownBattle";
import "../showdown/styles/showdown.css";
import "../showdown/react/replay.css";
import {
  GEN_CYAN,
  GEN_MAGENTA,
  POKE_CYAN,
  POKE_MAGENTA,
  POLY_BLUE,
  POLY_ORANGE,
  SimpleGeneralsIO,
  SimplePokemonIO,
  SimplePolytopiaIO,
} from "./SimpleIO";

// 3-Game Repeating Twitter Landscape Video (960x540 CSS @ 2x DPR = 1920x1080):
//   Loop 1 (0.0s – 13.5s):  Polytopia   — GPT-6 Astra trains -> zooms into GPT-6 Astra vs Starter
//   Loop 2 (13.5s – 27.0s): Generals    — GPT-6.1 Sol trains -> zooms into GPT-6.1 Sol vs DeepSeek V4 Pro
//   Loop 3 (27.0s – 41.5s): Pokémon 3v3 — GPT-6.1 Sol trains -> zooms into GPT-6.1 Sol vs GLM-5.3 (Houndoom sweep!)

const SEG_POLY_MS = 13500;
const SEG_GEN_MS = 13500;
const SEG_POKE_MS = 15500;

export const SEG1_START = 0;
export const SEG2_START = SEG_POLY_MS; // 13500
export const SEG3_START = SEG_POLY_MS + SEG_GEN_MS; // 27000
export const TOTAL_MS = SEG_POLY_MS + SEG_GEN_MS + SEG_POKE_MS; // 42500

// Per-segment shared choreography (local time within each loop):
const RL_RATE = 1.22; // 0..6600ms of ResearchLoop plays cleanly in 5.4s
const RL_MAX_MS = 6600;
const TRANS_START = 5450;
const TRANS_END = 6350;
const PLAY_START = 6150;
const FADE_OUT_MS = 380;

// Polytopia highlight window: Actions 127 -> 201.95 (Turn 8 Capture -> Turn 11 City Capture + Mountain Temple)
const POLY_START_ACT = 127;
const POLY_END_ACT = 201.95;
const POLY_DURATION: Record<number, number> = {
  1: 110, 2: 260, 3: 150, 4: 150, 5: 160, 6: 160, 7: 40, 8: 320, 9: 160, 10: 200,
};

// Generals highlight window: Ticks 142 -> 184 (General captured at Tick 184!)
const GEN_START_TICK = 142;
const GEN_END_TICK = 184;

// Spatial board anchors (in 960x540 CSS space) for pixel-locked right-column zooms
const BOARD_ANCHORS = {
  polytopia: { p1: { x: 805, y: 298, w: 224 }, p2: { x: 632, y: 282, w: 544 } },
  generals: { p1: { x: 805, y: 306, w: 222 }, p2: { x: 632, y: 284, w: 472 } },
  pokemon: { p1: { x: 805, y: 305, w: 222 }, p2: { x: 632, y: 282, w: 536 } },
} as const;

const clamp = (v: number, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const easeOutCubic = (t: number) => 1 - Math.pow(1 - clamp(t), 3);
const easeInOutCubic = (t: number) => {
  const x = clamp(t);
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
};

type PolyReplayMeta = { actions: [number, number, number, ...unknown[]][] };
type PokeFile = { id: string; title: string; players: [string, string]; winner: string; turns: number; log: string };

function buildPolyTimeline(actions: PolyReplayMeta["actions"], startIdx: number, endPos: number) {
  const startFloor = Math.floor(startIdx);
  const endFloor = Math.min(actions.length - 1, Math.floor(endPos));
  const weights: { idx: number; startW: number; endW: number; fracMax: number }[] = [];
  let acc = 0;
  for (let i = startFloor; i <= endFloor; i++) {
    const dur = POLY_DURATION[actions[i][2]] ?? 150;
    const fracMax = i === endFloor ? Math.max(0.01, endPos - endFloor) : 1;
    const w = dur * fracMax;
    weights.push({ idx: i, startW: acc, endW: acc + w, fracMax });
    acc += w;
  }
  return { weights, totalW: acc };
}

function zoomTransforms(localMs: number, segDur: number, isLastSeg: boolean, game: keyof typeof BOARD_ANCHORS) {
  const { p1, p2 } = BOARD_ANCHORS[game];
  const ratio = p2.w / p1.w;
  const rawU = clamp((localMs - TRANS_START) / (TRANS_END - TRANS_START));
  const u = easeInOutCubic(rawU);

  const cx = lerp(p1.x, p2.x, u);
  const cy = lerp(p1.y, p2.y, u);
  const curW = p1.w * Math.pow(ratio, u);

  const segIn = easeOutCubic(clamp(localMs / 360));
  const segOut = isLastSeg ? 0 : easeInOutCubic(clamp((localMs - (segDur - FADE_OUT_MS)) / FADE_OUT_MS));
  const segVis = 1 - segOut;

  const s1 = curW / p1.w;
  const tx1 = cx - p1.x * s1;
  const ty1 = cy - p1.y * s1;
  const act1Op = segIn * (1 - easeInOutCubic(clamp((rawU - 0.24) / 0.36))) * segVis;

  const s2 = curW / p2.w;
  const tx2 = cx - p2.x * s2;
  const ty2 = cy - p2.y * s2;
  const boardOp = easeOutCubic(clamp((rawU - 0.28) / 0.42)) * segVis;

  const panelIn = easeOutCubic(clamp((rawU - 0.52) / 0.48));
  const hudIn = easeOutCubic(clamp((rawU - 0.64) / 0.36));

  return {
    rlTime: Math.min(RL_MAX_MS, localMs * RL_RATE),
    tx1, ty1, s1, act1Op,
    tx2, ty2, s2, boardOp,
    panelIn, hudIn, segVis,
  };
}

declare global {
  interface Window {
    __setVideoTime?: (ms: number) => void;
    __videoReady?: boolean;
    __videoDurationMs?: number;
    __pokePlayStartMs?: number;
    __pokeSync?: () => void;
    __realClock?: {
      raf: typeof window.requestAnimationFrame;
    };
    __vclock?: {
      enable: () => void;
      disable: () => void;
      advance: (dtMs: number) => void;
    };
  }
}

export default function TwitterVideo() {
  const [tMs, setTMs] = useState(0);
  const [artReady, setArtReady] = useState(false);
  const [extraArtReady, setExtraArtReady] = useState(false);
  const [polyReady, setPolyReady] = useState(false);
  const [genReady, setGenReady] = useState(false);
  const [pokeReady, setPokeReady] = useState(false);
  const [fontsReady, setFontsReady] = useState(false);

  const [polyMeta, setPolyMeta] = useState<PolyReplayMeta | null>(null);
  const [pokeFile, setPokeFile] = useState<PokeFile | null>(null);
  const pokeStartedRef = useRef(false);

  const base = import.meta.env.BASE_URL;
  const polySrc = `${base}polytopia/polytopia-v3a-astra-vs-starter.json`;
  const genSrc = `${base}generals/generals-20261120-sol-vs-deepseek.json`;
  const pokeSrc = `${base}pokemon/sol61-a2-vs-glm53-a3.json`;

  // Preload JSONs, fonts, and hero-art images for Generals & Pokémon so zero images ever pop in late.
  useEffect(() => {
    let live = true;
    fetch(polySrc).then((r) => r.json()).then((d) => { if (live) setPolyMeta(d as PolyReplayMeta); });
    fetch(pokeSrc).then((r) => r.json()).then((d) => { if (live) setPokeFile(d as PokeFile); });
    document.fonts.ready.then(() => { if (live) setFontsReady(true); });

    const localHeroImgs = [
      `${base}hero-art/pokemon/bg-meadow.jpg`,
      `${base}hero-art/pokemon/groudon.png`,
      `${base}hero-art/pokemon/houndoom-f.png`,
      `${base}hero-art/generals/mountainie.png`,
      `${base}hero-art/generals/citie.png`,
      `${base}hero-art/generals/crownie.png`,
    ];
    const showdownSprites = [
      "https://play.pokemonshowdown.com/sprites/ani-back/houndoom-f.gif",
      "https://play.pokemonshowdown.com/sprites/ani/groudon.gif",
      "https://play.pokemonshowdown.com/sprites/ani/heracross.gif",
      "https://play.pokemonshowdown.com/sprites/ani/cacturne.gif",
    ];
    Promise.all(
      [...localHeroImgs, ...showdownSprites].map(
        (src) =>
          new Promise<void>((res) => {
            const img = new Image();
            img.onload = () => res();
            img.onerror = () => res();
            img.src = src;
          }),
      ),
    ).then(() => {
      if (live) setExtraArtReady(true);
    });

    return () => { live = false; };
  }, [base, polySrc, pokeSrc]);

  // Pokémon Showdown battle engine (mounted from the start so it preloads & seeks to Turn 1 cleanly)
  const pokeBattle = useShowdownBattle(pokeFile?.log ?? "", { enabled: !!pokeFile });
  const pokeSnapshots = useMemo(() => (pokeFile ? parsePokemonTimeline(pokeFile.log) : []), [pokeFile]);

  useEffect(() => {
    if (pokeBattle.state.status !== "ready") return;
    pokeBattle.setSpeed("fast");
    pokeBattle.seekTurn(1);
    const id = window.setTimeout(() => setPokeReady(true), 450);
    return () => window.clearTimeout(id);
  }, [pokeBattle.state.status]);

  // Trigger Pokémon Showdown playback right when Segment 3's zoom lands
  const pokePlayStartGlobal = SEG3_START + PLAY_START;
  useEffect(() => {
    if (tMs >= pokePlayStartGlobal && !pokeStartedRef.current && pokeBattle.state.status === "ready") {
      pokeStartedRef.current = true;
      pokeBattle.play();
    } else if (tMs < SEG3_START && pokeStartedRef.current && pokeBattle.state.status === "ready") {
      pokeStartedRef.current = false;
      pokeBattle.pause();
      pokeBattle.seekTurn(1);
    }
  }, [tMs, pokePlayStartGlobal, pokeBattle]);

  const polyTimeline = useMemo(
    () => (polyMeta ? buildPolyTimeline(polyMeta.actions, POLY_START_ACT, POLY_END_ACT) : null),
    [polyMeta],
  );

  useEffect(() => {
    window.__videoDurationMs = TOTAL_MS;
    window.__pokePlayStartMs = pokePlayStartGlobal;
    window.__pokeSync = () => pokeBattle.refresh();
    window.__setVideoTime = (ms: number) => {
      flushSync(() => {
        setTMs(clamp(ms, 0, TOTAL_MS));
      });
    };
    return () => {
      delete window.__setVideoTime;
      delete window.__pokeSync;
    };
  }, [pokeBattle, pokePlayStartGlobal]);

  useEffect(() => {
    window.__videoReady = artReady && extraArtReady && polyReady && genReady && pokeReady && fontsReady && !!polyTimeline;
  }, [artReady, extraArtReady, polyReady, genReady, pokeReady, fontsReady, polyTimeline]);

  // Optional live browser preview with ?autoplay=1
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("autoplay") !== "1") return;
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      setTMs((now - t0) % TOTAL_MS);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const onArtReady = useCallback(() => setArtReady(true), []);
  const onPolyReady = useCallback(() => setPolyReady(true), []);
  const onGenReady = useCallback(() => setGenReady(true), []);

  // Active segment index: 0 = Polytopia, 1 = Generals, 2 = Pokémon
  const seg = tMs < SEG2_START ? 0 : tMs < SEG3_START ? 1 : 2;
  const localMs = seg === 0 ? tMs : seg === 1 ? tMs - SEG2_START : tMs - SEG3_START;

  // Polytopia action pos
  const polyPos = useMemo(() => {
    if (!polyTimeline) return POLY_START_ACT;
    const lms = seg === 0 ? localMs : seg > 0 ? SEG_POLY_MS : 0;
    const endPlay = SEG_POLY_MS - 800;
    if (lms <= PLAY_START) return POLY_START_ACT;
    if (lms >= endPlay) return POLY_END_ACT;
    const frac = (lms - PLAY_START) / (endPlay - PLAY_START);
    const targetW = frac * polyTimeline.totalW;
    for (const s of polyTimeline.weights) {
      if (targetW <= s.endW) {
        const local = s.endW > s.startW ? (targetW - s.startW) / (s.endW - s.startW) : 0;
        return s.idx + local * s.fracMax;
      }
    }
    return POLY_END_ACT;
  }, [polyTimeline, seg, localMs]);

  // Generals tick pos
  const genPos = useMemo(() => {
    const lms = seg === 1 ? localMs : seg > 1 ? SEG_GEN_MS : 0;
    const endPlay = SEG_GEN_MS - 1050;
    if (lms <= PLAY_START) return GEN_START_TICK;
    if (lms >= endPlay) return GEN_END_TICK;
    const frac = (lms - PLAY_START) / (endPlay - PLAY_START);
    return GEN_START_TICK + frac * (GEN_END_TICK - GEN_START_TICK);
  }, [seg, localMs]);

  const polyT = zoomTransforms(seg === 0 ? localMs : 0, SEG_POLY_MS, false, "polytopia");
  const genT = zoomTransforms(seg === 1 ? localMs : 0, SEG_GEN_MS, false, "generals");
  const pokeT = zoomTransforms(seg === 2 ? localMs : 0, SEG_POKE_MS, true, "pokemon");

  const activeRl =
    seg === 0
      ? { scene: 6, ...polyT }
      : seg === 1
      ? { scene: 2, ...genT }
      : { scene: 4, ...pokeT };

  // Polytopia Part 2 custom layout
  const renderPolyCustom = (ctx: ReplayCustomContext) => {
    const actPlayer = ctx.action ? ctx.action[0] : 0;
    const actText = ctx.action ? ctx.labels[ctx.action[5]] : "";
    const actColor = actPlayer === 1 ? POLY_BLUE : POLY_ORANGE;
    const vis = seg === 0 && polyT.boardOp > 0.002;
    return (
      <div className="sp-root" style={{ visibility: seg === 0 ? "visible" : "hidden" }}>
        <div
          className="sp-board-zoom-layer"
          style={{
            opacity: polyT.boardOp,
            transform: `translate3d(${polyT.tx2.toFixed(2)}px, ${polyT.ty2.toFixed(2)}px, 0) scale(${polyT.s2.toFixed(4)})`,
            visibility: vis ? "visible" : "hidden",
          }}
        >
          <div className="sp-board-card" style={{ backgroundColor: `rgba(248, 248, 246, ${polyT.hudIn.toFixed(3)})` }}>
            <div className="sp-hud" style={{ opacity: polyT.hudIn }}>
              <div className="sp-score">
                <i style={{ background: POLY_BLUE }} />
                <span className="sp-score-name">GPT-6 Astra</span>
                <strong>{ctx.score[0].toLocaleString()}</strong>
              </div>
              <span className="sp-turn">Turn {ctx.turn}</span>
              <div className="sp-score">
                <strong>{ctx.score[1].toLocaleString()}</strong>
                <span className="sp-score-name">Starter</span>
                <i style={{ background: POLY_ORANGE }} />
              </div>
            </div>
            <div className="sp-board-wrap">{ctx.board}</div>
            {actText && (
              <div className="sp-act-pill" style={{ opacity: polyT.hudIn }}>
                <i style={{ background: actColor }} />
                <span>{actPlayer === 2 ? `Starter · ${actText}` : actText}</span>
              </div>
            )}
          </div>
        </div>

        <div
          className="sp-panel-layer"
          style={{
            opacity: polyT.panelIn * polyT.segVis,
            transform: `translate3d(${((1 - polyT.panelIn) * -32).toFixed(2)}px, 0, 0)`,
            visibility: seg === 0 && polyT.panelIn > 0.002 ? "visible" : "hidden",
          }}
        >
          <SimplePolytopiaIO
            size={ctx.size}
            tiles={ctx.tiles}
            units={ctx.units}
            cities={ctx.cities}
            action={ctx.action}
            labels={ctx.labels}
          />
        </div>
      </div>
    );
  };

  // Generals Part 2 custom layout
  const renderGenCustom = (ctx: GeneralsCustomContext) => {
    const solMove = ctx.moves.find((m) => m.player === 0);
    const pillText = ctx.winner
      ? `${ctx.winner} captures general`
      : solMove
      ? `Move (${solMove.fromCol}, ${solMove.fromRow}) ${solMove.split ? "½" : ""}${solMove.arrow}`
      : "Hold";
    const vis = seg === 1 && genT.boardOp > 0.002;
    return (
      <div className="sp-root" style={{ visibility: seg === 1 ? "visible" : "hidden" }}>
        <div
          className="sp-board-zoom-layer"
          style={{
            opacity: genT.boardOp,
            transform: `translate3d(${genT.tx2.toFixed(2)}px, ${genT.ty2.toFixed(2)}px, 0) scale(${genT.s2.toFixed(4)})`,
            visibility: vis ? "visible" : "hidden",
          }}
        >
          <div className="sp-board-card" style={{ backgroundColor: `rgba(248, 248, 246, ${genT.hudIn.toFixed(3)})` }}>
            <div className="sp-hud" style={{ opacity: genT.hudIn }}>
              <div className="sp-score">
                <i style={{ background: GEN_CYAN }} />
                <span className="sp-score-name">GPT-6.1 Sol</span>
                <strong>{ctx.scores[0].army}</strong>
              </div>
              <span className="sp-turn">Tick {ctx.tick}</span>
              <div className="sp-score">
                <strong>{ctx.scores[1].army}</strong>
                <span className="sp-score-name">DeepSeek V4 Pro</span>
                <i style={{ background: GEN_MAGENTA }} />
              </div>
            </div>
            <div className="sp-board-wrap sp-gen-board-wrap">{ctx.board}</div>
            <div className="sp-act-pill" style={{ opacity: genT.hudIn }}>
              <i style={{ background: GEN_CYAN }} />
              <span>{pillText}</span>
            </div>
          </div>
        </div>

        <div
          className="sp-panel-layer"
          style={{
            opacity: genT.panelIn * genT.segVis,
            transform: `translate3d(${((1 - genT.panelIn) * -32).toFixed(2)}px, 0, 0)`,
            visibility: seg === 1 && genT.panelIn > 0.002 ? "visible" : "hidden",
          }}
        >
          <SimpleGeneralsIO
            width={ctx.width}
            height={ctx.height}
            cells={ctx.cells}
            actions={ctx.actions}
            actionWidth={ctx.actionWidth}
          />
        </div>
      </div>
    );
  };

  // Pokémon Part 2 current turn
  const pokeTurn = pokeBattle.state.finished ? 4 : Math.max(1, pokeBattle.state.turn || 1);

  // Showdown arena scale inside 624x508 card: cropped 440x360 -> 552x452 (clean centered fit, zero overlap)
  const pokeScale = 552 / 440;

  return (
    <div className="tv-frame pt ar">
      {/* Hidden Showdown battle log node (required by Showdown engine) */}
      <div className="ps-battle" style={{ display: "none" }}>
        <div className="battle-log" ref={pokeBattle.logRef} />
      </div>

      {/* Game 1: Polytopia Viewer */}
      <ReplayViewer
        src={polySrc}
        names={["GPT-6 Astra", "Starter"]}
        controlledPos={polyPos}
        onReady={onPolyReady}
        renderCustom={renderPolyCustom}
      />

      {/* Game 2: Generals Viewer */}
      <GeneralsReplay
        src={genSrc}
        names={["GPT-6.1 Sol", "DeepSeek V4 Pro"]}
        controlledPos={genPos}
        onReady={onGenReady}
        renderCustom={renderGenCustom}
      />

      {/* Game 3: Pokémon Showdown Viewer (always mounted off-opacity so sprites & Turn 1 stay ready) */}
      <div className="sp-root" style={{ visibility: seg === 2 ? "visible" : "hidden" }}>
        <div
          className="sp-board-zoom-layer"
          style={{
            opacity: seg === 2 ? pokeT.boardOp : 0,
            transform: `translate3d(${pokeT.tx2.toFixed(2)}px, ${pokeT.ty2.toFixed(2)}px, 0) scale(${pokeT.s2.toFixed(4)})`,
            visibility: seg === 2 && pokeT.boardOp > 0.002 ? "visible" : "hidden",
          }}
        >
          <div className="sp-board-card" style={{ backgroundColor: `rgba(248, 248, 246, ${pokeT.hudIn.toFixed(3)})` }}>
            <div className="sp-hud" style={{ opacity: pokeT.hudIn }}>
              <div className="sp-score">
                <i style={{ background: POKE_CYAN }} />
                <span className="sp-score-name">GPT-6.1 Sol</span>
              </div>
              <span className="sp-turn">{pokeBattle.state.finished ? "Final" : `Turn ${pokeTurn}`}</span>
              <div className="sp-score">
                <span className="sp-score-name">GLM-5.3</span>
                <i style={{ background: POKE_MAGENTA }} />
              </div>
            </div>

            <div className="sp-board-wrap sp-poke-board-wrap">
              <div className="psr psr-crop sp-poke-psr">
                <div className="psr-stage sp-poke-stage" style={{ width: 552, height: Math.round(360 * pokeScale) }}>
                  <div
                    className="psr-scale ps-battle"
                    style={{ transform: `scale(${pokeScale}) translateX(-100px)` }}
                  >
                    <div className="battle" ref={pokeBattle.frameRef} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div
          className="sp-panel-layer"
          style={{
            opacity: seg === 2 ? pokeT.panelIn * pokeT.segVis : 0,
            transform: `translate3d(${((1 - pokeT.panelIn) * -32).toFixed(2)}px, 0, 0)`,
            visibility: seg === 2 && pokeT.panelIn > 0.002 ? "visible" : "hidden",
          }}
        >
          <SimplePokemonIO snapshots={pokeSnapshots} turn={pokeTurn} finished={pokeBattle.state.finished} />
        </div>
      </div>

      {/* Shared Part 1: ResearchLoop (switches cleanly between Polytopia, Generals, and Pokémon) */}
      <div
        className="tv-act1-layer"
        style={{
          opacity: activeRl.act1Op,
          transform: `translate3d(${activeRl.tx1.toFixed(2)}px, ${activeRl.ty1.toFixed(2)}px, 0) scale(${activeRl.s1.toFixed(4)})`,
          visibility: activeRl.act1Op > 0.002 ? "visible" : "hidden",
        }}
      >
        <div className="tv-rl-wrap">
          <ResearchLoop controlled={{ scene: activeRl.scene, t: activeRl.rlTime }} onArtReady={onArtReady} />
        </div>
      </div>
    </div>
  );
}
