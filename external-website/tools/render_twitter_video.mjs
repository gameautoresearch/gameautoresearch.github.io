#!/usr/bin/env node
// Deterministic 1920x1080 3-game Twitter video renderer (Polytopia -> Generals -> Pokémon).
// Drives Vite + headless Google Chrome over native CDP WebSocket with deterministic virtual-clock
// stepping for Pokémon Showdown animations, piping lossless PNG frames directly into ffmpeg.

import { spawn } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const args = process.argv.slice(2);
const flag = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
};

const keyframesDir = flag("--keyframes", null);
const outPath = resolve(ROOT, flag("--out", "../polytopia-twitter.mp4"));
const fps = Number(flag("--fps", "30"));
const vitePort = Number(flag("--vite-port", String(4179 + Math.floor(Math.random() * 400))));
const cdpPort = Number(flag("--cdp-port", String(9322 + Math.floor(Math.random() * 400))));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForHttp(url, timeoutMs = 15000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return res;
    } catch {}
    await sleep(120);
  }
  throw new Error(`Timed out waiting for ${url}`);
}

class CDPClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.ready = new Promise((resolveReady, rejectReady) => {
      this.ws.addEventListener("open", resolveReady, { once: true });
      this.ws.addEventListener("error", rejectReady, { once: true });
    });
    this.ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(String(ev.data));
      if (msg.id && this.pending.has(msg.id)) {
        const { res, rej } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) rej(new Error(msg.error.message || JSON.stringify(msg.error)));
        else res(msg.result);
      }
    });
  }

  async send(method, params = {}) {
    await this.ready;
    const id = this.nextId++;
    return new Promise((res, rej) => {
      this.pending.set(id, { res, rej });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async eval(expression) {
    const out = await this.send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (out.exceptionDetails) {
      throw new Error(
        `Page eval error: ${out.exceptionDetails.exception?.description || out.exceptionDetails.text}`,
      );
    }
    return out.result?.value;
  }

  close() {
    this.ws.close();
  }
}

async function main() {
  const profileDir = join(tmpdir(), `ag-chrome-video-${process.pid}`);
  mkdirSync(profileDir, { recursive: true });

  const vite = spawn("npx", ["vite", "--host", "127.0.0.1", "--port", String(vitePort), "--strictPort"], {
    cwd: ROOT,
    stdio: ["ignore", "pipe", "pipe"],
  });
  let viteErr = "";
  vite.stderr.on("data", (d) => (viteErr += d.toString()));

  let chrome = null;
  let cdp = null;
  try {
    await waitForHttp(`http://127.0.0.1:${vitePort}/video.html`);
    console.log(`[vite] ready on :${vitePort}`);

    chrome = spawn(
      CHROME,
      [
        "--headless=new",
        `--remote-debugging-port=${cdpPort}`,
        `--user-data-dir=${profileDir}`,
        "--window-size=960,540",
        "--force-device-scale-factor=2",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
        "about:blank",
      ],
      { stdio: ["ignore", "pipe", "pipe"] },
    );

    const targetsRes = await waitForHttp(`http://127.0.0.1:${cdpPort}/json/list`);
    const targets = await targetsRes.json();
    const pageTarget = targets.find((t) => t.type === "page") || targets[0];
    if (!pageTarget?.webSocketDebuggerUrl) {
      throw new Error("Could not find Chrome CDP page target");
    }

    cdp = new CDPClient(pageTarget.webSocketDebuggerUrl);
    await cdp.send("Page.enable");
    await cdp.send("Runtime.enable");
    await cdp.send("Emulation.setDeviceMetricsOverride", {
      width: 960,
      height: 540,
      deviceScaleFactor: 2,
      mobile: false,
    });

    await cdp.send("Page.navigate", { url: `http://127.0.0.1:${vitePort}/video.html?vclock=1` });

    // Wait for fonts, hero sprites, Polytopia, Generals, and Pokémon Showdown Turn 1 to finish loading
    const t0 = Date.now();
    while (Date.now() - t0 < 25000) {
      const ready = await cdp.eval("Boolean(window.__videoReady)");
      if (ready) break;
      await sleep(120);
    }
    const isReady = await cdp.eval("Boolean(window.__videoReady)");
    if (!isReady) throw new Error("Timed out waiting for window.__videoReady");

    const durationMs = await cdp.eval("window.__videoDurationMs || 41500");
    const pokePlayStartMs = await cdp.eval("window.__pokePlayStartMs || 33150");
    console.log(`[page] ready! duration=${durationMs}ms pokePlayStart=${pokePlayStartMs}ms`);

    // Enable deterministic virtual clock once all assets & Showdown Turn 1 are loaded
    await cdp.eval("window.__vclock && window.__vclock.enable()");

    const POKE_CLOCK_MULT = 2.25;
    let lastPokeAdvanceMs = pokePlayStartMs;

    const renderFrameAt = async (ms) => {
      const advancePoke = ms > pokePlayStartMs ? (ms - lastPokeAdvanceMs) * POKE_CLOCK_MULT : 0;
      if (ms > pokePlayStartMs) lastPokeAdvanceMs = ms;

      await cdp.eval(`
        new Promise((resolve) => {
          window.__setVideoTime(${ms});
          ${advancePoke > 0 ? `window.__vclock && window.__vclock.advance(${advancePoke}); window.__pokeSync && window.__pokeSync();` : ""}
          const raf = window.__realClock ? window.__realClock.raf : requestAnimationFrame;
          raf(() => raf(resolve));
        })
      `);
      const shot = await cdp.send("Page.captureScreenshot", {
        format: "png",
        fromSurface: true,
      });
      return Buffer.from(shot.data, "base64");
    };

    if (keyframesDir) {
      mkdirSync(keyframesDir, { recursive: true });
      const stamps = [36500, 39000, 40800, 42100];
      for (const ms of stamps) {
        const png = await renderFrameAt(ms);
        const f = join(keyframesDir, `frame_${String(ms).padStart(5, "0")}ms.png`);
        writeFileSync(f, png);
        console.log(`[keyframe] wrote ${f} (${png.length} bytes)`);
      }
      return;
    }

    const totalFrames = Math.round((durationMs / 1000) * fps);
    console.log(`[render] ${totalFrames} frames @ ${fps}fps -> ${outPath}`);

    const ff = spawn(
      "ffmpeg",
      [
        "-y",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-framerate", String(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level:v", "4.2",
        "-crf", "17",
        "-preset", "medium",
        "-movflags", "+faststart",
        outPath,
      ],
      { stdio: ["pipe", "inherit", "pipe"] },
    );

    let ffErr = "";
    ff.stderr.on("data", (d) => (ffErr += d.toString()));

    const renderStart = Date.now();
    for (let i = 0; i < totalFrames; i++) {
      const ms = (i / fps) * 1000;
      const png = await renderFrameAt(ms);
      if (!ff.stdin.write(png)) {
        await new Promise((r) => ff.stdin.once("drain", r));
      }
      if ((i + 1) % 90 === 0 || i === totalFrames - 1) {
        const elapsed = ((Date.now() - renderStart) / 1000).toFixed(1);
        console.log(`[render] ${i + 1}/${totalFrames} frames (${elapsed}s)`);
      }
    }
    ff.stdin.end();
    const exitCode = await new Promise((r) => ff.on("close", r));
    if (exitCode !== 0) {
      throw new Error(`ffmpeg failed (code ${exitCode}):\n${ffErr}`);
    }
    console.log(`[done] ${outPath}`);
  } finally {
    cdp?.close();
    chrome?.kill("SIGKILL");
    vite.kill("SIGKILL");
    await sleep(250);
    try {
      rmSync(profileDir, { recursive: true, force: true });
    } catch {}
    if (viteErr && viteErr.includes("Error")) {
      console.error("[vite stderr]", viteErr);
    }
  }
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
