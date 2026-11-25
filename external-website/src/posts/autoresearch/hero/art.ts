import { useEffect, useState } from "react";

type Sprite = { file: string; w: number; h: number; ax: number; ay: number; unit: number };
type UnitPart = [name: string, x: number, y: number, sx: number, sy: number, rotation: number, layer: number, tint: boolean];
type PolyManifest = { sprites: Record<string, Sprite>; units: Record<string, UnitPart[]> };
type Manifest = { polytopia: PolyManifest | null; generals: boolean; pokemon: boolean };
export type PolyArt = PolyManifest & { urls: Record<string, string>; tinted: Record<string, string> };
export type HeroArt = { polytopia: PolyArt | null; generals: boolean; pokemon: boolean };
const EMPTY_ART: HeroArt = { polytopia: null, generals: false, pokemon: false };
const base = `${import.meta.env.BASE_URL}__hero-art/`;

export const artUrl = (game: string, file: string) => `${base}${game}/${encodeURIComponent(file)}`;

async function loadImage(url: string) {
  const image = new Image();
  image.src = url;
  await image.decode();
  return image;
}

function tintedUrl(image: HTMLImageElement, color: string) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(image, 0, 0);
  ctx.globalCompositeOperation = "multiply";
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(image, 0, 0);
  return canvas.toDataURL("image/png");
}

async function loadPolytopia(manifest: PolyManifest): Promise<PolyArt> {
  const urls: PolyArt["urls"] = {};
  const tinted: PolyArt["tinted"] = {};
  await Promise.all(Object.entries(manifest.sprites).map(async ([name, { file }]) => {
    const url = artUrl("polytopia", file);
    const image = await loadImage(url);
    urls[name] = url;
    if (name.startsWith("Border") || name.includes("tint")) {
      for (const color of ["#00a0d6", "#e0218a"]) {
        const source = tintedUrl(image, color);
        if (source) tinted[`${name}:${color}`] = source;
      }
    }
  }));
  return { ...manifest, urls, tinted };
}

export function useHeroArt(enabled: boolean) {
  const [art, setArt] = useState(EMPTY_ART);

  useEffect(() => {
    if (!import.meta.env.DEV || !enabled) return;
    const controller = new AbortController();
    async function load() {
      const response = await fetch(`${base}manifest.json`, { signal: controller.signal });
      if (!response.ok) return;
      const manifest: Manifest = await response.json();
      if (controller.signal.aborted) return;
      setArt({ ...manifest, polytopia: null });
      if (manifest.polytopia) {
        const polytopia = await loadPolytopia(manifest.polytopia);
        if (!controller.signal.aborted) setArt({ ...manifest, polytopia });
      }
    }
    void load().catch(() => { /* The illustrative boards are available without the local art files. */ });
    return () => controller.abort();
  }, [enabled]);

  return art;
}
