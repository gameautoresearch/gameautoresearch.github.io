import { createReadStream, existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";
import { POLY_SPRITES } from "../src/posts/autoresearch/hero/polySprites";

const site = fileURLToPath(new URL("../", import.meta.url));
const types = { png: "image/png", jpg: "image/jpeg" } as const;

export function localHeroArt(): Plugin {
  return {
    name: "local-hero-art",
    apply: "serve",
    configureServer(server) {
      if (existsSync(resolve(server.config.publicDir, "hero-art/manifest.json"))) return;
      const repo = process.env.GAB_HERO_ART_ROOT || resolve(site, "..");
      const poly = resolve(repo, "simulators/polytopia/artifacts/mobile-assets/game");
      const generals = resolve(repo, "vendor/engines/generals-bots/generals/assets/images");
      const showdown = resolve(site, ".local/hero-art/showdown");
      const files = new Map<string, string>();
      const add = (group: string, dir: string, file: string) => {
        if (!/^[\w .-]+\.(png|jpg)$/.test(file) || file.startsWith(".")) return;
        const path = resolve(dir, file);
        if (existsSync(path)) files.set(`${group}/${file}`, path);
      };

      let polytopia = null;
      const manifestPath = resolve(poly, "game.json");
      if (existsSync(manifestPath)) {
        const { sprites, units } = JSON.parse(readFileSync(manifestPath, "utf8")) as {
          sprites: Record<string, { file: string }>;
          units: Record<string, unknown>;
        };
        const selected = Object.fromEntries<{ file: string }>(
          POLY_SPRITES.flatMap((name) => sprites[name] ? [[name, sprites[name]] as const] : []),
        );
        for (const { file } of Object.values(selected)) add("polytopia", poly, file);
        if (selected.ground_bardur && selected.ground_imperius && Object.values(selected).every(({ file }) => files.has(`polytopia/${file}`))) {
          polytopia = {
            sprites: selected,
            units: Object.fromEntries([2, 3, 4, 5, 9, 11, 12].map((kind) => [kind, units[kind]])),
          };
        }
      }

      const generalFiles = ["crownie.png", "citie.png", "mountainie.png"];
      const pokemonFiles = ["entei.png", "decidueye.png", "houndoom-f.png", "groudon.png", "bg-meadow.jpg", "bg-forest.jpg"];
      for (const file of generalFiles) add("generals", generals, file);
      for (const file of pokemonFiles) add("pokemon", showdown, file);

      const manifest = JSON.stringify({
        polytopia,
        generals: generalFiles.every((file) => files.has(`generals/${file}`)),
        pokemon: pokemonFiles.every((file) => files.has(`pokemon/${file}`)),
      });
      const prefix = `${server.config.base}hero-art/`;
      server.middlewares.use((req, res, next) => {
        const url = req.url?.split("?", 1)[0];
        if (!url?.startsWith(prefix)) return next();
        const key = url.slice(prefix.length);
        if (key === "manifest.json") {
          res.setHeader("Content-Type", "application/json");
          return res.end(manifest);
        }
        let path: string | undefined;
        try { path = files.get(decodeURIComponent(key)); } catch { /* Invalid URL encoding. */ }
        if (!path) {
          res.statusCode = 404;
          return res.end();
        }
        const extension = path.endsWith(".png") ? "png" : "jpg";
        res.setHeader("Content-Type", types[extension]);
        res.setHeader("Cache-Control", "private, max-age=3600");
        createReadStream(path).on("error", () => res.destroy()).pipe(res);
      });
    },
  };
}
