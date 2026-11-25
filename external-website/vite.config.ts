import { readdirSync, readFileSync } from "node:fs";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { metaTags, PAGE } from "./src/meta";
import { localHeroArt } from "./tools/heroArt";

const pagesBase = process.env.PAGES_BASE_PATH;
const vercelHost = process.env.VERCEL_PROJECT_PRODUCTION_URL;
const siteUrl = (process.env.PAGES_SITE_URL ?? (vercelHost ? `https://${vercelHost}` : "")).replace(/\/$/, "");

// Link previews: the page's title and tags go straight into index.html.
function linkPreviews(): Plugin {
  return {
    name: "link-previews",
    transformIndexHtml: (html) =>
      html.replace(/<!-- meta -->/, metaTags(siteUrl)).replace(/<title>.*<\/title>/, `<title>${PAGE.title}</title>`),
  };
}

// Pages serves files directly. Each replay needs an index for bookmarks and reloads.
function staticReplayPages(): Plugin {
  return {
    name: "static-replay-pages",
    apply: "build",
    generateBundle: {
      order: "post",
      handler(_options, bundle) {
        const page = bundle["index.html"];
        if (page?.type !== "asset") this.error("Article index was not generated.");
        const files = readdirSync(new URL("./public/pokemon/", import.meta.url));
        for (const file of files.filter((name) => /^[\w-]+\.json$/.test(name))) {
          this.emitFile({ type: "asset", fileName: `replays/pokemon/${file.slice(0, -5)}/index.html`, source: page.source });
        }
        this.emitFile({ type: "asset", fileName: "trajectories/index.html", source: page.source });
        const trajs = readdirSync(new URL("./public/trajectories/", import.meta.url));
        for (const file of trajs.filter((name) => /^[\w-]+\.json$/.test(name))) {
          this.emitFile({ type: "asset", fileName: `trajectories/${file.slice(0, -5)}/index.html`, source: page.source });
        }
        const notices = new URL("./src/showdown/licenses/", import.meta.url);
        for (const notice of readdirSync(notices)) {
          this.emitFile({
            type: "asset",
            fileName: `licenses/showdown/${notice}`,
            source: readFileSync(new URL(notice, notices)),
          });
        }
      },
    },
  };
}

export default defineConfig({
  base: pagesBase ? `${pagesBase.replace(/\/$/, "")}/` : "/",
  plugins: [react(), linkPreviews(), staticReplayPages(), localHeroArt()],
  // Pokémon Showdown's battle engine is one ~790 kB chunk, loaded only when a battle replay comes into view.
  build: { chunkSizeWarningLimit: 900 },
});
