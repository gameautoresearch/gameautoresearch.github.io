# External website

This folder contains the benchmark research article. GitHub Actions publishes it
at https://gameautoresearch.github.io/.

Run the following from this folder with Node.js 24:

```sh
npm ci --ignore-scripts
npm run dev
```

To build for GitHub Pages:

```sh
PAGES_SITE_URL=https://gameautoresearch.github.io npm run build
```

The build writes `dist/`, including a static entry point for each Pokémon replay.
Upstream Showdown sources and notices are under `src/showdown/`.

## Local hero artwork

The Vite development server can use the same Polytopia sprite export as the
editorial animation. If it lives in another worktree, start the server with
`GAB_HERO_ART_ROOT=/path/to/game-autoresearch-bench npm run dev`. The server
looks for the export at `simulators/polytopia/artifacts/mobile-assets/game/`
and reads the three Generals icons from the vendored engine.

The Pokémon preview uses local Showdown stills under
`.local/hero-art/showdown/`. The expected files are `entei.png` and
`houndoom-f.png` (back sprites), `decidueye.png` and `groudon.png` (front
sprites), plus `bg-forest.jpg` and `bg-meadow.jpg`. The preview PNGs are first
frames from the existing offline replay cache. Vite serves only the selected
artwork in development. The production build uses the illustrated boards and
contains no local game artwork.
