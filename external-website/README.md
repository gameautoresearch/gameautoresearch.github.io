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

## Hero artwork

The public build and local preview serve the same hero artwork from
`public/hero-art/`. The manifest describes the selected Polytopia sprites and
unit layers. The Generals icons come from the vendored engine, and the Pokémon
backgrounds and first-frame sprites come from Pokémon Showdown's battle art.

For a checkout without the bundled manifest, the Vite development server can
read the sprite export and vendored icons from
`GAB_HERO_ART_ROOT=/path/to/game-autoresearch-bench npm run dev`. The export
lives at `simulators/polytopia/artifacts/mobile-assets/game/` and the Showdown
stills go in `.local/hero-art/showdown/`.
