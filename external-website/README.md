# External website

This folder contains the benchmark research article. GitHub Actions publishes it
at https://gameautoresearch.github.io/game-autoresearch-bench/.

Run the following from this folder with Node.js 24:

```sh
npm ci --ignore-scripts
npm run dev
```

To build for the GitHub Pages project path:

```sh
PAGES_BASE_PATH=/game-autoresearch-bench \
  PAGES_SITE_URL=https://gameautoresearch.github.io/game-autoresearch-bench \
  npm run build
```

The build writes `dist/`, including a static entry point for each Pokémon replay.
Upstream Showdown sources and notices are under `src/showdown/`.
