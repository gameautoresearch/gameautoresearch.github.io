// Numbers for the post, copied from the readouts under reports/ in game-autoresearch-bench (paths beside each).
// Nothing here is a private evaluation seed, map, panel or network.

// reports/pokemon/poke3v3-rep-2026-11-15.md: 38 networks, 703 pairs, 179,968 games. Lift = 100 × mean score
// against three retrains of the unmodified starter − 50, 95% hierarchical bootstrap.
export const pokemonLift = [
  { model: "GPT-6.1 Sol", lift: 18.2, ci: [13.5, 22.3], attempts: [17.3, 20.4, 13.3, 22.0, 17.8], netElo: [111, 124, 109, 143, 117] },
  { model: "GPT-6 Astra", lift: 15.5, ci: [11.7, 19.4], attempts: [14.1, 12.9, 19.7, 15.1, 15.9], netElo: [97, 117, 109, 114, 123] },
  { model: "GPT-6 Luna", lift: 8.2, ci: [1.5, 14.5], attempts: [2.9, -2.3, 11.5, 14.1, 14.7], netElo: [-3, -5, 75, 95, 95] },
  { model: "Inkling 2 Small", lift: 6.7, ci: [2.0, 12.1], attempts: [4.9, 12.5, 1.3, 4.2, 10.5], netElo: [28, 88, -2, 24, 91] },
  { model: "DeepSeek V4 Pro", lift: 4.3, ci: [0.9, 7.7], attempts: [7.0, 3.3, 4.8, 0.7, 5.6], netElo: [66, 15, 54, 16, 24] },
  { model: "Kimi K3", lift: 3.7, ci: [-2.1, 9.9], attempts: [3.3, 3.6, -2.5, 14.7, -0.8], netElo: [9, 53, -8, 116, 30] },
  { model: "GLM-5.3", lift: 1.7, ci: [-3.2, 7.5], attempts: [-0.5, -0.7, 3.0, -4.4, 11.2], netElo: [0, -2, 19, -28, 66] },
] as const;

// reports/clash/clash-short-batch-2026-11-15.md
export const clashLift = [
  { model: "GLM-5.3", lift: -1.6, ci: [-8.1, 4.9] },
  { model: "Inkling 2 Small", lift: -7.9, ci: [-20.2, 3.0] },
  { model: "Kimi K3", lift: -10.3, ci: [-24.5, 2.1] },
  { model: "DeepSeek V4 Pro", lift: -18.9, ci: [-42.3, -1.4] },
] as const;

// reports/pokemon/poke3v3-compute-vs-intelligence-2026-11-15/analysis.json (3-attempt data): best research network
// so far, by session hour, mean of three attempts, Elo over the starter.
export const bestSoFar = {
  hours: [1.25, 2.25, 3.25, 4.75],
  series: [
    { model: "GPT-6.1 Sol", elo: [94, 121, 128, 145] },
    { model: "GPT-6 Astra", elo: [14, 97, 97, 113] },
    { model: "DeepSeek V4 Pro", elo: [-12, 60, 60, 66] },
    { model: "Inkling 2 Small", elo: [13, 21, 21, 43] },
    { model: "GPT-6 Luna", elo: [26, 30, 34, 34] },
    { model: "Kimi K3", elo: [-21, -7, -5, -5] },
    { model: "GLM-5.3", elo: [-19, 3, 3, 5] },
  ],
};

// Same analysis: the unmodified starter trained for different lengths (minutes of L40S, Elo over the 1 h starter).
// Fit: +11.7 Elo per doubling (95% 9.2–24.0).
export const starterCurve = {
  points: [
    { minutes: 4.1, elo: -37.2 },
    { minutes: 10.5, elo: -22.7 },
    { minutes: 25.4, elo: -26.3 },
    { minutes: 25.4, elo: -31.8 },
    { minutes: 25.6, elo: -7.2 },
    { minutes: 54.0, elo: 6.3 },
    { minutes: 55.4, elo: 9.0 },
    { minutes: 55.5, elo: 2.3 },
    { minutes: 55.5, elo: -1.5 },
    { minutes: 55.5, elo: -0.8 },
  ],
  intercept: -67.5,
  perDoubling: 11.7,
  sol: 112.8,
  astra: 108,
};

// Polytopia, Nov 15 recipe suite (reports/polytopia/recipe-suite-2026-11-15.md): private grade, 95% CI.
export const polytopiaGrades = [
  { who: "Claude Opus 5.5", grade: 67.9, ci: [65.0, 70.6] },
  { who: "DeepSeek V4.1 Flash", grade: 65.0, ci: [62.9, 67.1] },
  { who: "GPT-6.1 Terra", grade: 60.5, ci: [57.5, 63.5] },
  { who: "GPT-6 Astra", grade: 59.2, ci: [56.4, 61.7] },
  { who: "Claude Fable 5.2", grade: 59.1, ci: [56.3, 61.8] },
  { who: "GPT-6 Luna (unchanged starter)", grade: 59.0, ci: [56.5, 61.6] },
  { who: "GPT-6.1 Sol", grade: 54.2, ci: [51.4, 57.0] },
] as const;

// Cross-game tournament Elo (reports/suite-2026-11-13.md; reports/polytopia/open-tournament-2026-11-15.md).
export const crossGame = [
  { game: "Generals", rows: [["GPT-6 Astra", 265], ["Claude Opus 5.5", 260], ["GPT-6.1 Sol", 178], ["GPT-6.1 Terra", 112], ["DeepSeek V4.1 Flash", -383], ["GPT-6 Luna", -431]] },
  { game: "Hex 9×9 (MoHex = 0)", rows: [["GPT-6.1 Sol", 64], ["Claude Opus 5.5", 33], ["GPT-6 Astra", -241], ["DeepSeek V4.1 Flash", -507], ["GPT-6 Luna", -522], ["GPT-6.1 Terra", -723]] },
] as const;

// Pokémon 3v3 cost per run, mean of three attempts (poke3v3-cost-effectiveness-2026-11-15). API at list price.
export const pokemonCost = [
  { model: "GPT-6.1 Sol", api: 0.88, gpu: 14.05, eloPerDollar: 7.6 },
  { model: "GPT-6 Astra", api: 3.95, gpu: 13.98, eloPerDollar: 6.0 },
  { model: "Inkling 2 Small", api: 1.59, gpu: 13.6, eloPerDollar: 2.6 },
  { model: "DeepSeek V4 Pro", api: 7.44, gpu: 12.94, eloPerDollar: 2.2 },
  { model: "GPT-6 Luna", api: 0.41, gpu: 13.43, eloPerDollar: 1.6 },
  { model: "Kimi K3", api: 5.91, gpu: 13.62, eloPerDollar: 1.0 },
  { model: "GLM-5.3", api: 10.3, gpu: 12.86, eloPerDollar: 0.4 },
] as const;

// Polytopia compute frontier (reports/polytopia/compute-frontier-2026-11-17.md, data.json beside it): the unmodified
// starter trained for different lengths, one Bradley-Terry fit over 72,976 games, the 200M-sample starter = 0.
// Fresh finals: [hours, Elo]. Snapshots: [hours, Elo] inside the two 16 h runs (their finals were lost at hand-off).
export const frontier = {
  starter: [[0.5, -60.5], [0.5, -52.3], [1, 47.7], [1, 38.2], [1, 40.1], [2, 134.5], [2, 167.1], [2, 125.2], [2, 147.5], [4, 198.0], [4, 220.0], [4, 212.9]],
  snapshots: [
    [0.5, -56.4], [1, 42.1], [1.5, 65.5], [2, 96.9], [3, 131.4], [4, 167.8], [6, 273.1], [8, 305.6], [10, 323.1], [12, 334.4], [14, 356.7], [14.5, 347.4],
    [0.5, -2.8], [1, 55.3], [1.5, 126.0], [2, 152.5], [3, 164.8], [4, 212.9], [6, 239.4], [8, 280.6], [10, 298.4], [12, 327.9], [14, 336.4], [14.5, 338.2],
  ],
  opus: [[2, 200.2], [2, 244.2], [2, 228.4], [8, 345.2], [8, 345.4]],
  flash: [[2, 181.5], [2, 173.0], [2, 150.8], [8, 292.8], [8, 318.3]],
  random: -1008,
};

// Empty starter, best research network so far by research GPU-hours ([GPU-h when sealed, Elo]); `end` = GPU-hours
// used. Polytopia: reports/polytopia/empty15-2026-11-18.md (elo.json; 15-min PPO starter = 0, 753,600 games).
// Pokémon: reports/pokemon/poke3v3-empty15b-2026-11-17/README.md (checkpoint-elo/elo.json; mean of two 15-min PPO
// starters = 0, 366,336 games; campaign stopped at 3.7 h of its 5 h wall clock).
export const emptyPolytopia = {
  models: [
    { model: "GPT-6 Astra", end: 4.96, final: 206, best: [[0.078, -112], [0.208, 8], [0.636, 134], [0.85, 191], [1.063, 219], [2.981, 220], [4.045, 244], [4.681, 269]] },
    { model: "GPT-6.1 Sol", end: 4.91, final: 136, best: [[0.05, -110], [0.209, 86], [0.424, 94], [0.636, 110], [0.849, 147], [1.063, 159], [3.186, 167], [3.823, 185]] },
    { model: "Kimi K3.1 Pro", end: 4.58, final: -54, best: [[0.065, -1459], [0.146, -1029], [0.355, -386], [0.776, -224], [2.669, -98], [3.091, -79], [3.511, -72], [3.937, -62], [4.148, -53]] },
    { model: "Tencent Hy4", end: 4.52, final: -520, best: [[0.067, -552], [0.273, -526], [1.775, -522], [2.807, -518], [3.426, -516]] },
    { model: "GLM-5.3", end: 4.78, final: -758, best: [[0.116, -2938], [0.169, -1346], [0.378, -791], [1.637, -673], [2.262, -615]] },
    { model: "DeepSeek V4 Flash", end: 4.52, final: -1386, best: [[0.082, -2938], [0.167, -1588], [0.241, -1420], [1.496, -1410], [2.34, -1402], [2.75, -1382]] },
  ],
  refs: { starter200m: 238, random: -1315 },
};
export const emptyPokemon = {
  models: [
    { model: "GPT-6 Astra", end: 2.64, best: [[0.174, 58], [0.382, 92], [0.587, 96], [2.025, 108]] },
    { model: "Kimi K3.1 Pro", end: 2.36, best: [[0.056, -4], [0.313, 75], [0.521, 84], [0.727, 86], [2.155, 97]] },
    { model: "GLM-5.3", end: 2.24, best: [[0.178, 12], [0.802, 17], [1.006, 22], [1.832, 53]] },
    { model: "DeepSeek V4 Flash", end: 2.37, best: [[0.019, -52], [0.09, -20], [0.297, -2], [0.714, 2], [0.914, 21]] },
  ],
  emptyStarter: -328,
};
