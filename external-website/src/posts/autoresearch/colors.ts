// CMYK process palette for the post: the four printing plates on a white page. Cyan, magenta and yellow carry
// the models the story is about; key (black) and greys carry everyone else. Each model keeps one color in every
// chart. Cyan/magenta/yellow pass the colorblind checks of the dataviz validator against each other (all pairs).
export const CMYK = {
  c: "#00a0d6", cDeep: "#0081ad", cLight: "#7fcfea",
  m: "#e0218a", mDeep: "#b8156e",
  y: "#d4a000", yDeep: "#8f6c00",
  k: "#111111",
  grey: "#bdbdbd",
} as const;

// One color per researcher (no dashes: color alone identifies a model), used by every line and dot chart. Keys are matched in this order against the
// lower-cased model name, so more specific names come first ("6.2 astra" before "astra"). The extra hues (blue,
// orange, purple, teal) pass the dataviz validator together with cyan, magenta and yellow in legend order.
const MODEL: Record<string, { color: string }> = {
  "6.2 astra": { color: "#f08cc0" },
  neon: { color: "#2f55c4" },
  carbon: { color: "#7f9ae0" },
  hydrogen: { color: "#1f9e89" },
  fable: { color: "#e8740f" },
  muse: { color: "#8a5cc4" },
  sol: { color: CMYK.c },
  astra: { color: CMYK.m },
  opus: { color: CMYK.y },
  kimi: { color: CMYK.k },
  deepseek: { color: "#6b6b6b" },
  glm: { color: "#9a9a9a" },
  luna: { color: "#b5b5b5" },
  inkling: { color: "#b5b5b5" },
  hy4: { color: "#b5b5b5" },
};

export function modelStyle(name: string) {
  const n = name.toLowerCase();
  const key = Object.keys(MODEL).find((k) => n.includes(k));
  return key ? MODEL[key] : { color: CMYK.grey };
}

// Diverging fill for "better or worse than the starter": magenta below, grey around 0, cyan above.
export function liftFill(t: number) {
  const grey = [226, 226, 222], up = [0, 129, 173], down = [224, 33, 138];
  const to = t >= 0 ? up : down, k = Math.min(1, Math.abs(t));
  return `rgb(${grey.map((b, i) => Math.round(b + (to[i] - b) * k)).join(",")})`;
}
