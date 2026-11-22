// Bring Pokémon Showdown's battle engine into the site as TypeScript modules: src/showdown/.
//
// Source: smogon/pokemon-showdown-client at a pinned commit. The battle replay and animation engine (battle-*.ts) is
// MIT by its own headers; battle-animations-moves.ts is CC0; the Caja HTML sanitizer is Apache-2.0; the battle styles
// are GPL-2.0. Built data (Pokédex, moves, items, abilities, battle text, sprite sizes) comes from
// play.pokemonshowdown.com/data and is converted to JSON. Sprites, backgrounds, effects and sounds are game art and stay
// on Showdown's server (the repository does not publish game assets).
//
// Showdown's own build concatenates these modules into global scripts, so the source leans on globals in a few places.
// The patches below (PATCHES) make them work as real ES modules; client-main.ts and client-connection.ts are the
// site's small stand-ins for the parts of Showdown's client the engine imports, and engine/index.ts fixes the module
// evaluation order (the site imports the engine from there). Everything is recorded in
// src/showdown/MANIFEST.json.
//
// Run again only to update the copy on purpose: node tools/vendor_showdown.mjs
import { createHash } from "node:crypto";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const COMMIT = "1cf9b385a97ad5c77a25b91a1c799078827ec306";
const REPO = `https://raw.githubusercontent.com/smogon/pokemon-showdown-client/${COMMIT}`;
const PS = "https://play.pokemonshowdown.com";
const SITE = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(SITE, "src/showdown");

const ENGINE = ["battle.ts", "battle-animations.ts", "battle-animations-moves.ts", "battle-dex.ts", "battle-dex-data.ts",
  "battle-log.ts", "battle-text-parser.ts", "battle-tooltips.ts", "battle-sound.ts", "battle-scene-stub.ts",
  "battle-teams.ts", "globals.d.ts"];
const SCRIPTS = { "battle-log-misc.js": "play.pokemonshowdown.com/src/battle-log-misc.js",
  "html-sanitizer.js": "play.pokemonshowdown.com/js/lib/html-sanitizer-minified.js" };
const STYLES = ["battle.css", "battle-log.css"];
// data file -> the global it defines
const DATA = { "pokedex.js": "BattlePokedex", "moves.js": "BattleMovedex", "abilities.js": "BattleAbilities",
  "items.js": "BattleItems", "text.js": "BattleText", "aliases.js": "BattleAliases", "typechart.js": "BattleTypeChart",
  "pokedex-mini.js": "BattlePokemonSprites", "pokedex-mini-bw.js": "BattlePokemonSpritesBW" };
const LICENSES = {
  "LICENSE-MIT.txt": "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/LICENSE",
  "LICENSE-AGPL-3.0.txt": `${REPO}/LICENSE`,
  "LICENSE-GPL-2.0.txt": "https://www.gnu.org/licenses/old-licenses/gpl-2.0.txt",
  "LICENSE-Apache-2.0.txt": "https://www.apache.org/licenses/LICENSE-2.0.txt",
};

// [file, exact text, replacement, why]
const PATCHES = [
  ["battle-log.ts",
    "// defined in battle-log-misc\ndeclare function MD5(input: string): string;\ndeclare function formatText(input: string, isTrusted?: boolean): string;\nexport { MD5, formatText };",
    "// Site patch (ES modules): MD5 is defined by battle-log-misc.js, which load.ts runs as a global script. formatText\n// (chat formatting) belongs to Showdown's chat client, not the battle engine, so chat text is shown escaped.\nconst MD5 = (input: string): string => (window as any).MD5(input);\nconst formatText = (input: string, _isTrusted?: boolean): string => BattleLog.escapeHTML(input);\nexport { MD5, formatText };",
    "Showdown's build turns these declared globals into real ones; an ES module cannot export an undeclared name."],
];

const sha = (b) => createHash("sha256").update(b).digest("hex");
async function fetchText(url) {
  const r = await fetch(url, { headers: { "User-Agent": "showdown-vendor-script" } });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return await r.text();
}

// Prefix every selector so Showdown's styles apply only inside .ps-battle (and its tooltip, which it appends to body).
const SCOPE = ":where(.ps-battle, #tooltipwrapper)";
function scopeSelector(sel) {
  sel = sel.trim();
  if (!sel) return sel;
  if (/^(html|body)$/.test(sel)) return SCOPE;
  if (/^(html|body)\s/.test(sel)) return sel.replace(/^(html|body)\s+/, `${SCOPE} `);
  if (sel.startsWith("#tooltipwrapper")) return sel;
  return `${SCOPE} ${sel}`;
}
function scopeCss(css) {
  let out = "", i = 0;
  css = css.replace(/\/\*[\s\S]*?\*\//g, "");
  while (i < css.length) {
    const open = css.indexOf("{", i);
    if (open < 0) { out += css.slice(i); break; }
    const head = css.slice(i, open).trim();
    let depth = 1, j = open + 1;
    while (depth && j < css.length) { if (css[j] === "{") depth++; else if (css[j] === "}") depth--; j++; }
    const body = css.slice(open + 1, j - 1);
    if (head.startsWith("@media") || head.startsWith("@supports")) out += `${head} {\n${scopeCss(body)}}\n`;
    else if (head.startsWith("@")) out += `${head} {${body}}\n`;
    else out += `${head.split(",").map(scopeSelector).join(", ")} {${body}}\n`;
    i = j;
  }
  return out;
}

async function main() {
  // Replace only what this script writes; the site's own files in src/showdown (engine/client-*.ts, engine/index.ts,
  // load.ts, react/, index.ts)
  // stay.
  for (const d of ["data", "vendor", "styles", "licenses"]) await rm(join(OUT, d), { recursive: true, force: true });
  for (const f of ENGINE) await rm(join(OUT, "engine", f), { force: true });
  for (const d of ["engine", "data", "vendor", "styles", "licenses"]) await mkdir(join(OUT, d), { recursive: true });
  const manifest = { repository: "https://github.com/smogon/pokemon-showdown-client", commit: COMMIT,
    fetched: new Date().toISOString().slice(0, 10), files: {}, patches: [] };
  const record = (path, url, upstream, written, license) =>
    (manifest.files[path] = { url, license, sha256_upstream: sha(upstream), sha256: sha(written) });

  const engine = {};
  for (const f of ENGINE) engine[f] = await fetchText(`${REPO}/play.pokemonshowdown.com/src/${f}`);
  const original = { ...engine };
  for (const [f, from, to, why] of PATCHES) {
    if (!engine[f].includes(from)) throw new Error(`patch target missing in ${f}`);
    engine[f] = engine[f].replace(from, to);
    manifest.patches.push({ file: `engine/${f}`, why });
  }
  for (const f of ENGINE) {
    await writeFile(join(OUT, "engine", f), engine[f]);
    const lic = f === "battle-animations-moves.ts" ? "CC0-1.0" : f === "globals.d.ts" ? "AGPL-3.0" : "MIT";
    record(`engine/${f}`, `${REPO}/play.pokemonshowdown.com/src/${f}`, original[f], engine[f], lic);
  }
  for (const [name, path] of Object.entries(SCRIPTS)) {
    const text = await fetchText(`${REPO}/${path}`);
    await writeFile(join(OUT, "vendor", name), text);
    record(`vendor/${name}`, `${REPO}/${path}`, text, text, name.startsWith("html") ? "Apache-2.0" : "MIT");
  }

  let styles = "";
  for (const f of STYLES) {
    const text = await fetchText(`${REPO}/play.pokemonshowdown.com/style/${f}`);
    styles += `/* ${f} */\n` + text.replace(/@import[^;]*;/g, "").replace(/url\((['"]?)\.\.\/fx\//g, `url($1${PS}/fx/`);
    manifest.files[`styles/showdown.css (${f})`] = { url: `${REPO}/play.pokemonshowdown.com/style/${f}`, license: "GPL-2.0", sha256_upstream: sha(text) };
  }
  const scoped = "/* Pokémon Showdown's battle.css and battle-log.css (GPL-2.0), scoped to .ps-battle by tools/vendor_showdown.mjs;\n   ../fx/ image URLs point at play.pokemonshowdown.com. */\n" + scopeCss(styles);
  await writeFile(join(OUT, "styles/showdown.css"), scoped);

  for (const [file, global] of Object.entries(DATA)) {
    const text = await fetchText(`${PS}/data/${file}`);
    const sandbox = { exports: {}, window: {} };
    vm.runInNewContext(text, sandbox, { filename: file, timeout: 10000 });
    const value = sandbox.exports[global] ?? sandbox.window[global];
    if (!value) throw new Error(`${file} did not define ${global}`);
    const json = JSON.stringify(value);
    const name = file.replace(/\.js$/, ".json");
    await writeFile(join(OUT, "data", name), json);
    record(`data/${name}`, `${PS}/data/${file}`, text, json, global === "BattleText" || global.startsWith("BattlePokemonSprites") ? "AGPL-3.0 (Showdown client data)" : "MIT (Showdown server data)");
  }
  for (const [name, url] of Object.entries(LICENSES)) await writeFile(join(OUT, "licenses", name), await fetchText(url));
  await writeFile(join(OUT, "MANIFEST.json"), JSON.stringify(manifest, null, 1) + "\n");
  console.log(`engine ${ENGINE.length} files, data ${Object.keys(DATA).length} files -> ${OUT}`);
}

await main();
