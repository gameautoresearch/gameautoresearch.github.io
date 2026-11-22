# Vendored dependencies

`engines/` contains complete upstream source trees, including local improvements.
They are ordinary versioned files, not submodules. Keep their licenses, notices,
original names and frozen pins.

| Source | Owning simulator |
|---|---|
| `engines/crforge` | [Clash Royale](../simulators/clash/README.md) |
| `engines/DouZero` | [Dou Dizhu](../simulators/doudizhu/README.md) |
| `engines/generals-bots` | [Generals.io](../simulators/generals/README.md) |
| `engines/pokemon-showdown`, `engines/poke-env` | [Pokémon](../simulators/pokemon/README.md) |
| `engines/AutoTavern` | [Battlegrounds](../simulators/tavern/README.md) |
| `engines/benzene` | [Hex](../simulators/hex/README.md): MoHex 2.0, the evaluation opponent; two recorded local changes (Boost compatibility, a resistance-evaluation crash fix) |

The benchmark Polytopia engines are under `simulators/polytopia/`. AutoTavern’s
Rust worker and Python binding changes are retained in the vendored source.

`upstream-git/manifest.json` records upstream pins and source provenance. After
cloning, run `python3 tools/restore_upstream_metadata.py` to fetch each pinned
commit into a shallow local `.git` directory. Existing metadata and source files
are preserved. The historical `candidates`
path is a relative link to `engines/`.

The optional [AverageJoe reference track](../simulators/generals/averagejoe/README.md)
requires a separate upstream checkout.
