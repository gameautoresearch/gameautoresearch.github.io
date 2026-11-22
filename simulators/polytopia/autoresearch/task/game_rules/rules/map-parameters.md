# Map parameters

These are the generator ranges of the benchmark's one map type, not current live-game map
settings. Fractions are sampled within ranges; final discrete terrain counts are
affected by capitals, villages, repair passes and legal resource placement.

| Preset | Land fraction | Mountain of free land | Forest of remaining free land | Coast noise scale | Shape | Village spacing | Shared land route |
|---|---|---|---|---|---|---:|---|
|drylands|(0.93, 0.96)|(0.12, 0.16)|(0.35, 0.45)|1.0|noise|3|False|

The fraction of possible villages actually placed (village_keep) is(.8,1.0).
Every11×11 board uses4 ruins and minimum capital Chebyshev distance6.
Resource placement uses animal1 on forest, fish3 on shallow water, metal5 on
mountain, fruit6 on field and starfish8 on water. No land route between the
capitals is forced; in a check of3,032 generated layouts the capitals were always
exactly6 tiles apart (Chebyshev) and joined by land that crosses no mountain.

[Mode and maps](mode-and-maps.md) · [Wiki home](../README.md)
