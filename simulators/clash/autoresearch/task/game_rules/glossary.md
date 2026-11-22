# Units and data-field glossary

- **World unit:** 1/1,000 placement tile. Half-tile route cell: 500 units.
- **Tick/visit:** 50 ms of simulation. Ten ticks normally follow each joint action (the short match form).
- **Speed:** movement/projectile row budget used by integer movement. Projectile speed is displacement budget per 50 ms visit; troop motion additionally obeys routes, collision, avoidance, stun and pushes.
- **Range:** raw attack radius; actual reach includes collision-radius extensions and a separate minimum-range test.
- **Sight:** target acquisition window, not an omniscient policy view. SightClip trims rear/side candidate acceptance.
- **Hit speed:** interval field in milliseconds between normal attacks. **Load time** is the first-hit/windup input to the timer state machine; it is not an extra delay added to every hit.
- **Deploy time:** entity deployment countdown after a queued command runs. **Summon delay** is a separate stagger before entering deployment.
- **Mass/collision radius:** physical interaction and separation parameters. They are not HP or attack range.
- **Crown percentage:** additive change from 100% damage. −77 =23%; −80 =20%. Rounded up after scaling for projectile/direct damage.
- **Base HP/damage:** table value before rarity/tower scaling. **Effective level-11 value:** the actual value compiled into this mode's spec.
- **Rarity.relativeLevel / powerLevelMultiplier:** pinned scaling inputs. Rarity names do not grant different mechanics.
- **Projectile, areaEffect, on…Action, spawn…:** references to another effect definition or effect rule. `none` means absent; this deck's normal troops have no death, special ability or spawn effect rule.
- **onlyEnemies / attacksGround / attacksAir:** target filters, distinct from whether a unit itself is airborne.
- **king / summonerTower:** exported crown-tower role flags. Native PrincessTower directly owns its firing behavior.
- **gameTagsToSet / flags:** bitsets enabling/disabling state behavior. Entity flag values are listed in the schema dictionary.
- **Shield, lifetime, dash, jump, charge, clone, reflect, hidden/stealth fields:** generic data schema supports these concepts; zero/false/none on fixed8 rows means absent from this task, not that the field was forgotten.
- **unmodelledColumns:** exporter-reported fields not represented by the native row. Empty for the documented rows; this does not prove whole commercial-game equivalence.
- **Form / slot flags / evolvedSpells:** alternate content metadata. Plain form zero and slot flags zero are the only admitted state here; alternate rows are excluded.
- **Pending command:** a promise accepted earlier but not yet executed. **Pending projectile damage:** a distinct target-selection estimate for already-flying shots. Neither is interchangeable with HP already subtracted.
- **Health component present:** existence of the health object; different from alive HP>0 and from still being listed in the holder.

Every per-entity table preserves exact exporter property names so researchers can cross-check a word against the corresponding source without internet. All values are mode-pinned, not live-game recommendations.

[Wiki home](README.md)
