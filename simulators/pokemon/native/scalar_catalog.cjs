'use strict';
const path=require('node:path');
const {Dex}=require(path.resolve(__dirname,'../../../candidates/pokemon-showdown/dist/sim/dex'));
const {serializeMove}=require('./oracle_utils.cjs');
const sets=require(path.resolve(__dirname,'../../../candidates/pokemon-showdown/data/random-battles/gen9/sets.json'));
const dex=Dex.mod('gen9');
const reachable=[...new Set(Object.values(sets).flatMap(p=>p.sets.flatMap(s=>s.movepool)).map(name=>dex.moves.get(name).id))];
process.stdout.write(JSON.stringify({reachable,records:dex.moves.all().filter(m=>m.exists&&m.gen<=9&&!m.isZ&&!m.isMax).map(serializeMove)}));
