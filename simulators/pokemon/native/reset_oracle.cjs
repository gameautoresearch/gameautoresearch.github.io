'use strict';
// Independent constructor-only oracle. Native construction never invokes this.
const path=require('node:path');
const root=path.resolve(__dirname,'../../../candidates/pokemon-showdown/dist/sim');
const {Battle}=require(path.join(root,'battle'));
const {Dex}=require(path.join(root,'dex'));
const {formatId}=require('./oracle_utils.cjs');
const request=JSON.parse(require('node:fs').readFileSync(0,'utf8')),dex=Dex.mod('gen9');
const metadata=request.formes.map(name=>{
  const p=dex.species.get(name);
  return {name,id:p.id,types:p.types,stats:p.baseStats,fixed_hp:p.maxHP||0,nfe:!!p.nfe,weight_hg:p.weighthg,paradox_species:p.tags.includes('Paradox')};
});
const rows=request.teams.map(team=>{
  if(team.some(set=>set.nature))throw Error('Frozen neutral-nature contract changed');
  const battle=new Battle({formatid:formatId(request.team_size),seed:[1,2,3,4]});
  // A second player would start callbacks; stop at independent construction.
  battle.setPlayer('p1',{name:'p1',team:structuredClone(team)});
  const result=battle.sides[0].pokemon.map(p=>({species:p.species.id,types:p.getTypes(),level:p.level,tera:p.teraType,
    hp:p.hp,maxhp:p.maxhp,stats:['atk','def','spa','spd','spe'].map(k=>p.storedStats[k]),ability:p.ability,item:p.item,
    weight_hg:p.species.weighthg,nfe:!!p.baseSpecies.nfe,paradox_species:p.baseSpecies.tags.includes('Paradox'),
    moves:p.moveSlots.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp}))}));
  const seed=battle.prng.getSeed();battle.destroy();return {result,seed};
});
process.stdout.write(JSON.stringify({metadata,rows}));
