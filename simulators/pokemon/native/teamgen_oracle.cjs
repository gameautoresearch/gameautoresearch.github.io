'use strict';
// Pinned generator oracle only. Never linked into native reset/stepping.
const path=require('node:path');
const root=path.resolve(__dirname,'../../../candidates/pokemon-showdown');
const {Dex}=require(path.join(root,'dist/sim/dex'));
const {RandomTeams}=require(path.join(root,'dist/data/random-battles/gen9/teams'));
const {serializeMove,formatId}=require('./oracle_utils.cjs');
// POKEMON_TEAM_SIZE selects the Max Team Size of the full-team section; every other record is size-independent.
const teamSize=process.env.POKEMON_TEAM_SIZE===undefined?6:Number(process.env.POKEMON_TEAM_SIZE);
const sets=require(path.join(root,'data/random-battles/gen9/sets.json'));
const dex=Dex.mod('gen9'),names=Object.keys(sets),generator=new RandomTeams(formatId(teamSize),[1,2,3,4]);
const types=['Normal','Fire','Water','Grass','Electric','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Steel','Dark','Fairy','Stellar'];
const species=names.map(name=>{
  const s=dex.species.get(name),battle=typeof s.battleOnly==='string'?dex.species.get(s.battleOnly):s;
  return {id:s.id,name:s.name,base:s.baseSpecies,types:s.types,battle_types:battle.types,
    battle_only:s.battleOnly||null,cosmetic_formes:s.cosmeticFormes||null,other_formes:s.otherFormes||null,
    level:sets[name].level||80,sets:sets[name].sets,stats:s.baseStats,weight:s.weightkg,gender:s.gender,
    is_mega:!!s.isMega,abilities:Object.values(s.abilities),required_move:s.requiredMove?dex.moves.get(s.requiredMove).id:null,
    required_items:s.requiredItems||null,nfe:!!s.nfe,effectiveness:types.map(type=>dex.getEffectiveness(type,s))};
});
const pools=[];
for(const excluded of [[],names.filter((_,i)=>i%7===0)])for(const type of [-1,...types.keys()]){
  const [members,groups]=generator.getPokemonPool(type<0?'Water':types[type],excluded.map(species=>({species})),type>=0,names);
  pools.push({type,excluded,members,groups});
}
const formes=[];
for(let seed=1;seed<=8;seed++)for(const name of names){
  const before=[seed,seed+1,seed+2,seed+3];generator.setSeed(before);
  formes.push({species:name,seed:before,forme:generator.getForme(dex.species.get(name)),after:generator.prng.getSeed()});
}
const compatibility=names.map(name=>names.map(other=>generator.getPokemonCompatibility(dex.species.get(name),[{speciesId:other}])));
const moveNames=[...new Set([...Object.values(sets).flatMap(row=>row.sets.flatMap(set=>set.movepool)),
  ...species.map(row=>row.required_move).filter(Boolean),'terablast','relicsong','triattack'].map(name=>dex.moves.get(name).id))];
const moves=moveNames.map(name=>{const move=dex.moves.get(name);return {...serializeMove(move),
  power_callback:!!move.basePowerCallback,damage_callback:!!move.damageCallback,has_secondary:!!move.secondary};});
const queries=[];
for(const name of names)for(const set of sets[name].sets){
  const teraTypes=Array.isArray(set.teraTypes)?set.teraTypes:[set.teraTypes];
  for(const chosen of [set.movepool.slice(0,4),set.movepool.slice(-4)])for(const tera of [teraTypes[0],teraTypes.at(-1)]){
    const selected=[...new Set(chosen.map(name=>dex.moves.get(name).id))];
    const counter=generator.queryMoves(new Set(selected),dex.species.get(name),tera,set.abilities);
    queries.push({species:name,moves:selected,tera,abilities:set.abilities,role:set.role,counts:Object.fromEntries(counter),
      damaging:[...counter.damagingMoves].map(m=>m.id),power:[...counter.basePowerMoves].map(m=>m.id)});
  }
}
const abilityQueries=[];
for(const [index,query] of queries.entries()){
  const species=dex.species.get(query.species),selected=new Set(query.moves);
  const counter=generator.queryMoves(selected,species,query.tera,query.abilities);
  const weatherFlags=query.abilities.length>1?Array.from({length:16},(_,i)=>i):[0];
  for(const weather of weatherFlags)for(const seed of (query.abilities.length>1?[1,7]:[1])){
    const before=[seed,seed+1,seed+2,seed+3];generator.setSeed(before);
    const details={rain:weather&1?1:0,sun:weather&2?1:0,sand:weather&4?1:0,snow:weather&8?1:0};
    const chosen=generator.getAbility(new Set(species.types),selected,query.abilities,counter,details,species,false,false,query.tera,query.role);
    abilityQueries.push({query:index,weather,seed:before,chosen,after:generator.prng.getSeed()});
  }
}
const culls=[];
for(const name of names)for(const set of sets[name].sets){
  const original=set.movepool.map(name=>dex.moves.get(name).id),tera=Array.isArray(set.teraTypes)?set.teraTypes[0]:set.teraTypes;
  for(let count=0;count<=Math.min(3,original.length);count++)for(const flags of [0,16,32,64,128,256,512,1024,2048,4095]){
    const chosen=original.slice(0,count),pool=original.slice(count),before=[...pool],details={
      screens:flags&16?1:0,stickyWeb:flags&32?1:0,stealthRock:flags&64?1:0,defog:flags&128?1:0,
      rapidSpin:flags&256?1:0,toxicSpikes:flags&512?1:0,spikes:flags&1024?2:0,statusCure:flags&2048?1:0};
    const counter=generator.queryMoves(new Set(chosen),dex.species.get(name),tera,set.abilities);
    let fault=null;
    try{generator.cullMovePool(new Set(dex.species.get(name).types),new Set(chosen),set.abilities,counter,pool,details,dex.species.get(name),false,tera,set.role,false);}
    catch(error){fault=error.message;}
    culls.push({species:name,chosen,pool:before,result:pool,abilities:set.abilities,tera,team:flags,fault});
  }
}
const statusMoves=dex.moves.all().filter(move=>move.category==='Status').map(move=>move.id);
const movesets=[];
for(const name of names)for(const set of sets[name].sets){
  const tera=Array.isArray(set.teraTypes)?set.teraTypes[0]:set.teraTypes;
  for(const flags of [0,4095])for(const seed of [1,7,29,113]){
    const pool=set.movepool.map(name=>dex.moves.get(name).id),before=[...pool],details={
      screens:flags&16?1:0,stickyWeb:flags&32?1:0,stealthRock:flags&64?1:0,defog:flags&128?1:0,
      rapidSpin:flags&256?1:0,toxicSpikes:flags&512?1:0,spikes:flags&1024?2:0,statusCure:flags&2048?1:0};
    const start=[seed,seed+1,seed+2,seed+3];generator.setSeed(start);let chosen=null,fault=null;
    try{chosen=[...generator.randomMoveset(new Set(dex.species.get(name).types),set.abilities,details,dex.species.get(name),false,pool,tera,set.role,false)];}
    catch(error){fault=error.message;}
    movesets.push({species:name,pool:before,remaining:pool,abilities:set.abilities,tera,role:set.role,team:flags,seed:start,chosen,fault,after:generator.prng.getSeed()});
  }
}
const items=[{id:'',name:''},...dex.items.all().filter(item=>item.exists&&item.gen<=9).map(item=>({id:item.id,name:item.name}))];
const setStages=[];
for(const name of names)for(const set of sets[name].sets){
  const species=dex.species.get(name),tera=Array.isArray(set.teraTypes)?set.teraTypes[0]:set.teraTypes;
  for(const flags of [0,4095])for(const lead of [false,true])for(const seed of [1,7,29,113]){
    const pool=set.movepool.map(name=>dex.moves.get(name).id),before=[...pool],details={
      rain:flags&1?1:0,sun:flags&2?1:0,sand:flags&4?1:0,snow:flags&8?1:0,
      screens:flags&16?1:0,stickyWeb:flags&32?1:0,stealthRock:flags&64?1:0,defog:flags&128?1:0,
      rapidSpin:flags&256?1:0,toxicSpikes:flags&512?1:0,spikes:flags&1024?2:0,statusCure:flags&2048?1:0};
    const start=[seed,seed+1,seed+2,seed+3];generator.setSeed(start);let chosen=null,ability=null,item=null,fault=null;
    try{
      const selected=generator.randomMoveset(new Set(species.types),set.abilities,details,species,lead,pool,tera,set.role,false);
      chosen=[...selected];const counter=generator.queryMoves(selected,species,tera,set.abilities);
      ability=generator.getAbility(new Set(species.types),selected,set.abilities,counter,details,species,lead,false,tera,set.role);
      item=generator.getPriorityItem(ability,new Set(species.types),selected,counter,details,species,lead,tera,set.role,false);
      if(item===undefined)item=generator.getItem(ability,new Set(species.types),selected,counter,details,species,lead,tera,set.role);
    }catch(error){fault=error.message;}
    setStages.push({species:name,pool:before,remaining:pool,abilities:set.abilities,tera,role:set.role,team:flags,lead,seed:start,chosen,ability,item,fault,after:generator.prng.getSeed()});
  }
}
const fullSets=[];
for(const name of names)for(const flags of [0,4095,4096])for(const lead of [false,true])for(const seed of [1,7]){
  const details={rain:flags&1?1:0,sun:flags&2?1:0,sand:flags&4?1:0,snow:flags&8?1:0,
    screens:flags&16?1:0,stickyWeb:flags&32?1:0,stealthRock:flags&64?1:0,defog:flags&128?1:0,
    rapidSpin:flags&256?1:0,toxicSpikes:flags&512?1:0,spikes:flags&1024?2:0,statusCure:flags&2048?1:0,teraBlast:flags&4096?1:0};
  const start=[seed,seed+1,seed+2,seed+3];generator.setSeed(start);let result=null,fault=null;
  try{result=generator.randomSet(name,details,lead,false);}catch(error){fault=error.message;}
  fullSets.push({species:name,team:flags,lead,seed:start,result,fault,after:generator.prng.getSeed()});
}
const teams=[];
for(let seed=1;seed<=512;seed++){
  const start=[seed,(seed*251+7)%65536,(seed*997+19)%65536,(seed*1009+31)%65536];generator.setSeed(start);
  for(let sequence=0;sequence<2;sequence++){
    const before=generator.prng.getSeed();let result=null,fault=null;
    try{result=generator.randomTeam();}catch(error){fault=error.message;}
    teams.push({seed:before,result,fault,after:generator.prng.getSeed(),sequence});
  }
}
process.stdout.write(JSON.stringify({species,pools,formes,compatibility,types,moves,queries,ability_queries:abilityQueries,culls,status_moves:statusMoves,movesets,items,set_stages:setStages,full_sets:fullSets,teams,...(teamSize===6?{}:{team_size:teamSize})}));
