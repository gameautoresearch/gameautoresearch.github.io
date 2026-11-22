'use strict';
// Differential development oracle only. Never imported by native transitions.
const path=require('node:path');
const {PRNG}=require(path.resolve(__dirname,'../../../candidates/pokemon-showdown/dist/sim/prng'));
const {BattleStream}=require(path.resolve(__dirname,'../../../candidates/pokemon-showdown/dist/sim/battle-stream'));
const {Dex}=require(path.resolve(__dirname,'../../../candidates/pokemon-showdown/dist/sim/dex'));
const {serializeMove,formatId}=require('./oracle_utils.cjs');
const request=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
if(request.op==='rng') {
 const rows=request.seeds.map(seed=>{const p=new PRNG(seed);const values=[];for(let i=0;i<request.frames;i++)values.push(p.rng.next());return {values,seed:p.getSeed()};});
 process.stdout.write(JSON.stringify(rows));
} else if(request.op==='duel') {
 const frames=[];const stream=new BattleStream({noCatch:true});stream.pushMessage=()=>{};
 const moveIds=request.moves || [request.move || 'tackle',request.move || 'tackle'];
 const team=side=>[{name:'Test'+side,species:request.species?.[side] || 'Mew',ability:'No Ability',moves:[moveIds[side]],level:request.levels?.[side] || 100,teraType:request.tera_type || 'Normal',evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:request.speed_evs?.[side] ?? (side?0:4)},ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},nature:'Serious'}];
 stream._write('>start '+JSON.stringify({formatid:formatId(request.team_size),seed:request.seed})+'\n>player p1 '+JSON.stringify({name:'p1',team:team(0)})+'\n>player p2 '+JSON.stringify({name:'p2',team:team(1)}));
 const b=stream.battle;
 if(request.statuses)for(let side=0;side<2;side++)if(request.statuses[side])b.sides[side].active[0].setStatus(request.statuses[side]);
 if(request.boosts)for(let side=0;side<2;side++)Object.assign(b.sides[side].active[0].boosts,request.boosts[side]);
 if(request.hp_fraction)for(let side=0;side<2;side++){const p=b.sides[side].active[0];p.hp=Math.max(1,Math.floor(p.maxhp*request.hp_fraction[side]));}
 if(request.pp)for(let side=0;side<2;side++)b.sides[side].active[0].moveSlots[0].pp=request.pp[side];
 function state(){return {seed:b.prng.getSeed(),turn:b.turn,ended:b.ended,mons:b.sides.map(s=>s.pokemon.map(p=>({species:p.species.id,types:p.getTypes(),level:p.level,tera:p.teraType,hp:p.hp,maxhp:p.maxhp,stats:{...p.storedStats},status:p.status,statusState:{time:p.statusState.time || 0,stage:p.statusState.stage || 0},moves:p.moveSlots.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp})),boosts:{...p.boosts}}))) };}
 const initial=state();
 for(let i=0;i<request.turns&&!b.ended;i++) {
  const calls=[],original=b.prng.random;
  b.prng.random=function(...args){const result=original.apply(this,args);calls.push({args,result,seed:this.getSeed(),...(request.stack?{stack:new Error().stack.split('\n').slice(2,7)}:{})});return result;};
  stream._write('>p1 move 1'+(request.tera?.[0]&&i===0?' terastallize':'')+'\n>p2 move 1'+(request.tera?.[1]&&i===0?' terastallize':''));b.prng.random=original;
  frames.push({calls,state:state()});
 }
 const types=['Normal','Fire','Water','Grass','Electric','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Steel','Dark','Fairy','Stellar'];
 const chart=types.slice(0,18).map(atk=>types.slice(0,18).map(def=>{const e=Dex.mod('gen9').types.get(def).damageTaken[atk];return e===3?-127:e===1?1:e===2?-1:0;}));
 const moves=moveIds.map(id=>serializeMove(Dex.mod('gen9').moves.get(id)));
 process.stdout.write(JSON.stringify({initial,frames,types,chart,move:moves[0],moves}));stream._destroy();
} else throw Error('Unknown oracle command');
