'use strict';
// Independent battle-start oracle; never called by native startup/reset.
const path=require('node:path'),root=path.resolve(__dirname,'../../../candidates/pokemon-showdown/dist/sim');
const {BattleStream}=require(path.join(root,'battle-stream'));
const {extractChannelMessages}=require(path.join(root,'battle'));
const {Dex}=require(path.join(root,'dex'));
const {View}=require('../view.cjs');
const {formatId}=require('./oracle_utils.cjs');
const request=JSON.parse(require('node:fs').readFileSync(0,'utf8')),dex=Dex.mod('gen9');
const types=['Normal','Fire','Water','Grass','Electric','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Steel','Dark','Fairy'];
const chart=types.map(a=>types.map(d=>{const e=dex.types.get(d).damageTaken[a];return e===3?-127:e===1?1:e===2?-1:0;}));
function run(spec){
  const views=[new View('p1'),new View('p2')],stream=new BattleStream({noCatch:true});
  stream.pushMessage=(kind,data)=>{
    if(kind==='update'){const messages=extractChannelMessages(data,[1,2]);for(let i=0;i<2;i++)views[i].receive(messages[i+1].join('\n'));}
    else if(kind==='sideupdate'){const split=data.indexOf('\n');views[Number(data[1])-1].receive(data.slice(split+1));}
  };
  stream._write('>start '+JSON.stringify({formatid:formatId(spec.team_size),seed:spec.seed})+'\n>player p1 '+JSON.stringify({name:'p1',team:spec.teams[0]})+'\n>player p2 '+JSON.stringify({name:'p2',team:spec.teams[1]}));
  const b=stream.battle;
  const result={seed:b.prng.getSeed(),turn:b.turn,ended:b.ended,winner:b.winner||null,
    active:b.sides.map(s=>0),order:b.sides.map(s=>s.pokemon.map((p,i)=>i)),forced:views.map(v=>!!v.request?.forceSwitch?.[0]),masks:views.map(v=>v.encode().mask.map(Number)),
    weather:b.field.weather,weather_duration:b.field.weatherState.duration||0,terrain:b.field.terrain,terrain_duration:b.field.terrainState.duration||0,total_fainted:b.sides.map(s=>s.totalFainted),
    side_conditions:b.sides.map(s=>['stealthrock','spikes','toxicspikes','stickyweb','reflect','lightscreen','auroraveil','tailwind','safeguard','mist'].map(k=>[s.sideConditions[k]?(s.sideConditions[k].layers||1):0,s.sideConditions[k]?.duration||0])),
    mons:b.sides.map(s=>s.pokemon.map(p=>({hp:p.hp,maxhp:p.maxhp,status:p.status,stats:{...p.storedStats},boosts:{...p.boosts},ability:p.ability,item:p.item,fainted:p.fainted,terastallized:!!p.terastallized,types_now:p.getTypes(),
      statusState:{time:p.statusState.time||0,stage:p.statusState.stage||0},moves:p.moveSlots.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp})),times_attacked:p.timesAttacked,
      paradox_stat:(p.volatiles.protosynthesis||p.volatiles.quarkdrive)?.bestStat?['atk','def','spa','spd','spe'].indexOf((p.volatiles.protosynthesis||p.volatiles.quarkdrive).bestStat)+1:0,paradox_booster:!!(p.volatiles.protosynthesis||p.volatiles.quarkdrive)?.fromBooster,
      stats_raised:!!p.statsRaisedThisTurn,active_move_actions:p.activeMoveActions,protean_used:!!(p.abilityState.protean||p.abilityState.libero)}))),observations:views.map(v=>v.encode())};
  stream._destroy();return result;
}
process.stdout.write(JSON.stringify({chart,rows:request.cases.map(run)}));
