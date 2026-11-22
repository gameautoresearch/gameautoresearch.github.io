'use strict';
// Independent differential fixtures only. This file is never a native backend.
const path = require('node:path');
const {BattleStream} = require(path.resolve(__dirname, '../../../candidates/pokemon-showdown/dist/sim/battle-stream'));
const {extractChannelMessages} = require(path.resolve(__dirname, '../../../candidates/pokemon-showdown/dist/sim/battle'));
const {Dex} = require(path.resolve(__dirname, '../../../candidates/pokemon-showdown/dist/sim/dex'));
const {View, choice, token, health} = require('../view.cjs');
const {serializeMove,formatId}=require('./oracle_utils.cjs');
const dex = Dex.mod('gen9');
const types = ['Normal','Fire','Water','Grass','Electric','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Steel','Dark','Fairy','Stellar'];
const chart = types.slice(0,18).map(atk => types.slice(0,18).map(def => { const e=dex.types.get(def).damageTaken[atk]; return e===3?-127:e===1?1:e===2?-1:0; }));

function run(spec) {
  const views=[new View('p1'), new View('p2')];
  const stream=new BattleStream({noCatch:true});
  stream.pushMessage=(kind,data)=>{
    if(kind==='update') {
      const channels=extractChannelMessages(data,[1,2]);
      for(let side=0;side<2;side++)views[side].receive(channels[side+1].join('\n'));
    } else if(kind==='sideupdate') {
      const split=data.indexOf('\n');views[Number(data[1])-1].receive(data.slice(split+1));
    }
  };
  let teams=spec.team_seeds?null:spec.teams.map((team,side)=>team.map((p,slot)=>spec.generated?({...p,name:`S${side}P${slot}`}):({
    name:`S${side}P${slot}`,species:p.species||'Mew',ability:p.ability||'No Ability',item:p.item||'',
    moves:p.moves||['tackle'],level:p.level||100,teraType:p.tera||'Normal',nature:'Serious',
    evs:{hp:0,atk:0,def:0,spa:0,spd:0,spe:p.speed_ev??(side?0:4)},
    ivs:{hp:31,atk:31,def:31,spa:31,spd:31,spe:31},
  })));
  stream._write('>start '+JSON.stringify({formatid:formatId(spec.team_size),seed:spec.seed})+
    '\n>player p1 '+JSON.stringify(spec.team_seeds?{name:'p1',seed:spec.team_seeds[0]}:{name:'p1',team:teams[0]})+
    '\n>player p2 '+JSON.stringify(spec.team_seeds?{name:'p2',seed:spec.team_seeds[1]}:{name:'p2',team:teams[1]}));
  const battle=stream.battle;
  if(spec.team_seeds)teams=battle.sides.map(s=>s.pokemon.map(p=>p.set));
  for(let side=0;side<2;side++)for(let slot=0;slot<battle.sides[side].pokemon.length;slot++){
    const p=battle.sides[side].pokemon[slot];p.nativeSlot=slot;
    if(spec.generated)continue;
    const source=spec.teams[side][slot];
    if(source.status)p.setStatus(source.status);
    if(source.hp_fraction)p.hp=Math.max(1,Math.floor(p.maxhp*source.hp_fraction));
    if(source.boosts)Object.assign(p.boosts,source.boosts);
    if(source.pp)source.pp.forEach((pp,i)=>p.moveSlots[i].pp=pp);
    if(source.stall_counter){p.addVolatile('stall');p.volatiles.stall.counter=source.stall_counter;}
  }
  if(!spec.generated){
  battle.makeRequest('move');
  // Direct fixture mutations need a fresh private request even without a log line.
  // They do not manufacture a public HP event for the opposing player.
  for(const side of battle.sides)side.emitRequest();battle.sentRequests=true;battle.sendUpdates();
  }
  function snapshot(){
    const result={
      unavailable_choices:views.map(v=>v.errors.filter(e=>e.startsWith('[Unavailable choice]')).length),trick_room:battle.field.pseudoWeather.trickroom?.duration||0,reviving:battle.sides.map(s=>!!s.slotConditions[0].revivalblessing),seed:battle.prng.getSeed(),turn:battle.turn,ended:battle.ended,winner:battle.winner||null,
      weather:battle.field.weather,weather_duration:battle.field.weatherState.duration||0,
      total_fainted:battle.sides.map(s=>s.totalFainted),
      wish:battle.sides.map(s=>{const w=s.slotConditions[0].wish;return w?[1,w.startingTurn,Math.floor(w.hp)]:[0,0,0];}),
      terrain:battle.field.terrain,terrain_duration:battle.field.terrainState.duration||0,public_terrain:views[0].terrain,
      hazard_order:battle.sides.map(s=>['stealthrock','spikes','toxicspikes','stickyweb'].filter(k=>s.sideConditions[k]).sort((a,b)=>s.sideConditions[a].effectOrder-s.sideConditions[b].effectOrder)),
      side_conditions:battle.sides.map(s=>['stealthrock','spikes','toxicspikes','stickyweb','reflect','lightscreen','auroraveil','tailwind','safeguard','mist'].map(k=>[s.sideConditions[k]?(s.sideConditions[k].layers||1):0,s.sideConditions[k]?.duration||0])),
      active:battle.sides.map(s=>s.active[0].nativeSlot),
      order:battle.sides.map(s=>s.pokemon.map(p=>p.nativeSlot)),
      forced:views.map(v=>!!v.request?.forceSwitch?.[0]),
      masks:views.map(v=>battle.ended?Array(14).fill(0):v.encode().mask.map(Number)),
      tera_used:battle.sides.map(s=>!!s.terastallizeUsed),
      future:battle.sides.map(side=>{const f=side.slotConditions[0]?.futuremove;return f?[f.endingTurn+1,f.move,Number(f.source.side.id[1])-1,f.source.nativeSlot]:[0,'',0,0];}),mons:battle.sides.map(s=>[...s.pokemon].sort((a,b)=>a.nativeSlot-b.nativeSlot).map(p=>({
        known_type:p.knownType,got_attacked:p.attackedBy.length>0,illusion_slot:(p.illusion?.nativeSlot??-1)+1,original_ability:dex.abilities.get(p.set.ability).id,embodied:!!p.abilityState.embodied,tera_resisted:!!p.abilityState.resisted,original_species:dex.species.get(p.set.species).id,original_stats:{...p.baseStoredStats},original_types:dex.species.get(p.set.species).types,original_weight:dex.species.get(p.set.species).weighthg,original_moves:p.baseMoveSlots.map(m=>({...m})),transformed:!!p.transformed,form_busted:!!p.abilityState.busted,ivs:p.set.ivs,evs:p.set.evs,set_species:dex.species.get(p.set.species).id,gender:p.gender,attract_source:(p.volatiles.attract?.source?.nativeSlot??-1)+1,cud_berry:p.ability==='cudchew'?(p.abilityState.berry?.id||''):'',cud_counter:p.ability==='cudchew'?(p.abilityState.counter||0):0,base_ability:p.baseAbility,trace_seek:!!p.abilityState.seek,destiny_bond:!!p.volatiles.destinybond,mirror_coat:!!p.volatiles.mirrorcoat,mirror_damage:p.volatiles.mirrorcoat?.damage||0,base_attack:p.baseSpecies.baseStats.atk,bond_triggered:!!p.bondTriggered,chilly_reception:!!p.volatiles.chillyreception,focus_punch:p.volatiles.focuspunch?(p.volatiles.focuspunch.lostFocus?2:1):0,beak_blast:!!p.volatiles.beakblast,charged:!!p.volatiles.charge,recharge:p.volatiles.mustrecharge?.duration||0,partial_trap:p.volatiles.partiallytrapped?.duration||0,partial_source:(p.volatiles.partiallytrapped?.source?.nativeSlot??-1)+1,truant:!!p.volatiles.truant,no_retreat:!!p.volatiles.noretreat,trapped_volatile:!!p.volatiles.trapped,magnet_rise:p.volatiles.magnetrise?.duration||0,glaive_rush:!!p.volatiles.glaiverush,slow_start:p.ability==='slowstart'?(p.abilityState.counter||0):0,supreme_fallen:p.abilityState.fallen||0,unburden:!!p.volatiles.unburden,last_item:p.lastItem,unnerved:!!p.abilityState.unnerved,charge_duration:p.volatiles.twoturnmove?.duration||0,charge_move:p.volatiles.twoturnmove?.move||null,charge_released:!!p.volatiles.twoturnmove&&!p.volatiles[p.volatiles.twoturnmove.move],curse:!!p.volatiles.curse,salt_cure:!!p.volatiles.saltcure,last_result:p.moveLastTurnResult===false?1:p.moveLastTurnResult===true?2:p.moveLastTurnResult===null?3:0,current_result:p.moveThisTurnResult===false?1:p.moveThisTurnResult===true?2:p.moveThisTurnResult===null?3:0,flash_fire:!!p.volatiles.flashfire,once_boost:Number(!!p.swordBoost)|2*Number(!!p.shieldBoost),volatile_order:Object.keys(p.volatiles),slot:p.nativeSlot,species:p.species.id,types:p.species.types,level:p.level,tera:p.teraType,
        species_flags:(p.baseSpecies.baseSpecies==='Pikachu'?1:0)|([380,381].includes(p.baseSpecies.num)?2:0)|(p.baseSpecies.num===493?4:0),weight_hg:p.species.weighthg,stats_lowered:!!p.statsLoweredThisTurn,newly_switched:!!p.newlySwitched,hit_by_foe:p.attackedBy.filter(a=>a.thisTurn&&a.damage>0&&a.source.side!==p.side).reduce((mask,a)=>mask|(1<<a.source.nativeSlot),0),stats_raised:!!p.statsRaisedThisTurn,locked_duration:p.volatiles.lockedmove?.duration||0,locked_true:p.volatiles.lockedmove?.trueDuration||0,locked_move:p.volatiles.lockedmove?.move||null,
        heal_block:p.volatiles.healblock?.duration||0,throat_chop:p.volatiles.throatchop?.duration||0,nfe:!!p.baseSpecies.nfe,paradox_species:p.baseSpecies.tags.includes("Paradox"),
        paradox_stat:((p.volatiles.protosynthesis||p.volatiles.quarkdrive)?.bestStat)?['atk','def','spa','spd','spe'].indexOf((p.volatiles.protosynthesis||p.volatiles.quarkdrive).bestStat)+1:0,paradox_booster:!!(p.volatiles.protosynthesis||p.volatiles.quarkdrive)?.fromBooster,booster_started:!!p.itemState.started,roost:!!p.volatiles.roost,types_now:p.getTypes(),speed_cache:p.speed,hp:p.hp,maxhp:p.maxhp,times_attacked:p.timesAttacked,stats:{...p.storedStats},boosts:{...p.boosts},ability:p.ability,item:p.item,
        status:p.status,fainted:p.fainted,terastallized:!!p.terastallized,
        protect:!!p.volatiles.protect,stall_counter:p.volatiles.stall?.counter||0,stall_duration:p.volatiles.stall?.duration||0,
        stellar_types:[...new Set(p.stellarBoostedTypes)],
        confusion:p.volatiles.confusion?.time||0,flinch:!!p.volatiles.flinch,
        taunt:p.volatiles.taunt?.duration||0,encore:p.volatiles.encore?.duration||0,encore_move:p.volatiles.encore?.move||null,
        disable:p.volatiles.disable?.duration||0,disable_move:p.volatiles.disable?.move||null,
        protean_used:!!(p.abilityState.protean||p.abilityState.libero),active_move_actions:p.activeMoveActions,last_move:p.lastMove?.id||null,active_turns:p.activeTurns,
        leech_seed:!!p.volatiles.leechseed,yawn:p.volatiles.yawn?.duration||0,substitute:p.volatiles.substitute?.hp||0,
        statusState:{time:p.statusState.time||0,stage:p.statusState.stage||0},
        sleep_source:p.statusState.source?Number(p.statusState.source.side.id[1])-1:-1,
        moves:p.moveSlots.map(m=>({id:m.id,pp:m.pp,maxpp:m.maxpp})),
      }))),
    };
    if(spec.observe)result.observations=views.map(v=>v.encode());
    return result;
  }
  let policy=spec.policy_seed||173;const frames=[],scheduleViolations=[],initial=snapshot();
  const publicInitial=spec.observe?[0,1].map(side=>{
    const v=views[1-side],key=`p${side+1}`;
    return v.order[key].map(ident=>{
      const p=v.mons[ident],hp=health(p.condition);
      return {slot:spec.team_seeds?battle.sides[side].pokemon.find(p=>p.fullname===ident).nativeSlot:Number(ident.match(/P([0-5])$/)[1]),hp:Math.round(hp[0]*100),known_hp:hp[1],fainted:hp[2],
        status:((p.condition||'').match(/ (brn|par|slp|frz|psn|tox)$/)||['',''])[1],
        tera:!!p.terastallized,ability:(p.ability||'').toLowerCase().replace(/[^a-z0-9]/g,''),item:p.item||'',
        details_known:!!p.species,species:dex.species.get(p.species).id,level:p.level||100,moves:[...(p.moves||[])],last_move:p.lastMove||'',boosts:{...p.boosts}};
    });
  }):null;
  for(let phase=0;phase<(spec.phases||128)&&!battle.ended;phase++){
    const before=snapshot(),actions=before.masks.map((mask,side)=>{
      const legal=mask.flatMap((value,index)=>value?[index]:[]);if(!legal.length)return -1;
      const scheduled=spec.schedule?.[phase]?.[side];if(scheduled!==undefined&&scheduled!==null){if(mask[scheduled])return scheduled;scheduleViolations.push({phase,side,scheduled,mask});return legal[0];}
      policy=(Math.imul(policy,1664525)+1013904223)>>>0;
      if(spec.policy==='random')return legal[policy%legal.length];
      const switches=legal.filter(a=>a>=8),moves=legal.filter(a=>a<4);
      if(before.forced[side]||(!moves.length&&switches.length))return switches[policy%switches.length];
      if(spec.switch_every&&phase%spec.switch_every===side&&switches.length)return switches[policy%switches.length];
      if(spec.tera&&legal.includes(4))return 4;
      if(spec.policy==='scripted_public_power'&&moves.length){
        const obs=views[side].encode(),stats=moves.map(a=>[a,obs.actions[a]]);
        const statuses=stats.filter(([a,x])=>x[7]);
        if(phase%7===3&&statuses.length)return statuses[policy%statuses.length][0];
        const ranked=stats.map(([a,x])=>[a,x[0]*Math.max(.01,x[8])*(x[9]?1.5:1)]);
        ranked.sort((a,b)=>b[1]-a[1]||a[0]-b[0]);return ranked[0][0];
      }
      return moves[policy%moves.length]??legal[0];
    });
    if(actions.every(a=>a<0))throw Error('No ready seat in nonterminal oracle');
    const calls=[],original=battle.prng.random;
    battle.prng.random=function(...args){const result=original.apply(this,args);calls.push({args,result,seed:this.getSeed(),...(spec.stack?{stack:new Error().stack.split('\n').slice(2,7)}:{})});return result;};
    for(let side=0;side<2;side++)if(actions[side]>=0)views[side].request={...views[side].request,wait:true};
    stream._write(actions.map((action,side)=>action<0?'':`>p${side+1} ${choice(action)}`).filter(Boolean).join('\n'));
    battle.prng.random=original;
    if(views.some(v=>v.errors.some(e=>!e.startsWith('[Unavailable choice]'))))throw Error(JSON.stringify(views.map(v=>v.errors)));
    frames.push({actions,calls,state:snapshot()});
  }
  const names=[...new Set(teams.flatMap(team=>team.flatMap(p=>p.moves)))];
  const moves=names.map(name=>serializeMove(dex.moves.get(name)));
  const result={spec,initial,frames,moves,types,chart,schedule_violations:scheduleViolations};
  if(spec.observe){
    const species=['',...new Set([...teams.flatMap(team=>team.map(p=>dex.species.get(p.species).id)),...initial.mons.flatMap(team=>team.map(p=>p.species))])];
    const abilityNames=['No Ability','Natural Cure','Regenerator','Pressure','Huge Power','Pure Power','Guts','Levitate','Intimidate','Quick Feet','Marvel Scale','Skill Link','Own Tempo','Inner Focus','Oblivious','Insomnia','Vital Spirit','Immunity','Limber','Water Veil','Magma Armor','Shield Dust','Serene Grace','Soundproof','Bulletproof','Infiltrator','Drizzle','Drought','Sand Stream','Snow Warning','Swift Swim','Chlorophyll','Sand Rush','Slush Rush','Sand Veil','Snow Cloak','Rain Dish','Ice Body','Solar Power','Hydration','Air Lock','Cloud Nine','Technician','Thick Fat','Adaptability','Tinted Lens','Overgrow','Blaze','Torrent','Swarm','Iron Fist','Strong Jaw','Tough Claws','Fur Coat','Ice Scales','Filter','Solid Rock','Neuroforce','Sniper','Multiscale','Shadow Shield','Protosynthesis','Quark Drive','Clear Body','White Smoke','Full Metal Body','Defiant','Competitive','Contrary','Simple','Sheer Force','Sturdy','Battle Armor','Shell Armor','Water Absorb','Volt Absorb','Sap Sipper','Storm Drain','Lightning Rod','Motor Drive','Earth Eater','Well-Baked Body','Unaware','Prankster','Overcoat','Sticky Hold','Hustle','Compound Eyes','Sharpness','Mega Launcher','Reckless',"Dragon's Maw",'Transistor','Rocky Payload','Steelworker','Flare Boost','Toxic Boost','Protean','Libero','Suction Cups','Heavy Metal','Light Metal','Triage','Gale Wings','Flame Body','Static','Poison Point','Justified','Stamina','Weak Armor','Cursed Body','Speed Boost','Stakeout','Rock Head','Heatproof','Fluffy','Water Bubble','Punk Rock','Purifying Salt','Early Bird','Leaf Guard','Shed Skin','Frisk','Flash Fire','Synchronize','Thermal Exchange','Water Compaction','Rattled','Toxic Chain','Poison Touch','Aftermath','Rough Skin','Iron Barbs','Wind Rider','Psychic Surge','Grassy Surge','Electric Surge','Misty Surge','Download','Dauntless Shield','Intrepid Sword','No Guard','Power Spot','Plus','Minus','Mold Breaker','Teravolt','Turboblaze','Scrappy',"Mind's Eye",'Keen Eye','Big Pecks','Prism Armor','Surge Surfer','Liquid Voice','Pixilate','Galvanize','Corrosion','Unseen Fist','Good as Gold','Queenly Majesty','Analytic','Gooey','Tangling Hair','Toxic Debris','Seed Sower','Aroma Veil','Flower Veil','Poison Heal','Dry Skin','Super Luck','Magic Guard','Moxie','Chilling Neigh','Grim Neigh','Soul-Heart','Unburden','Cheek Pouch','Unnerve','Harvest','Comatose','Slow Start','Bad Dreams','Supreme Overlord','Liquid Ooze','Beads of Ruin','Tablets of Ruin','Vessel of Ruin','Sword of Ruin','Multitype','Arena Trap','Magnet Pull','Shadow Tag','Sand Force','Damp','Mirror Armor','Poison Puppeteer','Hadron Engine','Orichalcum Pulse','As One (Glastrier)','As One (Spectrier)','Truant','Mycelium Might','Effect Spore','Electromorphosis','Berserk','Anger Shell','Battle Bond','Magic Bounce','Magician','Pickpocket','Trace','Cute Charm','Cud Chew','Dancer','Disguise','Ice Face','Hunger Switch','Zero to Hero','Gulp Missile','Shields Down','Imposter','Tera Shift','Tera Shell','Teraform Zero','Embody Aspect (Teal)','Embody Aspect (Wellspring)','Embody Aspect (Hearthflame)','Embody Aspect (Cornerstone)','Illusion'];
    const itemNames=['','Leftovers','Choice Band','Choice Specs','Choice Scarf','Focus Sash','Sitrus Berry','Lum Berry','Loaded Dice','Life Orb','Assault Vest','Damp Rock','Heat Rock','Smooth Rock','Icy Rock','Light Clay','Heavy-Duty Boots','Terrain Extender','Chesto Berry','Eviolite','Wide Lens','Booster Energy','Safety Goggles','Float Stone','White Herb','Flame Orb','Toxic Orb','Weakness Policy','Rocky Helmet','Light Ball','Soul Dew','Silk Scarf','Charcoal','Mystic Water','Miracle Seed','Magnet','Never-Melt Ice','Black Belt','Poison Barb','Soft Sand','Sharp Beak','Twisted Spoon','Silver Powder','Hard Stone','Spell Tag','Dragon Fang','Metal Coat','Black Glasses','Fairy Feather','Air Balloon','Scope Lens','Power Herb','Leppa Berry','Throat Spray','Flame Plate','Splash Plate','Meadow Plate','Zap Plate','Icicle Plate','Fist Plate','Toxic Plate','Earth Plate','Sky Plate','Mind Plate','Insect Plate','Stone Plate','Spooky Plate','Draco Plate','Iron Plate','Dread Plate','Pixie Plate','Griseous Core','Lustrous Globe','Adamant Crystal','Lustrous Orb','Adamant Orb','Griseous Orb','Rusted Sword','Rusted Shield','Wellspring Mask','Hearthflame Mask','Cornerstone Mask','Expert Belt','Custap Berry'];
    result.public_initial=publicInitial;
    result.metadata={
      species:species.map(name=>{const p=dex.species.get(name);return {id:p.id,token:token('species',p.id==='greninjabond'?'Greninja':p.name),types:[0,1].map(i=>types.indexOf(p.types[i])),type_tokens:[0,1].map(i=>token('type',p.types[i])),stats:['atk','def','spa','spd','spe'].map(k=>p.baseStats[k])};}),
      moves:[...names,'struggle'].map(name=>{const m=dex.moves.get(name);return {id:m.id,token:token('move',m.id),type_token:token('type',m.type),category_token:token('category',m.category),type:types.indexOf(m.type),category:['Physical','Special','Status'].indexOf(m.category),power:m.basePower,accuracy:m.accuracy===true?100:m.accuracy,priority:m.priority};}),
      ability_tokens:abilityNames.map(s=>token('ability',s)),item_tokens:itemNames.map(s=>token('item',s)),
      type_tokens:types.map(s=>token('type',s)),status_tokens:['','brn','par','slp','frz','psn','tox'].map(s=>token('status',s)),
    };
  }
  stream._destroy();return result;
}
const request=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify(request.cases.map(run)));
