'use strict';
// Conservative mechanics obligations from the pinned Gen 9 oracle, not an
// executable fallback. Every callable remains an explicit porting obligation.
const path=require('node:path'),crypto=require('node:crypto');
const {Dex}=require(path.resolve(__dirname,'../../../candidates/pokemon-showdown/dist/sim/dex'));
const dex=Dex.mod('gen9'),digest=s=>crypto.createHash('sha256').update(s).digest('hex');
function callbacks(value,prefix='',out=[]) {
 if(!value||typeof value!=='object')return out;
 for(const [k,v] of Object.entries(value)) {
  const field=prefix?prefix+'.'+k:k;
  if(typeof v==='function')out.push({field,sha256:digest(v.toString())});
  else if(v&&typeof v==='object'&&!['effect','dex'].includes(k))callbacks(v,field,out);
 }
 return out;
}
const groups={moves:dex.moves.all(),abilities:dex.abilities.all(),items:dex.items.all(),conditions:Object.keys(dex.data.Conditions).map(id=>dex.conditions.get(id))};
const catalog={schema:1,task:'gen9randombattle',oracle_pin:'567c1cedddeaa2f35576005d3bfd526cf4f8a13e',full_task_admitted:false,
 scope:'Conservative Gen9 catalog (may include effects outside the random-team pool); do not shrink the supported task to the implemented subset.',groups:{}};
for(const [kind,entries] of Object.entries(groups))catalog.groups[kind]=entries.filter(e=>e.exists!==false&&!(e.gen>9)).map(e=>({id:e.id,name:e.name,gen:e.gen,nonstandard:e.isNonstandard || null,callbacks:callbacks(e)}));
catalog.obligation_count=Object.values(catalog.groups).flat().reduce((n,e)=>n+e.callbacks.length,0);
catalog.native_callback_handlers_qualified=0;
catalog.remaining_core_obligations=['native seeded random-team generator','full priority/event/forced-switch scheduler','all reachable move/ability/item/status/weather/terrain handlers','complete player-visible information tracker and encoder','full-task differential rollouts and policy transfer','full-task CPU/CUDA and checkpoint equivalence'];
process.stdout.write(JSON.stringify(catalog,null,2)+'\n');
