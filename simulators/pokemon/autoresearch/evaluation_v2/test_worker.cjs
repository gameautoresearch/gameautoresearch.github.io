'use strict';
const assert=require('node:assert/strict');
const {Game,normalizeTrace}=require('../../worker.cjs');
const {EvaluationGame,POLICY,recordChoices}=require('./worker.cjs');
const {View}=require('../../view.cjs');
const path=require('node:path');
const {BattleStream}=require(path.resolve(__dirname,'../../../../candidates/pokemon-showdown/dist/sim/battle-stream'));
const {extractChannelMessages}=require(path.resolve(__dirname,'../../../../candidates/pokemon-showdown/dist/sim/battle'));
function parity() {
 const a=new Game(900000081,512),b=new EvaluationGame(900000081);
 try {for(let k=0;k<100 && !a.terminated;k++) {
  const x=a.result(),y=b.result();assert.deepEqual(x.observations,y.observations);
  const actions=x.ready.map((r,s)=>r?x.observations[s].mask.findIndex(Boolean):null);a.step(actions);b.step(actions);
  assert.equal(a.trace.copy().digest('hex'),b.trace.copy().digest('hex'));
 }} finally {a.destroy();b.destroy();}
}
function fixture(teams) {
 const views=[new View('p1'),new View('p2')];let ended=false,stream=new BattleStream({noCatch:true});let messages=[];
 stream.pushMessage=(type,data)=>{if(type==='update') {const c=extractChannelMessages(data,[1,2]);for(let i=0;i<2;i++){views[i].receive(c[i+1].join('\n'));messages.push(...c[i+1]);}}else if(type==='sideupdate'){const i=Number(data[1])-1;views[i].receive(data.slice(data.indexOf('\n')+1));}else if(type==='end')ended=true;};
 stream._write('>start '+JSON.stringify({formatid:'gen9customgame',seed:[1,2,3,4]})+'\n'+teams.map((team,i)=>'>player p'+(i+1)+' '+JSON.stringify({name:'p'+(i+1),team})).join('\n'));
 return {views,stream,write:s=>stream._write(s),messages,get ended(){return ended;}};
}
const mon=(species,moves,ability='pressure')=>({species,moves,ability,level:100});
function mechanics() {
 const f=fixture([[mon('Pikachu',['thunderbolt']),mon('Eevee',['tackle'])],[mon('Wobbuffet',['counter'],'shadowtag'),mon('Eevee',['tackle'])]]);
 try {
  // Custom game has team preview; start explicitly. No engine-private fields are read.
  if(f.views[0].request?.teamPreview)f.write('>p1 team 12\n>p2 team 12');
  const before=f.views[0].encode();assert(before.mask[8+1]);assert(!f.views[0].request.active[0].trapped);
  f.write('>p1 switch 2');assert(f.views[0].errors.at(-1).startsWith('[Unavailable choice]'));
  const counter={views:f.views,attempted:[0,0],accepted:[0,0],retries:[0,0],consecutiveRetries:[15,0],retryLimit:16};
  assert(recordChoices(counter,{ready:[true,false]},[0,0]));assert.deepEqual(counter.accepted,[0,0]);assert.deepEqual(counter.attempted,[1,0]);assert.deepEqual(counter.consecutiveRetries,[16,0]);
  assert(f.views[0].request.active[0].trapped);assert(!f.views[0].encode().mask[9]);
  assert(f.views[0].encode().mask[4]);f.write('>p1 move 1 terastallize\n>p2 move 1');
  assert(f.views[0].tera.p1);assert(f.messages.some(m=>m.startsWith('|-terastallize|')));
 } finally {f.stream._destroy();}
 const pp=fixture([[mon('Magikarp',['splash'],'swiftswim')],[mon('Magikarp',['splash'],'swiftswim')]]);
 try {
  if(pp.views[0].request?.teamPreview)pp.write('>p1 team 1\n>p2 team 1');
  let struggle=false,forced=false;
  for(let i=0;i<200 && !pp.ended;i++) {
   for(const v of pp.views) {if(v.request?.active?.[0]?.moves?.[0]?.id==='struggle'){struggle=true;assert(v.encode().mask[0]);}if(v.request?.forceSwitch)forced=true;}
   pp.write('>p1 move 1\n>p2 move 1');
  }
  assert(struggle,'PP exhaustion must expose Struggle');assert(pp.ended,'Struggle battle must terminate naturally');
 } finally {pp.stream._destroy();}
 const fs=fixture([[mon('Shedinja',['splash'],'wonderguard'),mon('Eevee',['tackle'])],[mon('Tyranitar',['sandstorm'],'sandstream')]]);
 try {
  if(fs.views[0].request?.teamPreview)fs.write('>p1 team 12\n>p2 team 1');
  fs.write('>p1 move 1\n>p2 move 1');
  assert(fs.views[0].request.forceSwitch?.[0]);const mask=fs.views[0].encode().mask;
  assert(mask.slice(0,8).every(x=>!x));assert(mask[9]);
 } finally {fs.stream._destroy();}
}
function nativeLimit() {
 const g=new EvaluationGame(900000082,4096);
 try {
  while(!g.terminated&&!g.truncated) {
   const state=g.result();const actions=state.ready.map((r,s)=>r?state.observations[s].mask.findIndex((v,i)=>v&&i>=8):null);
   // Switch-only policies; any forced turn without a switch takes its first visible legal choice.
   for(let s=0;s<2;s++)if(actions[s]===-1)actions[s]=state.observations[s].mask.findIndex(Boolean);
   g.step(actions);
  }
  assert(g.terminated&&!g.truncated);assert.equal(g.result().terminal_reason,'engine_turn_limit_tie');
  assert(g.protocol[0].some(text=>text.includes('You have hit the turn limit!')));assert(g.steps>512);
 } finally {g.destroy();}
}
parity();mechanics();nativeLimit();console.log(JSON.stringify({passed:['100-phase projection and protocol parity','hidden trapping then own published rejection update','16retry safety gate with no accepted/fallback choice or waiting-seat duplication','Tera','PP exhaustion / Struggle natural completion','forced switch','native engine turn-limit tie without cycle draw'],policy:POLICY}));
