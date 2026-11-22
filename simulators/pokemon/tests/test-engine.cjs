'use strict';
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const path = require('node:path');
const {Game, seed, normalizeTrace} = require('../worker.cjs');
const {View, choice, SCHEMA} = require('../view.cjs');
const engine = path.resolve(__dirname,'../../../candidates/pokemon-showdown/dist/sim');
const {BattleStream,getPlayerStreams} = require(path.join(engine,'battle-stream'));
const {extractChannelMessages} = require(path.join(engine,'battle'));
function play(master, limit=512) {
 const g=new Game(master,limit); let decisions=0, x=master>>>0;
 while(!g.terminated&&!g.truncated) {
  const r=g.result();
  g.step(r.ready.map((ready,s)=> {
   if(!ready)return null;
   const legal=r.observations[s].mask.flatMap((m,i)=>m?[i]:[]);
   assert(legal.length>0);
   // Move-biased public random play; includes tera and ordinary/forced switches.
   x=(Math.imul(x,1664525)+1013904223)>>>0;
   const moves=legal.filter(a=>a<8), pool=moves.length&&x%5!==0?moves:legal;
   decisions++;return pool[x%pool.length];
  }));
 }
 return {g,decisions,hash:g.trace.copy().digest('hex')};
}
(async()=> {
 const a=play(438); const now=Date.now; Date.now=()=>now()+5000; const b=play(438); Date.now=now; const c=play(439);
 assert.equal(a.hash,b.hash);assert.notEqual(a.hash,c.hash);
 assert.deepEqual(a.g.spec,b.g.spec);assert.notEqual(seed(1,'battle'),seed(1,'team1'));
 assert(a.g.terminated);assert(!a.g.truncated);
 assert.deepEqual(a.g.result().returns.reduce((x,y)=>x+y,0),0);
 // Differential check: replay every exact choice through unmodified upstream
 // getPlayerStreams. Their individual visible byte streams must match the worker.
 const raw=new BattleStream({noCatch:true}), streams=getPlayerStreams(raw);
 const hashes=[crypto.createHash('sha256'),crypto.createHash('sha256')];
 const readers=['p1','p2'].map((p,i)=>(async()=>{for await(const chunk of streams[p])hashes[i].update(normalizeTrace(chunk));})());
 const drain=(async()=>{for await(const chunk of streams.omniscient){} })();
 for(const line of a.g.inputs) await raw.write(line);
 await Promise.all([...readers,drain]);

 assert.deepEqual(hashes.map(h=>h.digest('hex')),a.g.sideTrace.map(h=>h.copy().digest('hex')));
 raw._destroy();
 // Visibility: opponent-private split line must never reach p1; opponent hidden
 // slots must have no species/item/ability/move tokens before public revelation.
 const channels=extractChannelMessages('|split|p2\n|-damage|p2a: X|176/333\n|-damage|p2a: X|53/100',[1,2]);
 assert(!channels[1].join('\n').includes('176/333'));assert(channels[2].join('\n').includes('176/333'));
 const illusion = new View('p1'); illusion.receive('|switch|p2a: X|Pikachu, L80|56/100\n|replace|p2a: Zoroark|Zoroark, L80');
 assert.equal(illusion.mons['p2: Zoroark'].condition,'56/100');assert(illusion.encode().mons.flat().every(Number.isFinite));
 const fresh=new Game(99,1), r=fresh.result(), obs=r.observations[0];
 assert.equal(obs.tokens.slice(6).filter(t=>t[0]!==0).length,1);
 assert.equal(obs.tokens[6][1],0);assert.equal(obs.tokens[6][2],0);
 assert(obs.tokens[6].slice(7,11).every(v=>v===0));
 // Mutating the other view's private request cannot affect p1 encoding.
 const before=JSON.stringify(fresh.views[0].encode());
 fresh.views[1].request.side.pokemon[1].item='verysecret';
 assert.equal(JSON.stringify(fresh.views[0].encode()),before);
 fresh.step(r.ready.map((v,s)=>v?r.observations[s].mask.findIndex(Boolean):null));
 assert(fresh.truncated);assert(!fresh.terminated);assert.equal(fresh.result().reason,'decision_limit');
 assert.deepEqual(fresh.result().returns,[0,0]);assert.deepEqual(fresh.result().ready,[false,false]);
 assert.throws(()=>choice(14));assert.throws(()=>new Game(2,1).step([99,0]),/Masked/);
 let rev=new View('p1');
 rev.receive('|request|'+JSON.stringify({forceSwitch:[true],side:{pokemon:[{ident:'p1: A',details:'Pikachu',active:true,reviving:true,condition:'100/100',moves:[]},{ident:'p1: B',details:'Eevee',active:false,condition:'0 fnt',moves:[]},{ident:'p1: C',details:'Raichu',active:false,condition:'100/100',moves:[]}]}}));
 assert(rev.encode().mask[9]);assert(!rev.encode().mask[10]);
 rev.request.wait=true;assert(!rev.encode().mask.some(Boolean));
 let games=0,decisions=0,rejections=0;
 for(let i=0;i<40;i++){const x=play(1000+i); assert(x.g.terminated);games++;decisions+=x.decisions;rejections+=x.g.result().rejections.reduce((a,b)=>a+b);x.g.destroy();}
 for(const g of [a.g,b.g,c.g,fresh])g.destroy();
 console.log(JSON.stringify({passed:true,seed_replay_sha256:a.hash,differential_upstream_player_streams:true,visibility_boundary:true,mask_games:games,decisions,rejections,schema:SCHEMA}));
})().catch(e=>{console.error(e);process.exitCode=1;});
