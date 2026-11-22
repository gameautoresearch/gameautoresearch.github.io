'use strict';
// Simulator-only diagnostic. Choices depend only on visible action features.
const {performance}=require('node:perf_hooks');
const {Game}=require('./worker.cjs');
const crypto=require('node:crypto');
const cache=process.argv.includes('--cache'),memo=process.argv.includes('--memo'),fast=process.argv.includes('--callbacks'), count=Number(process.argv.find(x=>/^--games=/.test(x))?.split('=')[1] || 128);
const seedStart=Number(process.argv.find(x=>/^--seed-start=/.test(x))?.split('=')[1] || 23000);
const start=performance.now(), hashes=[];
let decisions=0,late=0,natural=0,truncated=0,resetMs=0,playMs=0,rng=8734;
for(let i=0;i<count;i++) {
  const begin=performance.now(),g=new Game(seedStart+i,512,false,true,cache,memo,fast);resetMs+=performance.now()-begin;
  const playing=performance.now();
  while(!g.terminated&&!g.truncated) {
    const state=g.result();
    const actions=state.ready.map((ready,seat)=> {
      if(!ready)return null;
      const obs=state.observations[seat],legal=obs.mask.flatMap((m,i)=>m?[i]:[]),moves=legal.filter(a=>a<8);
      rng=(Math.imul(rng,1664525)+1013904223)>>>0;
      let pool=legal;
      if(moves.length&&rng%7!==0) {
        const score=a=>{const f=obs.actions[a];return f[0]*f[1]*f[8]*(1+.5*f[9])+.002*f[2]-.001*f[10];};
        const best=Math.max(...moves.map(score));pool=moves.filter(a=>Math.abs(score(a)-best)<1e-9);
      }
      decisions++;if(state.turn>=50)late++;
      return pool[rng%pool.length];
    });
    g.step(actions);
  }
  playMs+=performance.now()-playing;natural+=Number(g.terminated);truncated+=Number(g.truncated);
  hashes.push(g.trace.copy().digest('hex'));g.destroy();
}
console.log(JSON.stringify({cache,memo,fast,count,decisions,late,natural,truncated,resetMs,playMs,seconds:(performance.now()-start)/1000,
  sha256:crypto.createHash('sha256').update(hashes.join(':')).digest('hex'),memory:process.memoryUsage()}));
