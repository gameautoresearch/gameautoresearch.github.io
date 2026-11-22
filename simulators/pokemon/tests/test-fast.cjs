'use strict';
const assert=require('node:assert/strict');
const {Game}=require('../worker.cjs');
const {installFastCallbacks}=require('../engine_fast.cjs');
let games=0,decisions=0,truncations=0;
for(let seed=30000;seed<30128;seed++) {
  const a=new Game(seed,512),b=new Game(seed,512,false,true,true,true,true);
  const original=a.stream.battle.getCallback;
  const target=a.stream.battle.p1.active[0];
  const effects=[target.getAbility(),target.getItem(),target.species,target.getStatus()];
  const events=['onSwitchIn','onStart','onAnySwitchIn','onResidual','onModifyAtk','onDoesNotExist'];
  installFastCallbacks(a.stream.battle);
  for(const effect of effects)for(const event of events)for(const holder of [target,target.side,a.stream.battle.field,a.stream.battle])
    assert.equal(a.stream.battle.getCallback(holder,effect,event),original.call(a.stream.battle,holder,effect,event));
  a.stream.battle.getCallback=original;
  let rng=seed;
  while(!a.terminated&&!a.truncated) {
    const x=a.result(),y=b.result();
    assert.deepEqual(x,y);
    const actions=x.ready.map((ready,s)=> {
      if(!ready)return null;
      const legal=x.observations[s].mask.flatMap((m,i)=>m?[i]:[]),moves=legal.filter(i=>i<8);
      rng=(Math.imul(rng,1664525)+1013904223)>>>0;
      const pool=moves.length&&rng%5!==0?moves:legal;decisions++;
      return pool[rng%pool.length];
    });
    a.step(actions);b.step(actions);
  }
  assert.deepEqual(a.result(),b.result());assert.equal(a.trace.copy().digest('hex'),b.trace.copy().digest('hex'));
  truncations+=Number(a.truncated);games++;a.destroy();b.destroy();
}
console.log(JSON.stringify({games,decisions,truncations,all_observations_and_visible_traces_exact:true,fast_callback_values_exact:true}));
