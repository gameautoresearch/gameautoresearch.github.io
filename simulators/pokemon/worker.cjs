'use strict';
const path = require('node:path');
const readline = require('node:readline');
const fs=require('node:fs');
const {performance} = require('node:perf_hooks');
const workerMetrics={engine_protocol_ms:0,observation_encode_ms:0,wire_serialize_ms:0,wire_bytes:0};
const crypto = require('node:crypto');
const {BattleStream} = require(path.resolve(__dirname,'../../candidates/pokemon-showdown/dist/sim/battle-stream'));
const {extractChannelMessages} = require(path.resolve(__dirname,'../../candidates/pokemon-showdown/dist/sim/battle'));
const {View, choice, SCHEMA} = require('./view.cjs');
// Wall-clock protocol timestamps are presentation metadata, not battle RNG.
function normalizeTrace(text) { return text.replace(/^\|t:\|[0-9]+$/gm, '|t:|0'); }
// The format this process plays: `--team-size=N` selects gen9randombattle with the Max Team Size custom rule
// (pinned Showdown source unchanged; its generator honours the rule); no argument is unmodified gen9randombattle.
const TEAM_SIZE=(()=>{const arg=process.argv.find(x=>x.startsWith('--team-size='));const n=arg?Number(arg.slice(12)):6;
  if(!Number.isInteger(n)||n<1||n>6)throw Error('Team size must be 1..6');return n;})();
const FORMAT=TEAM_SIZE===6?'gen9randombattle':`gen9randombattle@@@Max Team Size = ${TEAM_SIZE}`;
function seed(master, domain) {
  const b = crypto.createHash('sha256').update(`pokemon-v1:${master}:${domain}`).digest();
  return [0,2,4,6].map(i=>b.readUInt16LE(i)).join(',');
}
class Game {
  constructor(master, limit, swapped=false, optimized=true, tokenMemo=false,monMemo=false,fastCallbacks=false) {
    this.optimized=optimized; this.cached=null; this.activity={moves:0,damage:0,heals:0,switches:0,faints:0,late_damage:0,late_heals:0};
    this.master = master; this.limit = limit; this.steps = 0; this.terminated = false; this.truncated = false;
    this.views = [new View('p1',tokenMemo,monMemo),new View('p2',tokenMemo,monMemo)]; this.inputs = []; this.trace = crypto.createHash('sha256'); this.sideTrace = [crypto.createHash('sha256'),crypto.createHash('sha256')];
    this.spec = {battle:seed(master,'battle'),p1:seed(master,swapped?'team2':'team1'),p2:seed(master,swapped?'team1':'team2')};
    const game = this;
    this.stream = new BattleStream({noCatch:true});
    if(fastCallbacks) {
      const writeLine=this.stream._writeLine;
      this.stream._writeLine=function(type,message) {
        const result=writeLine.call(this,type,message);
        if(type==='start')require('./engine_fast.cjs').installFastCallbacks(this.battle);
        return result;
      };
    }
    this.stream.pushMessage = (type,data) => {
      if (type === 'update') {
        // Exact upstream getPlayerStreams split. Omniscient channel is never constructed.
        const c = extractChannelMessages(data,[1,2]);
        const late=game.views[0].turn>=50;
        for(const line of c[1]) {
          const type=line.split('|')[1];
          if(type==='move')game.activity.moves++;
          else if(type==='-damage'){game.activity.damage++;if(late)game.activity.late_damage++;}
          else if(type==='-heal'){game.activity.heals++;if(late)game.activity.late_heals++;}
          else if(type==='switch'||type==='drag')game.activity.switches++;
          else if(type==='faint')game.activity.faints++;
        }
        for (let s=0;s<2;s++) { const text=c[s+1].join('\n'); game.trace.update(`${s}:${normalizeTrace(text)}\n`); game.sideTrace[s].update(normalizeTrace(text)); game.views[s].receive(text); }
      } else if (type === 'sideupdate') {
        const i=data.indexOf('\n'), s=Number(data[1])-1, text=data.slice(i+1);
        game.trace.update(`${s}:${normalizeTrace(text)}\n`); game.sideTrace[s].update(normalizeTrace(text)); game.views[s].receive(text);
      } else if (type === 'end') {
        // end includes private teams; explicitly ignore the entire payload.
        game.terminated = true;
      }
    };
    this.write(`>start ${JSON.stringify({formatid:FORMAT,seed:this.spec.battle})}\n>player p1 ${JSON.stringify({name:'p1',seed:this.spec.p1})}\n>player p2 ${JSON.stringify({name:'p2',seed:this.spec.p2})}`);
  }
  write(text) { const t=performance.now(); this.cached=null; this.inputs.push(text); this.stream._write(text); workerMetrics.engine_protocol_ms+=performance.now()-t; }
  result() {
    if(this.optimized && this.cached) return this.cached;
    const encodeStarted=performance.now();
    const winner=this.views[0].winner;
    const result = {activity:{...this.activity},seed:this.master,seeds:this.spec,steps:this.steps,terminated:this.terminated,truncated:this.truncated,
      turn:this.views[0].turn,reason:this.truncated?'decision_limit':this.terminated?'natural':null,
      winner:this.terminated?winner:null, returns:this.terminated?(winner==='p1'?[1,-1]:winner==='p2'?[-1,1]:[0,0]):[0,0],
      ready:this.views.map(v=>!this.terminated&&!this.truncated&&!!v.request&&!v.request.wait),
      observations:this.views.map(v=>v.encode()), potential:this.views.map(v=>v.potential()),
      rejections:this.views.map(v=>v.errors.length),rejection_messages:this.views.map(v=>v.errors)};
    if(this.terminated||this.truncated)result.trace_sha256=this.trace.copy().digest('hex');
    workerMetrics.observation_encode_ms+=performance.now()-encodeStarted;
    if(this.optimized) this.cached=result;
    return result;
  }
  step(actions) {
    const before=this.result(); const lines=[];
    for(let s=0;s<2;s++) {
      if (!before.ready[s]) { if(actions[s] !== null) throw Error('Action for waiting seat'); continue; }
      const a=actions[s]; if(!before.observations[s].mask[a]) throw Error(`Masked action ${a}`);
      lines.push(`>p${s+1} ${choice(a)}`);
    }
    if(!lines.length) throw Error('No ready player: engine stalled');
    // Mark the submitted requests waiting; a new request replaces them after resolution.
    for(let s=0;s<2;s++) if(before.ready[s]) this.views[s].request = {...this.views[s].request,wait:true};
    this.write(lines.join('\n')); this.steps++; this.cached=null;
    const errors=this.views.flatMap(v=>v.errors);
    if(errors.some(e=>!e.startsWith('[Unavailable choice]'))) throw Error(`Engine rejected action: ${errors.at(-1)}`);
    if(!this.terminated && this.steps>=this.limit) this.truncated=true;
    return this.result();
  }
  destroy() { this.stream._destroy(); }
}
const observationKeys=['tokens','mons','global','global_tokens','action_tokens','actions','mask'];
function observationBuffer(obs) {
  // Native bulk typed-array copies avoid 700 per-number Buffer method calls.
  if(require('node:os').endianness()!=='LE')throw Error('Packed wire requires little endian host');
  const values=new Float64Array(700);let at=0;
  for(const key of observationKeys) {
    const entries=obs[key];
    if(Array.isArray(entries[0])) {
      for(const row of entries) {values.set(row,at);at+=row.length;}
    } else {values.set(entries,at);at+=entries.length;}
  }
  if(at!==values.length)throw Error('Packed observation schema length');
  return Buffer.from(values.buffer);
}
function packObservation(obs) { return {packed_f64le:observationBuffer(obs).toString('base64')}; }
function packResult(result) {
  if(!result || !result.observations)return result;
  return {...result,observations:result.observations.map(packObservation)};
}
let games=[];let packed=false;let sharedFd=null,sharedBytes=0;
async function handle(x) {
  if(x.op==='schema') return SCHEMA;
  if(x.op==='reset') {
    if(x.shared_path) {
      if(sharedFd!==null)fs.closeSync(sharedFd);
      sharedFd=fs.openSync(x.shared_path,'r+');sharedBytes=x.shared_bytes;
      if(fs.fstatSync(sharedFd).size!==sharedBytes)throw Error('Shared slab size mismatch');
    }
    packed=!!x.packed; games.forEach(g=>g.destroy()); games=x.seeds.map((s,i)=>new Game(s,x.limit || 512,x.swapped?.[i] || false,x.optimized !== false,!!x.token_cache,!!x.mon_cache,!!x.fast_callbacks)); return games.map(g=>g.result()); }
  if(x.op==='reset_slots') return games.map((g,i)=> {
    const u=x.updates[i];if(!u)return null;
    if(!g.terminated&&!g.truncated)throw Error('Cannot reset a live battle');
    g.destroy();games[i]=new Game(u.seed,u.limit || 512,!!u.swapped,g.optimized,!!x.token_cache,!!x.mon_cache,!!x.fast_callbacks);
    return games[i].result();
  });
  if(x.op==='step') return games.map((g,i)=>g.terminated||g.truncated?(g.optimized?null:g.result()):g.step(x.actions[i]));
  if(x.op==='hashes') return games.map(g=>({sha256:g.trace.copy().digest('hex'),turn:g.views[0].turn,steps:g.steps,terminated:g.terminated,truncated:g.truncated}));
  if(x.op==='trace') return games.map(g=>({sha256:g.trace.copy().digest('hex'),side_sha256:g.sideTrace.map(h=>h.copy().digest('hex')),inputs:g.inputs,seeds:g.spec,result:g.result()}));
  if(x.op==='close') { games.forEach(g=>g.destroy()); process.exit(0); }
  throw Error(`Unknown op ${x.op}`);
}
// Requests remain small JSON lines. Binary replies are length-delimited JSON
// metadata followed by lossless f64le observation buffers, in one pipe write.
const binary = process.argv.includes('--binary');
function reply(raw, op) {
  const start=performance.now(), buffers=[];
  const result=binary && ['reset','step','reset_slots'].includes(op) ? raw.map(game=> {
    if(!game?.observations)return game;
    return {...game,observations:game.observations.map(obs=> {
      const index=buffers.length;buffers.push(observationBuffer(obs));return {packed_index:index};
    })};
  }) : packed && ['reset','step','reset_slots'].includes(op) ? raw.map(packResult) : raw;
  const response={ok:true,result};
  workerMetrics.wire_serialize_ms+=performance.now()-start;
  sendResponse(response,buffers);
}
function sendResponse(response,buffers=[]) {
  const start=performance.now();
  response.worker_metrics=workerMetrics;
  if(binary) {
    const metadata=Buffer.from(JSON.stringify(response));
    const header=Buffer.allocUnsafe(8);header.writeUInt32LE(metadata.length,0);
    const bytes=buffers.reduce((n,b)=>n+b.length,0);header.writeUInt32LE(bytes,4);
    workerMetrics.wire_bytes+=8+metadata.length+bytes;
    let frame;
    if(sharedFd!==null) {
      if(bytes>sharedBytes)throw Error('Shared slab overflow');
      const payload=Buffer.concat(buffers);let written=0;
      while(written<payload.length)written+=fs.writeSync(sharedFd,payload,written,payload.length-written,written);
      frame=Buffer.concat([header,metadata]);
      workerMetrics.shared_bytes=(workerMetrics.shared_bytes || 0)+bytes;
    } else frame=Buffer.concat([header,metadata,...buffers]);
    workerMetrics.wire_serialize_ms+=performance.now()-start;
    process.stdout.write(frame);
  } else {
    const body=JSON.stringify(response)+'\n';workerMetrics.wire_bytes+=Buffer.byteLength(body);
    workerMetrics.wire_serialize_ms+=performance.now()-start;
    process.stdout.write(body);
  }
}
if(require.main===module) {
  const rl=readline.createInterface({input:process.stdin});
  rl.on('line',async line=>{try {const request=JSON.parse(line); reply(await handle(request),request.op);}
    catch(e) {const message=String(e.message); const category=/Masked action|Action for waiting/.test(message)?'policy_illegal_action':/engine stalled/.test(message)?'simulator_stall':/Engine rejected action/.test(message)?'engine_rejection':'simulator_or_adapter_fault'; sendResponse({ok:false,error:e.stack,category});}});
  // Parent disappearance closes stdin; never retain an orphan simulator.
  rl.on('close',()=>{games.forEach(g=>g.destroy());process.exit(0);});
}
module.exports={Game,seed,normalizeTrace,packObservation,TEAM_SIZE,FORMAT};
