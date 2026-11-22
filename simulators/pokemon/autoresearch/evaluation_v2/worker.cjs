'use strict';
// This wrapper delegates every transition/encoding to the unchanged canonical Game.
// Only player-channel protocol is retained. No battle object or end payload is exported.
const readline=require('node:readline');
const crypto=require('node:crypto');
const {Game,normalizeTrace}=require('../../worker.cjs');
const {View}=require('../../view.cjs');
const POLICY=Object.freeze({id:'showdown-completion-v2-draft1',max_adapter_phases:4096,max_consecutive_retries_per_seat:16,native_turn_limit:1000});
function recordChoices(game,before,errors) {
  for(let s=0;s<2;s++) {
    if(!before.ready[s])continue;
    game.attempted[s]++;
    const rejected=game.views[s].errors.slice(errors[s]).filter(e=>e.startsWith('[Unavailable choice]')).length;
    if(rejected>1)throw Error('Multiple rejections for one submitted choice');
    game.retries[s]+=rejected;game.accepted[s]+=1-rejected;
    game.consecutiveRetries[s]=rejected?game.consecutiveRetries[s]+rejected:0;
  }
  return game.consecutiveRetries.some(n=>n>=game.retryLimit);
}
class EvaluationGame extends Game {
  constructor(seed,limit=POLICY.max_adapter_phases,swapped=false) {
    // Capture initial public protocol during the synchronous base constructor.
    const logs={p1:[],p2:[]};const original=View.prototype.receive;
    View.prototype.receive=function(text){logs[this.side].push(text);return original.call(this,text);};
    try {super(seed,limit,swapped,true,false,false,false);} finally {View.prototype.receive=original;}
    this.protocol=[logs.p1,logs.p2];this.accepted=[0,0];this.attempted=[0,0];this.retries=[0,0];this.consecutiveRetries=[0,0];this.retryLimit=POLICY.max_consecutive_retries_per_seat;
    this.terminalReason=null;this.phaseTrace=[];this.publicHashes=[crypto.createHash('sha256'),crypto.createHash('sha256')];
    for(let s=0;s<2;s++) {
      for(const text of this.protocol[s])this.publicHashes[s].update(normalizeTrace(text)+'\n');
      const view=this.views[s],receive=view.receive;
      view.receive=(text)=>{this.protocol[s].push(text);this.publicHashes[s].update(normalizeTrace(text)+'\n');return receive.call(view,text);};
    }
    this.cached=null;
  }
  result() {
    const value=super.result();
    // Base constructor calls result only after fields initialized. Defensive early return.
    if(!this.accepted)return value;
    let naturalReason=value.winner===''?'engine_tie':'engine_win';
    if(value.terminated && value.winner==='') {
      if(this.protocol[0].some(t=>t.includes('You have hit the turn limit!')))naturalReason='engine_turn_limit_tie';
      else if(this.protocol[0].some(t=>t.includes('Endless Battle Clause activated!')))naturalReason='engine_endless_battle_clause_tie';
    }
    return {...value,termination_policy:POLICY.id,adapter_phases:this.steps,engine_turns:this.views[0].turn,
      choice_attempts:[...this.attempted],accepted_choices:[...this.accepted],unavailable_choice_retries:[...this.retries],
      consecutive_unavailable_retries:[...this.consecutiveRetries],
      terminal_reason:this.terminalReason || (value.terminated?naturalReason:value.truncated?'adapter_phase_safety_cap':null),
      natural_win:value.terminated&&value.winner!=='',natural_tie:value.terminated&&value.winner==='',
      protocol_hashes:this.publicHashes.map(h=>h.copy().digest('hex'))};
  }
  step(actions) {
    const before=this.result();const errors=this.views.map(v=>v.errors.length);
    super.step(actions);
    const retryCap=recordChoices(this,before,errors);
    if(!this.terminated && retryCap) {
      this.truncated=true;this.terminalReason='unavailable_choice_retry_safety_cap';
    }
    this.cached=null;
    const after=this.result();
    this.phaseTrace.push({phase:this.steps,turn:after.turn,actions:[...actions],ready:before.ready,
      rejections:[...after.rejections],accepted:[...this.accepted],trace_sha256:this.trace.copy().digest('hex'),
      observations_sha256:crypto.createHash('sha256').update(JSON.stringify(after.observations)).digest('hex')});
    return after;
  }
  visible(seat,offset=0) {
    if(seat!==0&&seat!==1)throw Error('Invalid player seat');
    return {seat,request:structuredClone(this.views[seat].request),protocol:this.protocol[seat].slice(offset),
      protocol_offset:this.protocol[seat].length,mask:this.result().observations[seat].mask,
      turn:this.views[seat].turn,termination_policy:POLICY.id};
  }
}
let games=[];
function handle(x) {
  if(x.op==='reset') {games.forEach(g=>g.destroy());games=x.seeds.map((s,i)=>new EvaluationGame(s,x.limit||POLICY.max_adapter_phases,x.swapped?.[i]||false));return games.map(g=>g.result());}
  if(x.op==='step')return games.map((g,i)=>g.terminated||g.truncated?null:g.step(x.actions[i]));
  if(x.op==='visible')return games.map((g,i)=>g.visible(x.seats[i],x.offsets?.[i]||0));
  if(x.op==='audit')return games.map(g=>({trace:g.phaseTrace,hash:g.trace.copy().digest('hex'),result:{...g.result(),observations:undefined,seeds:undefined,seed:undefined}}));
  throw Error('Unsupported evaluation-v2 operation');
}
if(require.main===module) {
  const rl=readline.createInterface({input:process.stdin});
  rl.on('line',line=>{try{process.stdout.write(JSON.stringify({ok:true,result:handle(JSON.parse(line))})+'\n');}
    catch(e){const message=String(e.message);process.stdout.write(JSON.stringify({ok:false,error:e.stack,category:/Masked action|Action for waiting/.test(message)?'policy_illegal_action':/engine stalled/.test(message)?'simulator_stall':/Engine rejected/.test(message)?'engine_rejection':'simulator_or_adapter_fault'})+'\n');}});
  rl.on('close',()=>{games.forEach(g=>g.destroy());process.exit(0);});
}
module.exports={EvaluationGame,POLICY,recordChoices};
