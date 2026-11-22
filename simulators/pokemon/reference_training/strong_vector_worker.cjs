'use strict';
// Training-only sparse public-channel transport. Every game delegates transitions,
// visible encoding, retry counters and termination to the qualified EvaluationGame.
const readline=require('node:readline');
const {EvaluationGame,POLICY}=require('../autoresearch/evaluation_v2/worker.cjs');
let games=[];
function handle(x) {
 if(x.op==='reset') {games.forEach(g=>g.destroy());games=x.seeds.map((s,i)=>new EvaluationGame(s,x.limit||POLICY.max_adapter_phases,x.swapped?.[i]||false));return games.map(g=>g.result());}
 if(x.op==='step')return games.map((g,i)=>g.terminated||g.truncated?null:g.step(x.actions[i]));
 if(x.op==='visible')return games.map((g,i)=>x.seats[i]===null?null:g.visible(x.seats[i],x.offsets?.[i]||0));
 if(x.op==='audit')return games.map(g=>({trace:g.phaseTrace,hash:g.trace.copy().digest('hex'),result:{...g.result(),observations:undefined,seeds:undefined,seed:undefined}}));
 throw Error('Unsupported sparse public training operation');
}
const rl=readline.createInterface({input:process.stdin});
rl.on('line',line=>{try{process.stdout.write(JSON.stringify({ok:true,result:handle(JSON.parse(line))})+'\n');}
 catch(e){const message=String(e.message);process.stdout.write(JSON.stringify({ok:false,error:e.stack,category:/Masked action|Action for waiting/.test(message)?'policy_illegal_action':/engine stalled/.test(message)?'simulator_stall':/Engine rejected/.test(message)?'engine_rejection':'simulator_or_adapter_fault'})+'\n');}});
rl.on('close',()=>{games.forEach(g=>g.destroy());process.exit(0);});
