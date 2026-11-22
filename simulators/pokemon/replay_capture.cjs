'use strict';
// Spectator projection runs beside the unchanged source-bound inference adapter.
const path=require('node:path');
const worker=path.resolve(process.argv[2]||path.join(__dirname,'worker.cjs'));
const {Game}=require(worker);
const {extractChannelMessages}=require(path.resolve(path.dirname(worker),'../../candidates/pokemon-showdown/dist/sim/battle'));
const readline=require('node:readline');
let game, events=[];
function snapshot() {
  const b=game.stream.battle;
  return {phase:game.steps,turn:b.turn,ended:game.terminated,truncated:game.truncated,
    winner:game.result().winner,events,
    sides:b.sides.map(side=>({name:side.name,team:side.pokemon.map(p=>({
      name:p.species.name,types:p.getTypes(),hp:p.hp,maxhp:p.maxhp,status:p.status,
      active:p.isActive,fainted:p.fainted,level:p.level,tera:p.terastallized||null,
    }))}))};
}
function reset(seed) {
  if(game)game.destroy();game=new Game(seed,512);events=[];
  const original=game.stream.pushMessage;
  game.stream.pushMessage=(type,data)=>{
    if(type==='update')for(const line of extractChannelMessages(data,[1,2])[1]){
      const k=line.split('|')[1];
      if(['move','switch','drag','faint','-supereffective','-resisted','-crit','-status','-terastallize','win','tie'].includes(k))events.push(line);
    }
    return original.call(game.stream,type,data);
  };
}
const rl=readline.createInterface({input:process.stdin});
rl.on('line',line=>{try{const x=JSON.parse(line);
  if(x.op==='reset')reset(x.seed);
  else if(x.op==='step'){events=[];game.step(x.actions);}
  else if(x.op==='verify'){
    const expected=game.trace.copy().digest('hex');const frames=[];
    reset(x.seed);frames.push(snapshot());
    for(const action of x.actions){events=[];game.step(action);frames.push(snapshot());}
    const actual=game.trace.copy().digest('hex');
    process.stdout.write(JSON.stringify({ok:true,trace:actual,matched:expected===actual,frames})+'\n');return;
  }else throw Error('Unknown capture request');
  process.stdout.write(JSON.stringify({ok:true,state:game.result(),frame:snapshot()})+'\n');
}catch(e){process.stdout.write(JSON.stringify({ok:false,error:String(e.stack)})+'\n');}});
rl.on('close',()=>{if(game)game.destroy();process.exit(0);});
