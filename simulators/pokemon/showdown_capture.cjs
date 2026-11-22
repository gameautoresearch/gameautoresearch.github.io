'use strict';
// Recorder only: full spectator protocol never enters either policy's observations.
const {BattleStream} = require('../../vendor/engines/pokemon-showdown/dist/sim/battle-stream');
const {extractChannelMessages} = require('../../vendor/engines/pokemon-showdown/dist/sim/battle');
const {EvaluationGame} = require('./autoresearch/evaluation_v2/worker.cjs');
const readline = require('node:readline');
const normalize = text => text.replace(/^\|t:\|[0-9]+$/gm, '|t:|0');
let game, updates = [];
const originalWrite = BattleStream.prototype._write;
BattleStream.prototype._write = function (message) {
  const push = this.pushMessage;
  this.pushMessage = function (type, data) {
    if (type === 'update') updates.push(...extractChannelMessages(data, [-1])[-1]);
    return push.call(this, type, data);
  };
  try { return originalWrite.call(this, message); }
  finally { this.pushMessage = push; }
};
function reset(seed) {
  game?.destroy(); updates = []; game = new EvaluationGame(seed, 4096);
}
readline.createInterface({input: process.stdin}).on('line', line => {
  try {
    const x = JSON.parse(line);
    if (x.op === 'reset') reset(x.seed);
    else if (x.op === 'step') game.step(x.actions);
    else if (x.op === 'verify') {
      const expected = game.trace.copy().digest('hex'), log = normalize(updates.join('\n'));
      reset(x.seed);
      for (const action of x.actions) game.step(action);
      const actual = game.trace.copy().digest('hex');
      process.stdout.write(JSON.stringify({ok:true, matched:expected === actual && log === normalize(updates.join('\n')), trace:actual, log:normalize(updates.join('\n')), state:game.result()})+'\n'); return;
    } else throw Error('Unknown capture operation');
    process.stdout.write(JSON.stringify({ok:true, state:game.result()})+'\n');
  } catch (e) { process.stdout.write(JSON.stringify({ok:false,error:String(e.stack)})+'\n'); }
}).on('close', () => {game?.destroy(); process.exit(0);});
