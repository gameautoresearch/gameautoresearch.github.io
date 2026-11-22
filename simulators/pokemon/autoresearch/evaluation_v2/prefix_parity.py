"""Full 700-value encoder/public protocol equality under recorded original actions."""
import argparse,json,subprocess
from pathlib import Path
from ..contract import write
JS=r'''
const fs=require('node:fs'),assert=require('node:assert/strict');
const {Game}=require('./integrations/pokemon/worker.cjs');
const {EvaluationGame}=require('./integrations/pokemon/autoresearch/evaluation_v2/worker.cjs');
const spec=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));let results=[];
for(const c of spec.cases) {
 const records=JSON.parse(fs.readFileSync(c.retained_trace,'utf8'));
 const a=new Game(c.seed,512),b=new EvaluationGame(c.seed,4096);
 let n=0;
 try {
  assert.deepEqual(a.result().observations,b.result().observations);
  for(const record of records) {
   a.step(record.actions);b.step(record.actions);n++;
   assert.deepEqual(a.result().observations,b.result().observations);
   assert.equal(a.trace.copy().digest('hex'),b.trace.copy().digest('hex'));
  }
  assert(n===512);assert(a.truncated&&!b.truncated);
  results.push({label:c.label,all700values_exact_each_phase:true,all_public_protocol_hashes_exact:true,phases:n,
   canonical512terminated:a.terminated,canonical512truncated:a.truncated,v2prefixtruncated:b.truncated});
 } finally {a.destroy();b.destroy();}
}
console.log(JSON.stringify({results,all_passed:true}));
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 result=subprocess.run(['node','-e',JS,a.request],capture_output=True,text=True,timeout=45)
 Path(a.output,'node-stderr.log').write_text(result.stderr)
 if result.returncode:raise RuntimeError(result.stderr)
 write(Path(a.output,'summary.json'),json.loads(result.stdout))
if __name__=='__main__':main()
