"""CPU-only diagnosis of earliest/current initialization rounding; no training or mutations."""
import json,modal
IMAGE='im-XaZA2ngN2DTPxrgJzHivWj'  # completed matched-l4-v1 image, all three source installs
image=modal.Image.from_id(IMAGE) if modal.is_local() else modal.Image.debian_slim(python_version='3.12')
app=modal.App('autoresearch-tavern-initialization-20261108',image=image)
CODE=r'''
import sys,json,hashlib,torch
sys.path.insert(0,'/workspace/earliest/integrations/tavern')
from model import Policy
states=[]
for threads in [4,1,4,1]:
 torch.set_num_threads(threads);torch.manual_seed(12345)
 p=Policy(hidden=128,embedding=8)
 states.append({k:v.detach().clone() for k,v in p.state_dict().items()})
result={'torch':torch.__version__,'gpu_used':False,'threads':[4,1,4,1],
 'parameter_checks':{k:{'same':torch.equal(states[0][k],states[1][k]),
 'max_abs':float((states[0][k]-states[1][k]).abs().max()),
 'changed':int((states[0][k]!=states[1][k]).sum()),
 'repeat_four_threads':torch.equal(states[0][k],states[2][k]),
 'repeat_one_thread':torch.equal(states[1][k],states[3][k])} for k in states[0]}}
print(json.dumps(result))
'''

@app.function(cpu=(4,4),memory=(8192,8192),timeout=120,max_containers=1,min_containers=0,
    buffer_containers=0,retries=0,single_use_containers=True)
def compare():
    import subprocess,time
    begin=time.perf_counter();p=subprocess.run(['/venvs/earliest/bin/python','-c',CODE],capture_output=True,text=True,timeout=100)
    return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'worker_seconds':time.perf_counter()-begin,
        'image':IMAGE,'gpu':None,'cpu_cores':4,'ram_gib':8}

@app.local_entrypoint()
def main():print(json.dumps(compare.remote()),flush=True)
