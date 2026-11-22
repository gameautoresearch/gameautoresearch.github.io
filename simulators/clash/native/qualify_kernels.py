"""Actual-device numerical qualification of optional fused arithmetic."""
from pathlib import Path
import json,sys,torch
from train import device_gae
from triton_kernels import gae

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(8151);rows=[]
    for t,n,gamma,lam in [(1,3,.9,.95),(5,4,.99,.95),(17,6,1.,1.),(31,8,.95,0.),(8,5,0.,.95),(32,512,.999,.97),(64,257,.999,.97)]:
        reward=torch.randn(t,n,device='cuda');value=torch.randn_like(reward);done=torch.rand(t,n,device='cuda')<.2;trunc=torch.randn_like(reward)*(torch.rand(t,n,device='cuda')<.5);last=torch.randn(n,device='cuda')
        expected=device_gae(reward,value,done,trunc,last,gamma,lam);actual=gae(reward,value,done,trunc,last,gamma,lam)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
        errors=[float((a-b).abs().max()) for a,b in zip(actual,expected)];rows.append({'ticks':t,'seats':n,'gamma':gamma,'lambda':lam,'max_absolute_errors':errors})
    result={'status':'passed','cuda_executed':True,'gpu':torch.cuda.get_device_name(),'cases':rows,'tolerance':{'rtol':1e-6,'atol':1e-6}};(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
