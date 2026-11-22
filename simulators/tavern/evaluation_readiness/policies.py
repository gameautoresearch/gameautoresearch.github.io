"""Strict source-compatible learned admission and visible-array inference adapters."""
import base64,importlib.util,io,json,os,selectors,subprocess,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'autoresearch'))
from contract import binding,sha,EDITABLE,validate

def visible_exchange(process,payload,seconds=15):
    """One deadline covers blocked input, startup and a partial output line."""
    incoming=bytearray();sent=0;end=time.monotonic()+seconds
    with selectors.DefaultSelector() as selector:
        for stream in (process.stdin,process.stdout):os.set_blocking(stream.fileno(),False)
        selector.register(process.stdin,selectors.EVENT_WRITE)
        selector.register(process.stdout,selectors.EVENT_READ)
        while True:
            remaining=end-time.monotonic()
            if remaining<=0:raise TimeoutError('Candidate forward deadline')
            events=selector.select(remaining)
            if not events:raise TimeoutError('Candidate forward deadline')
            for key,event in events:
                if key.fileobj is process.stdin:
                    try:sent+=os.write(key.fd,payload[sent:])
                    except BlockingIOError:continue
                    if sent==len(payload):selector.unregister(process.stdin)
                else:
                    try:chunk=os.read(key.fd,65536)
                    except BlockingIOError:continue
                    if not chunk:raise RuntimeError('Candidate process exited before complete reply; inspect stderr')
                    incoming.extend(chunk)
                    if len(incoming)>1000000:
                        from .lobby import IllegalPolicyAction
                        raise IllegalPolicyAction('Oversized candidate output')
                    if b'\n' in incoming:
                        line,extra=incoming.split(b'\n',1)
                        if extra.strip():
                            from .lobby import IllegalPolicyAction
                            raise IllegalPolicyAction('Unsolicited candidate output')
                        return line


def admit_bundle(bundle):
    bundle=Path(bundle);ident=json.loads((bundle/'identity.json').read_text());summary=json.loads((bundle/'summary.json').read_text())
    if ident['binding']!=binding():raise ValueError('Source/rules-bound checkpoint mismatch')
    validate(ident['config'])
    if any(sha(bundle/f)!=ident['source'][f] for f in EDITABLE):raise ValueError('Editable source identity mismatch')
    if sha(bundle/'identity.json')!=summary['identity_sha256'] or sha(bundle/'weights.npz')!=summary['weights_sha256']:raise ValueError('Weights/identity digest mismatch')
    if (bundle/'weights.npz').stat().st_size>32000000:raise ValueError('Oversized weights')
    with np.load(bundle/'weights.npz',allow_pickle=False) as z:
        if sum(v.size for v in z.values())>2000000 or any(not np.isfinite(v).all() for v in z.values()):raise ValueError('Nonfinite/oversized neural state')
    return ident,summary

class NeuralVisible:
    """Host-owned audited reference source; receives only visible feature/card tensors."""
    def __init__(self,bundle,policy_id):
        self.bundle=Path(bundle).resolve();self.policy_id=policy_id;ident,_=admit_bundle(self.bundle)
        spec=importlib.util.spec_from_file_location('reference_'+sha(self.bundle/'policy.py')[:16],self.bundle/'policy.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.model=module.Policy(**ident['config']['network']).cpu().eval()
        with np.load(self.bundle/'weights.npz',allow_pickle=False) as z:self.model.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files},strict=True)
    @torch.no_grad()
    def remote_act(self,obs,indices):
        f=torch.from_numpy(obs['features'][indices].copy());cards=torch.from_numpy(obs['cards'][indices].astype(np.int64,copy=True));mask=torch.from_numpy(obs['action_mask'][indices].copy()).bool()
        logits,_=self.model(f,cards)
        if logits.shape!=mask.shape or not torch.isfinite(logits[mask]).all() or not mask.any(-1).all():raise ValueError('Reference logits/mask invalid')
        actions=logits.masked_fill(~mask,-torch.inf).argmax(-1).numpy()
        return actions,np.zeros(len(indices),np.float32),np.zeros(len(indices),np.float32)

class IsolatedVisible:
    """Candidate: separate process/UID in root container, empty credential environment."""
    def __init__(self,bundle,policy_id,stderr=None):
        self.bundle=Path(bundle).resolve();self.policy_id=policy_id;admit_bundle(self.bundle)
        def demote():os.setgroups([]);os.setgid(65534);os.setuid(65534)
        self.p=subprocess.Popen([sys.executable,'-I',str(HERE.parent/'autoresearch/policy_worker.py'),str(self.bundle)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr or subprocess.DEVNULL,env={'PATH':'/usr/local/bin:/usr/bin:/bin','OMP_NUM_THREADS':'1'},cwd='/tmp',bufsize=0,preexec_fn=demote if os.geteuid()==0 else None)
    def remote_act(self,obs,indices):
        buf=io.BytesIO();np.savez(buf,features=obs['features'][indices],cards=obs['cards'][indices],mask=obs['action_mask'][indices]);line=visible_exchange(self.p,base64.b64encode(buf.getvalue())+b'\n')
        try:values=json.loads(line)
        except ValueError as exc:
            from .lobby import IllegalPolicyAction
            raise IllegalPolicyAction('Malformed candidate JSON') from exc
        if not isinstance(values,list) or any(type(v) is not int for v in values):
            from .lobby import IllegalPolicyAction
            raise IllegalPolicyAction('Candidate actions must be JSON integers')
        return np.asarray(values,np.int64),np.zeros(len(values),np.float32),np.zeros(len(values),np.float32)
    def close(self):
        if self.p.poll() is None:self.p.kill()
        self.p.wait(timeout=5)
