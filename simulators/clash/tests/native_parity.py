"""Pinned engine/native-pool determinism on occupied training-domain games."""
from pathlib import Path
import json,sys,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from game import HERE,make_vector,Session,dump
from runner import setup

def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
    cores=int(os.getenv('CLASH_CPU_CORES','4'))
    c=json.loads((HERE/'configs/profile-trimmed.json').read_text());c.update(envs=16,threads=cores)
    c['hardware_allocation']['cpu_threads']=cores;setup(c)
    maximum=0;late=0
    with make_vector(dict(c,threads=1)) as reference,make_vector(c) as parallel:
        a=Session(reference,dict(c,threads=1),np.random.default_rng(211))
        b=Session(parallel,dict(c,vector_bookkeeping=True),np.random.default_rng(211))
        rng=np.random.default_rng(811)
        for step in range(1500):
            actions=np.array([[rng.choice(np.flatnonzero(mask)) for mask in row] for row in reference.masks],np.int32)
            a.step(actions);b.step(actions)
            for name in ('observations','masks','infos','dones','rewards'):
                np.testing.assert_array_equal(getattr(reference,name),getattr(parallel,name))
            maximum=max(maximum,int(reference.observations[:,:,reference.schema.header['entity_count'][0]].max()))
            late+=int((reference.observations[:,:,reference.schema.header['tick'][0]]>=2400).sum())
            a.reset_done();b.reset_done()
        assert a.histories==b.histories
        assert a.counters==b.counters
        assert a.state()['observation_sha256']==b.state()['observation_sha256']
        for key in ('illegal_actions','unsupported_actions','entity_overflow_observations'):assert not a.counters[key],(key,a.counters[key])
        assert not reference.fault_counts()
        result={'status':'passed','threads_compared':[1,cores],'envs':16,'vector_steps':1500,'player_decisions':16*2*1500,'maximum_entities':maximum,'late_game_player_decisions':late,'counts':dict(a.counters),'checks':['all observations/masks/info/rewards/dones exact on every step','full histories','all counters','zero illegal/unsupported/fault/overflow','original source/table/jar pins checked']}
        dump(out/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
