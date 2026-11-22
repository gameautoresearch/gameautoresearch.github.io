"""Differential tests for the first native port modules, never a full-engine parity claim."""
from pathlib import Path
import ctypes,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from game import HERE,make_vector
from runner import setup
from build import build
import jpype

def main():
    output=Path(sys.argv[1]);output.mkdir(exist_ok=False)
    library=build();native=ctypes.CDLL(str(library));ptr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    native.cn_random.argtypes=[ctypes.c_int32,ptr,ctypes.c_int,ptr,ptr];native.cn_deck.argtypes=[ctypes.c_int32,ptr];native.cn_arithmetic.argtypes=[ptr,ctypes.c_int,ptr]
    c=json.loads((HERE/'configs/profile-trimmed.json').read_text());c.update(envs=1,threads=1);setup(c)
    rng=np.random.default_rng(781)
    with make_vector(c) as v:
        Random=jpype.JClass('org.crforge.core.battle.BattleRandom');Math=jpype.JClass('org.crforge.core.pathfinding.math.FixedMath')
        bounds=np.array([-1,0,1,2,3,7,999,2147483647]*20,np.int32)
        for seed in [0,1,-1,-2147483648,2147483647,*rng.integers(-2147483648,2147483647,50).tolist()]:
            draws=np.empty(len(bounds),np.int32);states=np.empty_like(draws);native.cn_random(seed,bounds,len(bounds),draws,states)
            reference=Random(seed)
            for bound,draw,state in zip(bounds,draws,states):
                assert reference.next(int(bound))==int(draw),(seed,bound,draw)
                assert reference.getState()==int(state)
        # DeckShuffle's constructor/next are private; test via its public order using
        # the actual eight MatchCard records, including the reference's card flags.
        Vector=jpype.JClass('org.crforge.env.VectorEnv');PathJ=jpype.JClass('java.nio.file.Path')
        from game import resolve
        tables=Vector.tables(PathJ.of(str(resolve(c['tables']))));records=jpype.JClass('org.crforge.core.battle.data.BattleRecords')(tables)
        cards=jpype.JClass('java.util.ArrayList')()
        for name in c['rules']['deck']:cards.add(records.matchCard(name))
        Shuffle=jpype.JClass('org.crforge.core.battle.match.DeckShuffle')
        for seed in [0,-1,2147483647,-2147483648,*rng.integers(-2147483648,2147483647,100).tolist()]:
            result=np.empty(8,np.int32);native.cn_deck(seed,result)
            np.testing.assert_array_equal(result,np.array(list(Shuffle.order(cards,seed)),np.int32))
        xy=rng.integers(-100000,100000,(2000,2),dtype=np.int32);xy[:4]=[[0,0],[46340,46340],[-46340,1],[-2147483648,-1]]
        result=np.empty((len(xy),4),np.int32);native.cn_arithmetic(xy,len(xy),result)
        for (x,y),row in zip(xy,result):
            expected=[Math.guardedSumOfSquares(int(x),int(y)),Math.guardedDistance(int(x),int(y)),Math.approxDistance(int(x),int(y)),Math.divOrZero(int(x),int(y))]
            np.testing.assert_array_equal(row,expected)
    summary={'status':'passed','library':str(library),'rng_draws':55*len(bounds),'deck_seeds':104,'arithmetic_vectors':len(xy),'full_task_parity':False,'scope':'native arithmetic, battle RNG and fixed-eight shuffle independently match pinned Java methods'}
    (output/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
