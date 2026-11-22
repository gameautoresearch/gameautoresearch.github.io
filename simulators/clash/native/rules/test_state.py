"""Differential state/king-action tests. No timings or paid jobs."""
from pathlib import Path
import ctypes,json,shutil,subprocess,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from game import HERE as GAME,make_vector,resolve
from runner import setup
import jpype

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    library=out/('unit-state.dylib' if sys.platform=='darwin' else 'unit-state.so')
    command=[shutil.which('clang++') or shutil.which('g++'),'-std=c++20','-O2','-fPIC','-dynamiclib' if sys.platform=='darwin' else '-shared',str(HERE/'test_state_bridge.cpp'),'-o',str(library)]
    subprocess.run(command,check=True);lib=ctypes.CDLL(str(library));ptr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    for name in ['rules_state_visit','rules_state_set','rules_combat_gate']:getattr(lib,name).argtypes=[ptr,ctypes.c_int,ptr]
    lib.rules_king_activation.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr]
    c=json.loads((GAME/'configs/profile-trimmed.json').read_text());c.update(envs=1,threads=1);setup(c);rng=np.random.default_rng(52791);counts={}
    with make_vector(c):
        J=jpype.JClass;Grid=J('org.crforge.core.pathfinding.GridEntity');Timers=J('org.crforge.core.pathfinding.state.StateTimers');Config=J('org.crforge.core.pathfinding.state.StateVisitConfig');Queries=J('org.crforge.core.pathfinding.state.StateQueries');Globals=J('org.crforge.core.pathfinding.state.StateVisitGlobals');Visit=J('org.crforge.core.pathfinding.state.EntityStateVisit');Setter=J('org.crforge.core.pathfinding.GridStateSetter');Target=J('org.crforge.core.pathfinding.target.TargetingState');Move=J('org.crforge.core.pathfinding.move.MovementState');Array=J('java.util.ArrayList')
        n=1000;inputs=np.zeros((n,12),np.int32)
        for p in inputs:p[:]=[rng.choice([0,1,2,4,11]),rng.choice([0,50,75,1000]),rng.integers(0,2001),rng.integers(0,6001),rng.integers(2),rng.integers(0,301),rng.integers(-500,100),1000,rng.integers(2),1,0,0]
        native=np.zeros((n,10),np.int32);lib.rules_state_visit(inputs,n,native)
        for i,p in enumerate(inputs):
            e=Grid();e.setState(int(p[0]));e.setDeployCountdown(int(p[1]));e.setDelay(int(p[2]));e.setPendingDamageDurationMs(int(p[3]));timers=Timers();timers.setSpawnImmune(bool(p[4]));timers.setSpawnImmuneElapsedMs(int(p[5]));timers.setDashImmunityRemainingMs(int(p[6]));q=Queries.forUnitWithRoute(0).withMayHoldRoute(bool(p[8]));cfg=Config.forGroundUnit(1000);chain=Array();setter=Setter(e,None,None,None,1000)
            Visit.stateVisit(e,timers,None,cfg,Globals.standard(),q,chain,setter)
            tail=int(len(chain)>0 and str(chain.get(len(chain)-1))=='visit_tail')
            # Gate calls are tested in the setter matrix separately.
            np.testing.assert_array_equal(native[i,:9],[e.getState(),e.getDeployCountdown(),e.getDelay(),e.getPendingDamageDurationMs(),timers.isSpawnImmune(),timers.getSpawnImmuneElapsedMs(),timers.getDashImmunityRemainingMs(),timers.isRemovalRequested(),tail],err_msg=f'state visit {i}')
        counts['state_visit_vectors']=n
        n=1000;inputs=np.zeros((n,9),np.int32)
        for p in inputs:p[:]=[rng.choice([0,1,2,4,11]),rng.choice([0,50,1000]),rng.choice([0,2,4,11]),rng.integers(0,100),rng.integers(2),1000,rng.integers(2),0,0]
        native=np.zeros((n,11),np.int32);lib.rules_state_set(inputs,n,native)
        for i,p in enumerate(inputs):
            e=Grid();e.setState(int(p[0]));e.setDeployCountdown(int(p[1]));e.setMovementComponent(True);e.setMovementActive(False);m=Move.forSide(0,3500,10000);m.getRoute().add(3);m.getRoute().add(5);m.setRouteLeadsAway(1);t=Target();t.setTargetLostTimerMs(int(p[3]));setter=Setter(e,m if p[6] else None,t,None,1000);setter.setTargetingSwitchedOn(jpype.JProxy('java.util.function.BooleanSupplier',dict(getAsBoolean=lambda p=p:bool(p[4]))))
            called=[0,0,0];setter.setTailGate(jpype.JProxy('java.lang.Runnable',dict(run=lambda:called.__setitem__(0,called[0]+1))));setter.setAttackingExit(jpype.JProxy('java.lang.Runnable',dict(run=lambda:called.__setitem__(1,called[1]+1))));setter.setDeployingEntry(jpype.JProxy('java.lang.Runnable',dict(run=lambda:called.__setitem__(2,called[2]+1))))
            old=e.getState();setter.setState(e,int(p[2]));changed=int(old!=e.getState())
            np.testing.assert_array_equal(native[i],[e.getState(),e.getDeployCountdown(),e.isMovementActive(),t.getTargetLostTimerMs(),m.getRoute().size(),m.getRouteLeadsAway(),0,*called,changed],err_msg=f'state setter {i}')
        counts['state_set_vectors']=n
        n=1000;inputs=np.zeros((n,7),np.int32)
        for p in inputs:p[:]=[rng.choice([0,1,2,4,11]),rng.choice([0,50,1000]),rng.integers(2),rng.integers(2),rng.integers(2),rng.choice([0,50]),rng.integers(2)]
        native=np.zeros((n,2),np.int32);lib.rules_combat_gate(inputs,n,native)
        Boolean=J('java.lang.Boolean').TYPE;Integer=J('java.lang.Integer').TYPE;Runnable=J('java.lang.Runnable');Gate=J('org.crforge.core.battle.unit.CombatGate')
        gate=Gate.class_.getDeclaredMethod('targetingOn',Grid.class_,Target.class_,Boolean,Boolean,Integer,Boolean,Runnable.class_,Runnable.class_,Boolean);gate.setAccessible(True)
        tables=J('org.crforge.env.VectorEnv').tables(J('java.nio.file.Path').of(str(resolve(c['tables']))));bits=J('org.crforge.core.pathfinding.EntityFlags').of(tables)
        TargetConfig=J('org.crforge.core.pathfinding.target.TargetingConfig');TargetView=J('org.crforge.core.pathfinding.target.TargetView');noop=jpype.JProxy('java.lang.Runnable',dict(run=lambda:None))
        for i,p in enumerate(inputs):
            e=Grid();e.setState(int(p[0]));e.setDeployCountdown(int(p[1]));e.setFlagBits(bits);e.setFlags(bits.inactive() if p[2] else jpype.JLong(0));t=Target();t.setOwner(e);t.setConfig(TargetConfig.forUnit(1200,5500,500,1200,700,True,False))
            if p[6]:t.setReference(TargetView(Grid(),TargetConfig.forUnit(1200,5500,500,1200,700,True,False)))
            on=gate.invoke(None,e,t,jpype.JBoolean(bool(p[3])),jpype.JBoolean(bool(p[4])),jpype.JInt(p[5]),jpype.JBoolean(True),noop,noop,jpype.JBoolean(False))
            np.testing.assert_array_equal(native[i],[on,t.getReference() is not None],err_msg=f'combat gate {i}')
        counts['combat_gate_vectors']=n
        # KingActivation: the same WaitToActivate + WithDuration actions, stepped
        # in all ActionHolder phases so finished tags remain for the extra pass.
        trigger=[False];active=J('org.crforge.core.battle.action.WithDuration')('wake',3300,jpype.JLong(2),None);wait=J('org.crforge.core.battle.action.WaitToActivate')('sleep',jpype.JProxy('java.util.function.BooleanSupplier',dict(getAsBoolean=lambda:trigger[0])),active,jpype.JLong(1));holder=J('org.crforge.core.battle.action.ActionHolder')();holder.schedule(wait,0,True)
        native=np.zeros((120,8),np.int32);lib.rules_king_activation(10,120,1,native)
        for tick in range(120):
            before=holder.tags();trigger[0]=tick>=10;holder.pendingPass(1);holder.runPass(tick);holder.pendingPass(2);holder.pendingPass(3);holder.endOfTick()
            assert tuple(native[tick,:2])==(before,holder.tags()),(tick,native[tick],holder.tags())
        counts['king_action_rows']=120
        # Real king starting row: first pending pass starts INACTIVE; damage on
        # tick10 activates the same full holder pipeline used by the simulator.
        Standard=J('org.crforge.core.battle.unit.Standard1v1Battle');b=Standard(tables,11,False);king=b.getWorld().kingTower(0);holder=king.actionHolder()
        native=np.zeros((120,8),np.int32);lib.rules_king_activation(10,120,0,native)
        def normalize(tags):return (1 if tags&bits.inactive() else 0)|(2 if tags&bits.activating() else 0)
        for tick in range(120):
            before=normalize(holder.tags())
            if tick==10:king.getHitPoints().setHitPoints(king.getHitPoints().getHitPoints()-1)
            b.getBattle().step()
            assert tuple(native[tick,:2])==(before,normalize(holder.tags())),(tick,native[tick],normalize(holder.tags()))
        counts['real_king_rows']=120

    result={'status':'passed',**counts,'oracle_commit':'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de','complete_task_parity':False,'command':command}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
