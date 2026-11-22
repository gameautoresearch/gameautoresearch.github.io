#!/usr/bin/env python3
"""600 complete visitor ticks against the pinned Java query composition; no timings.
Oracle crforge core/src/main/java/org/crforge/core/pathfinding/ @
f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de (https://github.com/voonhous/crforge).
This is an isolated routing/targeting world; damage, actual towers and timeline are not simulated.
"""
from pathlib import Path
import json, os, subprocess, sys, shlex
import jpype as jp
from crforge_env._jvm import start
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'evidence/clash-implementation/native-pathfinding-20261108/adapter-1'
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'result.json').exists():raise FileExistsError(OUT/'result.json')
os.environ.setdefault('JAVA_HOME','/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home')
compiler_flags=shlex.split(os.environ.get('CLASH_PATHFINDING_CXXFLAGS',''))
subprocess.run(['c++','-std=c++20','-Wall','-Wextra','-Werror',*compiler_flags,str(HERE/'adapter_smoke.cpp'),'-o',str(OUT/'probe')],check=True)
lines=subprocess.check_output([str(OUT/'probe'),'trace'],text=True).splitlines();native=[list(map(int,x.split(','))) for x in lines[:-1]]
start(ROOT/'candidates/crforge/env/build/install/env');J=lambda s:jp.JClass('org.crforge.core.pathfinding.'+s);IA=jp.JArray(jp.JInt)
E=J('GridEntity');TV=J('target.TargetView');TC=J('target.TargetingConfig');TS=J('target.TargetingState');MS=J('move.MovementState')
index=J('index.SpatialIndex')(36,64);cfg=TC.forUnit(1200,5500,500,1200,700,True,False)
e=E();e.setId(5000007);e.setX(9000);e.setY(1000);e.setState(1);e.setMovementActive(True);e.setMovementComponent(True);e.setTargetable(1);e.setCollisionRadius(500);e.setMass(6)
t=TS();t.setConfig(cfg);t.setOwner(e);t.setAliveCheckBypass(True);t.setMovementComponentActive(True);m=MS();m.setRouteDirY(256)
selection=J('target.SelectionChain')(index,t,64);tv=TV(e,cfg);selection.register(tv)
listed=jp.JClass('java.util.ArrayList')();listed.add(e);byid={}
for i in [1,2]:
    r=E();r.setId(5000000+i);r.setX(9000 if i==1 else 14000);r.setY(25000);r.setCollisionRadius(750);r.setMass(20);r.setSide(1);r.setBuilding(True);r.setCrownTower(True);r.setTargetable(1);view=TV(r,TC.tower('mock',0,0,0,0,0,False));listed.add(r);byid[r.getId()]=i
    if i==1:selection.registerTower(view)
    else:selection.register(view);selection.setSeed(view)
grid=J('grid.CellGrid')(J('grid.TileMap')(36,64,IA([0]*2304)),True,50);costs=J('grid.CellCosts').standard();mc=J('move.MovementConfig').forGroundUnit();mg=J('move.MovementGlobals').forStandardArena(36)
unit=J('GridUnitState')(e,m,t,J('state.StateTimers')(),mc,J('move.SpeedConfig').forGroundUnit(60),None,selection,tv)
lookup=jp.JProxy('java.util.function.Function',dict={'apply':lambda other:unit if other.getId()==e.getId() else None})
neighbours=J('IndexNeighbourQuery')(index)
def chain():
    q=J('GridMovementQueries')(unit,grid,costs,lookup).withBuffs(IA([]),100,0)
    return J('move.MovementChain')(m,e,grid,mc,mg,q.reference(),neighbours,q).withoutMarkers()
supplier=jp.JProxy('java.util.function.Supplier',dict={'get':chain});setter=J('GridStateSetter')(e,m,t,supplier)
selection.setStateSetter(setter);selection.setBuildingKeepsAttacking(True);selection.setStopsWithoutTarget(jp.JProxy('java.util.function.BooleanSupplier',dict={'getAsBoolean':lambda:True}))
selection.getOutcome().setRoutePreparer(jp.JProxy('java.lang.Runnable',dict={'run':lambda:setter.prepareRoute()}))
hits=[0]
def hit(*args):hits[0]+=1;t.setHitStarted(True);return False
selection.setHitSink(jp.JProxy(J('target.HitSink'),dict={'hit':hit}))
def turn(target):
    if target is None:return
    v=IA([target.x()-e.getX(),target.y()-e.getY()]);J('math.FixedMath').normalize(v,256);e.setDirX(v[0]);e.setDirY(v[1])
selection.setTurn(jp.JProxy('java.util.function.Consumer',dict={'accept':turn}))
for tick in range(600):
    index.rebuild(listed);selection.beginTick();t.setRouteLeadsAway(m.getRouteLeadsAway()!=0)
    J('target.TargetingVisit').targetingVisit(t,e,m,selection,selection.getOutcome())
    if selection.getOutcome().isResumeRequested():setter.setState(e,1)
    c=chain();q=J('GridMovementQueries')(unit,grid,costs,lookup).withBuffs(IA([]),100,0)
    J('move.MovementVisit').movementVisit(m,e,None,mc,None,q,False,c)
    ref=t.getReference();reference=-1 if ref is None else byid[ref.id()]
    actual=[e.getX(),e.getY(),e.getDirX(),e.getDirY(),e.getState(),reference,t.getAttackTimerMs(),t.getLoadTimerMs(),m.getRoute().size(),m.getRouteDirX(),m.getRouteDirY(),hits[0]]
    if native[tick]!=actual:raise AssertionError({'tick':tick,'native':native[tick],'java':actual})
result={'status':'passed','ticks':600,'integer_fields_compared':7200,'final':native[-1],'compiler_flags':compiler_flags,'scope':'isolated pathfinding world; no damage or timeline','oracle_commit':'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de','cuda_executed':False,'timing_comparison':False}
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
