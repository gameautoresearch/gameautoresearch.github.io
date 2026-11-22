#!/usr/bin/env python3
"""Correctness-only differential test; writes a new evidence directory, never timings.

Oracle: crforge core/src/main/java/org/crforge/core/pathfinding/
@ f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de. Uses the pinned installed Java jars.
Run with .venv-clash/bin/python integrations/clash/native/pathfinding/test_differential.py.
"""
from pathlib import Path
import ctypes, hashlib, json, os, random, subprocess, sys, shlex

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
PIN='f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de'
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'evidence/clash-implementation/native-pathfinding-20261108'
os.environ.setdefault('JAVA_HOME','/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home')
import jpype as jp
from crforge_env._jvm import start

def run():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'candidates/crforge',text=True).strip()==PIN
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'differential-result.json').exists():raise FileExistsError(OUT/'differential-result.json')
    libpath=OUT/'probe.dylib'
    compiler_flags=shlex.split(os.environ.get('CLASH_PATHFINDING_CXXFLAGS',''))
    subprocess.run(['c++','-std=c++20','-Wall','-Wextra','-Werror',*compiler_flags,'-shared','-fPIC',str(HERE/'differential.cpp'),'-o',str(libpath)],check=True)
    lib=ctypes.CDLL(str(libpath));probe=lib.pf_probe;probe.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)];probe.restype=ctypes.c_int
    def native(op,a):
        av=(ctypes.c_int*len(a))(*a);ov=(ctypes.c_int*8192)();n=probe(op,av,ov);return list(ov[:n])
    start(ROOT/'candidates/crforge/env/build/install/env')
    J=lambda p:jp.JClass('org.crforge.core.pathfinding.'+p)
    IA=jp.JArray(jp.JInt)
    FM=J('math.FixedMath');RS=J('grid.RouteSearch');CC=J('grid.CellCosts').standard();CF=J('grid.CellCostField')
    TM=J('grid.TileMap');CG=J('grid.CellGrid');GM=J('move.GridMove');GME=J('move.GridMoveEntity')
    E=J('GridEntity');MS=J('move.MovementState');MC=J('move.MovementConfig');MG=J('move.MovementGlobals')
    TS=J('target.TargetingState');TC=J('target.TargetingConfig');TV=J('target.TargetView');TO=J('target.TargetingOutcome')
    AT=J('target.AttackTimerAdvance');Visit=J('target.TargetingVisit');Disp=J('move.Displacement');Chain=J('move.MovementChain')
    rng=random.Random(8190801);checks={}
    def check(kind,actual,expect,case):
        actual=list(map(int,actual));expect=list(map(int,expect))
        if actual!=expect:raise AssertionError(json.dumps({'kind':kind,'input':case,'native':actual,'java':expect}))
        checks[kind]=checks.get(kind,0)+1
    for i in range(5000):
        x=rng.randint(-30000,30000);y=rng.randint(-30000,30000);angle=rng.randint(-2000000000,2000000000)
        vec=IA([x,y]);FM.rotate1024(vec,angle)
        check('trig',native(1,[x,y,angle]),[FM.sine1024(angle),FM.angleOfVector(x,y),*vec],[x,y,angle])
    for i in range(500):
        w=rng.randint(2,18);h=rng.randint(2,18);cost=[rng.choice([-1,5,5,7,8,50]) for _ in range(w*h)]
        a=[w,h,rng.randrange(w*h),rng.randrange(w*h),rng.randrange(4),rng.randrange(1,10),rng.randrange(2),rng.randrange(2),0,rng.choice([0,0,1,5,20])]
        # Reopening is nonstandard; include it only in nonaccumulating positive-cost mode.
        a[8]=rng.randrange(2) if not a[6] else 0
        r=RS.search(w,h,IA(cost),a[2],a[3],a[4],a[5],bool(a[6]),bool(a[7]),bool(a[8]),a[9])
        route=list(r.route().toArray());expected=[len(route),*route,*r.counters(),r.remainingBudget()]
        check('route-search',native(2,a+cost),expected,a+cost)
        check('route-search-into',native(18,a+cost),expected,a+cost)
    for i in range(3000):
        a=[36,64,rng.randrange(-2,38),rng.randrange(-2,66),rng.randrange(128),rng.randrange(2),rng.randrange(2),rng.randrange(2),rng.randrange(17),rng.randrange(3),rng.randrange(2),rng.randrange(100)]
        answer=CF.cellCost(*a[:5],CC,*[bool(v) for v in a[5:8]],*a[8:10],bool(a[10]),a[11])
        check('cell-cost',native(3,a),[answer],a)
    for i in range(2000):
        w=8;h=10;tiles=[rng.choice([0,0,0,32]) for _ in range(w*h)]
        a=[w,h,rng.randrange(0,w*500),rng.randrange(0,h*500),rng.randrange(-250,251),rng.randrange(-250,251),5,rng.choice([0,1,2,4,6]),rng.randrange(2),rng.randrange(2),rng.randrange(2),0]
        grid=CG(TM(w,h,IA(tiles)),True,50);p=IA(a[2:4]);GM.gridMove(grid,p,a[4],a[5],GME(a[6],a[7],bool(a[8]),bool(a[10])),a[9])
        check('grid-move',native(4,a+tiles),p,a+tiles)
    for i in range(3000):
        a=[rng.choice([600,800,1000,1200,1600,1800]),rng.choice([0,200,400,1000,2000]),rng.choice([50,100,150]),rng.choice([-1,1]),rng.choice([-1,0]),rng.choice([0,0,0,50,200,1600]),rng.choice([0,100,200,1500]),rng.choice([0,0,1,50,100]),rng.randrange(2),rng.randrange(2),0,rng.randrange(8)==0,rng.choice([0,50,100,150])]
        cfg=TC.forUnit(1000,5500,250,a[0],a[1],True,False)
        # Base sequence entry's multiplier is 100; varying multiplier needs its builder.
        a[2]=100
        t=TS();t.setConfig(cfg);t.setOwner(E());t.setReference(TV(E(),cfg) if a[3]>=0 else None)
        for method,value in [('AttackSequenceIndex',a[4]),('AttackTimerMs',a[5]),('LoadTimerMs',a[6]),('AttackStartDelayMs',a[7]),('ChargeStrike',bool(a[8])),('InstantHit',bool(a[9]))]:getattr(t,'set'+method)(value)
        q=jp.JProxy(J('target.TargetingQueries'),dict={'attackTimersHeld':lambda:bool(a[11]),'scaleTimeStep':lambda base:int(base*a[12]/100),'runSelection':lambda:None,'validateReference':lambda mode:True,'hitSink':lambda:None})
        AT.advance(t,cfg,q)
        expected=[t.getAttackTimerMs(),t.getLoadTimerMs(),t.getAttackStartDelayMs(),t.isChargeStrike(),t.isInstantHit(),t.getBurstProgressMs(),t.isHitInProgressWithoutReference(),t.isAttackTimeRoundedUp()]
        check('attack-timer',native(5,a),expected,a)
    grid=CG(TM(36,64,IA([0]*2304)),True,50);cfgm=MC.forGroundUnit();globals_=MG.forStandardArena(36)
    class MQ:
        def __init__(self,e,speed=60):self.e=e;self.speed=speed
        def speedBudget(self):return self.speed
        def entityView(self):return GME(5,self.e.getState(),False,False)
        def facingGate(self):return 1
        def avoidanceGate(self):return 0
        def pushGate(self):return 0
        def routeRequest(self):return 0
        def ownerSide(self):return 0
        def referenceAvailable(self):return 0
        def gridWidth(self):return 36
    empty=jp.JClass('java.util.ArrayList')()
    for i in range(2000):
        a=[rng.randint(500,17000),rng.randint(500,31000),rng.randint(-256,256),rng.randint(-256,256),rng.choice([0,1,2]),rng.randint(-256,256),rng.randint(-256,256),rng.randint(-256,256),rng.randint(-500,500),rng.randint(-500,500),rng.randrange(1,4),0,rng.randrange(2),rng.randint(500,17000),rng.randint(500,31000),rng.randint(-10,250),rng.randrange(2)]
        e=E();e.setX(a[0]);e.setY(a[1]);e.setDirX(a[2]);e.setDirY(a[3]);e.setState(a[4]);m=MS()
        for name,v in [('RouteDirX',a[5]),('RouteDirY',a[6]),('AvoidanceBlend',a[7]),('PushX',a[8]),('PushY',a[9]),('PushCount',a[10]),('PushStuck',a[11]),('PushUnclamped',a[12])]:getattr(m,'set'+name)(v)
        q=jp.JProxy(J('move.MovementQueries'),inst=MQ(e));chain=Chain(m,e,grid,cfgm,globals_,None,empty,q).withoutMarkers()
        step=Disp.displace(m,e,grid,cfgm,globals_,q,chain,a[13],a[14],a[15],a[16],0)
        check('displacement',native(6,a),[e.getX(),e.getY(),e.getDirX(),e.getDirY(),m.getWaypointReached(),m.getPushX(),m.getPushY(),m.getPushCount(),step],a)
    for i in range(2000):
        a=[9000,10000,rng.choice([0,1,2,4,6,7,9,10]),0,rng.randint(7000,11000),rng.randint(9000,14000),rng.choice([250,500,750]),rng.choice([1000,2000,5500]),rng.choice([250,500]),rng.choice([600,800,1200,1600]),rng.choice([0,200,400]),rng.randrange(2),rng.randrange(2),rng.choice([-1,1]),rng.choice([0,0,50,100,600]),rng.choice([0,0,200]),rng.randrange(2),rng.choice([0,0,0,50,100]),rng.choice([0,0,50]),rng.choice([0,0,100]),rng.choice([0,50]),rng.randrange(6)==0,rng.randrange(2),rng.randrange(2),rng.randrange(12)==0,rng.choice([0,50,100,150]),rng.randrange(10)==0,0]
        e=E();e.setX(a[0]);e.setY(a[1]);e.setState(a[2]);e.setDeployCountdown(a[3]);r=E();r.setX(a[4]);r.setY(a[5]);r.setCollisionRadius(a[6]);cfg=TC.forUnit(a[7],5500,a[8],a[9],a[10],True,False).toBuilder().hasProjectile(bool(a[11])).isBuilding(bool(a[12])).build();rv=TV(r,cfg)
        t=TS();t.setOwner(e);t.setConfig(cfg);t.setReference(rv if a[13]>=0 else None)
        for name,v in [('AttackTimerMs',a[14]),('LoadTimerMs',a[15]),('HitInProgress',bool(a[16])),('AttackBlockTimerMs',a[17]),('VisitHoldMs',a[18]),('ResumeDelayMs',a[19]),('ResumeDelayElapsedMs',a[20]),('VisitSuspended',bool(a[21]))]:getattr(t,'set'+name)(v)
        m=MS();m.setPushbackInFlight(int(a[26]));m.setBlockCountdown(a[27]);out=TO();counts=[0,0,0]
        def hit(*args):counts[0]+=1;return False
        def starting():counts[1]+=1
        def turn(*args):counts[2]+=1
        sink=jp.JProxy(J('target.HitSink'),dict={'hit':hit})
        q=jp.JProxy(J('target.TargetingQueries'),dict={'runSelection':lambda:rv if a[23] else None,'validateReference':lambda mode:bool(a[22]) and t.getReference() is not None,'attackTimersHeld':lambda:bool(a[24]),'scaleTimeStep':lambda base:int(base*a[25]/100),'hitSink':lambda:sink,'onStartingAttack':starting,'turnToward':turn,'dashRangeReached':lambda ref:False})
        Visit.targetingVisit(t,e,m,q,out)
        expected=[1 if t.getReference() is not None else -1,t.getAttackTimerMs(),t.getLoadTimerMs(),t.isHitInProgress(),t.getAttackBlockTimerMs(),t.getVisitHoldMs(),t.getResumeDelayMs(),t.getResumeDelayElapsedMs(),e.getState(),out.isResumeRequested(),*counts,t.isHitInProgressWithoutReference(),t.isAttackTimeRoundedUp()]
        check('targeting-visit',native(7,a),expected,a)
    for i in range(1000):
        a=[rng.randint(1000,16000),rng.randint(1000,30000),1,0,256,rng.randint(-256,256),rng.randint(-256,256),rng.randint(0,2000),rng.randint(-200,200),rng.choice([0,0,300,1000]),rng.choice([0,200,500]),rng.randrange(1,10),rng.choice([-100,0,60,120,250,300,500])]
        a += [rng.randrange(2304) for _ in range(a[11])]
        e=E();e.setX(a[0]);e.setY(a[1]);e.setState(a[2]);e.setDirX(a[3]);e.setDirY(a[4]);m=MS()
        m.setRouteDirX(a[5]);m.setRouteDirY(a[6]);m.setMoveTimeMs(a[7]);m.setAvoidanceBlend(a[8]);route=J('grid.Route')()
        for n in a[13:]:route.add(n)
        m.setRoute(route);cfg=MC.forGroundUnit(a[9],a[10]);q=jp.JProxy(J('move.MovementQueries'),inst=MQ(e,a[12]));chain=Chain(m,e,grid,cfg,globals_,None,empty,q).withoutMarkers()
        J('move.MovementVisit').movementVisit(m,e,None,cfg,None,q,False,chain)
        nodes=list(m.getRoute().toArray());expected=[e.getX(),e.getY(),e.getDirX(),e.getDirY(),m.getWaypointReached(),len(nodes),*nodes,m.getRouteDirX(),m.getRouteDirY(),m.getMoveTimeMs(),m.getMoveTimeMsCopy(),m.getAvoidanceBlend()]
        check('movement-visit',native(8,a),expected,a)
    for i in range(1500):
        n=rng.randint(1,12);route=[rng.randrange(2304) for _ in range(rng.randrange(5))]
        a=[n,rng.choice([-200,-100,0,100,200]),-1,len(route),36,*route];entities=[];raw=[]
        for k in range(n):
            values=[rng.randint(7500,10500),rng.randint(8500,11500),rng.choice([0,0,100]),rng.randint(-256,256),rng.randint(-256,256),rng.choice([250,500,750]),rng.choice([1,3,5,20]),rng.choice([0,1,2,4]),rng.randrange(2),rng.randrange(2),rng.randrange(2),rng.randrange(2),rng.choice([-200,0,200]),rng.randrange(8)>0,k,0]
            raw+=values;e=E()
            for name,v in [('X',values[0]),('Y',values[1]),('Z',values[2]),('DirX',values[3]),('DirY',values[4]),('CollisionRadius',values[5]),('Mass',values[6]),('State',values[7]),('MovementActive',bool(values[8])),('MovementComponent',bool(values[9])),('Targetable',values[10]),('Alive',bool(values[13])),('Id',k)]:getattr(e,'set'+name)(v)
            entities.append(e)
        listed=jp.JClass('java.util.ArrayList')()
        for e in entities:listed.add(e)
        e=entities[0];m=MS();m.setAvoidanceBlend(a[1]);r=J('grid.Route')()
        for node in route:r.add(node)
        m.setRoute(r)
        class NQ(MQ):
            def neighbourAcceptsContact(self,other):return J('move.ContactRule').avoidable(other)
            def neighbourSpecialLoadPending(self,other):return raw[other.getId()*16+11]
            def neighbourAvoidanceBlend(self,other):return raw[other.getId()*16+12]
        q=jp.JProxy(J('move.MovementQueries'),inst=NQ(e));chain=Chain(m,e,grid,cfgm,globals_,None,listed,q).withoutMarkers()
        J('move.AvoidanceHandler').avoidance(m,e,listed,q,chain)
        nodes=list(m.getRoute().toArray());check('avoidance',native(9,a+raw),[m.getAvoidanceBlend(),len(nodes),*nodes],a+raw)
        m=MS();chain=Chain(m,e,grid,cfgm,globals_,None,listed,q).withoutMarkers()
        J('move.PushPass').pushPass(m,e,listed,q,chain)
        check('physical-push',native(10,a+raw),[m.getPushX(),m.getPushY(),m.getPushCount()],a+raw)
    for i in range(1500):
        n=rng.randint(1,25);a=[n,rng.randint(-2000,20000),rng.randint(-2000,34000),rng.randint(0,10000),rng.choice([0,0,500,2000]),rng.randrange(2),rng.randrange(2),0,rng.choice([-1,0,1])];entities=[];raw=[]
        index=J('index.SpatialIndex')(36,64)
        for k in range(n):
            v=[rng.randint(-1000,19000),rng.randint(-1000,33000),0,0,0,rng.choice([0,250,500,750,1000]),1,1,rng.randrange(2),rng.randrange(2),1,0,0,1,k,rng.randrange(2)];raw+=v;e=E()
            for name,value in [('X',v[0]),('Y',v[1]),('CollisionRadius',v[5]),('MovementActive',bool(v[8])),('MovementComponent',bool(v[9])),('Id',k),('Side',v[15]),('CrownTower',k%3==0),('Building',not bool(v[9]))]:getattr(e,'set'+name)(value)
            entities.append(e);index.insert(e)
        found=index.query(J('index.SpatialQuery')(a[1],a[2],a[3],a[4],bool(a[5]),bool(a[6]),a[7],a[8]));ids=[v.getId() for v in found];index.release(found)
        check('spatial-query',native(11,a+raw),[len(ids),*ids],a+raw)
        check('spatial-query-into',native(24,a+raw),[len(ids),*ids],a+raw)
    for i in range(1000):
        w=12;h=16;tiles=[rng.choice([0,0,0,1,2,32]) for _ in range(w*h)];overlay=[rng.choice([0,0,0,0,50]) for _ in range(w*h)]
        a=[w,h,rng.randrange(0,w*500),rng.randrange(0,h*500),rng.randrange(0,w*500),rng.randrange(0,h*500),rng.choice([250,500,1000,2000]),rng.randrange(2),0,0,0,0]
        grid=CG(TM(w,h,IA(tiles)),True,50);grid.setCurrent(IA(overlay))
        accepted=jp.JProxy(J('grid.CellPredicate'),dict={'test':lambda c,r:int(0<=c<w and 0<=r<h)})
        d=jp.JProxy(J('grid.TargetDistanceSquared'),dict={'distanceSquared':lambda x,y:FM.squaredDistance(a[4],a[5],x,y)})
        water=jp.JProxy(J('grid.CellPredicate'),dict={'test':lambda c,r:grid.water(c,r)})
        blocked=jp.JProxy(J('grid.CellPredicate'),dict={'test':lambda c,r:J('grid.CellTests').overlayBlocks(grid,c,r)})
        expected=J('grid.ReferenceEndpoint').selectEndpoint(w,h,a[2],a[3],a[4]//500,a[5]//500,a[6],bool(a[7]),True,True,accepted,d,water,blocked,None)
        check('endpoint',native(12,a+tiles+overlay),[expected],a+tiles+overlay)
        a[4]=rng.choice([-1,rng.randrange(0,w*500)]);a[5]=rng.randrange(2)
        tilelookup=jp.JProxy(J('grid.TileLookup'),dict={'bits':lambda c,r:tiles[r*w+c]})
        expected=J('grid.LaneAssignment').lane(w,h,w,a[2],a[3],a[4],a[5],tilelookup)
        check('lane-assignment',native(13,a+tiles+overlay),[expected],a+tiles+overlay)
        a[4]=rng.choice([250,500,750,1000]);a[5]=rng.choice([250,500,750,1000]);a[6]=rng.choice([20,50,100])
        packed=J('grid.FootprintOverlay').rasterize(grid,a[2],a[3],a[4],a[5],a[6])
        check('footprint',native(14,a+tiles+overlay),[int(packed is not None),0 if packed is None else packed,*grid.getCurrent()],a+tiles+overlay)
        a[4]=rng.choice([-1,0,rng.randrange(0,h*500)]);wt=jp.JProxy(J('grid.WaterTest'),dict={'water':lambda c,r:grid.water(c,r)})
        expected=J('grid.Relocation').relocate(w,h,a[2],a[3],a[4],wt)
        check('relocation',native(15,a+tiles+overlay),[expected],a+tiles+overlay)
    for i in range(1500):
        a=[rng.randint(1000,17000),rng.randint(1000,31000),rng.randrange(1000),rng.randrange(2),rng.randrange(0,1000),0,0,rng.randint(1000,17000),rng.randint(1000,31000),rng.randint(0,50000),rng.randrange(2),rng.randrange(2),rng.randrange(2)]
        e=E();e.setX(a[0]);e.setY(a[1]);e.setId(a[2]);m=MS();m.setPushbackInFlight(a[3]);m.setPushbackBudget(a[4]);m.setTargetX(a[5]);m.setTargetY(a[6])
        query=jp.JProxy(J('move.PushbackQueries'),dict={'dashWindupMs':lambda:0})
        J('move.PushbackRequest').set(m,e,query,a[7],a[8],a[9],bool(a[10]),bool(a[11]),bool(a[12]))
        check('pushback-set',native(16,a),[m.getPushbackInFlight(),m.getPushbackBudget(),m.getTargetX(),m.getTargetY(),m.getAttackPushback()],a)
    for state in (0,1,2,4,6,7,11):
        for alive in (False,True):
            for crown in (False,True):
                for hidden in (False,True):
                    for accepted in (False,True):
                        for bypass in (False,True):
                            for mode in (0,1):
                                a=[state,alive,crown,hidden,accepted,bypass,mode]
                                owner=E();owner.setId(5000007);target=E();target.setId(5000008);target.setSide(1);target.setState(state);target.setAlive(alive);target.setTargetable(1)
                                cfg=TC.forUnit(1200,5500,500,1200,700,True,False);t=TS();t.setOwner(owner);t.setConfig(cfg);t.setAliveCheckBypass(bypass);view=TV(target,cfg)
                                # CharacterEntity's final acceptance, including its hidden() query,
                                # is composed into TargetView. Crowns use WorldEntity acceptance.
                                visible=crown or not (hidden or state in (6,11))
                                view.setAcceptsAttacker(accepted and visible)
                                expect=J('target.ReferenceValidator').validate(t,view,mode,J('target.ValidatorQueries').standard1v1())
                                check('reference-final-acceptance',native(17,a),[expect],a)
    for i in range(500):
        w=rng.randint(2,18);h=rng.randint(2,18);costs=[rng.choice([-1,5,5,7,8,50]) for _ in range(w*h)]
        a=[w,h,rng.randrange(-1,w+1),rng.randrange(-1,h+1),rng.randrange(-3,w+3),rng.randrange(-3,h+3),rng.randrange(2)]
        def cost(c,r):return costs[r*w+c] if 0<=c<w and 0<=r<h else -1
        lookup=jp.JProxy(J('grid.CellCostLookup'),dict={'cost':cost})
        nodes=J('grid.RouteSearchWrapper').searchWrapper(w,h,lookup,*a[2:])
        expected=[0,0,0,0,0,0]
        if nodes is not None:
            result=RS.search(w,h,IA(costs),nodes.startNode(),nodes.goalNode(),1,5,False,True,False,0)
            route=list(result.route().toArray());expected=[len(route),*route,*result.counters(),result.remainingBudget()]
        check('grid-search-control',native(20,a+costs),expected,a+costs)
        check('grid-search-into',native(19,a+costs),expected,a+costs)
    for i in range(500):
        nodes=[]
        for _ in range(rng.randrange(30)):
            node=rng.randrange(2304);nodes.extend([node]*rng.randrange(1,5))
        expected=[]
        for node in nodes:
            if not expected or expected[-1]!=node:expected.append(node)
        check('route-dedup-in-place',native(21,[len(nodes),*nodes]),[len(expected),*expected],nodes)
    for i in range(500):
        w=12;h=16;goal=rng.randrange(w*h);old=[];found=[]
        for target in (old,found):
            for _ in range(rng.randrange(1,10)):
                target.extend([rng.randrange(w*h)]*rng.randrange(1,4))
        if i%2:old[0]=goal
        a=[w,h,goal,rng.randrange(2),len(old),len(found),rng.randrange(-1,w*h),rng.choice([0,50]),rng.choice([0,50]),rng.randrange(w*500),rng.randrange(h*500),*old,*found]
        grid=CG(TM(w,h,IA([0]*(w*h))),True,50);grid.setChangeFlags(IA([a[3],0]));current=[0]*(w*h);previous=[0]*(w*h)
        if a[6]>=0:current[a[6]]=a[7];previous[a[6]]=a[8]
        grid.setCurrent(IA(current));grid.setPrevious(IA(previous));e=E();e.setX(a[9]);e.setY(a[10]);m=MS();m.setExplicitX((goal%w)*500+250);m.setExplicitY((goal//w)*500+250);m.setRouteLeadsAway(1)
        route=J('grid.Route')();forced=J('grid.Route')()
        for node in old:route.add(node)
        for node in found:forced.add(node)
        m.setRoute(route)
        class PQ(MQ):
            def search(self,*args):return forced
            def ground(self):return 1
        q=jp.JProxy(J('move.MovementQueries'),inst=PQ(e));globals_=MG.forStandardArena(w)
        chain=Chain(m,e,grid,cfgm,globals_,None,empty,q).withoutMarkers()
        J('move.RoutePreparation').prepareRoute(m,e,grid,globals_,None,q,chain)
        final=list(m.getRoute().toArray());expected=[len(final),*final,m.getRouteDirX(),m.getRouteDirY(),m.getRouteLeadsAway()]
        check('prepare-route-control',native(22,a),expected,a)
        check('prepare-route-scratch',native(23,a),expected,a)
    result={'oracle_commit':PIN,'checks':checks,'total_checks':sum(checks.values()),'status':'passed','compiler_flags':compiler_flags,'timing_comparison':False,'cuda_executed':False,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.hpp')}}
    target=OUT/'differential-result.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':run()
