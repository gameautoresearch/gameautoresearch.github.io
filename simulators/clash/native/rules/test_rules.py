"""Focused pinned-Java differential checks for independent native fixed8 rules.
Run from repository root with .venv-clash/bin/python .../test_rules.py NEW_EVIDENCE_DIR.
The oracle is read only; C++ is independently compiled and never calls Java.
"""
from pathlib import Path
import ctypes, hashlib, json, shutil, subprocess, sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from game import HERE as GAME,make_vector,resolve
from runner import setup
import jpype

def pad(values,n):
    assert len(values)<=n
    return list(values)+[0]*(n-len(values))
def config_values(row,records):
    return np.array([row.startingElixir(),records.globalNumber('MAX_MANA'),records.battleStartCooldownMs('Ladder')//50,records.endScreenDelayMs(),len(row.sectionLengths()),len(row.rateLengths()),len(row.cooldownLengths()),len(row.eventTimes()),*pad([1000*int(x) for x in row.sectionLengths()],8),*pad(list(row.sectionTypes()),8),*pad([1000*int(x) for x in row.rateLengths()],8),*pad(list(row.fullBarMs()),8),*pad([1000*int(x) for x in row.cooldownLengths()],8),*pad(list(row.cooldownMs()),8),*pad(sorted([1000*int(x) for x in row.eventTimes()]),16)],np.int32)
def card_values(c,records):
    u=c.unit()
    return np.array([records.matchCard(c.name()).cost(),records.matchCard(c.name()).elixirProductionStopTimeMs(),-1 if u is None else 1,c.count(),c.summonRadius(),c.summonWidth(),c.summonDeployDelayMs(),0 if u is None else u.deployTimeMs(),0 if u is None else u.collisionRadius(),0 if u is None else u.spawnRadius(),0 if u is None else u.spawnAngleShift(),0 if u is None else u.flyingHeight(),0 if u is None else u.spawnPathfindSpeed(),0 if u is None else u.tileSizeOverride(),c.deployWTileMargin(),c.deployStartY(),c.deployEndY(),c.canDeployOnEnemySide(),c.canPlaceOnBuildings(),c.canPlaceOnWater(),c.fullLaneDeploy(),False if u is None else u.building(),c.spellAsDeploy()],np.int32)
def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    compiler=shutil.which('clang++') or shutil.which('g++')
    libpath=out/('rules.dylib' if sys.platform=='darwin' else 'rules.so')
    compiler_command=[compiler,'-std=c++20','-O2','-fPIC','-fno-fast-math','-ffp-contract=off','-dynamiclib' if sys.platform=='darwin' else '-shared',str(HERE/'test_bridge.cpp'),'-o',str(libpath)]
    subprocess.run(compiler_command,check=True)
    lib=ctypes.CDLL(str(libpath));ptr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');byte=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    lib.rules_timeline.argtypes=[ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ptr]
    lib.rules_placement.argtypes=[ptr,ptr,ctypes.c_int,ctypes.c_int,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,byte,byte]
    lib.rules_side.argtypes=[ptr,ptr,ptr,ctypes.c_int,ptr]
    lib.rules_match.argtypes=[ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ptr]
    lib.rules_giving.argtypes=[ptr,ptr,ptr,ptr,ctypes.c_int,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr]
    lib.rules_command_run.argtypes=[ptr,ptr,ptr,ptr,ctypes.c_int,ptr,ctypes.c_int,ctypes.c_int,ptr,ptr]
    lib.rules_legal.argtypes=[ptr,ptr,ptr,ptr,ctypes.c_int,ptr,ptr,ctypes.c_int,ctypes.c_int,byte]
    c=json.loads((GAME/'configs/profile-trimmed.json').read_text());c.update(envs=1,threads=1);setup(c)
    J=jpype.JClass;rng=np.random.default_rng(8178)
    counts={'timeline_rows':0,'placement_cases':0,'side_rows':0}
    with make_vector(c):
        Array=J('java.util.ArrayList')
        records=J('org.crforge.core.battle.data.BattleRecords')(J('org.crforge.env.VectorEnv').tables(J('java.nio.file.Path').of(str(resolve(c['tables'])))))
        row=records.gameModeTimeline('Ladder');cv=config_values(row,records)
        Timeline=J('org.crforge.core.battle.match.Timeline')
        for ticks,eq,freeze in [(np.arange(0,7500,dtype=np.int32),np.ones(7500,np.int32),-1),(np.arange(0,7500,dtype=np.int32),np.zeros(7500,np.int32),-1),(np.array([0,0,1,2399,2400,3599,3600,4800,6000,7000],np.int32),np.ones(10,np.int32),4)]:
            native=np.zeros((len(ticks),10),np.int32);lib.rules_timeline(cv,ticks,eq,len(ticks),freeze,native);t=Timeline(row)
            for i,tick in enumerate(ticks):
                t.setCrownsEqual(bool(eq[i]));t.advance(int(tick))
                if i==freeze:t.freeze()
                expected=[t.getTimeMs(),t.getSection(),t.getRate(),t.getFullBarMs(),t.getNextCardCooldownMs(),t.sectionLeftMs(),t.overtime(),t.timeUp(),t.isFirstAdvance(),t.isFrozen()]
                np.testing.assert_array_equal(native[i],expected,err_msg=f'timeline tick {tick}');counts['timeline_rows']+=1
        TileMap=J('org.crforge.core.pathfinding.grid.TileMap');arena=TileMap.standard1v1();w=arena.width();h=arena.height();cells=np.array([arena.bits(x,y) for y in range(h) for x in range(w)],np.int32)
        Mask=J('org.crforge.core.battle.deploy.MaskEntity');Search=J('org.crforge.core.battle.deploy.PlacementSearch');Place=J('org.crforge.core.battle.deploy.CardPlacement')
        cards=[records.card(name) for name in c['rules']['deck']]
        sources={}
        for card in cards:sources[str(card.name())]=card_values(card,records).tolist()
        for case in range(500):
            card=cards[case%8];cp=card_values(card,records);side=case%2;entities=Array();rows=[]
            # Include standard king and princess boxes, and randomized surviving tower pockets.
            for team in range(2):
                for x,y,king in [(9000,2500,True),(3500,6500,False),(14500,6500,False)]:
                    if team:y=32000-y
                    alive=bool(rng.random()>.18);e=[alive,team,x,y,not king,king,not king,18 if king else 9,32 if king else 24,4 if king else 3]
                    rows.append(e);entities.add(Mask(*e))
            ep=np.array(rows,np.int32);x=int(rng.integers(-2000,20001));y=int(rng.integers(-2000,34001));sym=bool(case%3);seq=bool(case%5)
            native=np.zeros(24,np.int32);mask=np.empty(w*h//4,np.uint8);lands=np.empty_like(mask)
            lib.rules_placement(cp,cells,w,h,ep,len(rows),side,x,y,sym,seq,native,mask,lands)
            jm=Search.mask(arena,card,side,entities);np.testing.assert_array_equal(mask,list(jm),err_msg=f'mask {case}')
            np.testing.assert_array_equal(lands,list(Search.landingTiles(arena,card,jm)),err_msg=f'lands {case}')
            p=Place.place(arena,card,x,y,side,entities,sym,seq)
            assert native[0]==p.code(),(case,card.name(),native,p)
            if p.placed():
                assert tuple(native[1:3])==(p.x(),p.y()),(case,card.name(),native,p)
                if not card.spell():
                    interval=p.interval();assert bool(native[3])==(interval is not None)
                    if interval is not None:np.testing.assert_array_equal(native[4:6],list(interval))
                    assert native[6]==p.originLane()
                assert native[7]==len(p.units())
                for k,u in enumerate(p.units()):np.testing.assert_array_equal(native[8+8*k:16+8*k],[u.index(),u.dx(),u.dy(),u.x(),u.y(),u.lane(),u.start().state(),u.start().waitMs()],err_msg=f'unit {case} {card.name()}')
            counts['placement_cases']+=1
        # LadderMatch's full tower-only timeline/end/tiebreak progression, with
        # adversarial tower HP changes at known tick boundaries on both hosts.
        counts['match_rows']=0
        Standard=J('org.crforge.core.battle.unit.Standard1v1Battle');Tower=J('org.crforge.core.battle.unit.TowerEntity')
        deckj=Array()
        for name in c['rules']['deck']:deckj.add(name)
        tables=J('org.crforge.env.VectorEnv').tables(J('java.nio.file.Path').of(str(resolve(c['tables']))))
        scenarios=[([],6500),([[2,3,0]],100),([[2,0,0],[2,3,0]],180),([[5900,1,1000],[5900,4,800]],6400),([[3599,4,0]],3700)]
        for changes,steps in scenarios:
            b=Standard(tables,11,False);m=b.startLadderMatch(deckj,deckj,0,0);world=b.getWorld();battle=b.getBattle()
            towers=[e for e in list(world.getHolder().entities()) if isinstance(e,Tower)]
            initial=np.array([[e.getHitPoints().getHitPoints(),e.getView().getX(),e.getView().getY()] for e in towers],np.int32)
            events=np.array(changes,np.int32).reshape(-1,3);native=np.zeros((steps,23),np.int32)
            lib.rules_match(cv,initial,events,len(changes),steps,native)
            for i in range(steps):
                for tick,idx,hp in changes:
                    if tick==i:towers[idx].getHitPoints().setHitPoints(hp)
                battle.step();t=m.getTimeline()
                expected=[battle.getTick(),battle.getClockMs(),t.getTimeMs(),t.getSection(),t.getRate(),t.sectionLeftMs(),t.isFrozen(),m.getEndTimerMs(),m.getWinner(),m.getTiebreakMs(),m.isEnded(),m.isLastTicked(),m.isOver(),m.crowns(0),m.crowns(1),world.isHitsHeld(),world.isMatchEnded(),*[e.getHitPoints().getHitPoints() for e in towers]]
                np.testing.assert_array_equal(native[i],expected,err_msg=f'match {changes} step {i}')
                counts['match_rows']+=1
        counts['command_execution_rows']=0
        packed=np.array([card_values(card,records) for card in cards],np.int32)
        for seed in [1,27,819]:
            b=Standard(tables,11,False);b.getWorld().seed(seed);b.addPlayerData(1);b.addPlayerData(1)
            m=b.startLadderMatch(deckj,deckj,0,0);commands=J('org.crforge.core.battle.unit.PlayerCommands')(b)
            towers=[e for e in list(b.getWorld().getHolder().entities()) if isinstance(e,Tower)]
            ep=np.array([[True,e.side(),e.getView().getX(),e.getView().getY(),e.getData().summonerTower(),e.getData().king(),e.getData().building(),e.getData().noDeploySizeW(),e.getData().noDeploySizeH(),Search.footprint(e.getData())] for e in towers],np.int32)
            requests=np.array([[t,t%2,(t//3)%4,3500,5000 if t%2==0 else 27000] for t in [0,89,90,91,92,93,95,100,110,111,112,120,135,150]],np.int32)
            steps=200;gives=np.zeros((len(requests),7),np.int32);states=np.zeros((steps,26),np.int32)
            lib.rules_command_run(cv,packed,cells,ep,seed,requests,len(requests),steps,gives,states)
            for i in range(steps):
                for j,(tick,side,slot,x,y) in enumerate(requests):
                    if tick!=i:continue
                    g=commands.playCard(int(side),int(slot),11,int(x),int(y),f'exec-{j}')
                    np.testing.assert_array_equal(gives[j],[g.refusal().ordinal(),g.code(),g.givenTick(),g.runTick(),g.cost(),g.x(),g.y()],err_msg=f'execution giving {seed} {i}')
                b.getBattle().step()
                expected=[b.getBattle().getTick(),commands.promised(0),commands.promised(1),sum(int(p.matchCode()==0 and p.result() is not None and p.result().placed()) for p in b.getPlays())]
                for side in range(2):
                    ss=m.side(side);hand=ss.getHand();q=list(hand.queue());expected.extend([ss.getElixir(),ss.getSpent(),ss.getDeployCounter(),hand.getCooldownMs(),len(q),*list(hand.slots()),q[0],q[1]])
                np.testing.assert_array_equal(states[i],expected,err_msg=f'command execution seed {seed} tick {i}')
                counts['command_execution_rows']+=1
        # Native command giving against real evolving oracle matches, including
        # queued promise accounting, duplicate-card refusals and screen snapping.
        counts['command_cases']=0;counts['action_mask_cases']=0
        packed=np.array([card_values(card,records) for card in cards],np.int32)
        for seed in [1,27,819]:
            b=Standard(tables,11,False);b.getWorld().seed(seed);b.addPlayerData(1);b.addPlayerData(1)
            m=b.startLadderMatch(deckj,deckj,0,0);commands=J('org.crforge.core.battle.unit.PlayerCommands')(b);given=[]
            for i in range(400):
                if i%3==0:
                    side=i%2;slot=int(rng.integers(0,4));x=int(rng.integers(-1000,19000));y=int(rng.integers(-1000,33000));t=m.getTimeline()
                    hp=[e for e in list(b.getWorld().getHolder().entities()) if isinstance(e,Tower)]
                    ep=np.array([[True,e.side(),e.getView().getX(),e.getView().getY(),e.getData().summonerTower(),e.getData().king(),e.getData().building(),e.getData().noDeploySizeW(),e.getData().noDeploySizeH(),Search.footprint(e.getData())] for e in hp],np.int32)
                    section_elapsed=int(row.sectionLengths()[t.getSection()])*1000-t.sectionLeftMs()
                    snap=np.array([b.getBattle().getTick(),m.getEndTimerMs(),t.getSection(),section_elapsed,m.crowns(0)==m.crowns(1),b.getWorld().kingTower(0).getHitPoints().alive(),b.getWorld().kingTower(1).getHitPoints().alive(),2-m.crowns(1),2-m.crowns(0),m.side(0).getElixir(),m.side(1).getElixir(),*list(m.side(0).getHand().slots()),*list(m.side(1).getHand().slots())],np.int32)
                    plays={str(p.name()):p for p in b.getPlays()}
                    q=[]
                    for command in given:
                        entry=command.copy();p=plays.get(entry.pop('name'))
                        if p is not None:entry['code']=int(p.matchCode());entry['ran']=True;entry['placed']=bool(p.matchCode()==0 and p.result() is not None and p.result().placed())
                        if entry['ran'] and entry['placed'] and entry['run']<i:continue
                        if entry['ran'] and not entry['placed'] and entry['expire']+2<i:continue
                        q.append(list(entry.values()))
                    q=np.array(q,np.int32).reshape(-1,11);native=np.zeros(9,np.int32)
                    if i%15==0:
                        expected=np.zeros(2307,np.uint8);expected[0]=1;actual=np.zeros_like(expected)
                        lib.rules_legal(cv,packed,cells,ep,len(ep),snap,q,len(q),side,actual)
                        entitiesj=Array()
                        for er in ep:entitiesj.add(Mask(bool(er[0]),*[int(v) for v in er[1:4]],*[bool(v) for v in er[4:7]],*[int(v) for v in er[7:]]))
                        for sl in range(4):
                            if not commands.tryCard(side,sl,11).given():continue
                            idx=int(m.side(side).getHand().slots()[sl]);card=cards[idx];lands=Search.landingTiles(arena,card,Search.mask(arena,card,side,entitiesj))
                            for tile in range(576):
                                gx=J('org.crforge.env.Schema').gameX(side,tile%18)//1000;gy=J('org.crforge.env.Schema').gameY(side,tile//18)//1000
                                expected[1+sl*576+tile]=lands[gy*18+gx]
                        np.testing.assert_array_equal(actual,expected,err_msg=f'action mask seed {seed} tick {i}')
                        counts['action_mask_cases']+=1
                    lib.rules_giving(cv,packed,cells,ep,len(ep),snap,q,len(q),side,slot,x,y,native)
                    name=f'cmd-{i}';deck_idx=int(m.side(side).getHand().slots()[slot]);g=commands.playCard(side,slot,11,x,y,name)
                    expected=[g.refusal().ordinal(),g.code(),g.givenTick(),g.runTick(),g.cost(),g.x(),g.y(),commands.promised(0),commands.promised(1)]
                    np.testing.assert_array_equal(native,expected,err_msg=f'giving seed {seed} tick {i}')
                    if g.given():given.append(dict(side=side,card=deck_idx,given=g.givenTick(),run=g.runTick(),expire=i+60,cost=g.cost(),x=g.x(),y=g.y(),code=0,ran=False,placed=False,name=name))
                    counts['command_cases']+=1
                b.getBattle().step()
        # MatchSide constructor and visit/play have package visibility: use reflection
        # only to obtain oracle outputs; native never links or invokes the JVM.
        Side=J('org.crforge.core.battle.match.MatchSide');MatchCard=J('org.crforge.core.battle.match.MatchCard');List=J('java.util.List');Integer=J('java.lang.Integer').TYPE
        ctor=Side.class_.getDeclaredConstructor(List.class_,Integer);ctor.setAccessible(True)
        visit=Side.class_.getDeclaredMethod('visit',Timeline.class_,Integer);visit.setAccessible(True)
        play=Side.class_.getDeclaredMethod('play',Integer);play.setAccessible(True)
        deck=Array()
        for name in c['rules']['deck']:deck.add(records.matchCard(name))
        js=ctor.newInstance(deck,jpype.JInt(row.startingElixir()));order=np.arange(8,dtype=np.int32)
        deal=J('org.crforge.core.battle.match.Hand').class_.getDeclaredMethod('deal',List.class_);deal.setAccessible(True);orderj=Array()
        for k in range(8):orderj.add(jpype.JInt(k))
        deal.invoke(js.getHand(),orderj)
        n=7000;plays=np.zeros((n,3),np.int32);plays[:,0]=-1
        # Track valid cycling schedule with native-independent hand queue model.
        slots=[0,1,2,3];queue=[4,5,6,7];cool=0;timeline=Timeline(row)
        expected=[]
        for i in range(n):
            timeline.advance(i)
            if i%91==0:
                k=next((j for j,v in enumerate(slots) if v>=0),None)
                if k is not None:
                    card=slots[k];plays[i]=[card,records.matchCard(c['rules']['deck'][card]).cost(),0];play.invoke(js,jpype.JInt(card))
            filled=int(visit.invoke(js,timeline,jpype.JInt(records.globalNumber('MAX_MANA'))));hand=js.getHand();q=list(hand.queue());s=list(hand.slots())
            slots=s
            expected.append([filled,js.getElixir(),js.getSpent(),js.getWasted(),js.getProductionStopMs(),js.getDeployCounter(),-1 if js.lastPlayed() is None else c['rules']['deck'].index(str(js.lastPlayed().name())),hand.getCooldownMs(),len(q),*s,*pad(q,5)])
        native=np.zeros((n,18),np.int32);lib.rules_side(cv,order,plays,n,native)
        np.testing.assert_array_equal(native,np.array(expected,np.int32));counts['side_rows']=n
    result={'status':'passed',**counts,'complete_task_parity':False,'oracle_commit':'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de','scope':'timeline, MatchSide elixir/hand, full tower-only LadderMatch/end/tiebreak, commands/20-tick execution/promises, deploy/landing masks and full fixed8 placement against pinned Java','card_fields':sources,'config_fields':cv.tolist(),'command':compiler_command,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*') if p.is_file()}}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('card_fields','config_fields','source_sha256')}))
if __name__=='__main__':main()
