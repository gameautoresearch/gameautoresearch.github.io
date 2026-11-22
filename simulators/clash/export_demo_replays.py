"""Export fresh public-demo Clash films from native CPU, then independently replay.

No JVM, neural checkpoint, paid worker, private evaluation seed, commercial art,
raw world image or engine RNG state is included in the browser recordings.
"""
from pathlib import Path
import argparse,ctypes,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'native'),str(HERE)]
from runtime import NativeCPU
from evaluate import DATA,StaticSchema,PublicActor,no_jvm,public_outputs,public_digest,sha
from data_codegen import scaling

NAMES={'KingTower':'King tower','PrincessTower':'Princess tower','TowerPrincess':'Princess tower',
       'MiniPekka':'Mini P.E.K.K.A','Archer':'Archers','ArcherArrow':'Arrow','ArrowsSpell':'Arrows',
       'FireballSpell':'Fireball','KingProjectile':'King shot','MusketeerProjectile':'Musket shot',
       'TowerPrincessProjectile':'Tower arrow'}

def write(path,value):
    Path(path).write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class Recorder:
    def __init__(self,data):
        self.data=data;self.s=StaticSchema(data);self.unit_names=list(data['units']);self.shots=list(data['projectiles'])
        self.unit_hp=[scaling(u['hitpoints'],u,4 if u['king'] else 6 if u['summonerTower'] else 2) for u in data['units'].values()]
    def name(self,row):
        return NAMES.get(row,row)
    def view(self,o,seat):
        s=self.s;e=s.entity;result=[]
        for r in s.entities(o):
            k=int(r[e['kind']]);side=int(r[e['side']]);x=int(r[e['x']]);y=int(r[e['y']]);name=self.data['vocabulary'][int(r[e['row']])] if int(r[e['row']])>=0 else 'Unknown'
            result.append({'id':f'v{seat}-{int(r[e["id"]])}','kind':{1:'troop',2:'tower',3:'projectile',4:'area'}[k],
                'side':seat if side==0 else 1-seat if side==1 else 2,'name':self.name(name),
                'x':18000-x if seat==0 else x,'y':y if seat==0 else 32000-y,
                'hp':int(r[e['hp']]),'maxHp':int(r[e['max_hp']]),'state':int(r[e['state']]),
                'radius':int(r[e['radius']]),'deployMs':int(r[e['deploy_ms']]),
                'stunned':any(int(r[e[f'buff_{i}']])==self.data['buff_names'].index('ZapFreeze') for i in range(3))})
        return result
    def entities(self,env,views):
        lib=env.lib;lib.cn_world_debug.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int),ctypes.c_int];lib.cn_world_debug.restype=ctypes.c_int
        raw=np.empty(100000,np.int32);n=lib.cn_world_debug(env.handles[0],raw.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),len(raw))
        if n>len(raw):raise ValueError('debug output overflow')
        at=10;result=[]
        for _ in range(int(raw[4])):
            identifier,kind=map(int,raw[at:at+2]);at+=2
            if kind==5:
                q=list(map(int,raw[at:at+18]));at+=18;row,side,x,y,state,deploy,delay,hp=q[:8];name=self.unit_names[row]
                result.append({'id':f'w{identifier}','kind':'tower' if self.data['units'][name]['king'] or self.data['units'][name]['summonerTower'] else 'troop',
                    'side':side,'name':self.name(name),'x':x,'y':y,'hp':hp,'maxHp':self.unit_hp[row],
                    'state':state,'radius':self.data['units'][name]['collisionRadius'],'deployMs':delay if state==11 else deploy,
                    'stunned':any(v['kind']=='troop' and v['side']==side and v['x']==x and v['y']==y and v['name']==self.name(name) and v['stunned'] for v in views[side])})
            elif kind==4:
                q=list(map(int,raw[at:at+9]));at+=9;row,side,x,y,z,target,ax,ay,delay=q
                result.append({'id':f'w{identifier}','kind':'projectile','side':side,'name':self.name(self.shots[row]),'x':x,'y':y,'hp':0,'maxHp':0,'state':-1,'radius':0,'deployMs':delay,'stunned':False})
            else:at+=1
        if at!=n:raise ValueError('native debug ABI differs')
        result.extend(dict(x,id='a-'+x['id']) for x in views[0] if x['kind']=='area')
        return result
    def frame(self,env,commands):
        s=self.s;obs=env.observations[0];views=[self.view(obs[p],p) for p in (0,1)];info=env.infos[0]
        players=[]
        for p,o in enumerate(obs):
            cards=[]
            for slot in range(4):
                card=int(o[s.header['hand_card'][slot]])
                cards.append({'name':self.name(self.data['vocabulary'][card]) if card>=0 else 'Empty','cost':int(o[s.header['hand_cost'][slot]]),'pending':bool(o[s.header['hand_pending'][slot]])})
            players.append({'elixir':int(s.field(o,'elixir')),'reserved':int(s.field(o,'elixir_reserved')),
                'enemyElixir':int(s.field(o,'enemy_elixir')),'hand':cards,
                'next':self.name(self.data['vocabulary'][int(s.field(o,'next_card'))]) if int(s.field(o,'next_card'))>=0 else 'Unknown'})
        winner=int(info[s.info['winner']])
        return {'tick':int(info[s.info['tick']]),'elapsedMs':int(info[s.info['tick']])*50,
                'sectionLeftMs':int(s.field(obs[0],'section_left_ms')),'overtime':bool(s.field(obs[0],'overtime')),
                'crowns':[int(info[s.info[f'crowns_{p}']]) for p in (0,1)],'ended':winner>=0,
                'winner':winner if winner in (0,1) else None,'players':players,'entities':self.entities(env,views),'views':views,'commands':commands}
    def command(self,action,obs,side,tick):
        if action==0:return None
        if not 1<=action<=2304:raise ValueError('plain fixed8 demo cannot use champion abilities')
        n=action-1;slot=n//576;tile=n%576;col=tile%18;row=tile//18;name=self.data['vocabulary'][int(obs[self.s.header['hand_card'][slot]])]
        card=self.data['cards'][name];x=col*1000+500;y=row*1000+500
        return {'side':side,'action':action,'card':self.name(name),'kind':'spell' if card['unit'] is None else 'troop',
                'x':18000-x if side==0 else x,'y':y if side==0 else 32000-y,'radius':card['radius'],
                'tick':tick,'label':'Queued '+self.name(name)}

def generate(out):
    no_jvm();out=Path(out)
    if out.exists():raise FileExistsError('Use a new directory; replay evidence is immutable')
    out.mkdir(parents=True);data=json.loads(DATA.read_text());rec=Recorder(data);entries=[];receipts=[]
    specs=[('clash-scripted-duel','Scripted duel',('scripted','scripted'),2026110811),
           ('clash-scripted-random','Scripted vs random',('scripted','random'),2026110812),
           ('clash-random-scripted','Random vs scripted',('random','scripted'),2026110813)]
    for identifier,title,kinds,seed in specs:
        env=NativeCPU([seed]);actors=[PublicActor(kind,seed+p+100,data=data) for p,kind in enumerate(kinds)]
        frames=[];hashes=[];actions=[]
        try:
            frames.append(rec.frame(env,[]));hashes.append(public_digest(public_outputs(env)))
            for _ in range(3000):
                obs=env.observations[0].copy();mask=env.masks[0].copy();tick=int(env.infos[0,0]);a=[actors[p](obs[p],mask[p]) for p in (0,1)]
                if any(not mask[p,a[p]] for p in (0,1)):raise AssertionError('demo policy selected an illegal action')
                commands=[c for p in (0,1) if (c:=rec.command(a[p],obs[p],p,tick)) is not None]
                env.step([a]);actions.append(a);f=rec.frame(env,commands);frames.append(f);hashes.append(public_digest(public_outputs(env)))
                info=env.infos[0]
                if any(int(info[rec.s.info[k]]) for k in ('fault','not_modelled','illegal_0','illegal_1','truncated')):raise AssertionError('invalid demo episode')
                if f['ended']:break
            else:raise AssertionError('demo did not finish naturally; no truncated sample published')
            final=public_digest(public_outputs(env))
        finally:env.close()
        # Fresh world, no policy calls: independently check EVERY complete public
        # output and visual frame against the exact recorded action sequence.
        env=NativeCPU([seed])
        try:
            assert public_digest(public_outputs(env))==hashes[0] and rec.frame(env,[])==frames[0]
            for i,a in enumerate(actions,1):
                if any(not env.masks[0,p,a[p]] for p in (0,1)):raise AssertionError('replay closed action')
                env.step([a]);assert public_digest(public_outputs(env))==hashes[i],f'public replay mismatch {i}'
                assert rec.frame(env,frames[i]['commands'])==frames[i],f'visual replay mismatch {i}'
            assert public_digest(public_outputs(env))==final
        finally:env.close()
        no_jvm();last=frames[-1]
        movie={'schemaVersion':1,'gameId':'clash','id':identifier,'title':title,'subtitle':'Native C++ · fixed eight cards',
            'players':[{'name':('Scripted bot' if k=='scripted' else 'Random legal bot')+f' · {"Blue" if p==0 else "Red"}','kind':k} for p,k in enumerate(kinds)],
            'arena':{'width':18000,'height':32000,'decisionTicks':5,'tickMs':50,'deck':data['rules']['deck'],'towers':[e for e in frames[0]['entities'] if e['kind']=='tower']},
            'provenance':{'kind':'fresh-public-demo','engineCommit':data['engine_commit'] if 'engine_commit' in data else 'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de','ruleset':data['task'],'dataVersion':'16.635.8',
                'demoSeed':seed,'verification':'Every complete public output and visual frame independently replayed on native CPU; all selected actions legal; natural finish; no JVM',
                'verifiedFrames':len(frames),'trajectorySha256':digest(actions),'finalPublicSha256':final,'policy':'Pinned public-observation ScriptedBot / uniform random legal policy; no trained-policy strength claim'},
            'frames':frames,'outcome':{'kind':'natural','winner':last['winner'],'draw':last['winner'] is None,'tick':last['tick'],'decisions':len(actions),'crowns':last['crowns']}}
        write(out/(identifier+'.json'),movie)
        entries.append({'gameId':'clash','id':identifier,'title':title,'subtitle':f'{last["elapsedMs"]/1000:.1f}s · '+('Draw' if last['winner'] is None else ('Blue' if last['winner']==0 else 'Red')+' wins'),'moves':len(actions),'url':f'replays/clash/{identifier}.json'})
        receipts.append({'id':identifier,'frames':len(frames),'actions':len(actions),'sha256':sha(out/(identifier+'.json')),'finalPublicSha256':final,'outcome':movie['outcome']})
        print(json.dumps(receipts[-1]),flush=True)
    write(out/'index.json',entries);write(out/'verification.json',{'status':'passed','jvmStarted':False,'rulesLockSha256':sha(HERE/'rules.lock.json'),'exporterSha256':sha(__file__),'recordings':receipts})
    return entries

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();generate(a.output)
