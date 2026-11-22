"""Host-owned Clash session authority: allocation ledger, job queue, source-bound bundles, practice, submissions.

The session's budget mode (harness/budget.py) sets how many jobs may be active: one at a time in seconds mode, up to
the recorded concurrency limits in dollar mode. A training run is the fast-v4 launch: the editable source
is snapshotted, one finite L40S worker trains it, and the returned artifact is
accepted only if its source matches the snapshot and it shows a real learning
update; it is then sealed as an immutable bundle. Public practice plays a sealed
bundle on the release's public schedule in a CPU-only worker. The provider calls
block, so each job runs on its own thread; `tick` starts every queued job. Jobs share no mutable state: each has its
own source snapshot, job folder, ledger handle, worker and Volume.
"""
from pathlib import Path
import ast,fcntl,json,math,re,shutil,threading,time

from harness import budget as budgets, workers
from harness.backend import ACTIVE
from . import TRAIN_SHAPE
from simulators.clash.autoresearch.bundle import seal,verify
from simulators.clash.autoresearch.common import EDITABLE,MATCH_FORM,digest,dump,sha
from simulators.clash.autoresearch.short_release import check_panel, verify as verify_release
from simulators.clash.autoresearch.ledger import Ledger
from simulators.clash.autoresearch.readiness_contract import review_policy
from simulators.clash.autoresearch.release_v3 import resolve_schedule

NAME=re.compile(r'[a-z0-9][a-z0-9_-]{0,39}')
SOURCE=(*EDITABLE,'config.json')
HARDWARE={'cpu_threads':4,'ram_gib':16}  # fast-v4 contract: one L40S with 4 CPU cores and 16 GiB
PARAMETER_CAP=500_000  # a researcher's sealed policy (SPEC['parameter_cap']); the evaluator's own bound is wider
MAX_WORKER_SECONDS=3300  # contract limit for one launch's hard worker time
PRACTICE_SECONDS=600  # CPU-only practice worker timeout; not part of the GPU allocation ledger
GRADE_SECONDS=3600  # CPU-only development grade on the protected schedule; outside the allocation
UNRESOLVED=('reserved','running','uncertain')
PRACTICE_SHAPE={'gpu':0,'cpu_cores':8,'memory_mib':32768}  # the CPU-only practice worker (FastProvider.cpu_grade)
SHAPES={'train':TRAIN_SHAPE,'practice':PRACTICE_SHAPE}
RUN_SHAPE='run (L40S, 4 CPU, 16 GiB)'
RECOVERY_SECONDS=15  # how often an unknown worker end is asked for again
PRACTICE_SUMMARY=('games','raw_score_0_100','interval_95_percent','by_opponent','by_seat','natural_games','truncations',
                  'illegal_actions','simulator_faults','split','seconds')


def canonical_schedule(value):
    """The schedule as the practice worker sees it: panel bundles under /private, never host paths."""
    value=json.loads(json.dumps(value))
    for actor in value['opponents']:
        if actor['kind']=='checkpoint':actor['bundle']='/private/'+actor['name']
    return value


def preflight(files):
    """Refusals the trusted trainer, the bundle seal or the evaluator would make later, made before anything is spent.

    `files` maps policy.py, learner.py, train.py and config.json to paths.
    """
    if sum(Path(files[n]).stat().st_size for n in EDITABLE)>256000 or Path(files['config.json']).stat().st_size>65536:
        raise ValueError('The three training files may total at most 256,000 bytes and the config at most 65,536 bytes')
    c=json.loads(Path(files['config.json']).read_text())
    if not isinstance(c,dict):raise ValueError('The config must be a JSON object')
    if c.get('match_form')!=MATCH_FORM:
        raise ValueError(f'This benchmark plays the {MATCH_FORM} match form: keep "match_form": "{MATCH_FORM}" in the config')
    hardware=c.get('hardware_allocation')
    if not isinstance(hardware,dict) or any(hardware.get(k)!=v for k,v in HARDWARE.items()):
        raise ValueError('This session trains on one L40S with 4 CPU cores and 16 GiB: keep hardware_allocation.cpu_threads=4 and ram_gib=16')
    try:
        if c.get('max_ticks',0)!=0 or c.get('rules',{}).get('max_ticks',0)!=0:raise ValueError('Training uses the natural match horizon: keep max_ticks at 0')
        if not 1<=c['envs']<=2048 or not 1<=c['rollout_steps']<=256:raise ValueError('envs must be 1..2048 and rollout_steps 1..256')
        if c.get('training_route')=='native-fast-v1' and (c.get('warmstart_steps',0) or c.get('reward_shaping',0) or c.get('opponent_mode')!='selfplay'):
            raise ValueError('The native fast trainer requires warmstart_steps=0, reward_shaping=0 and opponent_mode="selfplay"')
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('The config lacks a required numeric field: '+str(exc)) from None
    # The evaluator admits only a direct greedy entry and pure-NN imports; an edited policy fails here, not after training.
    return {'hardware':dict(HARDWARE),'policy_review':review_policy(Path(files['policy.py']))}


class Account(Ledger):
    """The session's allocation ledger in its budget mode, which the ledger records and which never changes.

    seconds: the native ledger: one unresolved allocation at a time within `limit_seconds` training-worker seconds.
    usd: up to the recorded limits of unresolved allocations at once within `limit_usd`. Every allocation is a
    training worker, so the ledger keeps worker seconds (reserved, then charged from the provider receipt) and its
    dollars are those seconds at the recorded `usd_per_second`. A reservation holds its whole grant until it
    settles, so grants never add up past the budget. Milestones fall at the same fractions of the budget.
    """
    def __init__(self,path,seconds=None,usd=None):
        """`seconds` or `usd` creates (or checks) the ledger in that mode; neither reopens an existing one as it is."""
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=self.path.with_suffix('.lock').open('a+')
        asked=None if seconds is None and usd is None else budgets.USD if usd is not None else budgets.SECONDS
        fcntl.flock(self.lock,fcntl.LOCK_EX)
        try:
            if asked is None and not self.path.is_file():raise ValueError('No session ledger to reopen')
            if not self.path.exists():
                state={'allocations':[],'submissions':[],'milestones':{},'finalized':False,'budget_mode':asked}
                if asked==budgets.USD:
                    state.update(limit_usd=usd,usd_per_second=budgets.rate(TRAIN_SHAPE),prices=dict(budgets.PRICE),
                                 training_shape=dict(TRAIN_SHAPE),limits=budgets.limits(budgets.USD))
                else:state['limit_seconds']=seconds
                dump(self.path,state)
            state=self.read();self.mode=state.get('budget_mode',budgets.SECONDS)  # a ledger from before the modes
            if asked is not None and asked!=self.mode:
                raise ValueError(f'This session is budgeted in {self.mode} mode; it cannot be reopened in {asked} mode')
            if asked==budgets.SECONDS and state['limit_seconds']!=seconds or asked==budgets.USD and state['limit_usd']!=usd:
                raise ValueError('existing allocation ceiling cannot be reset')
        except BaseException:
            fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close();raise
        fcntl.flock(self.lock,fcntl.LOCK_UN)
    def limits(self,state=None):
        return dict((state or self.read()).get('limits') or budgets.limits(budgets.SECONDS))
    def reserve(self,name,seconds):
        if self.mode==budgets.SECONDS:return super().reserve(name,seconds)
        def go(s):
            # Every allocation is one training worker with one GPU.
            slots=min(s['limits']['max_concurrent_jobs'],s['limits']['max_concurrent_gpus'])
            if s['finalized'] or sum(r['status'] in UNRESOLVED for r in s['allocations'])>=slots:raise ValueError(f'closed session or {slots} unresolved active allocations')
            if any(r['name']==name for r in s['allocations']):raise ValueError('run name already reserved; historical attempts remain immutable')
            if not math.isfinite(seconds) or seconds<=0:raise ValueError('positive finite reservation required')
            rate=s['usd_per_second'];remaining=s['limit_usd']-self.used(s)*rate
            if seconds*rate>remaining+1e-9:raise ValueError(f'allocation needs ${seconds*rate:.4f} ({seconds:.2f} worker seconds including setup/cleanup reserve), but only ${remaining:.4f} remain; no GPU was launched')
            s['allocations'].append({'name':name,'reserved_seconds':seconds,'status':'reserved','at':time.time()})
        return self.mutate(go)
    @staticmethod
    def _tick_milestones(state,allocated_seconds):
        if state.get('budget_mode')==budgets.USD:  # the same fractions of the dollar budget; milestones stay shared
            state=dict(state,limit_seconds=state['limit_usd']/state['usd_per_second'])
        Ledger._tick_milestones(state,allocated_seconds)


class Controller:
    def __init__(self,root,release,provider,limit=None,usd=None,mode=None):
        """`limit` (seconds mode) or `usd` (dollar mode) opens a new session's ledger; neither reopens the existing
        one in its own mode, which must be `mode` when that is given."""
        self.root=Path(root);self.release=Path(release).resolve();self.provider=None
        self.binding=verify_release(self.release)
        self.interface=json.loads((self.release/'public/interface.json').read_text())
        self.pair_workers=3 if self.binding.get('grading')=='parallel3' else 1
        self.attach(provider)
        self.overhead=getattr(provider,'reservation_overhead_seconds',30)
        self.minimum=getattr(provider,'minimum_run_seconds',90)
        self.practice_limit=getattr(provider,'max_practice_evaluations',6)
        # Reopening keeps the existing mode and ceiling; neither is ever replaced.
        self.ledger=Account(self.root/'allocation-ledger.json',limit,usd)
        if mode is not None and mode!=self.ledger.mode:
            self.ledger.lock.close()
            raise ValueError(f'This session is budgeted in {self.ledger.mode} mode; it cannot be reopened in {mode} mode')
        bound=self.root/'fast-release-binding.json';value={'release_commitment':digest(self.binding),'release':str(self.release)}
        if bound.exists() and json.loads(bound.read_text())!=value:raise ValueError('session release cannot change')
        if not bound.exists():dump(bound,value)
        self.workspace=self.root/'workspace'  # the native trainer tree: policy.py, learner.py, train.py, config.json
        if not self.workspace.exists():
            self.workspace.mkdir()
            for n in SOURCE:shutil.copyfile(self.release/'public'/n,self.workspace/n)
        self.table=self.root/'session-jobs.json'
        if not self.table.exists():dump(self.table,[])
        self.lock=threading.RLock();self.recovering=threading.Lock();self.threads={};self.stops=set();self.recovered={}
        self.halts=set()  # stopped jobs whose worker is ended at once instead of at its next update (session close)

    def attach(self,provider):
        """Use `provider` for workers (a reopened session gets one only when it grades)."""
        if provider is not None:provider.interface=self.interface;provider.pair_workers=self.pair_workers
        self.provider=provider

    # job table
    def jobs(self):
        with self.lock:return json.loads(self.table.read_text())

    def job(self,name):
        return next((j for j in self.jobs() if j['name']==name),None)

    def _add(self,job):
        with self.lock:
            jobs=self.jobs();jobs.append(job);dump(self.table,jobs)

    def _update(self,name,**values):
        with self.lock:
            jobs=self.jobs();job=next(j for j in jobs if j['name']==name);job.update(values);dump(self.table,jobs);return job

    def active(self):
        return [j['name'] for j in self.jobs() if j['status'] in ACTIVE]

    def limits(self):
        return self.ledger.limits()

    def _idle(self):
        active=self.active()
        if active and self.limits()['max_concurrent_jobs']==1:raise ValueError('One job at a time; still active: '+', '.join(active))
        if active:raise ValueError('Every job must have ended first; still active: '+', '.join(active))

    def _admit(self,kind):
        """Refuse a new job of `kind` beyond the mode's concurrency. Queued, running and uncertain jobs all count."""
        held=[j for j in self.jobs() if j['status'] in ACTIVE];limits=self.limits();jobs=limits['max_concurrent_jobs']
        if jobs==1 and held:raise ValueError('One job at a time; still active: '+', '.join(j['name'] for j in held))
        if len(held)>=jobs:raise ValueError(f'At most {jobs} jobs at a time; still active: '+', '.join(j['name'] for j in held)+'. Stop one or wait (an unresolved worker counts)')
        for key,field,what in (('max_concurrent_gpus','gpu','GPUs'),('max_concurrent_cpu_cores','cpu_cores','CPU cores')):
            if limits.get(key) is not None and sum(SHAPES[j['kind']][field] for j in held)+SHAPES[kind][field]>limits[key]:
                raise ValueError(f'At most {limits[key]} {what} at a time across your active jobs (a training run takes 1 GPU and 4 cores, practice 8 cores); still active: '+', '.join(j['name'] for j in held))

    def _practice_used(self,state):
        """Practice requests counted against the limit: admitted ones plus active practice jobs not admitted yet."""
        requests=state.get('practice_requests',[]);admitted={r.get('job') for r in requests}
        return len(requests)+sum(j['kind']=='practice' and j['status'] in ACTIVE and j['name'] not in admitted for j in self.jobs())

    def _trained(self,name):
        job=self.job(name)
        if job is None or job['kind']!='train':raise ValueError('Unknown run')
        if job['status']!='complete' or not (self.root/'jobs'/name/'bundle/manifest.json').is_file():
            raise ValueError('This needs a completed training run with a sealed network')
        return job

    # accounting
    def budget(self):
        state=self.ledger.read();known={r['name'] for r in state['allocations']}
        # A queued run is a liability from the moment it is accepted, before its thread reserves it in the ledger.
        pending=sum(j['reserved_seconds'] for j in self.jobs() if j['kind']=='train' and j['status'] in ('queued','running') and j['name'] not in known)
        used=self.ledger.used(state)+pending;limits=self.ledger.limits(state)
        common={'min_run_seconds':self.minimum,'max_run_seconds':MAX_WORKER_SECONDS,
                'run_reserve_seconds':self.overhead,'concurrency':limits['max_concurrent_jobs'],
                'practice_evaluations_remaining':max(0,self.practice_limit-self._practice_used(state)),
                'practice_accounting':'CPU-only worker, limited by count, not charged to the allocation',
                'policy_parameter_cap':PARAMETER_CAP,
                'lifecycle':'finalized' if state['finalized'] else 'open','submissions':len(state['submissions'])}
        if self.ledger.mode==budgets.USD:
            rate=state['usd_per_second'];limit=state['limit_usd'];remaining=limit-used*rate;left=max(0.,remaining)
            return {'budget_mode':budgets.USD,'budget_usd':float(limit),'used_usd':float(used*rate),'remaining_usd':float(remaining),
                    'allocatable_usd':float(left),
                    # The longest single launch the unspent dollars still buy, its settlement reserve included.
                    'allocatable_seconds':float(max(0,min(left/rate-self.overhead,MAX_WORKER_SECONDS))),
                    'affordable_seconds':{RUN_SHAPE:int(left/rate+1e-6)},'usd_per_hour':{RUN_SHAPE:round(3600*rate,4)},
                    'usd_per_second':rate,'used_worker_seconds':float(used),
                    'concurrent_gpus':limits['max_concurrent_gpus'],'concurrent_cpu_cores':limits['max_concurrent_cpu_cores'],
                    'charging':'budget, used and remaining are US dollars: a training run is charged its worker seconds x '
                               f'${rate:.7f} (L40S, 4 CPU cores, 16 GiB at the frozen list prices). A queued or running '
                               'run holds its whole grant, settlement reserve included, until the provider reports its '
                               'real interval. At most two training runs (one GPU each) at once.',
                    **common}
        limit=state['limit_seconds']
        # Every launch also holds a settlement/teardown reserve until the provider reports its real interval.
        return {'budget_mode':budgets.SECONDS,'budget_seconds':float(limit),'used_seconds':float(used),'remaining_seconds':float(limit-used),
                'allocatable_seconds':float(max(0,min(limit-used-self.overhead,MAX_WORKER_SECONDS))),**common}

    # researcher operations
    def paths(self,root=None):
        root=self.workspace if root is None else Path(root)
        return {n:root/n for n in SOURCE}

    def run(self,name,seconds,config=None):
        state=self.ledger.read()
        if state['finalized']:raise ValueError('The session is finalized; no further training')
        if not isinstance(name,str) or not NAME.fullmatch(name) or name.startswith('eval-'):
            raise ValueError('Run names are 1-40 characters of a-z, 0-9, - and _; the eval- prefix is kept for practice jobs')
        if self.job(name) or any(r['name']==name for r in state['allocations']) or (self.root/'sources'/name).exists() or (self.root/'jobs'/name).exists():
            raise ValueError('Run name already used; failed and stopped runs keep their names')
        with self.lock:  # admission, the budget check and queueing are one step
            self._admit('train')
            if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not self.minimum<=seconds<=MAX_WORKER_SECONDS:
                raise ValueError(f'One launch takes {self.minimum} to {MAX_WORKER_SECONDS} worker seconds')
            reserved=seconds+self.overhead;budget=self.budget();dollars={}
            if self.ledger.mode==budgets.USD:
                rate=budget['usd_per_second'];remaining=budget['remaining_usd'];dollars={'reserved_usd':reserved*rate,'usd_per_second':rate}
                if reserved*rate>remaining+1e-9:
                    raise ValueError(f'This run reserves ${reserved*rate:.4f} ({seconds:g} worker seconds plus the {self.overhead}-second settlement reserve at ${rate:.7f} per second) but only ${remaining:.4f} remain; nothing was launched')
            elif reserved>budget['remaining_seconds']:
                raise ValueError(f'This run reserves {reserved:g} seconds ({seconds:g} plus the {self.overhead}-second settlement reserve) but only {budget["remaining_seconds"]:.1f} remain; nothing was launched')
            for n in EDITABLE:ast.parse((self.workspace/n).read_text(),filename=n)
            json.dumps(json.loads((self.workspace/'config.json').read_text()),allow_nan=False)
            preflight(self.paths())
            source=self.root/'sources'/name;source.mkdir(parents=True,exist_ok=False)
            for n in SOURCE:shutil.copyfile(self.workspace/n,source/n)
            self._add({'name':name,'kind':'train','status':'queued','seconds':seconds,'reserved_seconds':reserved,'config':config,
                       'queued_at':time.time(),'source':{n:sha(source/n) for n in SOURCE},**dollars})
        return {'queued':name,'worker_seconds':seconds,'reserved_seconds':reserved,**dollars}

    def evaluate(self,name):
        state=self.ledger.read()
        if state['finalized']:raise ValueError('The session is finalized; no new practice')
        self._trained(name);folder=self.root/'jobs'/name;manifest=verify(folder/'bundle')
        commitment=digest(canonical_schedule(resolve_schedule(self.release,json.loads((self.release/'public/practice.json').read_text()))))
        if (folder/'practice.json').exists():
            cached=json.loads((folder/'practice.json').read_text())
            if cached.get('submission_identity')!=manifest['identity'] or cached.get('schedule_commitment')!=commitment:raise ValueError('cached practice identity/schedule differs')
            return {'cached':True,'run':name,'practice':{k:v for k,v in cached.items() if k!='rows'}}
        with self.lock:
            self._admit('practice')
            # One practice per run at a time: a run's practice writes into that run's own folder.
            if any(j['kind']=='practice' and j['run']==name and j['status'] in ACTIVE for j in self.jobs()):
                raise ValueError(f'Practice of {name} is already queued or running; see ar status')
            used=self._practice_used(state)
            if used>=self.practice_limit:raise ValueError('Public practice request limit reached; completed practice results stay readable')
            attempts=sum(j['kind']=='practice' and j['run']==name for j in self.jobs())  # a failed practice may be asked again
            job='eval-'+name+('-'+str(attempts+1) if attempts else '')
            self._add({'name':job,'kind':'practice','run':name,'status':'queued','queued_at':time.time(),'identity':manifest['identity'],'schedule_commitment':commitment})
        return {'queued':job,'run':name,'timeout_seconds':PRACTICE_SECONDS,'charged_to_allocation':False,'practice_evaluations_remaining':self.practice_limit-used-1}

    def stop(self,name):
        job=self.job(name)
        if job is None:raise ValueError('Unknown run')
        if job['status']=='queued':self._update(name,status='stopped',finished_at=time.time(),error='cancelled before any worker was created')
        elif job['status']=='running':
            self.stops.add(name)
            if job['kind']=='train':
                return {'stopping':name,'note':'A run that is training ends at its next update and keeps the network '
                        'trained so far: it finishes as complete with stopped_early, or as stopped if no update had '
                        'finished, and the time until then is charged. A worker still waiting for a GPU is '
                        'cancelled uncharged. Check ar status.'}
        elif job['status']=='uncertain':return self._resolve(job)
        else:raise ValueError('Run is not active')
        return {'stopping':name}

    def _resolve(self,job,timeout=120):
        """Stop an uncertain job now: terminate its recorded worker and settle it from the provider's record."""
        name=job['name'];self.stops.add(name)
        thread=self.threads.get(name)
        if thread is not None and thread.is_alive():thread.join(timeout=timeout)  # a recovery already under way
        job=self.job(name)
        if job['status']=='uncertain' and hasattr(self.provider,'recover'):self._recover(job)
        job=self.job(name)
        if job['status']=='uncertain':
            return {'stopping':name,'resolved':False,'status':'uncertain',
                    'reason':'the provider has not reported this worker\'s end; its hold stays until it does',
                    **({'recovery_error':job['recovery_error']} if job.get('recovery_error') else {})}
        # The settled job's own failure is `job_error`: a top-level `error` would report the stop as failed.
        return {'stopped':name,'resolved':True,'status':job['status'],'charged_seconds':job.get('charged_seconds'),
                **({'job_error':job['error']} if job.get('error') else {})}

    def submit(self,name):
        self._trained(name);folder=self.root/'jobs'/name;manifest=verify(folder/'bundle');target=self.root/'submissions'/manifest['identity']
        if not target.exists():shutil.copytree(folder/'bundle',target)
        verify(target,manifest['identity']);row=self.ledger.submit(name,manifest['identity'])
        return {'submission':len(self.ledger.read()['submissions']),'identity':manifest['identity'],'parameters':manifest['parameters'],
                'practiced':(folder/'practice.json').is_file(),**row,**self._dollars(row['charged_or_reserved_seconds'],'charged_or_reserved_usd')}

    def finalize(self):
        self._idle();state=self.ledger.finalize()
        return {'lifecycle':'finalized','final':state['milestones']['final'],'submissions':len(state['submissions']),
                'charged_or_reserved_seconds':self.ledger.used(state),**self._dollars(self.ledger.used(state),'charged_or_reserved_usd')}

    def _dollars(self,seconds,key):
        """{key: the dollars of `seconds` of the training worker} in dollar mode, else {}."""
        return {key:seconds*self.ledger.read()['usd_per_second']} if self.ledger.mode==budgets.USD else {}

    def grade(self):
        """Development grade: the final submission plays the release's protected schedule once, on a CPU worker.

        fast-v4 has no private final bank, so this is labelled development, never official. The protected
        schedule is never shown to the researcher (practice uses the public one), but it is not a fresh
        private draw either.
        """
        state=self.ledger.read()
        if not state['finalized']:raise ValueError('Only a finalized session is graded')
        final=state['milestones'].get('final')
        if not final:raise ValueError('The session has no final submission')
        bundle=self.root/'submissions'/final['identity'];manifest=verify(bundle,final['identity'])
        schedule=check_panel(self.release,self.binding)  # schedule commitment and opponent identities re-checked
        canonical=canonical_schedule(schedule)
        folder=self.root/'grade';folder.mkdir(exist_ok=False);path=folder/'protected-schedule.json';dump(path,schedule)
        opponents={p['name']:p['bundle'] for p in schedule['opponents'] if p['kind']=='checkpoint'}
        result=self.provider.cpu_grade(bundle,path,opponents,folder/'result.json',seconds=GRADE_SECONDS,split='protected-development')
        if result.get('submission_identity')!=manifest['identity'] or result.get('schedule_commitment')!=digest(canonical):
            raise ValueError('grade result identity/schedule differs')
        return {'status':'development_only','schedule':'protected','official':False,'submission_identity':manifest['identity'],
                'schedule_commitment':digest(canonical),**{k:result[k] for k in PRACTICE_SUMMARY if k in result}}

    # read-only views
    def status(self):
        jobs=[]
        for job in self.jobs():
            job=dict(job)
            practice=self.root/'jobs'/job['name']/'practice.json'
            if job['kind']=='train' and practice.is_file():
                value=json.loads(practice.read_text());job['practice']={k:value[k] for k in PRACTICE_SUMMARY if k in value}
            jobs.append(job)
        return jobs

    def log(self,name,tail=80):
        job=self.job(name)
        if job is None:raise ValueError('Unknown run')
        result=dict(job);folder=self.root/'jobs'/(job['run'] if job['kind']=='practice' else name)
        # The job's own failure; a reply's top-level `error` would report this command as failed.
        failure=result.pop('error',None)
        if failure:result['job_error']=failure
        if job['kind']=='train':
            output=folder/'retained/output'
            for label,options in (('stdout',[folder/'stdout.log']),('stderr',[folder/'stderr.log']),('trainer',[output/'starter.log']),
                                  ('worker_error',[output/'worker-error.txt']),('progress',[folder/'artifact/progress.jsonl',output/'artifact/progress.jsonl'])):
                path=next((p for p in options if p.is_file()),None)
                if path:result[label]='\n'.join(path.read_text(errors='replace').splitlines()[-tail:])[-20000:]
            if (folder/'artifact/result.json').is_file():
                result['result']={k:v for k,v in json.loads((folder/'artifact/result.json').read_text()).items() if k!='binding'}
        if (folder/'practice.json').is_file() and (job['kind']=='train' or job['status']=='complete'):
            result['practice']={k:v for k,v in json.loads((folder/'practice.json').read_text()).items() if k!='rows'}
        return result

    # workers
    def tick(self):
        """Start every queued job on a thread of its own (admission already kept them within the mode's limits)."""
        with self.lock:
            for name in [n for n,t in self.threads.items() if not t.is_alive()]:del self.threads[name]
            for job in self.jobs():
                if job['name'] in self.threads:continue
                if job['status']=='queued':
                    self._update(job['name'],status='running',started_at=time.time())
                    self._start(job['name'],self._train if job['kind']=='train' else self._practice,job)
                elif job['status']=='uncertain' and hasattr(self.provider,'recover') and time.monotonic()-self.recovered.get(job['name'],-math.inf)>=RECOVERY_SECONDS:
                    # Only the recorded provider identity is consulted; nothing is relaunched and no hold is cleared without a receipt.
                    self.recovered[job['name']]=time.monotonic()
                    self._start(job['name'],self._recover,job)

    def _start(self,name,target,job):
        self.threads[name]=threading.Thread(target=target,args=(job,),daemon=True);self.threads[name].start()

    def wait(self,timeout):
        """Join every job thread for at most `timeout` seconds in all; the names still working."""
        deadline=time.monotonic()+timeout
        for thread in list(self.threads.values()):thread.join(timeout=max(0,deadline-time.monotonic()))
        return [n for n,t in list(self.threads.items()) if t.is_alive()]

    def _ledger(self):
        # Each thread locks through its own file description; the main thread keeps self.ledger.
        return Account(self.ledger.path)

    def close(self):
        self.ledger.lock.close()

    def _close(self,job,status,**values):
        """Record how a job ended. An allocation without a provider end stays active and keeps its hold."""
        name=job['name']
        if job['kind']=='train':
            row=next((r for r in self.ledger.read()['allocations'] if r['name']==name),None)
            if row is not None and row['status'] in UNRESOLVED:status='uncertain'
            if row is not None and 'charged_seconds' in row:
                values['charged_seconds']=row['charged_seconds'];values.update(self._dollars(row['charged_seconds'],'charged_usd'))
        self._update(name,status=status,finished_at=time.time(),**values)

    def _train(self,job):
        name=job['name'];source=self.root/'sources'/name;folder=self.root/'jobs'/name;ledger=self._ledger()
        try:
            result=self.provider.train(ledger,name,source,folder,job['seconds'],stop=lambda:'now' if name in self.halts else name in self.stops)
            for n in SOURCE:
                if sha(source/n)!=sha(folder/'artifact'/n):raise ValueError('trained artifact source mismatch')
            if not result['finite_losses'] or result['weight_l1_change']<=0 or result['updates']<1:raise ValueError('no verified learning update')
            checkpoint=folder/'artifact/checkpoint.pt'
            checkpoint_sha=sha(checkpoint) if checkpoint.exists() else result.get('checkpoint_sha256')
            if not checkpoint_sha or len(checkpoint_sha)!=64:raise ValueError('source-bound checkpoint commitment missing')
            m=seal(folder/'artifact',folder/'bundle',{'job':name,'checkpoint_sha256':checkpoint_sha,'checkpoint_storage':result.get('checkpoint_retained_remotely','local'),'training_result_sha256':sha(folder/'artifact/result.json')})
            if m['parameters']>PARAMETER_CAP:raise ValueError(f'Policy has {m["parameters"]:,} parameters; the cap is {PARAMETER_CAP:,}')
            summary={'identity':m['identity'],'parameters':m['parameters'],'loss_summary':result.get('loss_summary'),
                     **{k:result[k] for k in ('updates','samples','natural_games','optimizer_steps','finite_losses','weight_l1_change','seconds')}}
            # A run the researcher stopped still sealed a verified network: it is complete, and says it ended early.
            self._close(job,'complete',result=summary,**({'stopped_early':True} if name in self.stops else {}))
        except workers.NeverStarted as exc:
            # Created but never given a GPU: terminated and settled at 0 by the provider; the next run is accepted.
            self._close(job,'stopped' if name in self.stops else 'failed',error=str(exc),never_started=True)
        except BaseException as exc:
            self._close(job,'stopped' if name in self.stops else 'failed',error=(type(exc).__name__+': '+str(exc))[-3500:])
        finally:ledger.lock.close()

    def _practice(self,job):
        name=job['run'];folder=self.root/'jobs'/name;ledger=self._ledger()
        try:
            manifest=verify(folder/'bundle');schedule=resolve_schedule(self.release,json.loads((self.release/'public/practice.json').read_text()));canonical=canonical_schedule(schedule)
            def admit(s):
                if s['finalized']:raise ValueError('session finalized')
                requests=s.setdefault('practice_requests',[])
                if len(requests)>=self.practice_limit:raise ValueError('public practice request limit reached')
                requests.append({'name':name,'job':job['name'],'identity':manifest['identity'],'at':time.time(),'schedule_commitment':digest(canonical)})
            ledger.mutate(admit);path=folder/'resolved-practice.json';dump(path,schedule)
            opponents={p['name']:p['bundle'] for p in schedule['opponents'] if p['kind']=='checkpoint'}
            result=self.provider.cpu_grade(folder/'bundle',path,opponents,folder/'practice.json',seconds=PRACTICE_SECONDS,split='public-development',stop=lambda:job['name'] in self.stops)
            if result.get('submission_identity')!=manifest['identity'] or result.get('schedule_commitment')!=digest(canonical):raise ValueError('practice result identity/schedule differs')
            self._close(job,'complete',result={k:result[k] for k in PRACTICE_SUMMARY if k in result})
        except BaseException as exc:
            self._close(job,'stopped' if job['name'] in self.stops else 'failed',error=(type(exc).__name__+': '+str(exc))[-3500:])
        finally:ledger.lock.close()

    def _recover(self,job):
        with self.recovering:  # `stop` may resolve the same job from the request thread
            job=self.job(job['name'])
            if job['status']=='uncertain':self._recover_one(job)

    def _recover_one(self,job):
        ledger=self._ledger()
        try:
            reply=self.provider.recover(ledger,job['name'])
            if reply.get('resolved'):
                status='stopped' if job['name'] in self.stops else 'failed'
                if reply.get('never_started'):
                    self._close(job,status,error=(job.get('error') or '')+' | '+workers.NEVER_STARTED,never_started=True)
                else:
                    self._close(job,status,error=(job.get('error') or '')+' | allocation end confirmed later; its artifact is not adopted')
        except BaseException as exc:
            self._update(job['name'],recovery_error=(type(exc).__name__+': '+str(exc))[-1000:])
        finally:ledger.lock.close()

    def settle(self,timeout):
        """Cancel queued work, stop all running work, wait for every job thread and give unknown ends a last chance to resolve."""
        deadline=time.monotonic()+timeout
        with self.lock:  # nothing queued starts while the session closes
            for job in self.jobs():
                if job['status']=='queued':self._update(job['name'],status='stopped',finished_at=time.time(),error='cancelled before any worker was created')
                elif job['status']=='running':self.stops.add(job['name']);self.halts.add(job['name'])  # nothing unsubmitted is kept: end it now
        self.wait(max(0,deadline-time.monotonic()))
        # An unknown end is asked for again for at most two minutes; after that the hold simply stays.
        deadline=min(deadline,time.monotonic()+120)
        while hasattr(self.provider,'recover') and time.monotonic()<deadline and any(j['status']=='uncertain' for j in self.jobs()):
            self.recovered.clear();self.tick()
            self.wait(max(0,deadline-time.monotonic()))
            if any(j['status']=='uncertain' for j in self.jobs()):time.sleep(min(5,max(0,deadline-time.monotonic())))
        return self.active()
