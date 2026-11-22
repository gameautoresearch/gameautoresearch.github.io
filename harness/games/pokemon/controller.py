"""Host-owned Pokémon session authority: ledger, native workers, practice, submissions, grading.

Seconds mode runs one worker at a time; dollar mode up to the ledger's `jobs_at_once`. Every job has its own
record, reservation, operation folder (host and session Volume), receipt and following thread.
"""
import ast
import json
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
import uuid

from harness import workers
from simulators.pokemon.autoresearch.ledger import Ledger
from simulators.pokemon.autoresearch.runtime import bundle
from simulators.pokemon.autoresearch.runtime.sdk import EDITABLE, identity, sha, source_signature, validate_recipe
from .diagnostics import evaluation_feedback, stage_diagnostics
from .provider import GPU, STOPPED_QUEUED, assessment_files, private_write, runtime_files

RUN=re.compile(r'[A-Za-z0-9_-]{1,40}')
PRACTICE=re.compile(r'eval-[0-9a-f]{8}')
GRADE='private-grade'
MIN_TRAIN_SECONDS=180          # setup, export and publication headroom around a training loop
MIN_PRACTICE_SECONDS=240
TRAINING_RESERVE_SECONDS=120   # of every training lease, kept for setup and publication
LIVE_LOG_SECONDS=15            # a running trainer's log is re-read from the Volume at most this often
STOP_GRACE_SECONDS=180         # how long a trainer asked to stop may take to be sealed and published before termination
# A sealed run the trusted supervisor ended: on the researcher's stop request, or at the stage time limit.
EARLY=('stopped_early','cut_short')
STOPPING=('The trainer is ended and the last complete policy it wrote is packaged: the run finishes as complete with '
          'stopped_early and can be practised and submitted, or as stopped if it had not written a complete policy '
          'yet. The worker\'s time until then is charged. Check ar status.')
# The default practice lease. 173 retained 32-game panels took 38-76 s of worker time (median 53 s): 26-47 s of
# games and up to 39 s of setup and publication. Four times the games is about 160 s (worst seen, scaled: 230 s).
# Only the worker's measured time is charged; the lease is what is held while it runs.
PRACTICE_SECONDS=360
# One fixed public schedule, so practice scores of different runs are comparable: 16 seeds x both seats x the
# four-opponent panel = 128 games.
PRACTICE_SEEDS=tuple(range(900_420_000,900_420_016))
FORBIDDEN_IMPORTS=('base64','pickle','marshal','requests','urllib','socket')
# An empty-starter release (`starter` in release.json) keeps the project's reference learning code out of the
# researcher's box; its recipe may also import from the project only the runtime interface that box shows, and
# no dynamic imports. The names it may take from network.py are those the box's view of it keeps.
EMPTY_STARTER_MODULES={'integrations.pokemon.efficient':None,
    'integrations.pokemon.network':('KEYS','batch','scripted','scripted_batch'),
    'integrations.pokemon.autoresearch.runtime.sdk':None,
    'integrations.pokemon.autoresearch.runtime.graph_export':None}
PROJECT_ROOTS=('integrations','simulators','candidates','environments','harness','tools','vendor','evidence')
DYNAMIC_IMPORTS=('importlib','__import__')
RECOVER_EVERY=30.0             # seconds between attempts to resolve an unresolved allocation from its recorded worker
LATE_START=('The worker started only after its GPU queue wait was abandoned; it was terminated and is charged '
            'from its provider start/finish receipt')
RESOLVED=('Unresolved allocation resolved: its recorded worker was terminated and charged from its provider '
          'start/finish receipt')


def empty_starter_imports(name,tree):
    """An empty-starter recipe imports from the project only `EMPTY_STARTER_MODULES`, and nothing dynamically."""
    allowed=', '.join(sorted(EMPTY_STARTER_MODULES))
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for alias in node.names:
                root=alias.name.split('.')[0]
                if root in DYNAMIC_IMPORTS:raise ValueError(f'{name}: dynamic imports are not available in this task')
                if root in PROJECT_ROOTS and (alias.name not in EMPTY_STARTER_MODULES
                        or EMPTY_STARTER_MODULES[alias.name] is not None):
                    raise ValueError(f'{name}: import {alias.name} is not available in this task; from the project '
                                     f'you may import {allowed} (network.py: only its observation helpers)')
        elif isinstance(node,ast.ImportFrom):
            module=node.module or ''
            if node.level:continue  # relative to the editable files themselves
            if module.split('.')[0] in DYNAMIC_IMPORTS:raise ValueError(f'{name}: dynamic imports are not available in this task')
            if module.split('.')[0] in PROJECT_ROOTS:
                if module in EMPTY_STARTER_MODULES:
                    names=EMPTY_STARTER_MODULES[module]
                    if names is not None and any(alias.name not in names for alias in node.names):
                        raise ValueError(f'{name}: from {module} you may import only {", ".join(names)}')
                # `from package import module` names whole modules: each must be an unrestricted allowed module.
                elif any(EMPTY_STARTER_MODULES.get(module+'.'+alias.name,'not allowed') is not None
                         for alias in node.names):
                    raise ValueError(f'{name}: from {module} import ... is not available in this task; from the '
                                     f'project you may import {allowed}')
        elif isinstance(node,ast.Name) and node.id=='__import__':
            raise ValueError(f'{name}: dynamic imports are not available in this task')


def inspect_source(source,config,starter='ppo'):
    """Source-only admission: a recipe is code and settings, never encoded weights or state."""
    if set(source)!=set(EDITABLE) or any(not isinstance(text,str) for text in source.values()):
        raise ValueError('Exact editable sources required: '+', '.join(EDITABLE))
    for name in EDITABLE:
        text=source[name]
        if len(text.encode())>1024*1024:raise ValueError('Source file exceeds 1 MiB: '+name)
        try:
            tree=ast.parse(text,filename=name);compile(text,name,'exec',dont_inherit=True)
        except SyntaxError as exc:raise ValueError(f'{name}: line {exc.lineno}: {exc.msg}') from None
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant)and isinstance(node.value,(str,bytes))and len(node.value)>4096:
                raise ValueError(name+': encoded payload constant forbidden')
            if isinstance(node,(ast.List,ast.Tuple,ast.Set))and len(node.elts)>256:
                raise ValueError(name+': embedded tensor/weight tables forbidden')
            if isinstance(node,ast.Import)and any(a.name.split('.')[0]in FORBIDDEN_IMPORTS for a in node.names):
                raise ValueError(name+': external/encoded-state imports forbidden')
            if isinstance(node,ast.ImportFrom)and not node.level and (node.module or '').split('.')[0]in FORBIDDEN_IMPORTS:
                raise ValueError(name+': external/encoded-state imports forbidden')
        if starter=='empty':empty_starter_imports(name,tree)
    config=validate_recipe(config)
    def check(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if any(word in key.lower()for word in('checkpoint','resume','pretrain','weight_path','optimizer_state')):
                    raise ValueError('State paths forbidden in the config')
                check(item)
        elif isinstance(value,list):
            if len(value)>256:raise ValueError('Embedded config tensors forbidden')
            for item in value:check(item)
        elif isinstance(value,str)and(len(value)>1024 or value.endswith(('.pt','.pth','.safetensors','.bin'))):
            raise ValueError('State payload/path forbidden in the config')
    check(config)
    return config


class Controller:
    def __init__(self,release,root,provider,limits=None,poll=3.0,final=False):
        """`limits` binds a new session's allowances; None reopens an existing session unchanged.

        `final` makes a new session the fresh final training of a submitted recipe.
        A reopened session reads that from its own ledger, whatever is passed here.
        """
        self.release=Path(release);self.root=Path(root);self.provider=provider;self.poll=poll
        # The release's starter decides the source rules; a value this harness does not know is refused, not
        # treated as the PPO starter. (Unit-test stand-ins without a release manifest use the PPO rules.)
        manifest=self.release/'release.json'
        self.starter=json.loads(manifest.read_bytes()).get('starter','ppo') if manifest.is_file() else 'ppo'
        if self.starter not in ('ppo','empty'):raise ValueError('Unknown release starter: '+repr(self.starter))
        self.ledger=Ledger(self.root/'ledger.sqlite3',limits,fresh_final=final)
        self.final=self.ledger.fresh_final
        # Job name -> the thread working on it (its worker's launch and follow, or a resolution); `thread` is the
        # most recently started one.
        self.threads={};self.thread=None;self.stops=set();self.lock=threading.Lock();self.recovered_at=None
        self.peeked={};self.writing=threading.Lock();self.stop_grace=STOP_GRACE_SECONDS
        self.grace={}  # job name -> wall time after which a worker asked to stop is terminated

    # Records -------------------------------------------------------------------------------------------
    def record_path(self,name):
        folder='practice'if PRACTICE.fullmatch(name)else'grading'if name==GRADE else'stages'
        return self.root/folder/(name+'.json')

    def record(self,name):
        path=self.record_path(name)
        return json.loads(path.read_bytes())if path.is_file()else None

    def update(self,job,**fields):
        # The job's thread and a request (a stop) both write this record: one read-modify-write at a time.
        with self.writing:
            value=dict(self.record(job)or{},**fields);private_write(self.record_path(job),value);return value

    def running(self):
        """The jobs a thread of this host is working on, oldest first."""
        return [name for name,thread in list(self.threads.items())if thread.is_alive()]

    def busy(self):
        """The first job a thread is working on, or None."""
        return next(iter(self.running()),None)

    def grant(self,seconds):
        """What a lease of `seconds` holds until it settles, in the session's budget units."""
        reply=dict(reserved_seconds=seconds)
        if self.ledger.mode=='usd':reply['reserved_usd']=round(seconds*self.ledger.limits['usd_per_second'],6)
        return reply

    def state(self,job):
        """This job's state in the harness vocabulary, from the ledger row and its own record."""
        if job['status']=='reserved':return 'queued'
        if job['status']=='active':return 'running'
        if job['status']=='unknown':return 'uncertain'
        record=self.record(job['name'])or{}
        # The ledger settles a moment before the worker's outcome is written to its record: until then it is
        # still running, never a premature `failed`.
        if record.get('status')in('reserved','active'):return 'running'
        if record.get('successful'):return 'complete'
        return 'stopped'if record.get('stop_requested_at')else'failed'

    # Requests ------------------------------------------------------------------------------------------
    def check(self,source,config):
        inspect_source(source,config,self.starter)
        return dict(passed=True,syntax_checked=list(EDITABLE),config_valid=True,
            scope='Syntax, config and source-only rules; submitted code was not executed on the host')

    def admit(self,name):
        active=self.running()
        if len(active)>=self.ledger.capacity:
            if self.ledger.capacity==1:raise ValueError('One job at a time; active: '+active[0])
            raise ValueError(f'At most {self.ledger.capacity} jobs at once (each run or practice takes one '
                             f'{GPU} worker); active: '+', '.join(active))
        if (self.root/'ALLOCATION-BLOCKED.json').exists():
            raise ValueError('A worker rejected this release\'s shared inputs; this session cannot allocate again')
        if self.record_path(name).exists():raise ValueError('Use a new job name; failed attempts are retained')

    def run(self,name,seconds,source,config):
        """Reserve the whole lease, retain the exact source, then train on a native worker."""
        if not RUN.fullmatch(name)or PRACTICE.fullmatch(name)or name==GRADE:raise ValueError('Invalid job name')
        seconds=int(seconds)
        if seconds<MIN_TRAIN_SECONDS:raise ValueError(f'A training run needs a lease of at least {MIN_TRAIN_SECONDS} seconds')
        config=inspect_source(source,config,self.starter)
        with self.lock:
            self.admit(name)
            snapshot=self.root/'snapshots'/uuid.uuid4().hex;snapshot.mkdir(parents=True)
            for filename in EDITABLE:(snapshot/filename).write_text(source[filename])
            (snapshot/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            effective=config
            if self.final:
                # A recipe cannot be tuned to seeds it knows: fresh host secrets replace its own. They stay
                # in this root's private state and in the worker request, never in a reply or a log.
                effective=dict(config,seed=secrets.randbelow(700000000)+1,training_seed_start=secrets.randbelow(700000000)+1)
            spec=dict(action='train',source_sha256=source_signature(snapshot),config=effective,
                training_seconds=seconds-TRAINING_RESERVE_SECONDS)
            if self.final:
                recipe=dict(files={filename:sha(snapshot/filename)for filename in(*EDITABLE,'config.json')},
                    source_sha256=spec['source_sha256'],config_sha256=identity(config))
                spec.update(fresh_final=True,recipe_sha256=identity(recipe),no_research_state=True)
            files=[(snapshot/filename,'candidate/'+filename)for filename in(*EDITABLE,'config.json')]+runtime_files(self.release)
            try:operation=self.ledger.reserve(name,'research',seconds)
            except BaseException:
                shutil.rmtree(snapshot);raise
            if self.final:
                private_write(self.root/'final-training.json',dict(status='dispatch_intent',job=name,operation=operation,
                    recipe_sha256=spec['recipe_sha256'],effective_config=effective,spec=spec,seconds=seconds,
                    seed_source='fresh-host-secrets',checkpoint_delivered=False))
            self.update(name,name=name,kind='train',operation=operation,status='reserved',seconds=seconds,
                source_sha256=spec['source_sha256'],config_sha256=identity(effective),snapshot=str(snapshot),
                **(dict(fresh_final=True,recipe_config_sha256=identity(config),seed_source='fresh-host-secrets')if self.final else{}))
            self.start(name,operation,'research',spec,files,seconds)
        reply=dict(queued=name,**self.grant(seconds),training_seconds=spec['training_seconds'])
        return dict(reply,fresh_final=True,seed_source='fresh-host-secrets')if self.final else reply

    def evaluate(self,name,seconds=None):
        """Public practice of a completed run on the frozen panel; its lease counts against the allocation."""
        stage=self.record(name)if RUN.fullmatch(name)and not PRACTICE.fullmatch(name)else None
        if not stage or stage.get('kind')!='train'or not stage.get('successful'):raise ValueError('No completed registered run')
        seconds=PRACTICE_SECONDS if seconds is None else int(seconds)
        if seconds<MIN_PRACTICE_SECONDS:raise ValueError(f'Practice needs a lease of at least {MIN_PRACTICE_SECONDS} seconds')
        package=Path(stage['output'])/'stage/bundle';manifest=bundle.verify(package,stage['bundle_sha256'])
        with self.lock:
            label='eval-'+uuid.uuid4().hex[:8];self.admit(label)
            request,files=assessment_files(self.release)
            files.extend((package/filename,'candidate/'+filename)for filename in(*bundle.FILES,'manifest.json'))
            request.update(candidate=dict(kind='bundle-native',name=label,path='/pilotjob/input/candidate',
                bundle_sha256=manifest['bundle_sha256']),seeds=list(PRACTICE_SEEDS),public_development=True)
            operation=self.ledger.reserve(label,'research',seconds)
            self.update(label,name=label,kind='practice',run=name,operation=operation,status='reserved',seconds=seconds,
                source_sha256=stage['source_sha256'],config_sha256=stage['config_sha256'],bundle_sha256=manifest['bundle_sha256'])
            # Practice plays games on the CPU: a CPU worker (same cap), charged to the research allocation as before.
            self.start(label,operation,'practice',dict(action='assess',assessment=request),files,seconds)
        return dict(queued=label,run=name,**self.grant(seconds),games=2*len(PRACTICE_SEEDS)*len(request['panel']))

    def stop(self,name,now=False):
        """End the named active job. Whatever its worker was allocated stays charged.

        A training run whose worker has started is asked to stop: the trusted supervisor ends the trainer and
        seals the last complete policy it wrote, and the run then settles like a finished one (`stopped_early`).
        Its worker is terminated only if it has not finished within the grace. Everything else is terminated at
        once: a practice panel, a worker still queued, a run whose request could not be delivered, and every
        job when the session is being closed (`now`).

        An unresolved (`uncertain`) job is resolved now from its recorded worker instead: terminated, then
        settled at 0 if the provider never started it, or from its start/finish receipt if it did.
        """
        with self.lock:
            if name not in self.running():
                job=next((job for job in self.ledger.snapshot()['jobs']if job['name']==name),None)
                if job is None:raise ValueError('Run is not active')
                if job['status']=='reserved':job=self.reconcile_reserved(job)
                if job['status']!='unknown':raise ValueError('Run is not active')
                if not self.resolvable(job):
                    raise ValueError('Run is unresolved but no worker was recorded for it, so nothing can be '
                                     'terminated; its hold stays until the provider record is reconciled')
                self.stops.add(name);self.update(name,stop_requested_at=time.time())
                self.spawn(name,lambda:self.resolve(name,job['id']))
                return dict(stopping=name,resolving='Terminating its recorded worker and settling it from the '
                            'provider record: 0 seconds if it never started')
            self.stops.add(name);record=self.update(name,stop_requested_at=time.time())
        if not now and record.get('kind')=='train'and record.get('status')=='active'and self.ask(name,record['operation']):
            return dict(stopping=name,note=STOPPING)
        self.provider.stop(record['operation'])
        return dict(stopping=name)

    def ask(self,name,operation):
        """Deliver a stop request to a running trainer's worker and start its grace; False if it was not delivered."""
        if not hasattr(self.provider,'request_stop'):return False
        try:sent=bool(self.provider.request_stop(operation))
        except Exception as exc:
            private_write(self.root/'jobs'/operation/'stop-error.json',dict(error_type=type(exc).__name__));sent=False
        if sent:
            until=time.time()+self.stop_grace
            self.update(name,stop_request_sent_at=time.time(),stop_grace_until=until);self.grace[name]=until
        return sent

    def live(self,name):
        """What a running training job has printed so far, or why that is not known yet. Never raises."""
        record=self.record(name)or{}
        if record.get('kind')!='train'or self.final:return None
        if not (self.root/'jobs'/record.get('operation','')/'provider.json').is_file():
            return dict(available=False,reason='The worker has not been created yet')
        at,value=self.peeked.get(name,(0,None))
        if value is None or time.monotonic()-at>=LIVE_LOG_SECONDS:
            try:value=dict(self.provider.peek(record['operation']),available=True)
            except FileNotFoundError:
                value=dict(available=False,reason='The trainer has not started or its first output is not committed yet')
            except Exception as exc:
                value=dict(available=False,reason='Live output could not be read: '+type(exc).__name__+': '+str(exc)[:200])
            self.peeked[name]=(time.monotonic(),value)
        return dict(value,age_seconds=round(time.monotonic()-self.peeked[name][0],1))

    def submit(self,name):
        stage=self.record(name)if RUN.fullmatch(name)and not PRACTICE.fullmatch(name)else None
        if not stage or stage.get('kind')!='train'or not stage.get('successful'):raise ValueError('No completed registered run')
        source=Path(stage['output'])/'stage/bundle';manifest=bundle.verify(source,stage['bundle_sha256'])
        target=self.root/'submissions'/name
        # The same immutable publication for every submission; a label is never overwritten.
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,target)
        bundle.verify(target,manifest['bundle_sha256'])
        return self.ledger.submit(name,target,manifest['bundle_sha256'])

    def finalize(self):
        if self.running():raise ValueError('Resolve active work before finalizing: '+', '.join(self.running()))
        return self.ledger.finalize()

    # Workers -------------------------------------------------------------------------------------------
    def start(self,name,operation,kind,spec,files,seconds):
        def work():
            try:launched=self.provider.launch(self.ledger,operation,kind,spec,files,seconds,lambda:name in self.stops)
            except workers.NeverStarted as exc:
                # Created but never given a GPU: terminated and settled at 0 seconds, never left uncertain.
                self.update(name,status=self.ledger.job(operation)['status'],successful=False,never_started=True,
                    charged_seconds=0,error=str(exc)[:500])
                return
            except BaseException as exc:
                # The ledger already says what this attempt costs: a provable zero, or a held unknown.
                row=self.ledger.job(operation)
                self.update(name,status=row['status'],successful=False,error=type(exc).__name__+': '+str(exc)[:500])
                return
            fields=dict(status='active')
            # From here `stop` asks the trainer to end instead of terminating; one asked for earlier found
            # nothing training yet and ends the worker now.
            with self.lock:
                stopped=name in self.stops
                if (launched or{}).get('abandoned')and not stopped:fields['error']=LATE_START
                self.update(name,**fields)
            if stopped:self.stop_quietly(operation)
            self.follow(name,operation)
        self.spawn(name,work)

    def spawn(self,name,target):
        self.thread=self.threads[name]=threading.Thread(target=target,daemon=True,name='pokemon-'+name);self.thread.start()

    def join(self,timeout):
        """Wait for every job thread, at most `timeout` seconds in all."""
        deadline=time.monotonic()+timeout
        for thread in list(self.threads.values()):thread.join(timeout=max(0.0,deadline-time.monotonic()))

    def stop_quietly(self,operation):
        try:self.provider.stop(operation)
        except Exception as exc:private_write(self.root/'jobs'/operation/'stop-error.json',dict(error_type=type(exc).__name__))

    def follow(self,name,operation,deadline=None):
        """Reconcile one created worker until the provider says it stopped; a transient error never abandons it.

        A worker that was asked to stop and has not finished by the end of its grace is terminated here.
        """
        record=self.record(name)or{}  # a host restarted inside the grace keeps it
        if record.get('stop_grace_until')is not None and not record.get('stop_grace_expired_at'):
            self.grace.setdefault(name,record['stop_grace_until'])
        while True:
            try:result=self.provider.collect(self.ledger,operation)
            except Exception as exc:
                self.update(name,collect_error=type(exc).__name__+': '+str(exc)[:500])
                if self.ledger.job(operation)['status']=='unknown':
                    return self.update(name,status='unknown',successful=False)
                result=dict(status='active-or-unknown')
            if result['status']=='settled':
                self.grace.pop(name,None);return self.complete(name,result)
            if result['status']=='unknown':return self.update(name,status='unknown',successful=False)
            if deadline is not None and time.monotonic()>=deadline:return None
            if name in self.grace and time.time()>=self.grace[name]:
                del self.grace[name];self.update(name,stop_grace_expired_at=time.time());self.stop_quietly(operation)
            time.sleep(self.poll)

    def complete(self,name,result):
        record=self.record(name);output=Path(result['output'])if result.get('output')else None
        fields=dict(status='settled',successful=bool(result.get('successful')),output=str(output)if output else None,
            provider={k:result.get('provider',{}).get(k)for k in('provider_id','created_at','started_at','finished_at','gpu_type')})
        for key in('collection_failure','initialization_failure','shared_input_admission_failed','checkpoint_recovery'):
            if key in result:fields[key]=result[key]
        if result.get('never_started'):
            fields.update(never_started=True,charged_seconds=0,
                error=STOPPED_QUEUED if record.get('stop_requested_at')else workers.NEVER_STARTED)
        elif result.get('resolved')and not record.get('error'):fields['error']=RESOLVED
        if record['kind']=='train'and fields['successful']:
            try:
                fields.update(checkpoint_sha256=sha(output/'stage/checkpoint.pt'),
                    bundle_sha256=bundle.verify(output/'stage/bundle')['bundle_sha256'])
                json.loads((output/'stage/result.json').read_bytes())
                fields.update(self.early(output))
            except Exception as exc:fields.update(successful=False,error=type(exc).__name__+': '+str(exc)[:500])
            intent=self.root/'final-training.json'
            if self.final and intent.is_file():
                private_write(intent,dict(json.loads(intent.read_bytes()),status='settled',successful=fields['successful'],
                    bundle_sha256=fields.get('bundle_sha256'),**{flag:True for flag in EARLY if fields.get(flag)}))
        elif record['kind']in('practice','grade'):
            summary=json.loads((output/'summary.json').read_bytes())if output and(output/'summary.json').is_file()else{}
            feedback=evaluation_feedback(summary,fields['successful'],dict(bundle_sha256=record['bundle_sha256']))
            # A practice or grade is successful only as a complete valid score, never as mere execution.
            fields.update(feedback,successful=feedback['score_valid'])
        return self.update(name,**fields)

    @staticmethod
    def early(output):
        """How the trusted supervisor says a sealed run's trainer ended, if not by itself: its own receipt's flags."""
        path=output/'stage/supervisor.json'
        if path.is_symlink()or not path.is_file()or path.stat().st_size>1024*1024:return{}
        receipt=json.loads(path.read_bytes())
        return {flag:True for flag in EARLY if receipt.get(flag)is True}

    def tick(self):
        """After a host restart, reattach to this session's own recorded workers; never create a replacement.

        Also finishes an outcome record the host did not write after its ledger row settled, and resolves an
        unresolved allocation from its recorded worker (terminated; 0 seconds if the provider never started it).
        A job a thread already works on is left to it. In seconds mode one worker is followed at a time.
        """
        serial=self.ledger.capacity==1
        with self.lock:
            owned=set(self.running())
            if serial and owned:return
            for job in self.ledger.snapshot()['jobs']:
                name,operation=job['name'],job['id']
                if name in owned:continue
                if job['status']=='active':
                    self.spawn(name,lambda name=name,operation=operation:self.follow(name,operation))
                    if serial:return
                    continue
                if job['status']=='reserved':self.reconcile_reserved(job)
                elif (job['status']=='settled'and job['kind']=='research'
                        and(self.record(name)or{}).get('status')in('reserved','active')):
                    if (self.root/'jobs'/operation/'provider.json').is_file():
                        self.spawn(name,lambda name=name,operation=operation:self.follow(name,operation))
                        if serial:return
                        continue
                    self.update(name,status='settled',successful=False,error='Host stopped before a worker was created')
            if self.recovered_at is not None and time.monotonic()-self.recovered_at<RECOVER_EVERY:return
            self.recovered_at=time.monotonic()
            owned=set(self.running())
            for job in self.ledger.snapshot()['jobs']:
                if self.resolvable(job)and job['name']not in owned:
                    self.spawn(job['name'],lambda name=job['name'],operation=job['id']:self.resolve(name,operation))
                    if serial:return

    def reconcile_reserved(self,job):
        """A reservation no thread owns any more. Only a missing create attempt proves zero cost."""
        if (self.root/'jobs'/job['id']/'create-attempt.json').exists():
            self.ledger.unknown(job['id'],dict(reason='Provider creation uncertain after host restart; never duplicated'))
            self.update(job['name'],status='unknown',successful=False)
        else:
            self.ledger.cancel_unattached(job['id'],'host stopped before provider create')
            self.update(job['name'],status='settled',successful=False,error='Host stopped before a worker was created')
        return self.ledger.job(job['id'])

    def resolvable(self,job):
        """An unmeasured unresolved research allocation whose worker was recorded, so its provider record can decide."""
        return (job['status']=='unknown'and job['kind']=='research'and job['actual_seconds']is None
                and hasattr(self.provider,'resolve')and(self.root/'jobs'/job['id']/'provider.json').is_file())

    def resolve(self,name,operation):
        """Settle an unresolved allocation now from its recorded worker; it stays held only if that record is unreadable."""
        try:result=self.provider.resolve(self.ledger,operation)
        except Exception as exc:
            return self.update(name,resolve_error=type(exc).__name__+': '+str(exc)[:500])
        if result.get('status')!='settled':
            return self.update(name,resolve_error='The provider has not reported this worker stopped yet; retried later')
        return self.complete(name,dict(result,resolved=True))

    def settle(self,timeout=900):
        """Stop every active job and wait until each one's own worker is verified stopped and settled."""
        for name in self.running():
            try:self.stop(name,now=True)
            except Exception as exc:
                private_write(self.root/'settle-errors'/(name+'.json'),dict(job=name,error_type=type(exc).__name__,message=str(exc)[:500]))
        self.join(timeout)
        if not self.running():
            self.recovered_at=None  # closing resolves an unresolved allocation now, not after the retry interval
            self.tick()
            self.join(timeout)

    # Host-only private grading ---------------------------------------------------------------------------
    def grade(self,clusters,seconds):
        final=self.ledger.final()
        if final is None:raise ValueError('Finalize an explicit submission before private grading')
        package=Path(final['bundle']);manifest=bundle.verify(package,final['bundle_sha256'])
        record=self.record(GRADE)
        if record and record.get('status')in('settled','unknown'):raise FileExistsError('Private grading already ran for this session; it is never repeated')
        if record is None:
            # Seeds exist only now, after the final identity locked; a crash never creates a second bank.
            bank=self.ledger.private_bank(clusters)
            request,files=assessment_files(self.release)
            files.extend((package/filename,'candidate/'+filename)for filename in(*bundle.FILES,'manifest.json'))
            request.update(candidate=dict(kind='bundle-native',name='final',path='/pilotjob/input/candidate',
                bundle_sha256=manifest['bundle_sha256']),seeds=bank['primary'],public_development=False)
            operation=self.ledger.reserve(GRADE,'final',seconds)
            record=self.update(GRADE,name=GRADE,kind='grade',operation=operation,status='reserved',seconds=seconds,
                bundle_sha256=manifest['bundle_sha256'],private_bank_commitment=identity(bank))
            try:self.provider.launch(self.ledger,operation,'final',dict(action='assess',assessment=request),files,seconds)
            except BaseException as exc:
                self.update(GRADE,status=self.ledger.job(operation)['status'],successful=False,error=type(exc).__name__+': '+str(exc)[:500])
                raise
            self.update(GRADE,status='active')
        elif record['status']=='reserved':raise RuntimeError('Private grading dispatch is unresolved; reconcile it, never replay private games')
        record=self.follow(GRADE,record['operation'],deadline=time.monotonic()+seconds+900)
        if record is None or record['status']!='settled':raise RuntimeError('Private grading worker is unresolved; no score')
        # The graded policy must come from, and be graded on, the contract's fixed resources.
        trained=self.record(final['job'])
        self.provider.verify_profile(trained['operation'],GPU)
        self.provider.verify_profile(record['operation'],None)
        return self.update(GRADE,profile_verified=True)
