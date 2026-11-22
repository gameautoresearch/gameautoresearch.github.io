"""Host-owned Dou Dizhu session authority: ledger, job queue, practice and submissions.

The ledger records the session's budget mode: seconds mode runs one job at a time; dollar mode runs jobs side by
side up to the ledger's concurrency limits. Each job has its own worker thread, ledger connection, job folder,
sandbox and hand-off job name; nothing mutable is shared between jobs.
"""
from pathlib import Path
import json
import threading
import time
import uuid

from harness import workers
from simulators.doudizhu.autoresearch.contract import DEFAULT,EDITABLE
from simulators.doudizhu.autoresearch.ledger import Ledger
from simulators.doudizhu.autoresearch.bundle import verify
from simulators.doudizhu.autoresearch.manage import panel_paths

PRACTICE_MIN_SECONDS=10

class Controller:
    def __init__(self,root,release,provider,config=None):
        self.root=Path(root);self.release=Path(release);self.provider=provider
        self.ledger=Ledger(root,config);self.stops=set()
        self.threads={}  # job name -> the thread that owns its worker

    def claimed(self,name):
        """True while a worker thread owns this job, including before the provider reports its start."""
        thread=self.threads.get(name)
        return thread is not None and thread.is_alive()

    def busy(self):
        """True while any worker thread is alive."""
        return any(thread.is_alive() for thread in list(self.threads.values()))

    def join(self,timeout=None):
        """Wait for every worker thread, up to `timeout` seconds in total."""
        deadline=None if timeout is None else time.monotonic()+timeout
        for thread in list(self.threads.values()):
            thread.join(None if deadline is None else max(0.,deadline-time.monotonic()))

    def handle(self,request):
        # Run this in one broker thread. Source/config/timing returned by the guest
        # never replace session settings, immutable snapshots or provider receipts.
        action=request.get('action')
        if action in ('budget','status'):return self.ledger.public()
        if action=='run':
            import ast
            for name,text in request.get('code',{}).items():ast.parse(text,filename=name)
            limit=self.ledger.config.get('max_training_jobs')
            if limit is not None and sum(json.loads(j['payload']).get('kind')=='train' for j in self.ledger.jobs())>=limit:
                raise ValueError('Session training-job cap reached')
            if set(request.get('code',{}))!=set(EDITABLE):raise ValueError('Exact editable source whitelist required')
            if any(not isinstance(v,str) for v in request['code'].values()):raise ValueError('Invalid source payload')
            if sum(len(v.encode()) for v in request['code'].values())>DEFAULT['source_bytes_cap']:raise ValueError('Source transport cap exceeded')
            if len(json.dumps(request['config']).encode())>65536:raise ValueError('Recipe exceeds cap')
            name=request['name'];seconds=float(request['seconds'])
            if seconds>self.ledger.config.get('max_run_seconds',float('inf')):raise ValueError('Session per-run cap exceeded')
            self.ledger.queue(name,seconds,{'kind':'train','code':request['code'],'config':request['config'],'seed':int(request.get('seed',1))})
            return dict({'queued':name,'reserved_seconds':seconds},**self.dollars(seconds))
        if action=='eval':
            trained=next((r for r in self.ledger.jobs() if r['name']==request['name']),None)
            if not trained or not trained['artifact']:raise ValueError('Practice requires a completed registered run')
            verify(trained['artifact'],trained['artifact_sha'],check_tensors=False)
            name='eval-'+uuid.uuid4().hex[:12];seconds=float(request['seconds'])
            self.ledger.queue(name,seconds,{'kind':'practice','bundle':trained['artifact'],
                'panel':panel_paths(self.release,'practice'),'seeds':{'split':'practice','deal_seed':761943,'policy_seed':80921,
                    'deals':self.ledger.config['practice_deals']}},minimum_seconds=PRACTICE_MIN_SECONDS)
            return dict({'queued':name,'reserved_seconds':seconds,'budgeted_practice':True},**self.dollars(seconds))
        if action=='stop':
            r=next((r for r in self.ledger.jobs() if r['name']==request['name']),None)
            if not r:raise ValueError('Unknown run')
            settled=self.release_job(r)
            # Keep a resolved job's own failure out of the reply's top-level `error` (a failed command).
            if settled and 'error' in settled:settled['job_error']=settled.pop('error')
            return dict(settled or {},stopping=r['name'])
        if action=='submit':
            r=next((r for r in self.ledger.jobs() if r['name']==request['name']),None)
            if not r or not r['artifact']:raise ValueError('No registered trained artifact')
            verify(r['artifact'],r['artifact_sha'],check_tensors=False)
            return {'submission':self.ledger.submit(r['name']),'bundle_sha256':r['artifact_sha']}
        if action=='finalize':return self.ledger.finalize()
        raise ValueError('Unsupported broker action')

    def dollars(self,seconds):
        """A dollar-mode grant's price; nothing in seconds mode."""
        if self.ledger.mode()!='usd':return {}
        return {'usd_per_second':self.ledger.rate(),'reserved_usd':seconds*self.ledger.rate()}

    def release_job(self,job):
        """Cancel a job no worker has claimed (never allocated, uncharged); stop a claimed one through its worker.

        A claimed worker still waiting in the provider queue is abandoned uncharged; a started one is charged.
        An uncertain job is resolved now: its recorded worker is terminated and settled from the provider's
        record (never started: 0 charged; started: its exact start/finish lease).
        """
        if job['status']=='queued' and not self.claimed(job['name']):self.ledger.cancel_queued(job['name'])
        elif job['status'] in ('queued','running'):self.stops.add(job['name'])
        elif job['status']=='uncertain':
            if self.claimed(job['name']):return {'note':'Its worker thread is still settling it'}
            if not hasattr(self.provider,'recover'):raise ValueError('This provider cannot reconcile unresolved work')
            self.provider.recover(Ledger(self.root),job,stopped=True)
            settled=next(r for r in self.ledger.jobs() if r['name']==job['name'])
            return {'status':settled['status'],'charged':settled['charged'],'error':settled['error']}
        else:raise ValueError('Run is not active')
        return {}

    def tick(self):
        """Start queued jobs in ledger order, one worker thread each, while fewer threads than the session's job
        limit are alive (seconds mode: one, so a job starts only after the previous worker thread has ended).
        The ledger already refused any grant beyond the mode's concurrency and budget."""
        for name in [name for name,thread in self.threads.items() if not thread.is_alive()]:del self.threads[name]
        for queued in self.ledger.jobs():
            if len(self.threads)>=self.ledger.limit('max_concurrent_jobs'):return
            if queued['status']!='queued' or queued['name'] in self.threads:continue
            self.threads[queued['name']]=threading.Thread(target=self.work,args=(queued,),daemon=True)
            self.threads[queued['name']].start()

    def work(self,queued):
        # Each worker thread opens its own SQLite connection; writes serialize via BEGIN IMMEDIATE.
        ledger=Ledger(self.root)
        try:self.provider.execute(ledger,queued,lambda:queued['name'] in self.stops)
        except workers.NeverStarted as exc:
            # The provider terminated a worker that never got a GPU and settled it uncharged; nothing is held.
            current=next(j for j in ledger.jobs() if j['name']==queued['name'])
            if current['status'] in ('queued','uncertain'):
                ledger.abandon_unstarted(queued['name'],exc.receipt,str(exc),stopped=queued['name'] in self.stops)
        except BaseException as exc:ledger.uncertain(queued['name'],type(exc).__name__+': '+str(exc))
