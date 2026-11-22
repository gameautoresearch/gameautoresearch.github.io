from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from integrations.generals_autoresearch import training_limits as runtime


def test_training_privilege_drop_keeps_isolation_and_bounded_limits(monkeypatch):
    calls=[]
    monkeypatch.setattr(runtime.os,'setgroups',lambda v:calls.append(('groups',v)))
    monkeypatch.setattr(runtime.os,'setgid',lambda v:calls.append(('gid',v)))
    monkeypatch.setattr(runtime.os,'setuid',lambda v:calls.append(('uid',v)))
    monkeypatch.setattr(runtime.ctypes,'CDLL',lambda *a,**k:SimpleNamespace(prctl=lambda *args:calls.append(('prctl',args)) or 0))
    monkeypatch.setattr(runtime.resource,'setrlimit',lambda k,v:calls.append((k,v)))
    runtime.drop_training_privileges()
    assert calls[:4]==[('groups',[]),('gid',65534),('uid',65534),('prctl',(38,1,0,0,0))]
    assert calls[4:]==[(runtime.resource.RLIMIT_CORE,(0,0)),(runtime.resource.RLIMIT_NPROC,(256,256)),(runtime.resource.RLIMIT_NOFILE,(512,512))]


def test_failed_no_new_privileges_fails_closed(monkeypatch):
    for n in ('setgroups','setgid','setuid'):monkeypatch.setattr(runtime.os,n,lambda v:None)
    monkeypatch.setattr(runtime.ctypes,'CDLL',lambda *a,**k:SimpleNamespace(prctl=lambda *args:-1))
    with pytest.raises(OSError):runtime.drop_training_privileges()


def test_proc_sample_retains_actual_limits_and_identity(tmp_path):
    p=tmp_path/'123';p.mkdir();(p/'fd').mkdir();(p/'fd/0').touch()
    (p/'status').write_text('Uid:\t65534 65534 65534 65534\nGid:\t65534 65534 65534 65534\nThreads:\t87\nNoNewPrivs:\t1\n')
    (p/'limits').write_text('Max processes 256 256 processes\nMax open files 512 512 files\n')
    sample=runtime.process_sample(123,tmp_path)
    assert sample['Threads']=='87' and sample['NoNewPrivs']=='1' and sample['open_fds']==1
    assert '256' in sample['limits'] and '512' in sample['limits']


def test_monitor_retains_peak_and_child_exit(monkeypatch):
    monkeypatch.setattr(runtime,'process_sample',lambda pid:dict(Threads='88',open_fds=72,Uid='65534',NoNewPrivs='1'))
    record={};proc=SimpleNamespace(pid=123,poll=lambda:-6)
    assert runtime.wait_monitored(proc,180,record)==-6
    assert record['peak_threads']==88 and record['peak_open_fds']==72
    assert record['runtime_id']==runtime.RUNTIME_ID


def test_monitor_enforces_deadline(monkeypatch):
    monkeypatch.setattr(runtime,'process_sample',lambda pid:{})
    ticks=iter([1.,2.]);monkeypatch.setattr(runtime.time,'monotonic',lambda:next(ticks))
    with pytest.raises(subprocess.TimeoutExpired):runtime.wait_monitored(SimpleNamespace(pid=123,poll=lambda:None,args=['train']),.01,{})


def test_training_entry_does_not_use_inference_dropper():
    text=(Path(runtime.__file__).parent/'train_entry.py').read_text()
    assert 'from .arena import' not in text and 'wait_monitored(proc,a.seconds,runtime)' in text
