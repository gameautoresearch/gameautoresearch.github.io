"""Observer uses provider allocation start and the absolute campaign stop."""
import importlib,sys,types
from pathlib import Path
import pytest

@pytest.mark.parametrize('hard_stop,expected_min,expected_max',[(1000,210,220),(205,175,185)])
@pytest.mark.parametrize('module',['simulators.tavern.reference_training.launch','simulators.tavern.reference_training.strong.launch'])
def test_observer_queue_is_not_allocation_time(tmp_path,monkeypatch,hard_stop,expected_min,expected_max,module):
 import time,pathlib,threading
 now=[150.];termination=[];timers=[]
 class Timer:
  def __init__(self,seconds,callback):self.seconds=seconds;self.callback=callback;self.cancelled=False;timers.append(self)
  def start(self):pass
  def cancel(self):self.cancelled=True
 monkeypatch.setattr(threading,'Timer',Timer)
 class Volume:
  @staticmethod
  def from_name(*a,**k):return Volume()
  def commit(self):pass
 class Sandbox:
  @staticmethod
  def from_id(*a):return Sandbox()
  def terminate(self,**kw):termination.append(now[0])
 fake_modal=types.SimpleNamespace(Volume=Volume,Sandbox=Sandbox)
 monkeypatch.setitem(sys.modules,'modal',fake_modal)
 def info(*a):return dict(provider_id='sb-fixture',started_at=140.,finished_at=now[0] if termination else 0.,gpu_type='L4')
 monkeypatch.setitem(sys.modules,'simulators.tavern.autoresearch.provider',types.SimpleNamespace(configure_sandbox_waits=lambda:None,allocation_info=info))
 launch=importlib.import_module(module)
 monkeypatch.setattr(time,'time',lambda:now[0]);monkeypatch.setattr(time,'sleep',lambda seconds:now.__setitem__(0,now[0]+seconds))
 real_path=Path
 monkeypatch.setattr(pathlib,'Path',lambda p:tmp_path if str(p)=='/retained' else real_path(p))
 launch.observe('ap-fixture','sb-fixture','volume-fixture','attempt',100.,100.,hard_stop)
 assert termination and expected_min<=termination[0]<=expected_max
 assert (tmp_path/'attempt/control/allocation.json').exists()
 if module.endswith('strong.launch'):assert len(timers)==1 and timers[0].cancelled and timers[0].seconds==hard_stop-155

def test_strong_observer_watchdog_survives_status_rpc_failure(tmp_path,monkeypatch):
 import time,pathlib,threading,os
 timers=[];exits=[]
 class Timer:
  def __init__(self,seconds,callback):self.seconds=seconds;self.callback=callback;self.cancelled=False;timers.append(self)
  def start(self):pass
  def cancel(self):self.cancelled=True
 class Volume:
  @staticmethod
  def from_name(*a,**kw):return Volume()
  def commit(self):pass
 class Sandbox:
  @staticmethod
  def from_id(*a):return Sandbox()
 def failed_rpc(*a):raise RuntimeError('fixture status transport failed')
 monkeypatch.setitem(sys.modules,'modal',types.SimpleNamespace(Volume=Volume,Sandbox=Sandbox))
 monkeypatch.setitem(sys.modules,'simulators.tavern.autoresearch.provider',types.SimpleNamespace(configure_sandbox_waits=lambda:None,allocation_info=failed_rpc))
 launch=importlib.import_module('simulators.tavern.reference_training.strong.launch')
 monkeypatch.setattr(time,'time',lambda:100.);monkeypatch.setattr(threading,'Timer',Timer);monkeypatch.setattr(os,'_exit',exits.append)
 real_path=Path;monkeypatch.setattr(pathlib,'Path',lambda p:tmp_path if str(p)=='/retained' else real_path(p))
 with pytest.raises(RuntimeError,match='status transport failed'):launch.observe('ap-fixture','sb-fixture','volume-fixture','attempt',100.,100.,150.)
 assert len(timers)==1 and timers[0].seconds==45 and not timers[0].cancelled
 timers[0].callback();assert exits==[78]
