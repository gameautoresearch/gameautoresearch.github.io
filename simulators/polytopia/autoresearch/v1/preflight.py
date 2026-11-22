"""CPU-only gate: actual Rust state -> UID-isolated minimal policy -> legal transition."""
import os,sys,json
sys.path.insert(0,'/opt/v1')
from grade import Policy,RustGames,np,SIZE,PLAYERS,ROUND_LIMIT
p=Policy('/candidate',1)
try:
 engine=RustGames(SIZE,1,max_turns=ROUND_LIMIT,players=PLAYERS);state,prepared=engine.reset(np.array([91],np.int32))
 choice=p.choose(prepared.features,prepared.masks);tick,status=engine.tick(state,choice,np.array([92],np.int32))
 if int(np.asarray(status)[0]):raise RuntimeError('Initial policy action is illegal')
 import subprocess
 probe=subprocess.run(['/opt/venv/bin/python','-c',"from pathlib import Path; p=Path('/private/marker'); print(p.read_text())"],capture_output=True,text=True,user=65534,group=65534,extra_groups=[])
 if probe.returncode==0:raise RuntimeError('Unprivileged policy read private state')
 print(json.dumps({'legal_transition':True,'private_read_denied':True,'candidate_uid':65534}),flush=True)
finally:p.close()
