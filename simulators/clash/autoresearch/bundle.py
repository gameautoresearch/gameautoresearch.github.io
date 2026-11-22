"""Safe immutable submissions. Verification never imports entrant code."""
from pathlib import Path
import json,shutil,zipfile,math,re
import numpy as np
from .common import EDITABLE,dump,sha,digest,engine_binding
FILES=(*EDITABLE,'config.json','weights.npz')
def weights(path):
 p=Path(path)
 if p.is_symlink() or p.stat().st_size>16*1024**2:raise ValueError('weight size/symlink cap')
 count=0
 with zipfile.ZipFile(p) as z:
  infos=z.infolist()
  if not infos or len(infos)>1024 or len({i.filename for i in infos})!=len(infos):raise ValueError('invalid tensor archive')
  for info in infos:
   if info.file_size>16*1024**2:raise ValueError('expanded tensor cap')
   with z.open(info) as f:
    version=np.lib.format.read_magic(f)
    if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=16384)
    elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(f,max_header_size=16384)
    else:raise ValueError('tensor format')
    count+=math.prod(shape)
    if dtype!=np.float32 or len(shape)>8 or count>1000000 or f.tell()+math.prod(shape)*4!=info.file_size:raise ValueError('float32 shape/parameter cap')
 with np.load(p,allow_pickle=False) as a:
  for key in a.files:
   if not re.fullmatch(r'[A-Za-z0-9_.]+',key) or not np.isfinite(a[key]).all():raise ValueError('invalid/nonfinite weight')
 return count
def seal(source,destination,provenance=None):
 source=Path(source);dest=Path(destination)
 if dest.exists():raise FileExistsError(dest)
 for name in FILES:
  p=source/name
  if p.is_symlink() or not p.is_file():raise ValueError('regular bundle files required')
 if sum((source/n).stat().st_size for n in EDITABLE)>256000 or (source/'config.json').stat().st_size>65536:raise ValueError('source/config cap')
 m={'format':'clash-ar-bundle-v1','files':{n:sha(source/n) for n in FILES},'engine':engine_binding(),'parameters':weights(source/'weights.npz'),'provenance':provenance or {}}
 if json.loads((source/'config.json').read_text()).get('training_route')=='native-fast-v1':
  from .fast_entry import runtime_binding
  m['policy_runtime']=runtime_binding()
 m['identity']=digest(m);dest.mkdir(parents=True)
 for name in FILES:shutil.copyfile(source/name,dest/name);(dest/name).chmod(0o444)
 dump(dest/'manifest.json',m);(dest/'manifest.json').chmod(0o444);verify(dest);return m

def verify(root,expected=None):
 root=Path(root);m=json.loads((root/'manifest.json').read_text())
 if root.is_symlink() or m['format']!='clash-ar-bundle-v1' or digest({k:v for k,v in m.items() if k!='identity'})!=m['identity'] or (expected and expected!=m['identity']):raise ValueError('bundle manifest drift')
 if m['engine']!=engine_binding() or set(m['files'])!=set(FILES):raise ValueError('engine/source binding differs')
 for n,want in m['files'].items():
  if (root/n).is_symlink() or sha(root/n)!=want:raise ValueError('bundle bytes changed')
 if json.loads((root/'config.json').read_text()).get('training_route')=='native-fast-v1':
  from .fast_entry import runtime_binding
  if m.get('policy_runtime')!=runtime_binding():raise ValueError('fast policy runtime binding differs')
 if weights(root/'weights.npz')!=m['parameters']:raise ValueError('parameter count changed')
 return m
