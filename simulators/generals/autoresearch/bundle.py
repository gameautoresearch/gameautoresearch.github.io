"""Safe immutable policy packages. Verification never executes submitted Python."""
from pathlib import Path
import json
import re
import shutil
import zipfile
import math
import numpy as np
from .contract import DEFAULT,EDITABLE,PACKAGE,sha,digest,frozen_manifest,write_json

FILES=(*EDITABLE,'model.json','weights.npz')

def source_payload(directory):
 root=Path(directory);result={}
 for name in EDITABLE:
  p=root/name
  if p.is_symlink() or not p.is_file():raise ValueError('Regular editable source required: '+name)
  result[name]=p.read_text()
 if sum(len(v.encode()) for v in result.values())>DEFAULT['source_bytes_cap']:raise ValueError('Source cap exceeded')
 return result

def save_weights(params,path,cap=None):
 import jax
 from flax.traverse_util import flatten_dict
 arrays={'/'.join(key):np.asarray(value) for key,value in flatten_dict(jax.device_get(params)).items()}
 p=Path(path);tmp=p.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
 tmp.replace(p)
 validate_weights(p,cap)

def load_weights(path,cap=None):
 from flax.traverse_util import unflatten_dict
 validate_weights(path,cap)
 with np.load(path,allow_pickle=False) as a:
  return unflatten_dict({tuple(k.split('/')):a[k] for k in a.files})

def validate_weights(path,cap=None):
 p=Path(path);cap=cap or DEFAULT['policy_tensor_cap']
 if p.is_symlink() or p.stat().st_size>DEFAULT['artifact_bytes_cap']:raise ValueError('Oversized/symlinked weights')
 count=0
 with zipfile.ZipFile(p) as z:
  infos=z.infolist()
  if not infos or len(infos)>4096 or len({v.filename for v in infos})!=len(infos) or sum(v.file_size for v in infos)>DEFAULT['artifact_bytes_cap']:raise ValueError('Oversized/duplicate expanded arrays')
  for info in infos:
   if not info.filename.endswith('.npy'):raise ValueError('Only parameter arrays are permitted')
   with z.open(info) as f:
    version=np.lib.format.read_magic(f)
    if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=16384)
    elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(f,max_header_size=16384)
    else:raise ValueError('Unsupported array format')
    if dtype!=np.float32 or len(shape)>8 or any(type(d)!=int or d<0 for d in shape):raise ValueError('Finite float32 parameters required')
    n=math.prod(shape);count+=n
    if count>cap:raise ValueError('Parameter cap exceeded before allocation')
    if f.tell()+n*4!=info.file_size:raise ValueError('Parameter header/size mismatch')
 if count==0:raise ValueError('Empty parameter tree')
 loaded_count=0
 with np.load(p,allow_pickle=False) as a:
  if not a.files or len(a.files)!=len(set(a.files)):raise ValueError('Empty or duplicate weights')
  for name in a.files:
   if not re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*',name):raise ValueError('Invalid tensor path')
   v=a[name]
   if v.dtype!=np.float32 or v.ndim>8 or not np.isfinite(v).all():raise ValueError('Finite float32 parameter arrays required')
   loaded_count+=v.size
 if loaded_count!=count:raise ValueError('Parameter header changed')
 if count>cap:raise ValueError('Parameter cap exceeded')
 return count

def seal(source,destination,receipt=None):
 source=Path(source);dest=Path(destination)
 if dest.exists():raise FileExistsError('Immutable package exists')
 source_payload(source)
 for n in FILES:
  if (source/n).is_symlink() or not (source/n).is_file():raise ValueError('Regular bundle file required: '+n)
 if (source/'model.json').stat().st_size>65536:raise ValueError('Oversized architecture')
 model=json.loads((source/'model.json').read_text());count=validate_weights(source/'weights.npz')
 manifest=dict(format=1,contract=PACKAGE,files={n:sha(source/n) for n in FILES},
  tensors=count,engine=digest(frozen_manifest()),model=model,provenance=receipt or {})
 manifest['bundle_sha256']=digest(manifest);dest.mkdir(parents=True)
 for n in FILES:shutil.copy2(source/n,dest/n)
 write_json(dest/'manifest.json',manifest,exclusive=True)
 verify(dest)
 return manifest['bundle_sha256']

def verify(directory,expected=None,check_tensors=True,cap=None):
 root=Path(directory)
 if root.is_symlink():raise ValueError('Symlinked policy package')
 m=json.loads((root/'manifest.json').read_text());identity=m['bundle_sha256']
 if expected and identity!=expected:raise ValueError('Submitted identity changed')
 if digest({k:v for k,v in m.items() if k!='bundle_sha256'})!=identity:raise ValueError('Manifest drift')
 if m['contract']!=PACKAGE or m['engine']!=digest(frozen_manifest()):raise ValueError('Frozen rules/source mismatch')
 if set(m['files'])!=set(FILES):raise ValueError('Unexpected package members')
 for n,value in m['files'].items():
  if (root/n).is_symlink() or sha(root/n)!=value:raise ValueError('Package file drift: '+n)
 if check_tensors and validate_weights(root/'weights.npz',cap)!=m['tensors']:raise ValueError('Tensor count drift')
 return m

ARCHIVE='export.tar.gz'

def archive_bundle(directory,archive_path):
 """Worker side: the sealed package as one file, so the host fetches it in a single transfer."""
 import tarfile
 root=Path(directory);tmp=Path(str(archive_path)+'.tmp')
 with tarfile.open(tmp,'w:gz',compresslevel=1) as archive:
  for n in (*FILES,'manifest.json'):archive.add(root/n,arcname=n,recursive=False)
 tmp.replace(archive_path)

def unpack_bundle(archive_path,destination):
 """Host side: exactly the package members, as bounded regular files, verified before `destination` exists.

 A transfer that was cut off or an archive with anything else in it leaves no package directory behind.
 """
 import tarfile
 limits={**{n:DEFAULT['source_bytes_cap'] for n in EDITABLE},'model.json':65536,'weights.npz':DEFAULT['artifact_bytes_cap'],'manifest.json':1048576}
 if Path(archive_path).stat().st_size>sum(limits.values())+65536:raise ValueError('Oversized bundle archive')
 dest=Path(destination);staging=dest.with_name(dest.name+'.partial')
 if dest.exists() or staging.exists():raise FileExistsError('Immutable package exists')
 try:
  with tarfile.open(archive_path,'r:gz') as archive:
   members=archive.getmembers()
   if len(members)!=len(limits) or {m.name for m in members}!=set(limits):raise ValueError('Unexpected archive members')
   if any(not m.isfile() or m.size>limits[m.name] for m in members):raise ValueError('Invalid archive member')
   staging.mkdir(parents=True)
   for member in members:
    with archive.extractfile(member) as source,(staging/member.name).open('xb') as target:shutil.copyfileobj(source,target)
  identity=verify(staging,check_tensors=False)['bundle_sha256']
 except BaseException as exc:
  shutil.rmtree(staging,ignore_errors=True)
  if isinstance(exc,(tarfile.TarError,EOFError,OSError)) and not isinstance(exc,FileExistsError):raise ValueError('Incomplete or invalid bundle archive: '+type(exc).__name__) from exc
  raise
 staging.rename(dest)
 return identity

def export_existing(checkpoint,destination):
 import tempfile
 from integrations.generals.runtime import sources,read_checkpoint
 binding=sources();meta,data=read_checkpoint(checkpoint,binding=binding)
 with tempfile.TemporaryDirectory() as tmp:
  p=Path(tmp)
  for n in EDITABLE:shutil.copy2(Path(__file__).parent/'starter'/n,p/n)
  config={'width':meta['config']['network_width'],'compute_dtype':meta['config'].get('compute_dtype','float32')}
  write_json(p/'model.json',config);save_weights(data['params'],p/'weights.npz')
  return seal(p,destination,{'source_checkpoint_sha256':sha(checkpoint),'source_binding':binding['implementation_sha256'],
   'update':meta['update'],'qualification':'source-bound historical development bot; strength unqualified'})
