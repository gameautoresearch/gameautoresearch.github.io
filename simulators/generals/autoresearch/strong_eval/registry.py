"""Explicit immutable class/source/weight registry; historical bundles unchanged."""
from pathlib import Path
import json,shutil
from ..contract import sha,digest,write_json,ROOT
from ..bundle import verify
from .contract import CLASSES


def relative_name(name):
 p=Path(name)
 if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe manifest path')
 return p


def verify_source_manifest(source,training_source=None):
 data=json.loads(Path(source).read_text());payload=data['sources']
 if data['sha256']!=digest_legacy(payload):raise ValueError('AverageJoe source digest changed')
 bases={'averagejoe_files':ROOT/'vendor/engines/AverageJoe','engine_files':ROOT/'vendor/engines/generals-bots','framework_files':ROOT/'simulators/generals/averagejoe'}
 for field,base in bases.items():
  for name,value in payload[field].items():
   if (base/relative_name(name)).is_symlink() or sha(base/relative_name(name))!=value:raise ValueError('Checkpoint source incompatible: '+field+'/'+name)
 extensions={'strong_campaign_files':Path('simulators/generals/averagejoe/strong_campaign'),'accounting_files':Path('.')}
 known=set(bases)|set(extensions)
 if any(k.endswith('_files') and k not in known for k in payload):raise ValueError('Unknown training source manifest category')
 for field,base in extensions.items():
  if payload.get(field) and training_source is None:raise ValueError('Archived training extensions required')
  for name,value in payload.get(field,{}).items():
   if sha(Path(training_source)/base/relative_name(name))!=value:raise ValueError('Archived training extension changed: '+name)
 return data


def digest_legacy(value):
 import hashlib
 return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def register_entrant(source,destination,label,selection_eligible=True):
 source=Path(source);m=verify(source);dest=Path(destination)
 if dest.exists():raise FileExistsError('New immutable registry entry required')
 shutil.copytree(source,dest/'bundle')
 value=dict(format=2,label=label,selection_eligible=selection_eligible,inference_class='entrant-stateless-600k-v1',parameters=m['tensors'],bundle_sha256=m['bundle_sha256'],files={'bundle/'+k:v for k,v in m['files'].items()},provenance=dict(historical_bundle=str(source),historical_manifest_sha256=sha(source/'manifest.json')),strength_qualified=False)
 value['identity']=digest(value);write_json(dest/'registry.json',value,exclusive=True);return value


def register_averagejoe(checkpoint,config,source_manifest,destination,label,variant='model.eqx',archive_source=None):
 checkpoint=Path(checkpoint);dest=Path(destination);meta=json.loads((checkpoint/'state.json').read_text())
 if meta.get('exact_tensor_readback') is not True:raise ValueError('Verified checkpoint tensor readback required')
 source=verify_source_manifest(source_manifest,archive_source)
 if meta['source_sha256']!=source['sha256'] or meta['files'][variant]!=sha(checkpoint/variant):raise ValueError('AverageJoe checkpoint/source mismatch')
 if variant not in ('model.eqx','ema.eqx'):raise ValueError('Explicit current or EMA variant required')
 if dest.exists():raise FileExistsError('New immutable registry entry required')
 dest.mkdir(parents=True)
 for p,n in [(checkpoint/variant,'model.eqx'),(Path(config),'config.json'),(Path(source_manifest),'source.json'),(checkpoint/'state.json','state.json')]:shutil.copy2(p,dest/n)
 for field,base in {'strong_campaign_files':Path('simulators/generals/averagejoe/strong_campaign'),'accounting_files':Path('.')}.items():
  for name,value in source['sources'].get(field,{}).items():
   target=dest/'training_source'/base/relative_name(name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(archive_source)/base/name,target)
 value=dict(format=2,label=label,inference_class='trusted-averagejoe-history-16m-v1',parameters=15351761,files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()},provenance=dict(checkpoint=str(checkpoint),variant=variant,source_sha256=source['sha256'],iteration=meta['iteration'],stage=meta['stage'],training_general_trade=meta.get('general_trade',False),training_extensions=sorted(k for k in source['sources'] if k in ('strong_campaign_files','accounting_files')),parent_lineage=meta.get('parent'),neural_inference_only=True),strength_qualified=False)
 value['identity']=digest(value);write_json(dest/'registry.json',value,exclusive=True);return value


def verify_entry(directory):
 root=Path(directory);m=json.loads((root/'registry.json').read_text())
 if digest({k:v for k,v in m.items() if k!='identity'})!=m['identity']:raise ValueError('Registry digest drift')
 if m['inference_class'] not in CLASSES or m['parameters']>CLASSES[m['inference_class']]['parameters']:raise ValueError('Inference class/parameter cap mismatch')
 for n,value in m['files'].items():
  p=root/relative_name(n)
  if p.is_symlink() or not p.is_file() or sha(p)!=value:raise ValueError('Registry file drift: '+n)
 if m['inference_class']=='entrant-stateless-600k-v1':verify(root/'bundle',m['bundle_sha256'])
 else:verify_source_manifest(root/'source.json',root/'training_source')
 return m
