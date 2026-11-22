"""Guarded finite CPU final qualification; cannot train or start a GPU."""
from pathlib import Path
import argparse,io,json,os,shutil,tarfile,time
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .launch import CAMPAIGN,LEDGER,role_usage,observe
from simulators.clash.autoresearch.common import ROOT,GAME,dump,sha,digest

def launch():
 import modal
 from tools.overnight_budget import CampaignBudget
 from simulators.generals.autoresearch.provider import configure_sandbox_waits
 configure_sandbox_waits();root=CAMPAIGN/'clash/rl';selection=json.loads((root/'selection-close.json').read_text());private=root/'fresh-assessment';bank=json.loads((private/'bank.json').read_text());commit=json.loads((private/'commitment.json').read_text())
 if not selection['no_further_training_or_selection'] or sha(private/'bank.json')!=commit['bank_sha256']:raise ValueError('closed roster/committed bank required')
 target=root/'final-qualification-01';target.mkdir(exist_ok=False);inputs=target/'inputs';inputs.mkdir(mode=0o700)
 candidates=[]
 for x in selection['candidates']:
  item=dict(x);dest=inputs/'candidates'/x['name'];shutil.copytree(x['bundle'],dest);item['bundle']='candidates/'+x['name'];candidates.append(item)
 transported=dict(selection,candidates=candidates);dump(inputs/'selection.json',transported);shutil.copyfile(private/'bank.json',inputs/'bank.json')
 peer=json.loads((root.parent/'coordination/evaluator.json').read_text());release=Path(peer['public_release']);interface=json.loads((release/'interface.json').read_text());opponents=json.loads((release/'opponents.json').read_text())
 for x in opponents:
  if x['kind']=='checkpoint':
   dest=inputs/'opponents'/x['name'];shutil.copytree(x['bundle'],dest);x['bundle']='opponents/'+x['name']
 dump(inputs/'opponents.json',opponents);dump(inputs/'metadata.json',{'bank_sha256':sha(inputs/'bank.json'),'interface_sha256':interface['interface_sha256'],'selection_sha256':sha(root/'selection-close.json')})
 archive=target/'assessment-overlay.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  t.add(GAME/'reference_training/assessment_worker.py',arcname='simulators/clash/reference_training/assessment_worker.py')
  for n,want in interface['source_sha256'].items():
   if sha(release/n)!=want:raise ValueError('public evaluator release changed')
   t.add(release/n,arcname='simulators/clash/autoresearch/'+n)
 budget=CampaignBudget('clash',path=LEDGER)
 quote=.4642176+(4230*(8*.00003942+32*.00000667)+4290*(.25*.0000131+1*.00000222))*1.1
 if role_usage(budget)['resource_usd']+quote>32:raise ValueError('RL role cash subcap protects final qualification')
 build=budget.reserve_worker('clash/rl/final-cpu-image',600,cpu_cores=32,memory_gib=128,sandbox=False)
 worker=budget.reserve_worker('clash/rl/final-cpu-worker',4230,cpu_cores=8,memory_gib=32)
 observer_op=budget.reserve_worker('clash/rl/final-cpu-observer',4290,cpu_cores=.25,memory_gib=1,sandbox=False)
 dump(target/'reservations.json',{'build':build,'worker':worker,'observer':observer_op})
 volume_name='clash-reference-20261109-final-qualification-01';volume=modal.Volume.from_name(volume_name,environment_name='staging',create_if_missing=True,version=2)
 image_data=json.loads((root/'paired-v3-69193/image.json').read_text());image=(modal.Image.from_id(image_data['image_id']).add_local_file(archive,'/opt/assessment-overlay.tar.gz',copy=True).run_commands('timeout 30s tar -xzf /opt/assessment-overlay.tar.gz -C /opt/clash'))
 app=modal.App('clash-reference-20261109-final-qualification-01');observer=app.function(image=image,cpu=(.25,.25),memory=(1024,1024),volumes={'/retained':volume},timeout=4290,startup_timeout=120,retries=0,max_containers=1,min_containers=0,buffer_containers=0,single_use_containers=True,serialized=True,include_source=False,name='allocation_observer')(observe)
 with modal.enable_output(),app.run(detach=True,environment_name='staging'):
  dump(target/'app.json',{'app_id':app.app_id});image.build(app);budget.attach(build,image.object_id);dump(target/'image.json',{'image_id':image.object_id,'overlay_sha256':sha(archive),'build_hold_retained':True})
  with volume.batch_upload() as upload:upload.put_directory(inputs,'/inputs')
  dump(private/'unblinded.json',{'at_unix':time.time(),'bank_sha256':sha(private/'bank.json'),'selection_sha256':sha(root/'selection-close.json'),'training_closed':True})
  began=time.time();sb=modal.Sandbox.create('python','-u','-m','simulators.clash.reference_training.assessment_worker','--root','/retained',app=app,image=image,cpu=(8,8),memory=(32768,32768),timeout=4200,block_network=True,secrets=[],volumes={'/retained':volume},include_oidc_identity_token=False,environment_name='staging',tags={'campaign':'reference-readiness-20261109-0251Z','role':'clash-rl-final-cpu'})
  budget.attach(worker,sb.object_id);p={'app_id':app.app_id,'sandbox_id':sb.object_id,'volume_name':volume_name,'operation':worker,'observer_operation':observer_op,'build_operation':build,'provider_timeout':4200,'cpu_only':True};dump(target/'provider.json',p)
  call=observer.spawn(app.app_id,sb.object_id,volume_name,4200,began);budget.attach(observer_op,call.object_id);p['observer_call_id']=call.object_id;dump(target/'provider.json',p);sb.detach()
 return p

def collect():
 import modal
 from tools.overnight_budget import CampaignBudget
 from simulators.generals.autoresearch.provider import configure_sandbox_waits,allocation_info
 configure_sandbox_waits();target=CAMPAIGN/'clash/rl/final-qualification-01';p=json.loads((target/'provider.json').read_text());receipt=allocation_info(p['app_id'],p['sandbox_id']);dump(target/'provider-current.json',receipt)
 if not receipt or not receipt.get('finished_at'):return {'running':True,'provider':receipt}
 budget=CampaignBudget('clash',path=LEDGER);row=next(x for x in budget.snapshot()['operations'] if x['id']==p['operation'])
 if row['status']!='settled':budget.settle_worker(p['operation'],receipt)
 volume=modal.Volume.from_name(p['volume_name'],environment_name='staging');destination=target/'collected';destination.mkdir(exist_ok=True)
 for entry in volume.iterdir('/',recursive=True):
  if entry.type.name!='FILE' or entry.path.lstrip('/').startswith('inputs/'):continue
  path=destination/entry.path.lstrip('/');path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():continue
  temp=path.with_name(path.name+'.download-part')
  with temp.open('wb') as f:
   for block in volume.read_file(entry.path):f.write(block)
  temp.replace(path)
 result=json.loads((destination/'result.json').read_text()) if (destination/'result.json').exists() else None;dump(target/'settlement.json',{'provider':receipt,'result_present':result is not None,'cpu_only':True,'complete':bool(result and result['complete'])});return {'complete':bool(result and result['complete']),'natural_games':result.get('natural_games') if result else None,'provider':receipt}

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=('launch','collect'));a=p.parse_args();print(json.dumps(launch() if a.action=='launch' else collect(),indent=2))
if __name__=='__main__':main()
