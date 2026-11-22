"""Explicit bounded CPU pilot/screen. Public development seeds only until selection closes."""
import argparse,json,os
from pathlib import Path
from .campaign import BASE
from .remote import launch
from ..contract import ROOT,sha,write
from ..bundle import FILES,verify


def main():
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--stage',choices=['pilot','screen'],required=True);a=p.parse_args()
 os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
 build=json.loads((BASE/'jobs/pinned-search-build-002/output/summary.json').read_text());up=BASE/'upstream';mount=Path('/data')/a.label/'input'
 files=[(up/'foul-play-main.tar.gz','foul-play.tar.gz'),(up/'gen9randombattle-public.json','public-sets.json')]
 opponents=[{'name':'power','kind':'power'}]
 if a.stage=='screen':
  name='development-a';path=BASE/'public-development-opponents'/name
  for file in ['weights.pt','opponent.json']:files.append((path/file,name+'/'+file))
  opponents.append({'name':name,'kind':'checkpoint','path':str(mount/name)})
  old=ROOT/'evidence/overnight-20261108-1533Z/pokemon';ledger=json.loads((old/'research-glm/gpu-ledger.json').read_text());selected=ledger['submissions'][-1];bundle=Path(selected['bundle']);verify(bundle,selected['bundle_sha256'])
  for file in (*FILES,'manifest.json'):files.append((bundle/file,'retained-glm/'+file))
  opponents.append({'name':'retained-glm','kind':'bundle','path':str(mount/'retained-glm')})
 spec={'wheel_sha256':build['wheel_sha256'],'wheels':'/data/pinned-search-build-002/output/wheels',
  'source_archive':str(mount/'foul-play.tar.gz'),'source_archive_sha256':sha(up/'foul-play-main.tar.gz'),
  'public_sets':str(mount/'public-sets.json'),'public_sets_sha256':sha(up/'gen9randombattle-public.json'),
  'opponents':opponents,'seeds':[900000101] if a.stage=='pilot' else list(range(900000111,900000119)),
  'mode':'production','iterations':1000,'time_ms':10,'samples':2,'inference_timeout_seconds':1.0,'game_wall_seconds':120,
  'max_adapter_phases':4096,'stage':a.stage,'information_boundary':'Only one player request and normalized own protocol history; distinct inference UIDs, private schedule/other-player data denied'}
 print(launch(a.label,'qualification',spec,files,seconds=240 if a.stage=='pilot' else 1200))
if __name__=='__main__':main()
