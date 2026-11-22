"""After selection closes, commit one new private bank and execute the frozen evaluator."""
import argparse,json,os,secrets,time
from pathlib import Path
from .campaign import BASE,coordinate
from .remote import launch
from ..contract import ROOT,write,sha,identity
from ..bundle import FILES,verify


def prepare():
    binding=json.loads((BASE/'active-evaluator-release.json').read_text());release=Path(binding['path'])
    registry=json.loads((release/'opponent-registry.json').read_text())
    selection={'release_sha256':binding['release_sha256'],'registry_sha256':sha(release/'opponent-registry.json'),
        'foul_play_search_configuration':registry['search']['name'],'opponents':['power','development-a','retained-glm'],
        'both_seats':True,'selection_closed_at_unix':time.time(),'rule':'All three precommitted development-screen opponents; no best policy/checkpoint selection using this bank'}
    private=BASE/'private-final-bank';private.mkdir(mode=0o700,exist_ok=False)
    write(private/'selection.json',selection,exclusive=True)
    seeds=[]
    while len(seeds)<8:
        value=3_000_000_000+secrets.randbelow(800_000_000)
        if value not in seeds:seeds.append(value)
    plan={'selection_sha256':sha(private/'selection.json'),'release_path':str(release.resolve()),'release_sha256':binding['release_sha256'],
        'registry_sha256':selection['registry_sha256'],'schedule_commitment':identity(seeds),'scheduled_games':48,'seed_clusters':8,'created_at_unix':time.time(),
        'interpretation':'Fresh private seed bank against the same weak fixed opponents; opponent/harness validation, not independent expert calibration'}
    plan['plan_sha256']=identity(plan);write(private/'seeds.json',{'seeds':seeds},exclusive=True);write(private/'plan.json',plan,exclusive=True)
    write(BASE/'final-bank-commitment.json',plan,exclusive=True)
    return plan


def execute():
    private=BASE/'private-final-bank';plan=json.loads((private/'plan.json').read_text());seeds=json.loads((private/'seeds.json').read_text())['seeds']
    if identity({k:v for k,v in plan.items() if k!='plan_sha256'})!=plan['plan_sha256'] or identity(seeds)!=plan['schedule_commitment']:raise ValueError('Fresh bank commitment changed')
    release=Path(plan['release_path'])
    if sha(release/'opponent-registry.json')!=plan['registry_sha256']:raise ValueError('Frozen opposition registry changed')
    write(private/'opened.json',{'plan_sha256':plan['plan_sha256'],'opened_at_unix':time.time()},exclusive=True)
    label='frozen-fresh-bank-001';mount=Path('/data')/label/'input';up=BASE/'upstream'
    build=json.loads((BASE/'jobs/pinned-search-build-002/output/summary.json').read_text())
    files=[(up/'foul-play-main.tar.gz','foul-play.tar.gz'),(up/'gen9randombattle-public.json','public-sets.json')]
    opponents=[{'name':'power','kind':'power'}]
    name='development-a';root=BASE/'public-development-opponents'/name
    for filename in ['weights.pt','opponent.json']:files.append((root/filename,name+'/'+filename))
    opponents.append({'name':name,'kind':'checkpoint','path':str(mount/name)})
    old=ROOT/'evidence/overnight-20261108-1533Z/pokemon';selected=json.loads((old/'research-glm/gpu-ledger.json').read_text())['submissions'][-1];bundle=Path(selected['bundle']);verify(bundle,selected['bundle_sha256'])
    for filename in (*FILES,'manifest.json'):files.append((bundle/filename,'retained-glm/'+filename))
    opponents.append({'name':'retained-glm','kind':'bundle','path':str(mount/'retained-glm')})
    spec={'wheel_sha256':build['wheel_sha256'],'source_archive':str(mount/'foul-play.tar.gz'),'source_archive_sha256':sha(up/'foul-play-main.tar.gz'),
        'public_sets':str(mount/'public-sets.json'),'public_sets_sha256':sha(up/'gen9randombattle-public.json'),'opponents':opponents,'seeds':seeds,
        'mode':'production','iterations':1000,'time_ms':10,'samples':2,'inference_timeout_seconds':1.0,'game_wall_seconds':120,'max_adapter_phases':4096,
        'stage':'frozen-private-final-bank','plan_sha256':plan['plan_sha256'],'information_boundary':'Own request and own player-channel history only; private schedule denied to separate inference UIDs'}
    coordinate('frozen-fresh-bank-running',fresh_bank={'plan_sha256':plan['plan_sha256'],'schedule_commitment':plan['schedule_commitment'],'scheduled_games':48,'private_seeds_not_shared':True})
    return launch(label,'qualification',spec,files,seconds=1200,source_release=release)


def main():
 os.environ.setdefault('MODAL_DISABLE_API_PROXY','1');p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute']);a=p.parse_args()
 result=prepare() if a.command=='prepare' else execute();print(json.dumps(result))
if __name__=='__main__':main()
