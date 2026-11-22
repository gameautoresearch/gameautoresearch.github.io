"""CPU-only private retained raw-state analysis; never launches a GPU or rebundles."""
from pathlib import Path
import argparse,hashlib,json,sys,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import modal
VOLUME_NAME='clash-structural-artifacts-20261108'
if modal.is_local():
    from structural_modal import image,BUNDLE
else:
    # The container mounts this standalone helper, not its sibling launcher.
    # Actual execution still uses the host-selected frozen structural image.
    image=modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu24.04')
    BUNDLE=Path('/opt/upload.tar.gz')
volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=False)
app=modal.App('clash-native-state-analysis-20261108')


@app.function(image=image,cpu=(1,1),memory=(4096,4096),volumes={'/artifacts':volume},max_containers=1,min_containers=0,scaledown_window=2,retries=0,timeout=600)
def analyze(run_id:str,analysis_id:str,expected_bundle_sha:str):
    import json,time,tarfile,tempfile,hashlib,subprocess,re,shutil,os
    from pathlib import Path
    started=time.perf_counter();cpu_started=time.process_time()
    import torch,numpy as np
    bundle=Path('/opt/upload.tar.gz');digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
    if digest!=expected_bundle_sha:raise ValueError('frozen source archive hash differs')
    folder=Path('/artifacts')/run_id;out=folder/analysis_id;out.mkdir(parents=False,exist_ok=False)
    receipt={'gpu_count':0,'cpu_physical_cores':1,'ram_gib':4,'run_id':run_id,'analysis_id':analysis_id,'frozen_bundle_sha256':digest,'scope':'CPU-only retained checkpoint byte analysis; no simulation/training, no GPU allocation, no source archive rewrite','rate_usd_per_second':.0000131+4*.00000222}
    try:
        with tempfile.TemporaryDirectory(prefix='clash-layout-') as temporary:
            root=Path(temporary)
            # Extract native headers only. No generated game data/assets needed.
            with tarfile.open(bundle) as archive:
                members=[m for m in archive.getmembers() if m.isfile() and m.name.startswith('integrations/clash/native/') and m.name.endswith(('.hpp','.cuh'))]
                archive.extractall(root,members=members,filter='data')
            native=root/'integrations/clash/native';world=(native/'world.hpp').read_text();prefix=world.split('struct NativeWorld {',1)[1].split('CN_HD',1)[0]
            world_names='spec holder match commands random entities movements targets timers health buffs projectiles areas combatTargets rows projectileRows areaCard chainUse removeRequested chains hits hitCount hitSerial present presentCount overlay gridCurrent arenaGrid costs routeScratch savedRouteScratch spatial kings kingSlots lastRefusal lastRunRefusal illegalTotal illegalNow publicPending publicPendingCount episodeStep firstSide faultCode matchEnded hitsHeld justReset observer observationEntities observationOwn observationClock observationCount renderedObs renderedInfo renderedMasks renderedRewards rendered spatialResultScratch spatialKeysScratch placementScratch'.split()
            world_names=[name for name in world_names if re.search(r'\b'+name+r'\b',prefix)]
            entity_names='id kind side row x y hp max_hp shield state level target_id deploy_ms attack_ms load_ms speed_percent hit_speed_percent ability_ran_tick aim_x aim_y time_left_ms radius deploy_index alive air building king clone awake no_move no_attack charged hidden invisible underground waiting has_hp bar_always bar_damaged bar_rolling ground_trail buff_count buffs'.split()
            source=['#include <cstdio>','#include <cstddef>','#include "'+str(native/'world.hpp')+'"','using namespace clash_native;','int main(){']
            for group,type_,names in [('world','NativeWorld',world_names),('observation_entity','ObservationEntity',entity_names)]:
                source.append('std::printf("size\\t'+group+'\\t%zu\\t%zu\\n",sizeof('+type_+'),alignof('+type_+'));')
                for name in names:source.append('std::printf("'+group+'\\t'+name+'\\t%zu\\t%zu\\n",offsetof('+type_+','+name+'),sizeof((('+type_+'*)nullptr)->'+name+'));')
            for name in ('tiles','current','previous'):source.append('std::printf("grid\\t'+name+'\\t%zu\\t%zu\\n",offsetof(GridView,'+name+'),sizeof(((GridView*)nullptr)->'+name+'));')
            source.append('}');cpp=root/'layout.cpp';cpp.write_text('\n'.join(source)+'\n');binary=root/'layout'
            compiler=shutil.which('g++') or shutil.which('clang++')
            command=[compiler,'-std=c++20',str(cpp),'-o',str(binary)];subprocess.run(command,check=True,capture_output=True,text=True)
            output=subprocess.check_output([str(binary)],text=True);(out/'layout.tsv').write_text(output)
            layout={};sizes={}
            for line in output.splitlines():
                group,name,offset,size=line.split('\t')
                if group=='size':sizes[name]={'size':int(offset),'alignment':int(size)}
                else:layout.setdefault(group,[]).append({'field':name,'offset':int(offset),'size':int(size)})
            expected=json.loads((folder/'resume-qualification/pointer-layout.json').read_text());wb=sizes['world']['size'];assert wb==expected['world_bytes'],(wb,expected)
            top={row['field']:row for row in layout['world']};grid={row['field']:row for row in layout['grid']}
            actual_pointers=[top['spec']['offset']]+[top['arenaGrid']['offset']+grid[name]['offset'] for name in ('tiles','current','previous')]
            assert actual_pointers==expected['pointer_offsets'],(actual_pointers,expected)
            entity=sizes['observation_entity']['size'];members=sorted(layout['observation_entity'],key=lambda x:x['offset']);padding=[];end=0
            for row in members:
                if row['offset']>end:padding.append([end,row['offset']])
                end=max(end,row['offset']+row['size'])
            if end<entity:padding.append([end,entity])
            layout_report={'sizes':sizes,'fields':layout,'observation_entity_padding':padding,'pointer_layout_verified':expected,'compiler':subprocess.check_output([compiler,'--version'],text=True),'command':command}
            (out/'layout.json').write_text(json.dumps(layout_report,indent=2)+'\n')
            checkpoints={key:folder/('resume-qualification/live/'+key+'/checkpoint.pt') for key in ('full','resumed')}
            loaded={key:torch.load(path,map_location='cpu',weights_only=False) for key,path in checkpoints.items()}
            a=loaded['full']['engine_state'].numpy();b=loaded['resumed']['engine_state'].numpy();assert a.shape==b.shape and a.shape[1]==wb
            all_rows=[];groups={};unknown=[];pointer_count=0;entity_padding_count=0
            for lane,offset in np.argwhere(a!=b):
                lane=int(lane);offset=int(offset);row={'lane':lane,'offset':offset,'full':int(a[lane,offset]),'resumed':int(b[lane,offset])}
                pointer=next((p for p in actual_pointers if p<=offset<p+expected['pointer_bytes']),None)
                field=next((r for r in layout['world'] if r['offset']<=offset<r['offset']+r['size']),None)
                row['world_field']=field['field'] if field else 'NativeWorld padding';row['field_offset']=offset-field['offset'] if field else None
                if pointer is not None:row['classification']='rebound pointer';pointer_count+=1
                elif field and field['field']=='observationEntities':
                    index,within=divmod(offset-field['offset'],entity);row.update(entity_index=index,entity_byte_offset=within)
                    member=next((r for r in members if r['offset']<=within<r['offset']+r['size']),None)
                    row['entity_member']=member['field'] if member else None
                    if any(lo<=within<hi for lo,hi in padding):row['classification']='compiler-proven ObservationEntity padding';entity_padding_count+=1
                    else:row['classification']='semantic/unknown byte';unknown.append(row)
                else:row['classification']='semantic/unknown byte';unknown.append(row)
                key=row['world_field']+' / '+row['classification'];groups[key]=groups.get(key,0)+1;all_rows.append(row)
            (out/'all-byte-differences.json').write_text(json.dumps(all_rows,indent=2)+'\n')
            summary={'status':'classified','run_id':run_id,'analysis_id':analysis_id,'frozen_bundle_sha256':digest,'checkpoint_sha256':{key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in checkpoints.items()},'world_bytes':wb,'lanes':a.shape[0],'raw_differing_bytes':len(all_rows),'rebound_pointer_bytes':pointer_count,'compiler_proven_observation_padding_bytes':entity_padding_count,'unknown_or_semantic_bytes':len(unknown),'all_nonpointer_differences_proven_observation_padding':bool(all_rows) and not unknown,'groups':groups,'observation_entity_padding':padding,'first_differences':all_rows[:24],'unknown_differences':unknown[:60],'normalization_applied':False}
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    finally:
        receipt.update(function_body_seconds=time.perf_counter()-started,cpu_process_seconds=time.process_time()-cpu_started,conservative_load_headroom_seconds=60)
        receipt['estimated_usd_with_headroom']=(receipt['function_body_seconds']+60)*receipt['rate_usd_per_second'];(out/'cpu-analysis-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');volume.commit()
    receipt['function_seconds_including_commit']=time.perf_counter()-started
    return {'summary':summary,'receipt':receipt,'volume':VOLUME_NAME,'remote_directory':run_id+'/'+analysis_id}


@app.local_entrypoint()
def main(run_id:str='native-optimized-1',analysis_id:str='raw-state-analysis-1'):
    import re
    from modal.config import config
    if config.get('environment')!='staging':raise RuntimeError('staging only')
    if not re.fullmatch('[a-z0-9-]+',run_id) or not re.fullmatch('[a-z0-9-]+',analysis_id):raise ValueError('new safe artifact ID required')
    out=HERE.parents[2]/'evidence/clash-implementation/native-training-20261108'/analysis_id;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();call=analyze.spawn(run_id,analysis_id,hashlib.sha256(BUNDLE.read_bytes()).hexdigest());(out/'submitted.json').write_text(json.dumps({'function_call_id':call.object_id,'gpu_count':0,'run_id':run_id,'analysis_id':analysis_id,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n')
    result=call.get();result['controller_seconds']=time.perf_counter()-started;result['function_call_id']=call.object_id;(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
