"""Evidence-bound full-format admission; no mutable completion switch.

All receipts must name the same immutable candidate binary and runtime source.
Certification is separate from compilation and never launches experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .bridge import HERE, Native
from .prepare_runtime import format_id

EXTERNAL=('network.py','efficient.py','env.py','rl.py','autoresearch/starter/learner.py','autoresearch/contract.py')
# The six-Pokémon plan; a certificate for another team size names its own predeclared plan.
PLAN='reports/pokemon/native-cpu-qualification.json'
# Receipts that play generated random battles and so must all be of the certified team size.
SIZED=('ppo_shadow','resume','vector_api','ordinary_cli_resume','performance')
BANKS={'development_random':2048,'development_power':2048,'development_learned':4096,
       'independent_random':1366,'independent_power':1366,'independent_learned':1364}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_certificate(receipt,sources):
    if receipt.get('schema_version')!=1 or receipt.get('full_gen9_admission')is not True:raise ValueError('Complete native admission certificate required')
    if receipt['qualified_sources']!=sources:raise ValueError('Native source changed after qualification')
    for name,value in receipt['external_runtime_sha256'].items():
        if digest(HERE.parent/name)!=value:raise ValueError(f'External native runtime source changed: {name}')
    for name,value in receipt['receipt_sha256'].items():
        if hashlib.sha256(receipt['receipt_contents'][name].encode()).hexdigest()!=value:raise ValueError(f'Qualification evidence changed: {name}')
    if digest(HERE.parents[2]/receipt.get('qualification_plan',PLAN))!=receipt['qualification_plan_sha256']:raise ValueError('Native qualification declaration changed')
    if receipt.get('format','gen9randombattle')!=format_id(receipt.get('team_size',6)):raise ValueError('Certified format and team size disagree')
    return receipt


def certify(output,runtime_path,evidence_path,plan=PLAN):
    output=Path(output);runtime_path=Path(runtime_path);runtime=json.loads(runtime_path.read_bytes());team_size=runtime.get('team_size',6)
    declared=json.loads((HERE.parents[2]/plan).read_bytes())
    if declared.get('team_size',6)!=team_size or declared.get('format','gen9randombattle')!=format_id(team_size):raise ValueError('Qualification plan declares another format')
    library=runtime_path.parent/runtime['library'];manifest=json.loads((library.parent/'build.json').read_bytes())
    if digest(library)!=runtime['sha256']['library']or digest(library)!=manifest['binary_sha256']:raise ValueError('Candidate binary changed')
    with Native(library)as native:signature=native.source_signature
    receipts=json.loads(Path(evidence_path).read_bytes());expected=set(BANKS)|{'matrix','historical','ppo_shadow','resume','vector_api','ordinary_cli_resume','performance'}
    if set(receipts)!=expected:raise ValueError('Admission evidence map differs from predeclared checks')
    loaded={name:json.loads(Path(path).read_bytes())for name,path in receipts.items()}
    for name,row in loaded.items():
        if row.get('library_sha256')!=manifest['binary_sha256']or row.get('source_signature')!=signature:raise ValueError(f'{name}: candidate binary/source identity differs')
    for name in (*BANKS,*SIZED):
        if loaded[name].get('team_size',6)!=team_size:raise ValueError(f'{name}: receipt is for another team size')
    for name,count in BANKS.items():
        row=loaded[name]
        if row.get('cases')!=count or row.get('exact')!=count or row.get('natural_oracle_games',row.get('oracle_natural_games'))!=count:raise ValueError(f'{name}: complete unfiltered natural bank required')
        if row.get('unsupported_faults')!=0 or row.get('bounded_oracle_games')!=0 or row.get('schedule_violations')!=0:raise ValueError(f'{name}: unsupported, capped or substituted policy actions')
    for name,count in(('matrix',1464),('historical',7055)):
        row=loaded[name]
        if row.get('cases')!=count or row.get('exact')!=count:raise ValueError(f'{name}: complete interaction regression required')
    row=loaded['ppo_shadow']
    if row.get('exact_phases',0)<1024 or row.get('faint_reward')!=0 or row.get('cuda_initialized'):raise ValueError('Original terminal-reward CPU PPO shadow required')
    if len(row['rows'])<32 or sum(r['samples']for r in row['rows'])<100000 or sum(r['natural']for r in row['rows'])<1000 or any(r['faults']or r['truncations']for r in row['rows']):raise ValueError('Incomplete natural-game backward shadow')
    row=loaded['resume']
    if not(row.get('exact_serial_parallel_learning')and row.get('exact_model_optimizer_rng_environment_and_cursor_resume')):raise ValueError('Complete source-bound thread/resume qualification required')
    row=loaded['vector_api']
    if not row.get('exact_export_logits_and_values')or not all(r.get('exact_all_700')and r.get('exact_complete_vector_restore')and r.get('exact_slot_resets')for r in row['rows']):raise ValueError('Ordinary vector/export parity required')
    row=loaded['ordinary_cli_resume']
    if not row.get('exact_complete_cli_checkpoint_resume')or row.get('faint_reward')!=0 or not row.get('original_recipe'):raise ValueError('Ordinary terminal-reward recipe/resume qualification required')
    row=loaded['performance']
    required={'simulator','observation','opponent','inference','transfers','GAE','PPO','checkpoint','process','allocated_worker'}
    if not required<=set(row.get('phases',{}))or not row.get('complete_unfiltered_natural_games')or not row.get('retained_learned_policy')or not row.get('owned_remote_workers_stopped'):raise ValueError('Full-workload phase/lifecycle performance receipt required')
    # A remote performance worker's reserved seconds stay within the same 1,200-second ceiling, on the L4 of the
    # six-Pokémon certificate or on any later declared GPU (L40S is the repository default since 2026-11-11).
    if row.get('conservative_total_remote_reserved_seconds',row.get('conservative_total_remote_reserved_L4_seconds',1201))>1200:raise ValueError('Remote continuation ceiling exceeded')
    if not isinstance(row.get('hardware',{}).get('gpu'),str):raise ValueError('Performance receipt must name its GPU')
    receipt=dict(schema_version=1,full_gen9_admission=True,backend='native_cpp_cpu',format=format_id(team_size),showdown_pin='567c1cedddeaa2f35576005d3bfd526cf4f8a13e',
        qualified_binary_sha256=manifest['binary_sha256'],qualified_source_signature=signature,qualified_sources=manifest['source_sha256'],
        external_runtime_sha256={name:digest(HERE.parent/name)for name in EXTERNAL},runtime_data_sha256={k:runtime['sha256'][k]for k in('generators','constructors','chart')},
        receipt_sha256={name:digest(path)for name,path in receipts.items()},receipt_contents={name:Path(path).read_text()for name,path in receipts.items()},receipt_source_paths={name:str(Path(path).resolve())for name,path in receipts.items()},
        qualification_plan_sha256=digest(HERE.parents[2]/plan),
        **({} if team_size==6 else {'team_size':team_size,'qualification_plan':plan}),
        development_games=8192,independent_public_games=4096,minimum_throughput_required=None)
    verify_certificate(receipt,manifest['source_sha256'])
    with output.open('x')as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
    output.chmod(0o444);print(json.dumps(dict(full_gen9_admission=True,certificate=str(output),sha256=digest(output)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--runtime',required=True);p.add_argument('--evidence-map',required=True);p.add_argument('--plan',default=PLAN);a=p.parse_args();certify(a.output,a.runtime,a.evidence_map,a.plan)
