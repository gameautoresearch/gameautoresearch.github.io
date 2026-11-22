"""Freeze qualified public-information opposition before sampling any new private bank."""
import json,shutil,time,tarfile
from pathlib import Path
from .campaign import BASE,coordinate
from ..contract import HERE,RUNTIME,sha,identity,write


def main():
    required={'screen':'foul-play-eight-seed-screen-003','continuation':'retained-cap-continuation-004','runtime_fixtures':'runtime-fixtures-002','reference_admission':'reference-admission-001'}
    results={key:json.loads((BASE/'jobs'/name/'output/summary.json').read_text()) for key,name in required.items()}
    if results['screen']['natural_games']!=48 or results['screen']['faults'] or results['screen']['administrative_caps']:raise ValueError('Screen not complete')
    if not results['continuation']['all_original_prefixes_exact'] or not results['continuation']['all_reached_engine_outcome']:raise ValueError('Retained cases not admitted')
    if not results['runtime_fixtures']['passed'] or not results['reference_admission']['passed']:raise ValueError('Runtime/admission fixture incomplete')
    src=Path(__file__).parent;files={p.name:sha(p) for p in sorted(src.iterdir()) if p.suffix in ('.py','.cjs','.patch')}
    up=BASE/'upstream';build=json.loads((BASE/'jobs/pinned-search-build-002/output/summary.json').read_text())
    with tarfile.open(up/'foul-play-main.tar.gz') as tar:commit=tar.pax_headers['comment']
    policy={'id':'showdown-completion-v2-draft1','schema_version':2,'status':'frozen-qualified-development','max_adapter_phases':4096,'max_consecutive_unavailable_retries_per_seat':16,'native_engine_turn_limit':1000,
        'natural_ties':'Only authoritative Showdown tie protocol; turn-limit and Endless Battle Clause reasons retained','administrative_caps':'Unresolved; primary null; total score bounds/completion rate reported','retries':'No hidden trapping leaked in masks; only own updated request after engine rejection. No automatic fallback action.','counters':['engine_turns','adapter_phases','choice_attempts_by_seat','accepted_choices_by_seat','unavailable_choice_retries_by_seat'],'legacy_v1':'Original512-phase outcomes and source-bound artifacts unchanged; exact700value/protocol-prefix migration receipts required'}
    registry={'schema':'pokemon-public-information-opponent-registry-v2','rules_pin':'567c1cedddeaa2f35576005d3bfd526cf4f8a13e','format':'gen9randombattle','transition_authority':'Pinned CPU Showdown only','observation_interface':{'neural_values':700,'masked_actions':14,'search':'Own request and own player-channel history only; unknown opponent sets sampled from pinned public dataset, never true engine state'},
        'search':{'name':'foul-play-10ms-two-beliefs-v1','kind':'public-information-search','learned_reference':False,'foul_play_commit':commit,'foul_play_archive_sha256':sha(up/'foul-play-main.tar.gz'),'foul_play_license':'GPL-3.0','poke_engine_version':'0.0.48','original_sdist_sha256':sha(up/'poke_engine-0.0.48.tar.gz'),'engine_patch_sha256':sha(src/'poke-engine-reproducibility.patch'),'patched_source_archive_sha256':build['engine_archive_sha256'],
          'public_set_url':'https://pkmn.github.io/randbats/data/full/gen9randombattle.json','public_set_sha256':sha(up/'gen9randombattle-public.json'),'wheel_sha256':build['wheel_sha256'],
          'fixed_work_test':{'belief_samples':2,'iterations_per_sample':1000,'threads':1,'seed':'Own normalized public-input hash plus fixed adapter seed20261109; pinned Rust StdRng patch'},
          'production':{'belief_samples':2,'soft_search_ms_per_sample':10,'threads':1,'hard_end_to_end_choice_seconds':1,'whole_game_wall_safety_seconds':120,'time_caution':'Upstream time check is every1000iterations; measured overshoot retained. Hard process deadline remains authoritative.'},
          'isolation':{'uid':65533,'private_schedule_access':False,'network':False,'candidate_uid':65534,'volume_scope':'Only own private job subdirectory; no sibling jobs mounted'},'strength':'Qualified development search opponent at this reduced budget; expert/superhuman claims unqualified'},
        'baseline_opponents':['public-power','development-a short-trained canonical Policy','retained original GLM explicit final'],
        'public_information_difference':'Search sees full legal protocol/history and semantic species/move names, own request stats/items/abilities/PP and revealed opponent details. The700value neural encoder hashes/categories and bounded features lose some legal public history. No opponent-private requests, actual hidden team, RNG or state clones cross the interface.',
        'flags':{'harness_qualified':True,'learned_reference_qualified':False,'strong_panel_qualified':False,'superhuman_evidence':False,'normalized_reference_score':None},'qualification_receipts':{key:{'job':name,'summary_sha256':sha(BASE/'jobs'/name/'output/summary.json')} for key,name in required.items()}}
    manifest={'schema':'pokemon-evaluator-release-v2','created_at_unix':time.time(),'source_file_sha256':files,'canonical_runtime_file_sha256':{n:sha(HERE.parent/n) for n in RUNTIME},'cached_image_id':'im-JwNTde85zO3cAKA9qnqInr','termination_policy':policy,'opponent_registry':registry,'selection_closed':True,'final_bank_opened':False}
    manifest['release_sha256']=identity(manifest);release=BASE/'releases'/manifest['release_sha256'];(release/'source').mkdir(parents=True,exist_ok=False)
    for name in files:shutil.copyfile(src/name,release/'source'/name)
    write(release/'release.json',manifest,exclusive=True);write(release/'opponent-registry.json',registry,exclusive=True);write(release/'termination-policy.json',policy,exclusive=True)
    for path in release.rglob('*'):
        if path.is_file():path.chmod(0o444)
    write(BASE/'active-evaluator-release.json',{'path':str(release.resolve()),'release_sha256':manifest['release_sha256']},exclusive=True)
    coordinate('qualified-development-panel-frozen',evaluator_release={'path':str(release.resolve()),'release_sha256':manifest['release_sha256'],'opponent_registry_sha256':sha(release/'opponent-registry.json'),'termination_policy_sha256':sha(release/'termination-policy.json')},readiness_flags=registry['flags'],public_search_opponent={'source_archive':str((up/'foul-play-main.tar.gz').resolve()),'public_sets':str((up/'gen9randombattle-public.json').resolve()),'registry':str((release/'opponent-registry.json').resolve()),'source':str((release/'source').resolve()),'development_screen':str(BASE/'jobs/foul-play-eight-seed-screen-003/output/summary.json')},requested_changes=['No evaluatorGPUneeded; remaining1800GPUseconds stay unused unless parent explicitly reallocates role subcaps. Use reference_inference.load(directory, trusted_export_manifest_sha256) for standalone strict tensor admission in the cached image; reference_training module not needed for inference.'])
    print(release)
if __name__=='__main__':main()
