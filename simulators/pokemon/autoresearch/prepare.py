"""Offline immutable native release: frozen worker source, admitted engine data, evaluator inputs, workspace.

Nothing here allocates compute or creates private evaluation seeds. The
retained inputs are read from local evidence and verified against the digests
their own records declare before they are copied into the release.
"""
from pathlib import Path
import ast
import json
import os
import shutil

from .contract import HERE, ROOT, PIN, sha, identity, write
from .bundle import FILES, verify as original_verify

NATIVE = ROOT/'evidence/pokemon-implementation/native-cpu-completion-20261109'
EVALUATOR = ROOT/'evidence/strong-bots-20261109-0516Z/pokemon/evaluator'
# The three-Pokémon form: its admitted runtime (certificate + public engine data) and its two learned opponents.
NATIVE_3V3 = ROOT/'evidence/pokemon-3v3-20261113/admitted-runtime'
PANEL_3V3 = ROOT/'evidence/pokemon-3v3-20261113/panel'
LEARNED_3V3 = (('retained-3v3-a','learned-3v3-a'),('retained-3v3-b','learned-3v3-b'))
TEAM_SIZE = json.loads((HERE/'benchmark.json').read_bytes())['task']['team_size']
HARDWARE = dict(gpu='L40S', physical_cpu_cores=4, memory_gib=16)


def audit(root):
    root=Path(root);manifest=json.loads((root/'release.json').read_text())
    if identity({k:v for k,v in manifest.items() if k!='release_sha256'})!=manifest['release_sha256']:
        raise ValueError('Release manifest identity changed')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='release.json'}
    if actual!=set(manifest['files']):raise ValueError('Release file set changed')
    for name,value in manifest['files'].items():
        path=root/name
        if path.is_symlink() or sha(path)!=value:raise ValueError('Release source changed: '+name)
    return manifest


def retained(path):
    """A path recorded by an earlier local run, wherever this checkout now lives."""
    path=Path(path)
    if path.exists():return path
    text=str(path)
    if '/evidence/' not in text:raise FileNotFoundError('Retained evaluator input is missing: '+text)
    moved=ROOT/text[text.index('/evidence/')+1:]
    if not moved.exists():raise FileNotFoundError('Retained evaluator input is missing: '+text)
    return moved


def checked(source,target,expected):
    source=Path(source)
    if source.is_symlink() or not source.is_file() or sha(source)!=expected:
        raise ValueError('Retained input differs from its recorded digest: '+source.name)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)


def certified_team_size(certificate):
    if certificate.get('team_size',6)!=TEAM_SIZE:
        raise ValueError(f"Native certificate qualifies {certificate.get('team_size',6)}-Pokémon battles; this benchmark is {TEAM_SIZE}")


def native_runtime(target):
    """The admission certificate and the public engine data it certifies."""
    if TEAM_SIZE==6:runtime=NATIVE/'admitted-runtime-1';certificate_path=NATIVE/'admission-1.json'
    else:runtime=NATIVE_3V3;certificate_path=NATIVE_3V3/'admission.json'
    manifest=json.loads((runtime/'runtime.json').read_bytes())
    target.mkdir(parents=True)
    certificate=json.loads(certificate_path.read_bytes())
    if certificate.get('full_gen9_admission') is not True or certificate['showdown_pin']!=PIN:
        raise ValueError('Complete native admission for the pinned rules required')
    certified_team_size(certificate)
    # The certificate pins these files of this checkout; a worker re-verifies them from its own source tree.
    for name,value in certificate['external_runtime_sha256'].items():
        if sha(HERE.parent/name)!=value:raise ValueError('Native-certified source changed: '+name)
    shutil.copyfile(certificate_path,target/'native-admission.json')
    for name in ('generators','constructors','chart'):
        if certificate['runtime_data_sha256'][name]!=manifest['sha256'][name]:raise ValueError('Admitted engine data differs: '+name)
        checked(runtime/manifest[name],target/(name+'.json'),manifest['sha256'][name])


def evaluator(target):
    """Frozen four-opponent panel and the pinned public-information search opponent's inputs."""
    from .runtime.sdk import BENCHMARK_ID
    target.mkdir(parents=True)
    source=EVALUATOR/'v4-registry-candidate-002.json';registry=json.loads(source.read_bytes())
    deep=json.loads((EVALUATOR/'retained-deepseek1729.json').read_bytes())
    glm=next(x for x in registry['panel'] if x['name']=='retained-glm128')
    panel=[dict(kind='power',name='power')]
    for index,value in enumerate((deep,glm),1):
        bundle=retained(value['path']);manifest=original_verify(bundle,value['bundle_sha256'])
        for name in (*FILES,'manifest.json'):
            (target/f'peer-{index}').mkdir(exist_ok=True);shutil.copyfile(bundle/name,target/f'peer-{index}'/name)
        original_verify(target/f'peer-{index}',manifest['bundle_sha256'])
        panel.append(dict(kind='sealed-original-bundle',name=value['name'],path=f'peer-{index}',bundle_sha256=manifest['bundle_sha256']))
    panel.append(dict(kind='search',name='foul-play-two-beliefs-10ms'))
    request=dict(registry['evaluator_request'])
    checked(retained(request['source_archive']),target/'foul-play.tar.gz',request['source_archive_sha256'])
    checked(retained(request['public_sets']),target/'public-sets.json',request['public_sets_sha256'])
    for name,digest in request['wheel_sha256'].items():
        checked(retained(request['wheels'])/name,target/'wheels'/name,digest)
    request.update(source_archive='foul-play.tar.gz',public_sets='public-sets.json',wheels='wheels')
    # The evaluator refuses to run unless its qualified files are byte-identical in the worker's source tree.
    for folder,key in (('evaluation_v2','qualified_core_file_sha256'),('strong_bots','search_extension_file_sha256')):
        for name,digest in request[key].items():
            if sha(HERE/folder/name)!=digest:raise ValueError(f'Qualified evaluator file changed: {folder}/{name}')
    registry.update(benchmark_id=BENCHMARK_ID,panel=panel,evaluator_request=request,
        qualification='Runtime and search qualified; panel difficulty and learned reference strength unqualified',
        original_registry_sha256=sha(source),no_strength_normalization=True)
    write(target/'registry.json',registry)


def native_runtime_from(base,target):
    """The admission certificate and engine data sealed in an earlier release, re-checked against this checkout."""
    source=Path(base)/'frozen/runtime';target.mkdir(parents=True)
    certificate=json.loads((source/'native-admission.json').read_bytes())
    if certificate.get('full_gen9_admission') is not True or certificate['showdown_pin']!=PIN:
        raise ValueError('Complete native admission for the pinned rules required')
    certified_team_size(certificate)
    for name,value in certificate['external_runtime_sha256'].items():
        if sha(HERE.parent/name)!=value:raise ValueError('Native-certified source changed: '+name)
    shutil.copyfile(source/'native-admission.json',target/'native-admission.json')
    for name in ('generators','constructors','chart'):
        checked(source/(name+'.json'),target/(name+'.json'),certificate['runtime_data_sha256'][name])


def evaluator_from(base,target):
    """The frozen panel and search-opponent inputs sealed in an earlier release, every digest re-checked."""
    from .runtime.sdk import BENCHMARK_ID
    source=Path(base)/'private/assessment';target.mkdir(parents=True)
    registry=json.loads((source/'registry.json').read_bytes())
    if registry.get('benchmark_id')!=BENCHMARK_ID:
        if TEAM_SIZE!=6:return evaluator_for_team_size(source,registry,target)
        raise ValueError('Base evaluator registry belongs to another benchmark')
    for entry in registry['panel']:
        if entry['kind']=='sealed-original-bundle':
            shutil.copytree(source/entry['path'],target/entry['path'])
            original_verify(target/entry['path'],entry['bundle_sha256'])
        elif entry['kind']=='checkpoint':  # a smaller-team release's frozen self-play opponents
            checked(source/entry['path']/'opponent.json',target/entry['path']/'opponent.json',entry['metadata_sha256'])
            meta=json.loads((target/entry['path']/'opponent.json').read_bytes())
            if meta.get('team_size',6)!=TEAM_SIZE:raise ValueError(f"{entry['name']} was trained for another team size")
            checked(source/entry['path']/'weights.pt',target/entry['path']/'weights.pt',meta['weights_sha256'])
    request=registry['evaluator_request']
    checked(source/request['source_archive'],target/request['source_archive'],request['source_archive_sha256'])
    checked(source/request['public_sets'],target/request['public_sets'],request['public_sets_sha256'])
    for name,digest in request['wheel_sha256'].items():
        checked(source/request['wheels']/name,target/request['wheels']/name,digest)
    for folder,key in (('evaluation_v2','qualified_core_file_sha256'),('strong_bots','search_extension_file_sha256')):
        for name,digest in request[key].items():
            if sha(HERE/folder/name)!=digest:raise ValueError(f'Qualified evaluator file changed: {folder}/{name}')
    shutil.copyfile(source/'registry.json',target/'registry.json')
    actual={str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}
    expected={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    if actual!=expected:raise ValueError('Base evaluator inputs hold unexpected files: '+', '.join(sorted(expected-actual)))


def evaluator_for_team_size(source,base,target):
    """A smaller-team panel from a six-Pokémon release's evaluator inputs (every digest re-checked).

    Foul Play's pinned source, public sets and wheels are reused. The two learned opponents are this form's
    frozen self-play checkpoints. The evaluator files changed only to carry the team size (worker argument,
    Foul Play beliefs sized to the published team), so their digests are bound afresh and the six-Pokémon
    digests are kept as `inherited_from`; the new panel still needs its own live qualification."""
    from .runtime.sdk import BENCHMARK_ID
    from .contract import RUNTIME
    request=dict(base['evaluator_request'])
    checked(source/request['source_archive'],target/'foul-play.tar.gz',request['source_archive_sha256'])
    checked(source/request['public_sets'],target/'public-sets.json',request['public_sets_sha256'])
    for name,digest in request['wheel_sha256'].items():
        checked(source/request['wheels']/name,target/'wheels'/name,digest)
    panel=[dict(kind='power',name='power')]
    for index,(name,folder) in enumerate(LEARNED_3V3,1):
        origin=PANEL_3V3/folder;meta=json.loads((origin/'opponent.json').read_bytes())
        if meta.get('team_size')!=TEAM_SIZE:raise ValueError(f'{name} was trained for another team size')
        checked(origin/'weights.pt',target/f'peer-{index}'/'weights.pt',meta['weights_sha256'])
        shutil.copyfile(origin/'opponent.json',target/f'peer-{index}'/'opponent.json')
        panel.append(dict(kind='checkpoint',name=name,path=f'peer-{index}',metadata_sha256=sha(target/f'peer-{index}'/'opponent.json')))
    panel.append(dict(kind='search',name='foul-play-two-beliefs-10ms'))
    inherited={key:request[key] for key in ('qualified_core_file_sha256','search_extension_file_sha256','qualified_core_release_sha256','search_extension_release_sha256')}
    request.update(source_archive='foul-play.tar.gz',public_sets='public-sets.json',wheels='wheels',team_size=TEAM_SIZE,
        qualified_core_file_sha256={name:sha(HERE/'evaluation_v2'/name) for name in base['evaluator_request']['qualified_core_file_sha256']},
        search_extension_file_sha256={name:sha(HERE/'strong_bots'/name) for name in base['evaluator_request']['search_extension_file_sha256']},
        inherited_from=inherited)
    registry=dict(base,benchmark_id=BENCHMARK_ID,panel=panel,evaluator_request=request,team_size=TEAM_SIZE,
        canonical_runtime_file_sha256={name:sha(HERE.parent/name) for name in RUNTIME},
        qualification=f'{TEAM_SIZE}-Pokémon panel and team-size evaluator changes not yet live-qualified; learned opponents '
                      'are frozen self-play checkpoints of unqualified strength',
        derived_from_registry_sha256=sha(source/'registry.json'),no_strength_normalization=True)
    write(target/'registry.json',registry)


# Reference learning code in the read-only simulator sources: optimizers, backward passes, network modules and
# the project's PPO development scripts. An empty-starter release withholds every such file from the box.
LEARNING_CODE = (r'torch\.optim|\.backward\(|decision_gae|clip_grad|nn\.Module|\bPPO\b|\bppo\b|policy gradient'
                 r'|Policy\(hidden=')
NETWORK = 'simulators/pokemon/network.py'
# Where the worker's unprivileged trainer imports that module from (`bootstrap.PUBLIC_SOURCE_FILES`).
PUBLIC_NETWORK = 'integrations/pokemon/network.py'


def network_view(path=None):
    """`network.py` as an empty-starter box shows it: the observation keys and batching helpers the runtime
    exposes (`efficient.py` imports them), without its docstring and the reference `Policy` network, which is the
    PPO starter's architecture. Workers still run the full module."""
    text=Path(path or ROOT/NETWORK).read_text();tree=ast.parse(text)
    lines=text.splitlines(keepends=True);drop=[]
    for node in tree.body:
        docstring=isinstance(node,ast.Expr) and isinstance(getattr(node,'value',None),ast.Constant) and isinstance(node.value.value,str)
        if isinstance(node,ast.ClassDef) or (docstring and node is tree.body[0]):
            drop.append(((node.decorator_list[0].lineno if getattr(node,'decorator_list',None) else node.lineno)-1,node.end_lineno))
    for start,end in sorted(drop,reverse=True):del lines[start:end]
    return '# integrations/pokemon/network.py: the observation keys and helpers of the runtime.\n'+''.join(lines)


def reduce_public_network(frozen,manifest):
    """An empty-starter release's workers get `network_view()` as the public `network.py` their unprivileged
    trainer imports, so recipe code cannot reach the reference `Policy` there either. Only the public import copy
    changes: the protected tree (engine admission, evaluator, arena) keeps the certified module byte for byte, and
    the release's source manifest and pack are rebuilt to bind the reduced copy."""
    from .runtime import bootstrap
    path=Path(frozen)/'public-source'/PUBLIC_NETWORK
    path.chmod(0o644);path.write_text(network_view());path.chmod(0o444)
    public={name:sha(Path(frozen)/'public-source'/name) for name in manifest['public_files']}
    manifest=dict(manifest,public_files=public,source_sha256=identity(dict(protected=manifest['files'],public=public)))
    write(Path(frozen)/'source-manifest.json',manifest)
    return manifest,bootstrap.packed_source(frozen,manifest)


# What the researcher starts from. `ppo` is the working clipped-PPO starter; `empty` is boilerplate only (run
# plumbing, a learner with no update and a one-layer placeholder policy), with its own brief and recipes, and a
# box without the reference learning code (`LEARNING_CODE`, `network_view`).
STARTERS = {'ppo': dict(source='starter', recipes={'starter': 'starter', 'smoke': 'smoke'}, brief='brief.md'),
            'empty': dict(source='starter_empty', recipes={'starter': 'empty', 'smoke': 'empty-smoke'},
                          brief='brief-empty.md', withhold=LEARNING_CODE, views={NETWORK: network_view})}


def prepare(output,base_release=None,starter='ppo'):
    """A new release. `base_release` reuses the native runtime and evaluator inputs sealed in an earlier
    release (audited, every digest re-checked) when the local evidence they came from is gone. `starter`
    chooses the researcher's starting code (`STARTERS`); the release seals its brief as `public/brief.md`."""
    from .runtime import bootstrap, sdk
    if starter not in STARTERS:raise ValueError('starter must be one of: '+', '.join(STARTERS))
    chosen=STARTERS[starter]
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    public=out/'public';public.mkdir()
    for name in sdk.EDITABLE:
        target=public/'agent'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(sdk.HERE/chosen['source']/name,target)
    for name,native in chosen['recipes'].items():
        target=public/'recipes'/f'{name}.json';target.parent.mkdir(exist_ok=True)
        sdk.validate_recipe(json.loads((HERE/'recipes'/f'{native}.json').read_bytes()))
        shutil.copyfile(HERE/'recipes'/f'{native}.json',target)
    shutil.copyfile(HERE/'task'/chosen['brief'],public/'brief.md')
    for name in ('benchmark.json','CONTRACT.md'):
        shutil.copyfile(HERE/name,public/name)
    shutil.copyfile(sdk.HERE/'GRAPH_CONTRACT.md',public/'GRAPH_CONTRACT.md')
    # The researcher sees only the canonical workspace built from this public tree.
    from harness.workspace import stage as stage_workspace
    stage_workspace('pokemon',public,withhold=chosen.get('withhold'),
        views={name:make() for name,make in chosen.get('views',{}).items()})
    # One source snapshot for every worker of this release; later jobs stage exactly these bytes.
    frozen=out/'frozen';frozen.mkdir()
    _,manifest=bootstrap.source_snapshot(frozen)
    pack=bootstrap.packed_source(frozen,manifest)
    if chosen.get('views'):manifest,pack=reduce_public_network(frozen,manifest)
    base=audit(Path(base_release).resolve()) if base_release else None
    # A base release of another team size (the six-Pokémon releases, for the 3v3 form) lends only its evaluator
    # inputs; the native runtime then comes from this form's local certificate.
    base_native=base and json.loads((Path(base_release)/'frozen/runtime/native-admission.json').read_bytes()).get('team_size',6)==TEAM_SIZE
    (native_runtime_from(Path(base_release),frozen/'runtime') if base_native else native_runtime(frozen/'runtime'))
    protected={name:sha(frozen/'source/integrations/pokemon/autoresearch/runtime'/name) for name in sdk.PROTECTED}
    if identity(protected)!=sdk.runtime_signature():raise ValueError('Frozen runtime differs from the live runtime identity')
    private=out/'private';private.mkdir(mode=0o700)
    (evaluator_from(Path(base_release),private/'assessment') if base else evaluator(private/'assessment'))
    key=private/'env-state-key.bin'
    with key.open('xb') as stream:stream.write(os.urandom(32))
    release={'schema_version':2,'benchmark_id':sdk.BENCHMARK_ID,'showdown_pin':PIN,'image_id':bootstrap.IMAGE,
        'hardware':HARDWARE,'source_sha256':manifest['source_sha256'],'source_pack_sha256':sha(pack),
        'protected_runtime_sha256':sdk.runtime_signature(),'starter':starter,
        'starter_source_sha256':sdk.source_signature(public/'agent'),
        'native_admission_sha256':sha(frozen/'runtime/native-admission.json'),
        'evaluator_registry_sha256':sha(private/'assessment/registry.json'),
        'private_seeds_created':False,'calibration_qualified':False,
        **({'inputs_from_release_sha256':base['release_sha256']} if base else {}),
        'files':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    release['release_sha256']=identity(release)
    write(out/'release.json',release,exclusive=True)
    for path in out.rglob('*'):
        if path.is_file():path.chmod(0o400 if private in path.parents else 0o444)
    return audit(out)
