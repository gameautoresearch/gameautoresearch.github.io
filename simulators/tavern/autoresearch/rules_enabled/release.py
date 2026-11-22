"""Release constructor for the research harness (four-player small form, mixed-lobby evaluator).

Builds a sealed release: the public starter and configs, the researcher workspace
(harness.workspace), the frozen worker source that becomes the training and
evaluation image, and a private copy of the selected reference registry.
Offline: no cloud work, no model calls.
"""
import json,shutil,subprocess
from pathlib import Path
from simulators.tavern.autoresearch.evaluation_v2.release import HERE,AUTO,GAME,ROOT,RUNTIME,EVALUATOR_ID,RECEIPT,sha,write,verify
# Public practice: 4 seed groups x 4 seats x 3 profiles = 48 lobbies (the eight-player form used 2 x 8 x 3).
PRACTICE_SEEDS=[50101,50102,50103,50104]
EXTENDED_SEEDS=list(range(50111,50127))
FINAL_GROUPS=24

def native_binding():
    """The task/native source binding. taverngym is installed in the game interpreter, not the research one."""
    try:import taverngym
    except ImportError:
        code="import json,sys;sys.path.insert(0,'simulators/tavern/autoresearch');from contract import binding;print(json.dumps(binding()))"
        try:done=subprocess.run([str(ROOT/'.venv-tavern/bin/python'),'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=120)
        except OSError as exc:raise ValueError('The native binding needs the game interpreter .venv-tavern: '+str(exc)) from None
        if done.returncode:raise ValueError('The native binding needs .venv-tavern with a current AutoTavern build: '+done.stderr[-2000:])
        return json.loads(done.stdout)
    from simulators.tavern.autoresearch.contract import binding
    return binding()

def prepare(out,registry=None):
    """`registry` selects a compatible frozen reference pool; the default is the one named by the selected receipt."""
    from simulators.tavern.autoresearch.contract import EDITABLE,ENGINE
    from harness.workspace import stage as stage_workspace
    receipt=json.loads((HERE/RECEIPT).read_text())
    registry=Path(registry or ROOT/receipt['path']).resolve()
    if sha(registry/'manifest.json')!=receipt['manifest_sha256']:raise ValueError('Selected pool receipt differs; create an explicit new selection version')
    reg=json.loads((registry/'registry.json').read_text());pin=native_binding()
    if reg['binding']!=pin:raise ValueError('Selected pool not compatible with current task/native binding')
    entries=json.loads((registry/'manifest.json').read_text())
    if any(sha(registry/f)!=h for f,h in entries.items()):raise ValueError('Selected pool asset drift')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);public=out/'public';public.mkdir()
    for folder in ('starter','recipes'):(public/folder).mkdir()
    for f in EDITABLE:shutil.copy2(AUTO/'starter'/f,public/'starter'/f)
    for f in ('starter.json','smoke.json'):shutil.copy2(AUTO/'recipes'/f,public/'recipes'/f)
    shutil.copy2(HERE/'CONTRACT.md',public/'CONTRACT.md')
    shutil.copy2(HERE/'benchmark.json',public/'benchmark.json')
    write(public/'practice.json',dict(label=EVALUATOR_ID+'-PUBLIC-practice',seeds=PRACTICE_SEEDS))
    write(public/'practice-full.json',dict(label=EVALUATOR_ID+'-PUBLIC-extended',seeds=EXTENDED_SEEDS))
    # The researcher sees only the canonical workspace built from this public tree.
    stage_workspace('tavern',public)
    frozen=out/'frozen';shutil.copytree(ENGINE,frozen/'vendor/engines/AutoTavern',ignore=shutil.ignore_patterns('target','.git','__pycache__','*.so','*.dylib','*.pyc','.pytest_cache','.venv'))
    game=frozen/'simulators/tavern';game.mkdir(parents=True)
    for f in GAME.glob('*.py'):shutil.copy2(f,game/f.name)
    shutil.copytree(AUTO,game/'autoresearch',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.cache'))
    shutil.copytree(GAME/'evaluation_readiness',game/'evaluation_readiness',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(GAME/'recipes',game/'recipes')
    tools=frozen/'tools';tools.mkdir()
    for f in ('open_credit_campaign.py','overnight_budget.py'):shutil.copy2(ROOT/'tools'/f,tools/f)
    private=out/'private';private.mkdir(mode=0o700);dest=private/'registry';dest.mkdir()
    for rel,h in entries.items():
        p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(registry/rel,p)
    shutil.copy2(registry/'manifest.json',dest/'manifest.json')
    write(private/'panel-qualification.json',dict(label='PUBLIC-engineering-qualification-NOT-private-final',seeds=PRACTICE_SEEDS))
    write(out/'binding.json',pin)
    seats=int(pin['rules']['num_players'])
    manifest=dict(schema_version=2,evaluator_id=EVALUATOR_ID,benchmark_id=EVALUATOR_ID,artifact_binding_benchmark_id=pin['benchmark_id'],players=seats,frozen_files={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()},binding_sha256=sha(out/'binding.json'),registry_manifest_sha256=sha(dest/'manifest.json'),selected_registry_receipt_sha256=sha(HERE/RECEIPT),decision_source={f:sha(GAME/'evaluation_readiness'/f) for f in RUNTIME},qualification_panel_sha256=sha(private/'panel-qualification.json'),final_bank_status='not created; requires frozen selection',full_private_lobbies=FINAL_GROUPS*seats*3,public_practice_lobbies=len(PRACTICE_SEEDS)*seats*3,all_seats_natural_required=True,strong_panel_qualified=False,superhuman_evidence=False)
    manifest['public_files']={str(p.relative_to(public)):sha(p) for p in public.rglob('*') if p.is_file()}
    write(out/'release.json',manifest);verify(out)
    return dict(release=str(out),evaluator_id=manifest['evaluator_id'],frozen_files=len(manifest['frozen_files']),mixed_profiles=list(reg['profiles']),private_final_bank_created=False)
