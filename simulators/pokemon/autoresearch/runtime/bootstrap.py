"""Trusted worker bootstrap and the one source snapshot every job of a release stages.

The bootstrap runs as the sandbox command before any source is admitted: it
verifies the uploaded pack against its manifest byte for byte, quarantines the
pinned image's own Python source, restores only the image's Showdown engine
closure, and then starts the trusted worker from the admitted tree.
"""
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import textwrap

from ..contract import ROOT,sha,identity,write

# Retained immutable runtime image: Python, torch, Node and the pinned Showdown engine closure.
IMAGE='im-JwNTde85zO3cAKA9qnqInr'
PUBLIC_PYTHONPATH='/opt/pilot-public'
# The only trusted modules a researcher's unprivileged trainer or policy process can import.
PUBLIC_SOURCE_FILES=(
    'integrations/pokemon/network.py',
    'integrations/pokemon/efficient.py',
    'integrations/pokemon/autoresearch/contract.py',
    'integrations/pokemon/autoresearch/runtime/sdk.py',
    'integrations/pokemon/autoresearch/runtime/env_client.py',
    'integrations/pokemon/autoresearch/runtime/graph_export.py',
    'integrations/pokemon/autoresearch/runtime/graph_worker.py',
    'integrations/pokemon/autoresearch/policy_worker.py',
)
# ENTRY is bound below after the shared archive validator is defined.
_ENTRY_BODY=r'''
import hashlib,json,os,pathlib,shutil,subprocess,time,tarfile
from pathlib import Path, PurePosixPath
os.umask(0o077)
campaign=pathlib.Path('/campaign');campaign.chmod(0o700)
operation=os.environ['POKEMON_OPERATION_ID']
if not operation or any(c not in '0123456789abcdef' for c in operation):raise ValueError('Bounded operation identity required')
job=campaign/operation;job.chmod(0o700)
link=pathlib.Path('/pilotjob')
if link.is_symlink():raise ValueError('Unexpected image job symlink')
if link.exists():
 if not link.is_dir():raise ValueError('Unexpected image job file')
 cached=pathlib.Path('/image-pilot-cache')
 if cached.exists()or cached.is_symlink():raise ValueError('Image cache quarantine already exists')
 link.rename(cached);cached.chmod(0o700)
link.symlink_to(job,target_is_directory=True)
original_request=job/'input/request.json'
spec=json.loads(original_request.read_bytes())
timing=worker_deadline(spec,allocation_entry_unix)
effective=dict(spec,absolute_stop_unix=timing['effective_absolute_stop_unix'])
(job/'input/effective-request.json').write_text(json.dumps(effective))
output=job/'output';output.mkdir(mode=0o700,exist_ok=True)
(output/'effective-worker-deadline.json').write_text(json.dumps(dict(timing,original_request_sha256=hashlib.sha256(original_request.read_bytes()).hexdigest())))
layout=json.loads((job/'input/input-layout.json').read_bytes())
manifest_path=job/'input/source-manifest.json'
if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=layout['manifest_sha256']:raise ValueError('Source manifest changed')
manifest=json.loads(manifest_path.read_bytes())
shared=campaign/layout['source_pack']['path']
unpack_source_pack(shared,job/'input',manifest,layout['source_pack']['sha256'])
for name,item in layout['shared_files'].items():
 rel=PurePosixPath(name)
 if rel.is_absolute()or'..'in rel.parts or str(rel)!=name or'\\'in name:raise ValueError('Shared asset target invalid')
 src=campaign/item['path']
 if src.is_symlink()or not src.is_file()or src.stat().st_size!=item['bytes']:raise ValueError('Shared asset missing or changed size')
 with src.open('rb')as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 if actual!=item['sha256']:raise ValueError('Shared public asset changed')
 dst=job/'input'/name;dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists()or dst.is_symlink():raise ValueError('Shared asset destination already exists')
 shutil.copyfile(src,dst);dst.chmod(0o444)
# Preserve image caches under root-only quarantine; neither protected imports
# nor public imports can inherit files absent from this exact source manifest.
for original,quarantine in (('/opt/pk','/opt/pk-image-cache'),('/opt/pilot-public','/opt/pilot-public-image-cache')):
 p=pathlib.Path(original);cached=pathlib.Path(quarantine)
 if p.is_symlink():raise ValueError('Unexpected image source symlink')
 if p.exists():
  if not p.is_dir()or cached.exists()or cached.is_symlink():raise ValueError('Unexpected image source cache')
  p.rename(cached);cached.chmod(0o700)
# Hide cached historical source/recipes as well as this job's trusted source.
# Certified native files remain intact and are only read by root processes.
for old in ('/opt/pk','/job','/private','/runtime'):
 p=pathlib.Path(old)
 if p.exists():p.chmod(0o700)
manifest=json.loads((job/'input/source-manifest.json').read_bytes())
for name,digest in manifest['files'].items():
 p=job/'input/source'/name
 if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Uploaded protected source differs')
 dst=pathlib.Path('/opt/pk')/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);dst.chmod(0o444)
engine_proof=restore_image_engine('/opt/pk-image-cache','/opt/pk')
(output/'image-engine-closure.json').write_text(json.dumps(dict(engine_proof,image_id=manifest['image_id'])))
private=pathlib.Path('/opt/pk');os.chown(private,0,65533);private.chmod(0o750)
# The pinned Foul Play helper imports ROOT=/opt/pk under its trusted reference
# uid/gid65533. Only that reference group may traverse the protected tree;
# researcher65534 and crossplay65532 remain denied at its root.
for directory in private.rglob('*'):
 if directory.is_dir():directory.chmod(0o755)
public=pathlib.Path('/opt/pilot-public');public.mkdir(mode=0o755)
for name,digest in manifest['public_files'].items():
 src=job/'input/public-source'/name
 if hashlib.sha256(src.read_bytes()).hexdigest()!=digest:raise ValueError('Uploaded public import closure differs')
 dst=public/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);dst.chmod(0o444)
 for parent in dst.parents:
  if parent==public.parent:break
  parent.chmod(0o755)
os.chdir('/opt/pk');os.umask(0o022)
os.environ['PYTHONPATH']='/opt/pk'
os.environ['POKEMON_PILOT_PUBLIC_PYTHONPATH']=str(public)
p=subprocess.run(['python','-m','integrations.pokemon.autoresearch.runtime.worker','--job','/pilotjob'])
raise SystemExit(p.returncode)
'''




def unpack_source_pack(pack,destination,manifest,expected_sha256):
    """Validate every archive byte/member before extracting into root-only input.

    This exact function is included in the trusted bootstrap. A pack never
    replaces manifest verification, and neither its names nor links are trusted.
    """
    pack=Path(pack);destination=Path(destination)
    if pack.is_symlink()or not pack.is_file()or pack.stat().st_size>256*1024*1024:
        raise ValueError('Bounded regular source pack required')
    with pack.open('rb')as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=expected_sha256:raise ValueError('Packed source SHA256 differs')
    expected={}
    for prefix,key in (('source','files'),('public-source','public_files')):
        for name,digest in manifest[key].items():
            rel=PurePosixPath(name)
            if not name or rel.is_absolute()or '..'in rel.parts or str(rel)!=name or '\\'in name:
                raise ValueError('Canonical source manifest member required')
            expected[prefix+'/'+name]=digest
    if len(expected)>8192:raise ValueError('Source pack member ceiling')
    # All members are checked before any source destination is created.
    with tarfile.open(pack,'r:')as archive:
        members=[];seen=set();total=0
        for member in archive:
            name=member.name;rel=PurePosixPath(name)
            if (not member.isfile()or rel.is_absolute()or '..'in rel.parts or str(rel)!=name
                    or '\\'in name or name not in expected or name in seen
                    or member.size<0 or member.size>64*1024*1024):
                raise ValueError('Unexpected or unsafe packed source member')
            total+=member.size
            if total>256*1024*1024:raise ValueError('Packed source aggregate ceiling')
            with archive.extractfile(member)as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=expected[name]:raise ValueError('Packed source member SHA256 differs')
            seen.add(name);members.append(member)
        if seen!=set(expected):raise ValueError('Packed source exact file set differs')
        for member in members:
            dest=destination/member.name
            for parent in dest.parents:
                if parent.is_symlink():raise ValueError('Symlinked source destination')
                if parent==destination:break
            if dest.exists()or dest.is_symlink():raise ValueError('Fresh source destination required')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member)as stream,dest.open('xb')as output:
                shutil.copyfileobj(stream,output,1024*1024)
            with dest.open('rb')as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=expected[member.name]:raise ValueError('Extracted source member changed')
            dest.chmod(0o444)
    return dict(source_sha256=manifest['source_sha256'],files=len(expected),bytes=total)


def worker_deadline(spec,entry_unix):
    """Bind a worker lease to allocation entry, never to upload preparation."""
    seconds=spec['allocation_seconds'];margin=spec['allocation_margin_seconds']
    if (type(seconds)not in (int,float)or not math.isfinite(seconds)or not 60<=seconds<=86400
            or type(margin)not in (int,float)or not math.isfinite(margin)or not 0<margin<seconds
            or not math.isfinite(entry_unix)):
        raise ValueError('Finite allocated worker lease required')
    end=entry_unix+seconds-margin
    original=spec.get('original_campaign_deadline_unix')
    if original is not None:
        if not math.isfinite(original):raise ValueError('Finite original campaign deadline required')
        end=min(end,original)
    return dict(allocation_entry_unix=entry_unix,allocation_seconds=seconds,
        allocation_margin_seconds=margin,original_campaign_deadline_unix=original,
        effective_absolute_stop_unix=end,upload_preparation_excluded=True,bootstrap_setup_included=True)


def restore_image_engine(image_root,destination_root):
    """Preserve the pinned image's non-Python Showdown runtime closure only.

    Source snapshots deliberately do not upload upstream engine assets or Node
    dependencies. They already exist in the pinned image. Quarantining Python
    imports must not discard that separately pinned execution dependency.
    """
    image_root=Path(image_root).resolve();destination_root=Path(destination_root)
    engine=None
    for relative in ('candidates/pokemon-showdown','vendor/engines/pokemon-showdown'):
        candidate=image_root/relative
        if candidate.is_dir():
            candidate=candidate.resolve()
            if not candidate.is_relative_to(image_root):raise ValueError('Image engine escapes pinned image tree')
            engine=candidate;break
    if engine is None:raise ValueError('Pinned image Showdown engine closure missing')
    for required in ('dist/sim/battle-stream.js','dist/sim/battle.js','dist/sim/index.js','node_modules'):
        if not(engine/required).exists():raise ValueError('Incomplete pinned image engine closure: '+required)
    # Never revive the image's old integration/recipe Python source. Only the
    # whitelisted upstream engine subtree and optional root pin stamp survive.
    def inventory(tree):
        rows={}
        for path in sorted(tree.rglob('*')):
            name=path.relative_to(tree).as_posix()
            if path.is_symlink():
                resolved=path.resolve()
                if not resolved.is_relative_to(tree.resolve()):raise ValueError('Image engine link escapes closure')
                rows[name]=dict(link=os.readlink(path))
            elif path.is_file():
                with path.open('rb')as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                rows[name]=dict(sha256=digest,bytes=path.stat().st_size)
        return rows
    original=inventory(engine)
    target=destination_root/'candidates/pokemon-showdown'
    if target.exists()or target.is_symlink():raise ValueError('Fresh pinned image engine destination required')
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(engine,target,symlinks=True)
    copied=inventory(target)
    if copied!=original:raise ValueError('Pinned image engine changed during staging')
    stamp=image_root/'FROZEN.json';stamp_sha=None
    if stamp.exists():
        if stamp.is_symlink()or not stamp.is_file():raise ValueError('Regular image pin stamp required')
        with stamp.open('rb')as stream:stamp_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        shutil.copyfile(stamp,destination_root/'FROZEN.json')
        with(destination_root/'FROZEN.json').open('rb')as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=stamp_sha:raise ValueError('Image pin stamp changed')
        (destination_root/'FROZEN.json').chmod(0o444)
    encoded=json.dumps(original,sort_keys=True,separators=(',',':')).encode()
    return dict(engine='pokemon-showdown',source='pinned-image-engine-only',
        copied_files=len(copied),closure_sha256=hashlib.sha256(encoded).hexdigest(),
        original_and_copied_bytes_identical=True,pin_stamp_sha256=stamp_sha,files=copied)


_bootstrap,_worker_start=_ENTRY_BODY.split('p=subprocess.run',1)
ENTRY=("import hashlib,json,math,os,pathlib,shutil,subprocess,tarfile,time,traceback\n"
       "from pathlib import Path,PurePosixPath\nallocation_entry_unix=time.time()\n"+
       inspect.getsource(unpack_source_pack)+'\n'+inspect.getsource(worker_deadline)+'\n'+inspect.getsource(restore_image_engine)+
       '\ntry:\n'+textwrap.indent(_bootstrap,'    ')+r'''
except BaseException:
    # No candidate code has run. Preserve a typed shared-layout failure even
    # when protected source could not be admitted, without importing that code.
    operation=os.environ.get('POKEMON_OPERATION_ID','')
    if operation and all(c in '0123456789abcdef'for c in operation):
        output=pathlib.Path('/campaign')/operation/'output';output.mkdir(mode=0o700,exist_ok=True)
        (output/'bootstrap-error.log').write_text(traceback.format_exc()[-20000:])
        (output/'bootstrap-failure.json').write_text(json.dumps({'shared_input_admission_failed':True,'candidate_executed':False}))
        with tarfile.open(output/'results.tar','w')as archive:
            for name in ('bootstrap-error.log','bootstrap-failure.json'):archive.add(output/name,arcname=name,recursive=False)
    raise
'''+'\np=subprocess.run'+_worker_start)


def packed_source(target,manifest):
    """Deterministic pack: one immutable content address for the whole release."""
    target=Path(target);pack=target/'source-pack.tar'
    expected={prefix+'/'+name:digest for prefix,key in (('source','files'),('public-source','public_files'))
        for name,digest in manifest[key].items()}
    temporary=pack.with_suffix('.partial')
    with tarfile.open(temporary,'w',format=tarfile.PAX_FORMAT)as archive:
        for name,digest in sorted(expected.items()):
            source=target/name
            if source.is_symlink()or not source.is_file()or sha(source)!=digest:
                raise ValueError('Source changed while packing')
            info=tarfile.TarInfo(name);info.size=source.stat().st_size;info.mode=0o444
            info.uid=info.gid=info.mtime=0;info.uname=info.gname=''
            with source.open('rb')as stream:archive.addfile(info,stream)
    os.replace(temporary,pack)
    # Structural validation before any upload, using a temporary root-only tree.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='.pack-check-',dir=target)as temp:
        unpack_source_pack(pack,temp,manifest,sha(pack))
    return pack


def source_snapshot(target):
    """Copy the current protected and public source trees into a new release, with their manifest."""
    target=Path(target)
    source=target/'source';source.mkdir()
    for p in (ROOT/'simulators/pokemon').rglob('*'):
        if p.is_file()and p.suffix in ('.py','.cjs','.cpp','.hpp','.cu','.json')and '__pycache__'not in p.parts:
            dest=source/'integrations/pokemon'/p.relative_to(ROOT/'simulators/pokemon');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    # Native admission verifies the certificate's declared plan at execution time: the six-Pokémon original or a
    # team-size form's own (native-cpu-qualification-3v3.json).
    for plan in sorted((ROOT/'reports/pokemon').glob('native-cpu-qualification*.json')):
        dest=source/'reports/pokemon'/plan.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(plan,dest)
    files={str(p.relative_to(source)):sha(p)for p in source.rglob('*')if p.is_file()}
    public=target/'public-source';public.mkdir()
    for name in PUBLIC_SOURCE_FILES:
        dest=public/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if name.endswith('/autoresearch/contract.py'):
            # Public identity/file helpers only. The full native-certified
            # declaration stays byte-identical in the root-only source tree.
            raw=(source/name).read_text();tree=ast.parse(raw)
            helpers=[ast.get_source_segment(raw,node)for node in tree.body
                if isinstance(node,ast.FunctionDef)and node.name in ('sha','identity','write','source_signature')]
            if len(helpers)!=4:raise ValueError('Public identity helper closure changed')
            dest.write_text("from pathlib import Path\nimport hashlib,json,os,tempfile\nEDITABLE=('policy.py','learner.py','train.py')\n\n"+'\n\n'.join(helpers)+'\n')
        else:shutil.copyfile(source/name,dest)
        dest.chmod(0o444)
    public_files={str(p.relative_to(public)):sha(p)for p in public.rglob('*')if p.is_file()}
    manifest=dict(files=files,public_files=public_files,
        source_sha256=identity(dict(protected=files,public=public_files)),image_id=IMAGE)
    write(target/'source-manifest.json',manifest)
    for p in source.rglob('*'):
        if p.is_file():p.chmod(0o444)
    return source,manifest
