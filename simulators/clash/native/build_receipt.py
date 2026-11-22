"""Strict native build identities and receipts; no simulator/JVM/GPU execution.

A receipt is required for EVERY selected binary, including local build hits.
Identities contain actual compile inputs; binary bytes receive a separate hash.
Older build artifacts are never rewritten or silently repaired.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import errno,hashlib,json,os,platform,shutil,subprocess,sys,tempfile

FORMAT=2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity_sha256(identity):return hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def receipt_path(library):return Path(library).with_suffix('.json')


def compiler_identity(compiler):
    compiler=str(Path(compiler).resolve())
    return {'path':compiler,'version':subprocess.check_output([compiler,'--version'],text=True)}


def build_identity(*,backend,sources,data_binding,compiler,flags,translation_units,extra=None):
    value={'backend':backend,'source_sha256':dict(sorted(sources.items())),'data_binding':data_binding,'compiler':compiler,'platform':{'system':platform.system(),'machine':platform.machine(),'sys_platform':sys.platform},'flags':list(flags),'translation_units':list(translation_units)}
    if extra is not None:value['extra']=extra
    # Round-trip rejects unserializable values and snapshots caller-owned objects.
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def validate_receipt(library,expected_identity=None):
    """Validate selected bytes, receipt integrity, and complete expected inputs."""
    library=Path(library);path=receipt_path(library)
    if not library.is_file():raise ValueError('native build library is missing: '+str(library))
    if not path.is_file():raise ValueError('native build receipt is missing: '+str(path))
    try:receipt=json.loads(path.read_text())
    except (OSError,ValueError) as exc:raise ValueError('native build receipt is unreadable: '+str(path)) from exc
    if receipt.get('format')!=FORMAT:raise ValueError('native build receipt format differs: '+str(path))
    identity=receipt.get('identity')
    required={'backend','source_sha256','data_binding','compiler','platform','flags','translation_units'}
    if not isinstance(identity,dict) or not required.issubset(identity):raise ValueError('native build receipt identity is incomplete')
    if receipt.get('identity_sha256')!=identity_sha256(identity):raise ValueError('native build receipt identity checksum differs')
    for key in ('source_sha256','data_binding'):
        if receipt.get(key)!=identity[key]:raise ValueError('native build receipt '+key+' alias differs from identity')
    if expected_identity is not None and identity!=expected_identity:
        fields=sorted(key for key in set(identity)|set(expected_identity) if identity.get(key)!=expected_identity.get(key))
        raise ValueError('native build receipt inputs differ: '+','.join(fields))
    if receipt.get('library_sha256')!=sha(library):raise ValueError('native build receipt library hash differs: '+str(library))
    command=receipt.get('command')
    if not isinstance(command,list) or not command or not all(isinstance(v,str) for v in command):raise ValueError('native build receipt command is missing')
    try:prefix=[identity['compiler']['path'],*identity['flags']];units=identity['translation_units']
    except (KeyError,TypeError) as exc:raise ValueError('native build receipt compiler/flags schema differs') from exc
    if len(command)!=len(prefix)+len(units)+2 or command[:len(prefix)]!=prefix or command[-2]!='-o' or Path(command[-1]).name!=library.name:raise ValueError('native build receipt command/flags differ from identity')
    actual_units=command[len(prefix):-2]
    if any(not (Path(actual).as_posix()==unit or Path(actual).as_posix().endswith('/'+unit)) for actual,unit in zip(actual_units,units)):raise ValueError('native build receipt translation units differ from identity')
    return receipt


def backend_build_binding(library,expected_identity=None):
    """Checkpoint-facing actual binary/build binding, after strict validation.

    Build entry points already supply the current expected identity. A caller
    retaining that identity may pass it here to repeat the same input check.
    """
    receipt=validate_receipt(library,expected_identity)
    return {key:receipt[key] for key in ('identity_sha256','library_sha256','identity')}


@contextmanager
def _lock(library):
    import fcntl
    path=Path(str(library)+'.lock');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield


def _present(library):return os.path.lexists(library) or os.path.lexists(receipt_path(library))


def _publish_file(source,destination):
    """Publish one staged file while its caller holds the artifact flock.

    Some volume/FUSE implementations reject hard links. Both staged and final
    files live under the destination parent, so a same-filesystem atomic rename
    is the portable fallback. Every cooperating publisher holds the SAME lock;
    explicit lexists checks protect valid files, partial files and broken links.
    """
    if os.path.lexists(destination):raise FileExistsError(errno.EEXIST,'native build destination already exists',str(destination))
    try:os.link(source,destination)
    except OSError as exc:
        unsupported={errno.EPERM,errno.EOPNOTSUPP,errno.ENOTSUP,errno.EXDEV}
        if exc.errno not in unsupported:raise
        if os.path.lexists(destination):raise FileExistsError(errno.EEXIST,'native build destination appeared before rename',str(destination)) from exc
        os.rename(source,destination)


def _publish(library,identity,make_command,source=None):
    """Publish a new pair only. Caller owns its destination lock."""
    library=Path(library)
    if _present(library):raise FileExistsError(errno.EEXIST,'native build artifact or receipt already exists',str(library))
    with tempfile.TemporaryDirectory(prefix='.native-build-',dir=library.parent) as folder:
        temporary=Path(folder)/library.name
        if source is None:
            command=list(make_command(temporary));subprocess.run(command,check=True)
            receipt={'format':FORMAT,'identity':identity,'identity_sha256':identity_sha256(identity),'library_sha256':sha(temporary),'command':command,'source_sha256':identity['source_sha256'],'data_binding':identity['data_binding']}
            receipt_path(temporary).write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
        else:
            validate_receipt(source,identity)
            shutil.copyfile(source,temporary);shutil.copyfile(receipt_path(source),receipt_path(temporary))
        validate_receipt(temporary,identity)
        # The receipt is published LAST. Partial/failed artifacts remain visible
        # and fail closed on the next load; they are never silently repaired.
        if _present(library):raise FileExistsError(errno.EEXIST,'native build destination appeared while compiling',str(library))
        _publish_file(temporary,library)
        _publish_file(receipt_path(temporary),receipt_path(library))
    return validate_receipt(library,identity)


def materialize_build(library,identity,make_command,*,cache=None):
    """Build/reuse a strictly validated artifact without repairing stale files."""
    library=Path(library);library.parent.mkdir(parents=True,exist_ok=True)
    with _lock(library):
        if _present(library):
            validate_receipt(library,identity)
            return library
        cached=None if cache is None else Path(cache)/library.name
        if cached is not None:
            cached.parent.mkdir(parents=True,exist_ok=True)
            with _lock(cached):
                if _present(cached):
                    validate_receipt(cached,identity)
                    _publish(library,identity,make_command,source=cached)
                    return library
        _publish(library,identity,make_command)
        if cached is not None:
            with _lock(cached):
                if _present(cached):validate_receipt(cached,identity)
                else:_publish(cached,identity,make_command,source=library)
        validate_receipt(library,identity)
        return library
