"""Fail-closed build receipt tests on isolated copies; no native cache mutation."""
from pathlib import Path
import argparse,copy,errno,json,os,sys,tempfile
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from build_receipt import backend_build_binding,build_identity,identity_sha256,materialize_build,receipt_path,validate_receipt


def main(out):
    out.mkdir(parents=True,exist_ok=False);cases=[]
    with tempfile.TemporaryDirectory(prefix='clash-receipts-') as folder:
        root=Path(folder)
        flags=['-c','import pathlib,sys;pathlib.Path(sys.argv[-1]).write_bytes(b"isolated test artifact")']
        identity=build_identity(backend='test-only',sources={'world.hpp':'a'*64},data_binding={'task_data_sha256':'b'*64,'config_sha256':'c'*64},compiler={'path':sys.executable,'version':sys.version},flags=flags,translation_units=[])
        library=root/'local'/'library.bin';cache=root/'cache'
        def command(path):return [sys.executable,*flags,'-o',str(path)]
        materialize_build(library,identity,command,cache=cache);receipt=validate_receipt(library,identity)
        binding=backend_build_binding(library,identity);assert binding['identity_sha256']==identity_sha256(identity)
        before=(library.read_bytes(),receipt_path(library).read_bytes())
        materialize_build(library,identity,lambda _:(_ for _ in ()).throw(AssertionError('local hit compiled')))
        assert before==(library.read_bytes(),receipt_path(library).read_bytes());cases.append('valid-local-hit-no-rebuild-no-overwrite')
        second=root/'other'/'library.bin';materialize_build(second,identity,lambda _:(_ for _ in ()).throw(AssertionError('cache hit compiled')),cache=cache)
        assert second.read_bytes()==before[0];cases.append('valid-shared-cache-copy')
        def rejection(name,change_library=None,change_receipt=None,remove_receipt=False,remove_library=False,expected=identity,via_cache=False):
            case=root/name;case.mkdir();target=case/'library.bin';target.write_bytes(before[0]);receipt_path(target).write_bytes(before[1])
            if change_library is not None:target.write_bytes(change_library)
            if change_receipt is not None:
                value=copy.deepcopy(receipt);change_receipt(value);receipt_path(target).write_text(json.dumps(value))
            if remove_receipt:receipt_path(target).unlink()
            if remove_library:target.unlink()
            snapshot={p.name:p.read_bytes() for p in case.iterdir()}
            try:
                if via_cache:materialize_build(root/(name+'-destination')/'library.bin',identity,command,cache=case)
                else:materialize_build(target,expected,command)
            except ValueError as exc:cases.append({'case':name,'error':str(exc)})
            else:raise AssertionError('accepted '+name)
            after={p.name:p.read_bytes() for p in case.iterdir() if p.suffix!='.lock'}
            assert snapshot==after,name+' repaired/overwrote failed artifacts'
        rejection('missing-receipt',remove_receipt=True);rejection('missing-library',remove_library=True)
        rejection('changed-library-bytes',change_library=b'tampered')
        rejection('old-receipt-format',change_receipt=lambda r:r.update(format=1))
        rejection('identity-checksum',change_receipt=lambda r:r.update(identity_sha256='0'*64))
        rejection('library-checksum',change_receipt=lambda r:r.update(library_sha256='0'*64))
        for name in ('source_sha256','compiler','flags','data_binding','platform','translation_units'):
            changed=copy.deepcopy(identity)
            if name=='flags':changed[name]=flags+['--different']
            elif name=='translation_units':changed[name]=['different.cpp']
            else:changed[name]=dict(changed[name],test_mismatch='different')
            assert identity_sha256(changed)!=identity_sha256(identity)
            rejection('different-'+name,expected=changed)
        rejection('command-flags-mismatch',change_receipt=lambda r:r['command'].__setitem__(1,'--different'))
        rejection('command-compiler-mismatch',change_receipt=lambda r:r['command'].__setitem__(0,'/different/compiler'))
        rejection('shared-cache-missing-receipt',remove_receipt=True,via_cache=True)
        rejection('shared-cache-binary-tamper',change_library=b'tampered',via_cache=True)
        for error_name in ('EPERM','EOPNOTSUPP','ENOTSUP','EXDEV'):
            code=getattr(errno,error_name);destination=root/('fallback-'+error_name)/'library.bin';fallback_cache=root/('fallback-cache-'+error_name)
            renames=[];rename=os.rename
            def traced_rename(source,target):
                renames.append(Path(target).name)
                return rename(source,target)
            with patch('build_receipt.os.link',side_effect=OSError(code,'isolated hard-link rejection')),patch('build_receipt.os.rename',side_effect=traced_rename):
                materialize_build(destination,identity,command,cache=fallback_cache)
                validate_receipt(destination,identity);validate_receipt(fallback_cache/'library.bin',identity)
                saved=(destination.read_bytes(),receipt_path(destination).read_bytes())
                materialize_build(destination,identity,lambda _:(_ for _ in ()).throw(AssertionError('fallback local hit compiled')))
                assert saved==(destination.read_bytes(),receipt_path(destination).read_bytes())
                cached_copy=root/('fallback-copy-'+error_name)/'library.bin'
                materialize_build(cached_copy,identity,lambda _:(_ for _ in ()).throw(AssertionError('fallback cache hit compiled')),cache=fallback_cache)
                validate_receipt(cached_copy,identity)
            assert renames==['library.bin','library.json']*3,(error_name,renames)
            cases.append('atomic-rename-fallback-'+error_name+'-fresh-local-cache-reload-receipt-last')
        with patch('build_receipt.os.link',side_effect=OSError(errno.EPERM,'isolated FUSE rejection')):
            rejection('fallback-existing-bad-cache-preserved',change_library=b'bad existing cache',via_cache=True)
            rejection('fallback-existing-missing-cache-receipt-preserved',remove_receipt=True,via_cache=True)
        # A destination appearing at the failed link must NOT be overwritten by
        # fallback rename, even if that writer did not cooperate with our lock.
        collision=root/'fallback-collision'/'library.bin'
        def collide(source,target):
            Path(target).write_bytes(b'collision sentinel')
            raise OSError(errno.EPERM,'isolated link rejection plus collision')
        with patch('build_receipt.os.link',side_effect=collide),patch('build_receipt.os.rename') as rename_mock:
            try:materialize_build(collision,identity,command)
            except FileExistsError:pass
            else:raise AssertionError('fallback overwrote collision')
            rename_mock.assert_not_called()
        assert collision.read_bytes()==b'collision sentinel' and not receipt_path(collision).exists()
        cases.append('fallback-racing-destination-preserved-no-rename')
        for code in (errno.EACCES,errno.EEXIST,errno.ENOSPC):
            destination=root/('no-fallback-'+str(code))/'library.bin'
            with patch('build_receipt.os.link',side_effect=OSError(code,'nonportable-error-must-propagate')),patch('build_receipt.os.rename') as rename_mock:
                try:materialize_build(destination,identity,command)
                except OSError as exc:assert exc.errno==code
                else:raise AssertionError('unexpected hard-link error accepted')
                rename_mock.assert_not_called()
            assert not destination.exists() and not receipt_path(destination).exists()
            cases.append('no-fallback-for-errno-'+str(code))
        assert before==(library.read_bytes(),receipt_path(library).read_bytes())
    result={'status':'passed','cases':cases,'case_count':len(cases),'isolated_fixtures_only':True,'real_build_cache_modified':False,'JVM_executed':False,'GPU_executed':False,'timing_comparison':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args().out)
