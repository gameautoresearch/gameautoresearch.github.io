"""Build game-owned C++ code; the pinned Java oracle is never linked into it."""
from pathlib import Path
import hashlib,shutil,sys
HERE=Path(__file__).resolve().parent


def build(form='full'):
    """Build (or reuse) the CPU library for a match form (native/match_forms.py); `full` is the pinned form."""
    from data_binding import validate_data
    from build_receipt import build_identity,compiler_identity,identity_sha256,materialize_build
    import match_forms
    data_binding=validate_data()
    target=HERE/'.build';target.mkdir(exist_ok=True)
    compiler=shutil.which('clang++') or shutil.which('g++')
    if not compiler:raise RuntimeError('C++ compiler required')
    compiler=compiler_identity(compiler)
    sources=sorted(p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.hpp','.cpp') and '.build' not in p.parts)
    hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    flags=['-std=c++20','-O3','-pthread','-fPIC','-fno-fast-math','-ffp-contract=off','-dynamiclib' if sys.platform=='darwin' else '-shared',*match_forms.flags(form)]
    units=['primitives.cpp','engine.cpp','cpu_batch.cpp']
    # The pinned form keeps its historical identity shape; any other form is named in it too.
    extra=None if form=='full' else {'match_form':match_forms.describe(form)}
    identity=build_identity(backend='native-cpu',sources=hashes,data_binding=data_binding,compiler=compiler,flags=flags,translation_units=units,extra=extra)
    digest=identity_sha256(identity)
    output=target/('libclash-native-v2'+match_forms.library_suffix(form)+'-'+digest[:16]+('.dylib' if sys.platform=='darwin' else '.so'))
    library=materialize_build(output,identity,lambda path:[compiler['path'],*flags,*[str(HERE/name) for name in units],'-o',str(path)])
    match_forms.check_library(library,form)
    return library

if __name__=='__main__':print(build(sys.argv[1] if len(sys.argv)>1 else 'full'))
