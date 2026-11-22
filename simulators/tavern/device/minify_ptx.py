"""Losslessly shorten internal Rust PTX identifiers, preserving public CUDA ABI.

Quoted strings/comments and all nonidentifier bytes stay exact. A mandatory reverse
substitution proves byte-for-byte recovery before writing a new output directory.
Actual device differential gates are still required for each resulting binary.
"""
import argparse,hashlib,json,re
from pathlib import Path
TOKEN=re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/|(?P<id>[A-Za-z_$][A-Za-z_0-9.$]*)',re.S)
PREFIX='_tavern_internal_'


def shorten(text):
    if PREFIX in text:raise ValueError('reserved generated identifier namespace already present')
    mapping={};reverse={}
    def replace(m):
        name=m.group('id')
        if name is None or not name.startswith(('_R','_ZN')):return m.group(0)
        if name not in mapping:
            renamed=PREFIX+str(len(mapping));mapping[name]=renamed;reverse[renamed]=name
        return mapping[name]
    result=TOKEN.sub(replace,text)
    restored=TOKEN.sub(lambda m:reverse.get(m.group('id'),m.group(0)),result)
    if restored!=text:raise AssertionError('PTX identifier transformation is not byte reversible')
    return result,reverse


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);raw=a.input.read_bytes();result,mapping=shorten(raw.decode())
    output=result.encode();(a.out/a.input.name).write_bytes(output)
    (a.out/'symbols.json').write_text(json.dumps(mapping,indent=2)+'\n')
    receipt={'input':str(a.input.resolve()),'raw_sha256':hashlib.sha256(raw).hexdigest(),
        'output_sha256':hashlib.sha256(output).hexdigest(),'raw_bytes':len(raw),'output_bytes':len(output),
        'internal_identifiers':len(mapping),'byte_exact_reverse_verified':True,
        'public_kernels_preserved':True,'quoted_data_and_comments_preserved':True}
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
