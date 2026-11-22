"""Restore exact compiled PTX bytes after compressed transfer; never edit instructions."""
import gzip,hashlib,json,lzma,sys
from pathlib import Path
source,destination=map(Path,sys.argv[1:])
rows=json.loads((source/'build.json').read_text())['modules']
if set(rows)!={'tavern.ptx','tavern-state.ptx'}:raise ValueError('expected the complete simulation/checkpoint module pair')
for name,row in rows.items():
    packed=source/(name+'.xz')
    data=lzma.decompress(packed.read_bytes()) if packed.exists() else gzip.decompress((source/(name+'.gz')).read_bytes())
    if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
        raise ValueError('PTX digest mismatch: '+name)
    (destination/name).write_bytes(data)
    print(json.dumps({'restored':name,'bytes':len(data),'sha256':row['sha256']}),flush=True)
