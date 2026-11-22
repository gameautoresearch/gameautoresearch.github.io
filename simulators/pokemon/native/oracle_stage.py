"""Freeze oracle wrappers alongside a native build before generating fixtures."""
import hashlib
import json
from pathlib import Path

from .bridge import HERE


def stage_oracle(output,build_source):
    output=Path(output);root=output/'oracle-source'
    native=root/'integrations/pokemon/native';native.mkdir(parents=True,exist_ok=False)
    hashes={}
    for source in sorted(Path(build_source).glob('*.cjs')):
        data=source.read_bytes();target=native/source.name;target.write_bytes(data);target.chmod(0o444)
        hashes[str(target.relative_to(root))]=hashlib.sha256(data).hexdigest()
    view=HERE.parent/'view.cjs';data=view.read_bytes();target=native.parent/'view.cjs'
    target.write_bytes(data);target.chmod(0o444);hashes[str(target.relative_to(root))]=hashlib.sha256(data).hexdigest()
    candidates=HERE.parents[2]/'candidates';(root/'candidates').symlink_to(candidates,target_is_directory=True)
    (output/'oracle-wrapper-manifest.json').write_text(json.dumps(dict(source_sha256=hashes,upstream_root=str(candidates)),indent=2)+'\n')
    return native
