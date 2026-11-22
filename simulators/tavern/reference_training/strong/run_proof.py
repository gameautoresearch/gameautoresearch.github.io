import argparse,json
from pathlib import Path
from .proof import validate
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(validate(json.loads(a.config.read_text()),a.inputs,a.out))
