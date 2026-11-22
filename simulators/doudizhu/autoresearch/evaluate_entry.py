from pathlib import Path
import argparse
import json
from .arena import evaluate,PolicyFailure
from .contract import write_json
from .grading import summary,grade
from .bundle import unpack_bundle

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',default='/private/evaluation.json');a=p.parse_args()
    c=json.loads(Path(a.request).read_text())
    for target in [c['candidate']]+[p['bundle'] for p in c['panel'] if p['kind']=='checkpoint']:
        archive=Path(target+'.tar')
        if archive.is_file():unpack_bundle(archive,target)
    try:result=evaluate(c['candidate'],c['panel'],deal_seed=c['deal_seed'],policy_seed=c['policy_seed'],deals=c['deals'],split=c['split'])
    except PolicyFailure as exc:
        result={'status':'invalid_policy','reward':0,'error_type':type(exc).__name__,'error':str(exc),'natural_losses_awarded':0}
        write_json('/private/evaluation-private.json',result);write_json('/job/evaluation-public.json',result);return
    write_json('/private/evaluation-private.json',result)
    public=grade(result['rows'],c.get('anchors',{}),development=c.get('development',False))
    write_json('/job/evaluation-public.json',public)

if __name__=='__main__':main()
