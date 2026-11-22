"""Strict tensor bridge export, executed inside the checkpoint's original source."""
import json
from pathlib import Path
import torch
from ..autoresearch.evaluation_v2.reference_bridge import export
from .strong_campaign import sha,write

def run(spec,job):
 out=job/'exports';out.mkdir();records={}
 for name,path in spec['checkpoints'].items():
  cp=torch.load(path,map_location='cpu',weights_only=False);meta=export(path,cp['config'],out/name)
  records[name]={'path':str(out/name),'manifest_sha256':meta['manifest_sha256'],'weights_sha256':meta['weights_sha256'],'model_sha256':meta['model_sha256'],'checkpoint_sha256':sha(path),'network':meta['network'],'original_reference_source_signature':meta['original_reference_source_signature']}
 write(job/'export-index.json',records);return {'source_admitted':True,'exports':records}
