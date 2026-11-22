"""Reuse an owned prebuilt public runtime; no package/Rust build workers recur."""
import json,os
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .budget import ROOT,CAMPAIGN
from simulators.tavern.autoresearch.evaluation_v2.release import sha,write

RECEIPT=CAMPAIGN/'tavern/evaluator/public-runtime-v1.json'
def freeze_public_runtime():
    stage=CAMPAIGN/'tavern/evaluator/workflow-aeaa24ee-qualification';build=json.loads((stage/'image-build.json').read_text());provider=json.loads((stage/'provider.json').read_text());inputs=json.loads((stage/'inputs.json').read_text());life=json.loads((stage/'last-status.json').read_text())['lifecycle.json']
    release=CAMPAIGN/'tavern/evaluator/benchmark-release-bootstrap-v1';manifest=json.loads((release/'release.json').read_text())
    if provider['kind']!='entrant-training' or provider['private_opposition_in_training_image'] is not False or life['status']!='completed':raise ValueError('Public runtime lacks completed native/CUDA training provenance')
    if any(Path(name).suffix in ('.pt','.npz','.pkl','.onnx') for name in inputs) or any(Path(name).suffix in ('.pt','.npz','.pkl','.onnx') for name in manifest['frozen_files']):raise ValueError('Do not reuse a training image containing weights/private data')
    data=dict(schema='owned-qualified-public-runtime-v1',image_id=build['image_id'],original_provider_app=provider['app_id'],original_provider_sandbox=provider['sandbox_id'],original_release_manifest_sha256=sha(release/'release.json'),original_inputs_sha256=sha(stage/'inputs.json'),original_host_source_manifest_sha256=sha(stage/'host-source-files.json'),native_fingerprint='217577b83fc4525e',private_reference_weights_or_panels_present=False,checkpoint_weights_in_image=False,entrant_inputs='Only public starter source and smoke recipe; no resume/checkpoint included',future_layer_policy='COPY exact source/assets only; private asset/panel chmod and probes run inside finite Sandbox before candidate startup')
    if RECEIPT.exists():raise FileExistsError(RECEIPT)
    write(RECEIPT,data);RECEIPT.chmod(0o444);return data

def public_image():
    import modal
    data=json.loads(RECEIPT.read_text())
    if data['native_fingerprint']!='217577b83fc4525e' or data['private_reference_weights_or_panels_present'] or data['checkpoint_weights_in_image']:raise ValueError('Public runtime provenance differs')
    return modal.Image.from_id(data['image_id'])

if __name__=='__main__':print(json.dumps(freeze_public_runtime(),indent=2))
