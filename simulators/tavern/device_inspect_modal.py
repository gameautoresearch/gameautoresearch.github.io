"""Read-only inspection of one retained Tavern compiler image, without a GPU."""
import json,os,modal
IMAGE=os.environ.get('TAVERN_INSPECT_IMAGE','im-ULlepvVlW6smIR6RxwBb2t')
image=modal.Image.from_id(IMAGE) if modal.is_local() else modal.Image.debian_slim(python_version='3.12')
app=modal.App('autoresearch-tavern-device-inspect-20261108',image=image)

@app.function(cpu=(2,2),memory=(8192,8192),timeout=120,max_containers=1,min_containers=0,
    buffer_containers=0,retries=0,single_use_containers=True)
def inspect(image_id:str):
    import hashlib,time
    from pathlib import Path
    start=time.perf_counter();rows=[]
    for p in sorted(Path('/workspace/integrations/tavern/device').glob('tavern*.ptx')):
        raw=p.read_bytes()
        rows.append({'name':p.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
            'lines':raw.count(b'\n'),'functions':raw.count(b'.func '),'kernels':raw.count(b'.entry ')})
    return {'image':image_id,'gpu':None,'cpu_cores':2,'ram_gib':8,'worker_seconds':time.perf_counter()-start,'modules':rows}

@app.local_entrypoint()
def main():print(json.dumps(inspect.remote(IMAGE)),flush=True)
