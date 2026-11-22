"""CPU identity/alias/RNG checks for explicit optional neural precision modes."""
from pathlib import Path
import argparse,copy,json,random,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))


def main(out):
    import numpy as np,torch
    from model import Policy
    from train import schema
    from precision import configure_precision,_rng,_exact
    data=json.loads((HERE/'generated-v2/task-data.json').read_text());s=schema(data)
    torch.set_num_threads(1);random.seed(81108);np.random.seed(81108);torch.manual_seed(81108)
    model=Policy(s,len(data['vocabulary']),len(data['buff_names']),64,12)
    fixture=HERE.parents[2]/'evidence/clash-implementation/native-training-20261108/compact-embedding-cpu-3/real-observations.npz'
    raw=np.load(fixture);batch=model.pack(raw['observations'][::7],raw['masks'][::7],trim=False)
    forward=model.forward.__func__;values=model.values.__func__;before={name:v.clone() for name,v in model.state_dict().items()}
    aliases=[(name,id(p),p.data_ptr(),p.dtype) for name,p in model.named_parameters()];rng=_rng()
    with torch.no_grad():expected=model(batch);expected_values=model.values(batch)
    metadata=configure_precision(model);again=configure_precision(model,'fp32',device='cpu')
    assert metadata==again and not metadata['cuda_matmul_allow_tf32']
    assert model.forward.__func__ is forward and model.values.__func__ is values
    assert aliases==[(name,id(p),p.data_ptr(),p.dtype) for name,p in model.named_parameters()]
    _exact(model.state_dict(),before,'state dict identity')
    with torch.no_grad():actual=model(batch);actual_values=model.values(batch)
    _exact(actual,expected,'default forward');_exact(actual_values,expected_values,'default values');_exact(_rng(),rng,'configure/forward RNG')
    cloned=copy.deepcopy(model);configure_precision(cloned,'fp32')
    assert cloned.forward.__self__ is cloned and cloned.values.__self__ is cloned
    with torch.no_grad():_exact(cloned(batch),expected,'deepcopy method binding')
    rejected=[]
    for mode in ('tf32','bf16','unknown'):
        state=_rng();old_flag=torch.backends.cuda.matmul.allow_tf32
        try:configure_precision(model,mode,device='cpu')
        except ValueError as exc:rejected.append({'mode':mode,'error':str(exc)})
        else:raise AssertionError('invalid CPU precision accepted '+mode)
        assert torch.backends.cuda.matmul.allow_tf32==old_flag;_exact(_rng(),state,'rejection RNG')
        assert model.forward.__func__ is forward and model.values.__func__ is values
        assert aliases==[(name,id(p),p.data_ptr(),p.dtype) for name,p in model.named_parameters()]
        _exact(model.state_dict(),before,'rejection state dict')
    # Explicit strict FP32 also clears an already-enabled global TF32 flag.
    torch.backends.cuda.matmul.allow_tf32=True;configure_precision(model,'fp32');assert not torch.backends.cuda.matmul.allow_tf32
    report={'status':'passed','device':'cpu','default_forward_and_values_exact':True,'parameter_objects_storage_dtype_and_state_dict_names_exact':True,'all_rng_streams_unchanged':True,'idempotent':True,'deepcopy_bound_methods_point_to_clone':True,'strict_fp32_disables_global_tf32':True,'rejected_modes':rejected,'metadata':metadata,'actual_gpu_qualification_executed':False,'timing_comparison':False}
    out.mkdir(parents=True,exist_ok=False);(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);main(parser.parse_args().out)
