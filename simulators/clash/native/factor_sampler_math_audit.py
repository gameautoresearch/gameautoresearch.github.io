"""Read-only math-domain audit of the proposed two-level Torch sampler.

No sampler is installed or benchmarked. Positive support is a hard contract:
a counterexample rejects this candidate rather than weakening the tolerance.
--cuda only uses an already allocated device; this file launches no worker.
"""
from pathlib import Path
import argparse,hashlib,json,platform


def factor_probabilities(logits):
    import torch
    from torch.distributions import Categorical
    original=Categorical(logits=logits,validate_args=False)
    flat=original.logits
    groups=flat[:,1:2305].reshape(-1,4,576)
    slot_logits=torch.cat((flat[:,:1],groups.logsumexp(-1),flat[:,2305:]),-1)
    slots=Categorical(logits=slot_logits,validate_args=False)
    # These neutral distributions are only a full-probability diagnostic for
    # zero-mass groups, never a change to a selectable conditional probability.
    has_mass=torch.isfinite(groups).any(-1)
    safe_groups=torch.where(has_mass[:,:,None],groups,torch.zeros_like(groups))
    tile=Categorical(logits=safe_groups,validate_args=False)
    support=torch.cat((slots.probs[:,:1]>0,
        ((slots.probs[:,1:5,None]>0)&(tile.probs>0)).flatten(1),slots.probs[:,5:]>0),-1)
    # Double product distinguishes true factor support loss from float32
    # underflow in a purely diagnostic multiplication of two positive masses.
    decomposed=torch.cat((slots.probs[:,:1].double(),
        (slots.probs[:,1:5,None].double()*tile.probs.double()).flatten(1),slots.probs[:,5:].double()),-1)
    return original,flat,slots,tile,support,decomposed


def run(out,cuda=False):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    device='cuda' if cuda else 'cpu'
    if cuda and not torch.cuda.is_available():raise RuntimeError('actual allocated CUDA GPU required')
    # Generate on CPU for identical retained input bits across CPU/GPU audits.
    generator=torch.Generator(device='cpu');generator.manual_seed(81108)
    inputs=torch.randn((256,2307),generator=generator,dtype=torch.float32)*25
    inputs[:,0]=0.;inputs[:,1:16]=torch.linspace(-110,-80,15)
    inputs=inputs.to(device)
    original,flat,slots,tiles,support,decomposed=factor_probabilities(inputs)
    lost=torch.nonzero((original.probs>0)&~support)
    failures=[]
    for row,action in lost.cpu().tolist():
        slot=0 if action==0 else (5+action-2305 if action>=2305 else 1+(action-1)//576)
        record={'row':row,'action':action,'slot':slot,'raw_logit':float(inputs[row,action]),
                'original_normalized_flat_logit':float(flat[row,action]),
                'original_positive_probability':float(original.probs[row,action]),
                'slot_probability':float(slots.probs[row,slot]),'tile_probability':None}
        if 1<=action<2305:record['tile_probability']=float(tiles.probs[row,(action-1)//576,(action-1)%576])
        failures.append(record)
    error=float((original.probs.double()-decomposed).abs().max())
    # Also retain simple requested cases; these do not replace the blocker.
    controls={}
    for label in ('small-positive-tail','large-common-offset','forced-wait','ability-only','first-index-tie'):
        x=torch.full((1,2307),-1e9,device=device)
        if label=='small-positive-tail':x[0,:3]=torch.tensor([0.,-20.,-20.],device=device)
        elif label=='large-common-offset':x.fill_(1e20)
        elif label=='forced-wait':x[0,0]=0.
        elif label=='ability-only':x.fill_(-float('inf'));x[0,-2:]=torch.tensor([0.,-1.],device=device)
        else:x[0,[0,1,2305]]=0.
        d,f,s,t,sp,p=factor_probabilities(x)
        controls[label]={'maximum_probability_absolute_error':float((d.probs.double()-p).abs().max()),
            'lost_positive_support':int(((d.probs>0)&~sp).sum()),
            'original_first_index_greedy':int(f.argmax(-1)),
            'exact_returned_logp_if_flat_gathered':bool(torch.equal(f.gather(1,f.argmax(-1)[:,None]).squeeze(1),d.log_prob(f.argmax(-1))))}
    out.mkdir(parents=True,exist_ok=False)
    torch.save({'logits':inputs.cpu(),'seed':81108,'source':'Torch CPU Generator; standard normal*25; explicit first-column/tail assignments'},out/'audit-inputs.pt')
    if failures:torch.save({'logits':inputs[failures[0]['row']].cpu(),'counterexample':failures[0]},out/'counterexample.pt')
    source=Path(__file__);(out/source.name).write_bytes(source.read_bytes())
    result={'status':'rejected_positive_support_contract' if failures else 'no_counterexample_on_this_runtime',
        'device':device,'torch':torch.__version__,'python':platform.python_version(),
        'gpu':torch.cuda.get_device_name() if cuda else None,'actual_gpu_executed':cuda,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'seed':81108,
        'rows':len(inputs),'actions':2307,'positive_support_losses':failures,
        'maximum_probability_absolute_error':error,'controls':controls,
        'candidate_implemented_or_integrated':False,'encoder_sources_changed':False,
        'timing_comparison':False,'reason':'Additional float32 group logsumexp/slot normalization can turn an original positive subnormal probability into zero. Absolute-probability tolerance does not satisfy the separate positive-support contract.'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--cuda',action='store_true')
    args=parser.parse_args();run(args.out,args.cuda)
