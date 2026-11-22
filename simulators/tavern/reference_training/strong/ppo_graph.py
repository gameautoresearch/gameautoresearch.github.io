"""Bounded full-minibatch backward graph with the original eager Adam step.

Captures no optimizer update or RNG. Finite loss and the exact mean(old-logp)
KL are read together before any optimizer step; early stopping stays after it.
"""
import time
import torch


def terms(model,values,c):
    f,cards,masks,actions,old,adv,ret=values
    dist,v=model.distribution(f,cards,masks,numeric_encoded=c.get('preprocess_numeric',False))
    logp=dist.log_prob(actions);ratio=(logp-old).exp()
    loss=(-torch.minimum(ratio*adv,ratio.clamp(1-c['clip'],1+c['clip'])*adv).mean()
          +c['value_coef']*((v-ret)**2).mean()-c['entropy_coef']*dist.entropy().mean())
    kl=(old-logp).mean()
    return loss,kl


def backward(model,values,c):
    key=(len(values[0]),c['clip'],c['value_coef'],c['entropy_coef'],c['max_grad_norm'],c.get('preprocess_numeric',False))
    cache=getattr(model,'_strong_backward_graph',None)
    if cache is None or cache[0]!=key:
        started=time.perf_counter();static=tuple(torch.empty_like(v) for v in values)
        for a,b in zip(static,values):a.copy_(b)
        for p in model.parameters():p.grad=torch.zeros_like(p)
        def compute():
            for p in model.parameters():p.grad.zero_()
            loss,kl=terms(model,static,c);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm'])
            return torch.stack((loss.detach(),kl.detach(),torch.isfinite(loss).float()))
        stream=torch.cuda.Stream(device=values[0].device);stream.wait_stream(torch.cuda.current_stream(values[0].device))
        with torch.cuda.stream(stream):
            for _ in range(3):compute()
        torch.cuda.current_stream(values[0].device).wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):output=compute()
        model._strong_backward_graph=cache=(key,static,graph,output,tuple(p.grad for p in model.parameters()))
        model._strong_graph_build_seconds=getattr(model,'_strong_graph_build_seconds',0.)+time.perf_counter()-started
    _,static,graph,output,grads=cache
    # The eager remainder minibatch may have called zero_grad(set_to_none=True).
    for parameter,gradient in zip(model.parameters(),grads):parameter.grad=gradient
    for a,b in zip(static,values):a.copy_(b)
    graph.replay()
    return output
