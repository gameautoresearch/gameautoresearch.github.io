"""One bounded full-minibatch CUDA graph for forward/backward/gradient clipping.

Adam remains eager with the same optimizer state/math and the loss is checked before
its step. The graph never samples, changes the SGD schedule or pads/repeats samples.
Short remainder minibatches use eager execution and share stable gradient buffers.
"""
import time
import torch


def loss_terms(model, inputs, config, encoded):
    f,c,m,actions,oldlp,adv,ret=inputs
    dist,value=model.distribution(f,c,m,numeric_encoded=encoded)
    logp=dist.log_prob(actions)
    ratio=(logp-oldlp).exp()
    policy_loss=-torch.minimum(ratio*adv,
        ratio.clamp(1-config['clip'],1+config['clip'])*adv).mean()
    value_loss=((value-ret)**2).mean()
    entropy=dist.entropy().mean()
    loss=policy_loss+config['value_coef']*value_loss-config['entropy_coef']*entropy
    kl=((ratio-1)-(logp-oldlp)).mean().detach()
    return loss,policy_loss,value_loss,entropy,kl


def backward_graph(model, inputs, config, encoded):
    if not getattr(model,'fast_inference',False):
        raise ValueError('graph learner requires prevalidated masks (fast_inference)')
    key=(len(inputs[0]),encoded,config['clip'],config['value_coef'],config['entropy_coef'],config['max_grad_norm'])
    cache=getattr(model,'_learner_graph',None)
    if cache is None or cache[0]!=key:
        start=time.perf_counter()
        # Keep exactly one capture per model. Normal short final minibatches do not
        # build graphs, so varying completed-cohort lengths cannot grow the cache.
        static=tuple(torch.empty_like(x) for x in inputs)
        for dst,src in zip(static,inputs):dst.copy_(src)
        for p in model.parameters():
            if p.grad is None:p.grad=torch.zeros_like(p)
        def compute():
            for p in model.parameters():p.grad.zero_()
            terms=loss_terms(model,static,config,encoded)
            terms[0].backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'])
            return (*terms,grad)
        stream=torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
        with torch.cuda.stream(stream):
            for _ in range(3):compute()
        torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):outputs=compute()
        cache=model._learner_graph=(key,static,graph,outputs)
        model.learner_graph_build_seconds=getattr(model,'learner_graph_build_seconds',0.)+time.perf_counter()-start
    _,static,graph,outputs=cache
    for dst,src in zip(static,inputs):dst.copy_(src)
    graph.replay()
    return outputs
