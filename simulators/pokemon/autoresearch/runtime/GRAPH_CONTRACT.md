# Common neural inference contract

The exported tensor graph is the scored policy. Every policy trained on the
native runtime passes this identical admission boundary. Submitted Python is
never imported by official policy inference.

`policy.create(config['network'])` returns a PyTorch module whose `forward(x)`
returns `(logits, value)`. The frozen exporter calls `model.eval().cpu()` and
strict `torch.export.export` on the public seven-key observation dictionary at
batch size one. Call the common `graph_export.export(copy.deepcopy(model),
out/'graph.json')` as soon as the model is built, as the starter does: the graph is
the architecture only, so a run that is stopped or cut off at its time limit can
still be sealed from its last checkpoint. Export a copy, because the exporter moves
the module it is given to the CPU in eval mode. Keep any tensor constants
as registered persistent buffers. The trusted sealer checks this JSON DAG against
the exact checkpoint state tensors; it never imports your Python.

Limits: a researcher's policy has at most 1,000,000 parameter/buffer tensor elements (the graph
validator itself admits 10,000,000); 2,048 graph nodes;
2 MiB graph JSON; rank at most eight; 10M elements per activation; 64M aggregate
activation elements. State tensors must be dense, finite CPU tensors at export.
Only the existing 700-value observation projection enters inference. There is
no private seed, opponent decision, simulator state, process, file or network
interface. Host code applies the legal mask and chooses the maximum legal logit.

Supported pure tensor operations include embeddings, dense linear layers,
matrix/batch matrix multiplication, elementwise arithmetic and comparisons,
normalization, common activations, masking, attention with zero dropout,
reductions, reshape/view/transpose/permute, slices/index selection, concatenation
and stack/split. The exact admitted overloads are fixed in the frozen
`graph_export.py` registry. An unsupported operation produces an explicit error;
there is no Python fallback. Randomness, mutation, custom operators, higher-order
control flow, training-mode dropout/batchnorm, lifted nonpersistent constants,
simulator calls and test-time search are rejected.

Architecture and training code remain editable within this interface. Preserve
the checkpoint's exact source/config/runtime/model/Adam/Python/dedicated-NumPy/
Torch/CUDA RNG and authenticated opaque environment state for full resume.
An explicit tensors-only fork records changed source/config lineage and starts
fresh optimizer/RNG/native states; it never relabels a full-state continuation.

## Exact frozen operator overloads

aten._log_softmax.default, aten._native_batch_norm_legit_no_training.default, aten._softmax.default, aten._to_copy.default, aten._unsafe_view.default, aten.abs.default, aten.add.Scalar, aten.add.Tensor, aten.addmm.default, aten.alias.default, aten.amax.default, aten.amin.default, aten.batch_norm.default, aten.bitwise_and.Tensor, aten.bitwise_not.default, aten.bitwise_or.Tensor, aten.bmm.default, aten.cat.default, aten.clamp.Tensor, aten.clamp.default, aten.clamp_max.default, aten.clamp_min.default, aten.clone.default, aten.contiguous.default, aten.detach.default, aten.div.Scalar, aten.div.Tensor, aten.dropout.default, aten.elu.default, aten.embedding.default, aten.eq.Scalar, aten.eq.Tensor, aten.exp.default, aten.expand.default, aten.expand_as.default, aten.flatten.using_ints, aten.ge.Scalar, aten.ge.Tensor, aten.gelu.default, aten.group_norm.default, aten.gt.Scalar, aten.gt.Tensor, aten.layer_norm.default, aten.le.Scalar, aten.le.Tensor, aten.leaky_relu.default, aten.linear.default, aten.log.default, aten.log1p.default, aten.log_softmax.int, aten.logical_and.default, aten.logical_not.default, aten.logical_or.default, aten.lt.Scalar, aten.lt.Tensor, aten.masked_fill.Scalar, aten.masked_fill.Tensor, aten.matmul.default, aten.maximum.default, aten.mean.default, aten.mean.dim, aten.minimum.default, aten.mm.default, aten.mul.Scalar, aten.mul.Tensor, aten.native_group_norm.default, aten.native_layer_norm.default, aten.ne.Scalar, aten.ne.Tensor, aten.neg.default, aten.permute.default, aten.pow.Tensor_Scalar, aten.relu.default, aten.relu6.default, aten.repeat.default, aten.reshape.default, aten.rsqrt.default, aten.scaled_dot_product_attention.default, aten.select.int, aten.sigmoid.default, aten.silu.default, aten.slice.Tensor, aten.softmax.int, aten.softplus.default, aten.split.Tensor, aten.split_with_sizes.default, aten.sqrt.default, aten.square.default, aten.squeeze.default, aten.squeeze.dim, aten.squeeze.dims, aten.stack.default, aten.sub.Scalar, aten.sub.Tensor, aten.sum.default, aten.sum.dim_IntList, aten.t.default, aten.tanh.default, aten.to.device, aten.to.dtype, aten.transpose.int, aten.unbind.int, aten.unsqueeze.default, aten.view.default, aten.where.self, getitem.
