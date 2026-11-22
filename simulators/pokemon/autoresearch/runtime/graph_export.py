"""Pilot tensor DAG boundary. Export ONLY in an isolated, untrusted child.

Trusted admission/inference never imports a submitted module or torch export
archive. It accepts bounded JSON and ordinary checkpoint state tensors, resolves
operators through this fixed registry, and interprets the admitted DAG directly.
Python export-time behavior is outside this API's trust boundary. The scored
forward is exactly the admitted graph, not the submitted Python forward.

API: export(model, json_path=None) -> JSON dict; validate(graph, state_dict) ->
None; GraphPolicy(graph, state_dict)(public_tensor_dict) -> (logits, values).
run(graph, state_dict, observations) is a one-shot convenience (validates again).
Inference uses batch size one and the existing seven-key/700-value contract.
Register tensor constants as persistent buffers; lifted non-state constants,
mutations, randomness, higher-order/control-flow and custom operators are refused.
"""
from pathlib import Path
from collections import OrderedDict
import copy
import json
import math
import operator
import re
from types import MappingProxyType

import torch

SCHEMA = 'pokemon-pilot-pure-tensor-dag-v1'
MAX_GRAPH_BYTES = 2 * 1024 * 1024
MAX_NODES = 2048
MAX_TENSOR_ELEMENTS = 10_000_000
MAX_INTERMEDIATE_ELEMENTS = 64_000_000
MAX_DIMENSIONS = 8
CONTRACT = MappingProxyType({
    'tokens': ((1, 12, 12), torch.int64),
    'mons': ((1, 12, 24), torch.float32),
    'global': ((1, 40), torch.float32),
    'global_tokens': ((1, 4), torch.int64),
    'action_tokens': ((1, 14, 3), torch.int64),
    'actions': ((1, 14, 12), torch.float32),
    'mask': ((1, 14), torch.bool),
})
_DTYPES = MappingProxyType({str(x).removeprefix('torch.'): x for x in (
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)})
# Exact overloads only: no attribute lookup from untrusted target strings.
_NAMES = '''
embedding.default linear.default addmm.default mm.default bmm.default matmul.default
add.Tensor add.Scalar sub.Tensor sub.Scalar mul.Tensor mul.Scalar div.Tensor div.Scalar
pow.Tensor_Scalar neg.default abs.default square.default sqrt.default rsqrt.default
exp.default log.default log1p.default tanh.default sigmoid.default relu.default
relu6.default gelu.default silu.default leaky_relu.default elu.default softplus.default
clamp.default clamp.Tensor clamp_min.default clamp_max.default maximum.default minimum.default
where.self masked_fill.Scalar masked_fill.Tensor
logical_not.default logical_and.default logical_or.default bitwise_not.default
bitwise_and.Tensor bitwise_or.Tensor eq.Tensor eq.Scalar ne.Tensor ne.Scalar
lt.Tensor lt.Scalar le.Tensor le.Scalar gt.Tensor gt.Scalar ge.Tensor ge.Scalar
view.default reshape.default _unsafe_view.default flatten.using_ints
unsqueeze.default squeeze.default squeeze.dim squeeze.dims
transpose.int t.default permute.default contiguous.default clone.default detach.default alias.default
slice.Tensor select.int expand.default expand_as.default repeat.default
cat.default stack.default split.Tensor split_with_sizes.default unbind.int
sum.default sum.dim_IntList mean.default mean.dim amax.default amin.default
softmax.int log_softmax.int _softmax.default _log_softmax.default
layer_norm.default native_layer_norm.default group_norm.default native_group_norm.default
batch_norm.default _native_batch_norm_legit_no_training.default
_to_copy.default to.dtype to.device
scaled_dot_product_attention.default dropout.default
'''.split()
_ops = {'getitem': operator.getitem}
for _name in _NAMES:
    _base, _overload = _name.split('.')
    if hasattr(torch.ops.aten, _base) and hasattr(getattr(torch.ops.aten, _base), _overload):
        _ops['aten.' + _name] = getattr(getattr(torch.ops.aten, _base), _overload)
OPS = MappingProxyType(_ops)
_NAME = re.compile(r'^[A-Za-z_][A-Za-z_0-9]{0,127}$')


class GraphError(ValueError):
    """An unsupported or invalid submission, never a request to run Python."""


def example_observations():
    return {key: torch.ones(shape, dtype=dtype) if key == 'mask'
            else torch.zeros(shape, dtype=dtype) for key, (shape, dtype) in CONTRACT.items()}


def _tensor(tensor, *, finite=False):
    if type(tensor) is not torch.Tensor or tensor.layout != torch.strided or tensor.device.type != 'cpu':
        raise GraphError('Only ordinary dense CPU state tensors are admitted')
    if tensor.dtype not in _DTYPES.values() or tensor.ndim > MAX_DIMENSIONS or tensor.numel() > MAX_TENSOR_ELEMENTS:
        raise GraphError('Unsupported tensor dtype, rank or element count')
    if finite and not bool(torch.isfinite(tensor).all()):
        raise GraphError('Nonfinite state tensor')


def _state(weights):
    if type(weights) not in (dict, OrderedDict):
        raise GraphError('State must be a tensor dictionary')
    if not weights or len(weights) > MAX_NODES:
        raise GraphError('Empty or oversized state dictionary')
    for name, value in weights.items():
        if type(name) is not str or not 0 < len(name) <= 256:
            raise GraphError('Invalid state tensor name')
        _tensor(value, finite=True)
    if sum(value.numel() for value in weights.values()) > MAX_TENSOR_ELEMENTS:
        raise GraphError('State exceeds ten million parameter/buffer elements')


def _encode(value):
    from torch.fx import Node
    if isinstance(value, Node):
        return {'node': value.name}
    if isinstance(value, (tuple, list)):
        return {'tuple' if isinstance(value, tuple) else 'list': [_encode(x) for x in value]}
    if type(value) in (bool, int, str) or value is None:
        return value
    if type(value) is float:
        if math.isnan(value):
            raise GraphError('NaN literal is unsupported')
        return {'float': 'inf' if value > 0 else '-inf'} if math.isinf(value) else value
    if isinstance(value, torch.dtype) and value in _DTYPES.values():
        return {'dtype': str(value).removeprefix('torch.')}
    if isinstance(value, torch.device) and value.type == 'cpu' and value.index is None:
        return {'device': 'cpu'}
    raise GraphError('Unsupported graph constant; register tensor constants as persistent buffers')


def _value(value, known, depth=0):
    if depth > 16:
        raise GraphError('Graph argument nesting limit')
    if value is None or type(value) is bool:
        return
    if type(value) is int and -(2**63) <= value < 2**63:
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is str and len(value) <= 128:
        return
    if type(value) is not dict or len(value) != 1:
        raise GraphError('Only JSON literals and explicitly tagged graph arguments are admitted')
    tag, item = next(iter(value.items()))
    if tag == 'node' and type(item) is str and item in known:
        return
    if tag in ('tuple', 'list') and type(item) is list and len(item) <= MAX_NODES:
        for x in item:
            _value(x, known, depth + 1)
        return
    if tag == 'dtype' and type(item) is str and item in _DTYPES:
        return
    if tag == 'device' and item == 'cpu':
        return
    if tag == 'float' and item in ('inf', '-inf'):
        return
    raise GraphError('Invalid argument, forward reference, device or constant')


def _decode(value, values):
    if type(value) is not dict:
        return value
    tag, item = next(iter(value.items()))
    if tag == 'node':
        return values[item]
    if tag in ('tuple', 'list'):
        items = [_decode(x, values) for x in item]
        return tuple(items) if tag == 'tuple' else items
    if tag == 'dtype':
        return _DTYPES[item]
    if tag == 'device':
        return torch.device('cpu')
    return float(item)


def _structure(graph, weights):
    if type(graph) is not dict or set(graph) != {'schema', 'inputs', 'nodes', 'outputs'} or graph['schema'] != SCHEMA:
        raise GraphError('Unknown tensor graph schema')
    try:
        if len(json.dumps(graph, allow_nan=False).encode()) > MAX_GRAPH_BYTES:
            raise GraphError('Oversized JSON graph')
    except (TypeError, ValueError, RecursionError) as exc:
        raise GraphError('Graph must be bounded ordinary JSON') from exc
    if type(graph['inputs']) is not list or type(graph['nodes']) is not list or not 1 <= len(graph['nodes']) <= MAX_NODES:
        raise GraphError('Invalid graph inputs or node count')
    if len(graph['inputs']) > MAX_NODES:
        raise GraphError('Too many graph inputs')
    known = set(); observations = set()
    for item in graph['inputs']:
        if type(item) is not dict or item.get('kind') not in ('observation', 'state'):
            raise GraphError('Only public observations and state tensor inputs are admitted')
        fields = {'name', 'kind', 'key'} if item['kind'] == 'observation' else {'name', 'kind', 'key', 'shape', 'dtype'}
        if set(item) != fields or type(item['name']) is not str or not _NAME.fullmatch(item['name']) or item['name'] in known:
            raise GraphError('Invalid/duplicate graph input')
        key = item['key']
        if type(key) is not str:
            raise GraphError('Invalid graph input key')
        if item['kind'] == 'observation':
            if key not in CONTRACT or key in observations:
                raise GraphError('Exact public seven-key input schema required')
            observations.add(key)
        elif key not in weights or type(item['shape']) is not list or item['shape'] != list(weights[key].shape) or item['dtype'] != str(weights[key].dtype).removeprefix('torch.'):
            raise GraphError('State tensor name, shape or dtype differs from graph export')
        known.add(item['name'])
    if observations != set(CONTRACT):
        raise GraphError('Exact public seven-key input schema required')
    for node in graph['nodes']:
        if type(node) is not dict or set(node) != {'name', 'target', 'args', 'kwargs'}:
            raise GraphError('Only pure function DAG nodes are admitted')
        name = node['name']
        if type(name) is not str or not _NAME.fullmatch(name) or name in known:
            raise GraphError('Invalid/duplicate DAG node name')
        if type(node['target']) is not str or node['target'] not in OPS:
            raise GraphError('Unsupported operation: only allowlisted pure tensor operations are admitted')
        _value(node['args'], known)
        if type(node['args']) is not dict or set(node['args']) != {'tuple'}:
            raise GraphError('Positional arguments must be an explicit tuple')
        if type(node['kwargs']) is not dict or len(node['kwargs']) > 32:
            raise GraphError('Invalid operation keyword arguments')
        for key, value in node['kwargs'].items():
            if type(key) is not str or not _NAME.fullmatch(key):
                raise GraphError('Invalid operation keyword name')
            _value(value, known)
        known.add(name)
    if type(graph['outputs']) is not list or len(graph['outputs']) != 2:
        raise GraphError('Graph must return logits and values')
    for value in graph['outputs']:
        if type(value) is not dict or set(value) != {'node'}:
            raise GraphError('Graph outputs must be tensor node references')
        _value(value, known)


def _guard_operation(target, args, kwargs):
    if target == 'getitem' and (len(args) != 2 or type(args[0]) not in (tuple, list) or type(args[1]) is not int):
        raise GraphError('getitem only selects an integer element of a tensor tuple/list')
    def argument(index, key, default):
        return args[index] if len(args) > index else kwargs.get(key, default)
    if target == 'aten.dropout.default' and argument(2, 'train', True) is not False:
        raise GraphError('Training/random dropout is not admitted')
    if target == 'aten.scaled_dot_product_attention.default' and argument(4, 'dropout_p', 0.) != 0:
        raise GraphError('Random attention dropout is not admitted')
    if target == 'aten.batch_norm.default' and argument(5, 'training', True) is not False:
        raise GraphError('Mutating/training batch normalization is not admitted')
    # to.device can encode a device as a string rather than a tagged device.
    if target in ('aten.to.device', 'aten._to_copy.default'):
        device = argument(1, 'device', None) if target == 'aten.to.device' else kwargs.get('device')
        if device is not None and str(device) != 'cpu':
            raise GraphError('Inference graph may use only the CPU device')
    if target == 'aten._to_copy.default' and kwargs.get('pin_memory'):
        raise GraphError('Pinned allocator/device hooks are unsupported')


def _size(value):
    if isinstance(value, torch.Tensor):
        if value.ndim > MAX_DIMENSIONS or value.numel() > MAX_TENSOR_ELEMENTS:
            raise GraphError('Tensor graph activation exceeds resource bound')
        return value.numel()
    if type(value) in (tuple, list) and len(value) <= MAX_NODES:
        return sum(_size(x) for x in value)
    if value is None:
        return 0
    raise GraphError('An operation returned a non-tensor value')


def _execute(graph, weights, observations):
    values = {x['name']: observations[x['key']] if x['kind'] == 'observation' else weights[x['key']] for x in graph['inputs']}
    elements = 0
    for node in graph['nodes']:
        args = _decode(node['args'], values)
        kwargs = {key: _decode(value, values) for key, value in node['kwargs'].items()}
        _guard_operation(node['target'], args, kwargs)
        value = OPS[node['target']](*args, **kwargs)
        elements += _size(value)
        if elements > MAX_INTERMEDIATE_ELEMENTS:
            raise GraphError('Tensor graph total activation bound exceeded')
        values[node['name']] = value
    result = tuple(_decode(x, values) for x in graph['outputs'])
    if any(not isinstance(x, torch.Tensor) or not x.is_floating_point() for x in result) or tuple(result[0].shape) != (1, 14) or tuple(result[1].shape) != (1,):
        raise GraphError('Graph output contract is floating logits[1,14], values[1]')
    return result


def validate(graph, weights):
    """Trusted admission: no candidate imports, export archives or Python hooks."""
    _state(weights)
    _structure(graph, weights)
    # Infer allocation sizes with metadata-only tensors before executing any real
    # graph operation. This refuses malicious reshape/repeat/expand allocations.
    from torch._subclasses.fake_tensor import FakeTensorMode
    try:
        with FakeTensorMode() as mode:
            state = {k: mode.from_tensor(v, static_shapes=True) for k, v in weights.items()}
            observations = {k: mode.from_tensor(v, static_shapes=True) for k, v in example_observations().items()}
            _execute(graph, state, observations)
    except GraphError:
        raise
    except Exception as exc:
        raise GraphError('Invalid or unsupported static tensor graph: ' + type(exc).__name__) from exc


class GraphPolicy:
    """Validate once, then run the fixed pure DAG without submitted Python."""
    def __init__(self, graph, weights):
        validate(graph, weights)
        self.graph = copy.deepcopy(graph)
        self.weights = {key: value.detach().clone() for key, value in weights.items()}

    def __call__(self, observations):
        if type(observations) is not dict or set(observations) != set(CONTRACT):
            raise GraphError('Exact public observation dictionary required')
        for key, (shape, dtype) in CONTRACT.items():
            tensor = observations[key]
            _tensor(tensor)
            if tuple(tensor.shape) != shape or tensor.dtype != dtype or not bool(torch.isfinite(tensor).all()):
                raise GraphError('Observation shape/dtype/finite contract differs: ' + key)
        if not bool(observations['mask'].any()):
            raise GraphError('Waiting/terminal observation cannot request a decision')
        with torch.no_grad():
            logits, values = _execute(self.graph, self.weights, observations)
        if not bool(torch.isfinite(logits[observations['mask']]).all()) or not bool(torch.isfinite(values).all()):
            raise GraphError('Nonfinite legal logits or values')
        return logits, values


def run(graph, weights, observations):
    return GraphPolicy(graph, weights)(observations)


def load_graph(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_GRAPH_BYTES:
        raise GraphError('Expected bounded regular JSON graph file')
    try:
        return json.loads(path.read_text())
    except (ValueError, RecursionError) as exc:
        raise GraphError('Invalid graph JSON') from exc


def export(model, output_path=None):
    """UNTRUSTED CHILD ONLY: trace model, returning/saving the supported JSON DAG.

    State placeholders refer to original state_dict names. No separate weight
    format or tensor archive is introduced. The trusted sealer must call validate
    with its independently loaded weights and bind graph.json into its manifest.
    """
    model.eval().cpu()
    weights = model.state_dict()
    _state(weights)
    try:
        program = torch.export.export(model, (example_observations(),), strict=True)
    except Exception as exc:
        raise GraphError('Cannot export a static pure tensor forward: ' + type(exc).__name__) from exc
    specs = program.graph_signature
    if any(x.kind.name != 'USER_OUTPUT' for x in specs.output_specs) or len(specs.output_specs) != 2:
        raise GraphError('Mutation or non-logits/value outputs are unsupported')
    inputs = []; observation_index = 0
    for item in specs.input_specs:
        name = item.arg.name
        if item.kind.name in ('PARAMETER', 'BUFFER') and item.target in weights:
            tensor = weights[item.target]
            inputs.append(dict(name=name, kind='state', key=item.target, shape=list(tensor.shape), dtype=str(tensor.dtype).removeprefix('torch.')))
        elif item.kind.name == 'USER_INPUT' and observation_index < len(CONTRACT):
            inputs.append(dict(name=name, kind='observation', key=tuple(CONTRACT)[observation_index]))
            observation_index += 1
        else:
            raise GraphError('Lifted constants/nonpersistent buffers are unsupported; use a persistent registered buffer')
    nodes = []; outputs = None
    for node in program.graph_module.graph.nodes:
        if node.op == 'placeholder':
            continue
        if node.op == 'output':
            outputs = [_encode(x) for x in node.args[0]]
            continue
        if node.op != 'call_function':
            raise GraphError('Only tensor function DAG nodes can be exported')
        target = 'getitem' if node.target is operator.getitem else str(node.target)
        if target not in OPS:
            raise GraphError('Unsupported tensor operation: ' + target)
        nodes.append(dict(name=node.name, target=target, args=_encode(node.args), kwargs={key: _encode(value) for key, value in node.kwargs.items()}))
    graph = dict(schema=SCHEMA, inputs=inputs, nodes=nodes, outputs=outputs)
    validate(graph, weights)
    if output_path is not None:
        path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(graph, indent=2, allow_nan=False) + '\n'
        if len(encoded.encode()) > MAX_GRAPH_BYTES:
            raise GraphError('Serialized graph exceeds the JSON file bound')
        with path.open('x') as stream:
            stream.write(encoded)
    return graph
