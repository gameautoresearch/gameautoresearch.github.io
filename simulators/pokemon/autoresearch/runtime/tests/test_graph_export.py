"""Trusted infrastructure fixtures only; no game simulation or candidate research."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from simulators.pokemon.autoresearch.runtime import graph_export as ge
from simulators.pokemon.autoresearch.runtime.starter.policy import create


class TensorGraphBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The common untrained starter is an infrastructure parity fixture.
        with torch.random.fork_rng():
            torch.manual_seed(4101)
            starter = Path(__file__).resolve().parents[2]/'recipes/starter.json'
            cls.model = create(json.loads(starter.read_text())['network']).eval()
            cls.weights = cls.model.state_dict()
            cls.graph = ge.export(cls.model)

    def observations(self):
        x = ge.example_observations()
        x['mons'].fill_(.25); x['actions'].fill_(.125)
        x['tokens'].fill_(17); x['global_tokens'].fill_(23); x['action_tokens'].fill_(11)
        x['mask'][0, [1, 3, 5, 8, 13]] = False
        return x

    def test_starter_exact_logits_values_and_masked_actions(self):
        policy = ge.GraphPolicy(self.graph, self.weights)
        for x in (ge.example_observations(), self.observations()):
            with torch.no_grad(): expected = self.model(x)
            actual = policy(x)
            for a, b in zip(actual, expected): self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(actual[0].masked_fill(~x['mask'], -torch.inf).argmax(-1),
                                        expected[0].masked_fill(~x['mask'], -torch.inf).argmax(-1)))

    def test_json_roundtrip_and_state_key_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'graph.json'; graph = ge.export(self.model, path)
            self.assertEqual(ge.load_graph(path), graph)
            ge.validate(json.loads(json.dumps(graph)), self.weights)
            with self.assertRaises(FileExistsError): ge.export(self.model, path)
        state = copy.deepcopy(self.weights); state.pop(next(iter(state)))
        with self.assertRaises(ge.GraphError): ge.validate(self.graph, state)
        wrong = copy.deepcopy(self.graph)
        first = next(x for x in wrong['inputs'] if x['kind'] == 'state'); first['dtype'] = 'float64'
        with self.assertRaises(ge.GraphError): ge.validate(wrong, self.weights)

    def test_inference_never_imports_or_exports_candidate(self):
        with (patch('importlib.util.spec_from_file_location', side_effect=AssertionError('No candidate import')),
              patch('torch.export.export', side_effect=AssertionError('No inference-time export'))):
            policy = ge.GraphPolicy(self.graph, self.weights)
            policy(self.observations())

    def test_forbidden_python_environment_search_io_random_mutation_and_controlflow(self):
        for target in ('os.system', 'subprocess.run', 'builtins.open', 'torch.load',
                       'simulators.pokemon.env.Vector.step', 'higher_order.cond',
                       'higher_order.while_loop', 'aten.rand.default', 'aten.copy_.default',
                       'aten.empty.memory_format', 'getattr', '__import__'):
            with self.subTest(target=target):
                graph = copy.deepcopy(self.graph); graph['nodes'][0]['target'] = target
                with self.assertRaisesRegex(ge.GraphError, 'allowlisted'): ge.validate(graph, self.weights)

    def test_reject_forward_references_unknown_input_and_duplicate_nodes(self):
        graph = copy.deepcopy(self.graph); graph['nodes'][0]['args'] = {'tuple': [{'node': 'future'}]}
        with self.assertRaises(ge.GraphError): ge.validate(graph, self.weights)
        graph = copy.deepcopy(self.graph)
        next(x for x in graph['inputs'] if x['kind'] == 'observation')['key'] = 'private_seed'
        with self.assertRaises(ge.GraphError): ge.validate(graph, self.weights)
        graph = copy.deepcopy(self.graph); graph['nodes'][1]['name'] = graph['nodes'][0]['name']
        with self.assertRaises(ge.GraphError): ge.validate(graph, self.weights)

    def test_state_nonfinite_and_ten_million_element_cap(self):
        state = copy.deepcopy(self.weights); state[next(iter(state))].flatten()[0] = float('nan')
        with self.assertRaisesRegex(ge.GraphError, 'Nonfinite'): ge.validate(self.graph, state)
        with self.assertRaisesRegex(ge.GraphError, 'element count'):
            ge.validate(self.graph, {'oversized': torch.zeros(ge.MAX_TENSOR_ELEMENTS + 1)})
        with self.assertRaisesRegex(ge.GraphError, 'ten million'):
            ge.validate(self.graph, {'a': torch.zeros(6_000_000), 'b': torch.zeros(6_000_000)})

    def test_fake_admission_rejects_huge_allocation_before_real_execution(self):
        graph = copy.deepcopy(self.graph)
        # A repeat would allocate billions of elements if called on real tensors.
        graph['nodes'].insert(0, dict(name='huge', target='aten.repeat.default',
            args={'tuple': [{'node': 'x_global'}, {'list': [100_000_000, 1]}]}, kwargs={}))
        with self.assertRaisesRegex(ge.GraphError, 'resource bound'): ge.validate(graph, self.weights)

    def test_graph_bounds_and_output_shape(self):
        graph = copy.deepcopy(self.graph); graph['outputs'] = [graph['outputs'][1], graph['outputs'][0]]
        with self.assertRaisesRegex(ge.GraphError, 'output contract'): ge.validate(graph, self.weights)
        graph = copy.deepcopy(self.graph); graph['nodes'] *= ge.MAX_NODES
        with self.assertRaises(ge.GraphError): ge.validate(graph, self.weights)
        graph = copy.deepcopy(self.graph); graph['schema'] = 'python-pickle'
        with self.assertRaises(ge.GraphError): ge.validate(graph, self.weights)

    def test_training_random_ops_and_cuda_device_are_forbidden(self):
        for target, args, kwargs in (
            ('aten.dropout.default', [{'node': 'x_global'}, .5, True], {}),
            ('aten._to_copy.default', [{'node': 'x_global'}], {'device': 'cuda:0'}),
        ):
            graph = copy.deepcopy(self.graph)
            graph['nodes'].insert(0, dict(name='forbidden', target=target, args={'tuple': args}, kwargs=kwargs))
            with self.assertRaises(ge.GraphError): ge.validate(graph, self.weights)

    def test_observation_shape_dtype_extra_private_fields_and_terminal_masks(self):
        policy = ge.GraphPolicy(self.graph, self.weights)
        for mutate in (lambda x: x.update(seed=10),
                       lambda x: x.update(tokens=x['tokens'].float()),
                       lambda x: x.update(mons=torch.zeros(2, 12, 24)),
                       lambda x: x['mask'].zero_(),
                       lambda x: x['global'].fill_(float('nan'))):
            x = self.observations(); mutate(x)
            with self.assertRaises(ge.GraphError): policy(x)

    def test_persistent_buffer_and_layernorm_architecture(self):
        class Fixture(nn.Module):
            def __init__(self):
                super().__init__(); self.norm = nn.LayerNorm(40); self.head = nn.Linear(40, 15)
                self.register_buffer('offset', torch.ones(40))
            def forward(self, x):
                z = self.head(self.norm(x['global'] + self.offset))
                return z[:, :14].masked_fill(~x['mask'], -torch.inf), z[:, 14]
        model = Fixture().eval(); graph = ge.export(model); x = self.observations()
        actual = ge.run(graph, model.state_dict(), x)
        with torch.no_grad(): expected = model(x)
        for a, b in zip(actual, expected): self.assertTrue(torch.equal(a, b))

    def test_lifted_tensor_constants_are_rejected_with_actionable_message(self):
        class Fixture(nn.Module):
            def __init__(self):
                super().__init__(); self.head = nn.Linear(40, 15)
            def forward(self, x):
                z = self.head(x['global'] + torch.tensor([1.] * 40))
                return z[:, :14], z[:, 14]
        with self.assertRaisesRegex(ge.GraphError, 'persistent registered buffer'): ge.export(Fixture())


if __name__ == '__main__': unittest.main()
