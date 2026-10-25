"""Proposal modes, per-search generators and arity-correct argument sums."""

from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Batch

from maths_ai.gnn_inference.atp_lean_gnn.actor_critic import action_arity
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_premise_mask, build_vocab, dag_to_pyg
from maths_ai.gnn_inference.tests.model_helpers import actor_critic
from maths_ai.hybrid_reasoner.proposal import (
    ProposalSpec,
    derive_search_seed,
    input_identity,
    proposal_specs,
)
from maths_ai.lean_replay import seed_from_text

ID_TO_TACTIC = {0: "exact", 1: "have", 2: "trivial", 3: "apply", 4: "rw"}


def _graphs(*states: str):
    dags = [proof_state_to_dag(state) for state in states]
    vocab = build_vocab(dags)
    datas = []
    for dag in dags:
        data = dag_to_pyg(dag, vocab, add_reverse_edges=True)
        data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
        state_matches = (data.x == vocab.get("State", 0)).nonzero(as_tuple=False).view(-1)
        data.state_node_index = state_matches[-1:]
        datas.append(data)
    return datas, vocab


def _only(*tactic_ids: int) -> torch.Tensor:
    """A tactic mask that forces row b onto ``tactic_ids[b]``."""
    mask = torch.zeros(len(tactic_ids), len(ID_TO_TACTIC), dtype=torch.bool)
    for row, tactic_id in enumerate(tactic_ids):
        mask[row, tactic_id] = True
    return mask


STATES = ("n : Nat\nh : Even n\n⊢ Even n", "m : Nat\nk : Nat\nh : m = k\n⊢ k = m")


class ProposalModeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.datas, vocab = _graphs(*STATES)
        self.model = actor_critic(len(vocab), len(ID_TO_TACTIC))
        self.model.eval()
        self.batch = Batch.from_data_list(self.datas)

    def act(self, batch=None, **kwargs):
        with torch.no_grad():
            return self.model.act(batch or self.batch, id_to_tactic=ID_TO_TACTIC, **kwargs)

    def test_greedy_chooses_the_tactic_and_every_argument_by_argmax(self):
        # An argument-bearing tactic: greedy must not sample its pointer step.
        samples = [
            self.act(greedy=True, tactic_mask=_only(1, 1), generator=torch.Generator().manual_seed(seed))
            for seed in range(5)
        ]
        for sample in samples[1:]:
            self.assertTrue(torch.equal(sample.tactic_action, samples[0].tactic_action))
            for step, first in zip(sample.arg_actions, samples[0].arg_actions):
                self.assertTrue(torch.equal(step, first))
        self.assertTrue(torch.equal(samples[0].tactic_action, torch.tensor([1, 1])))
        self.assertEqual(len(samples[0].arg_actions), 2)

    def test_greedy_takes_the_argmax_family(self):
        with torch.no_grad():
            _nodes, _state, logits, _values = self.model.encode(self.batch)
        sample = self.act(greedy=True)
        self.assertTrue(torch.equal(sample.tactic_action, logits.argmax(dim=1)))

    def test_sampling_explores_tactic_families_and_arguments(self):
        families, arguments = set(), set()
        generator = torch.Generator().manual_seed(0)
        for _ in range(64):
            sample = self.act(generator=generator)
            families.add(int(sample.tactic_action[0]))
            forced = self.act(tactic_mask=_only(0, 0), generator=generator)
            arguments.add(int(forced.arg_actions[0][0]))
        self.assertGreater(len(families), 1)
        self.assertGreater(len(arguments), 1)

    def test_a_seed_reproduces_its_draws_without_touching_the_global_rng(self):
        before = torch.get_rng_state()
        first = self.act(generator=torch.Generator().manual_seed(7))
        second = self.act(generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(torch.equal(first.tactic_action, second.tactic_action))
        for a, b in zip(first.arg_actions, second.arg_actions):
            self.assertTrue(torch.equal(a, b))


class ArityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.datas, self.vocab = _graphs(*STATES)
        self.model = actor_critic(len(self.vocab), len(ID_TO_TACTIC))
        self.model.eval()

    def act(self, datas, mask, **kwargs):
        with torch.no_grad():
            return self.model.act(
                Batch.from_data_list(datas),
                id_to_tactic=ID_TO_TACTIC,
                tactic_mask=mask,
                greedy=True,
                **kwargs,
            )

    def test_arity_resolution(self):
        self.assertEqual(action_arity("trivial", 3), (0, True))
        self.assertEqual(action_arity("exact", 3), (1, True))
        self.assertEqual(action_arity("have", 3), (2, True))
        self.assertEqual(action_arity("have", 1), (1, False))
        self.assertEqual(action_arity("<UNK>", 3), (0, False))
        self.assertEqual(action_arity(None, 3), (0, False))

    def test_a_shorter_row_sums_only_its_own_steps(self):
        alone = self.act(self.datas[:1], _only(0))
        mixed = self.act(self.datas, _only(0, 1))
        self.assertEqual(mixed.arities.tolist(), [1, 2])
        self.assertEqual(len(mixed.arg_actions), 2)
        # A higher-arity row in the batch adds a step the exact row does not own.
        self.assertAlmostEqual(float(mixed.arg_logp[0]), float(alone.arg_logp[0]), places=5)
        self.assertAlmostEqual(
            float(mixed.arg_logp[1]),
            float(mixed.arg_step_logp[0][1] + mixed.arg_step_logp[1][1]),
            places=5,
        )

    def test_a_zero_argument_action_has_no_argument_probability(self):
        sample = self.act(self.datas, _only(2, 0))
        self.assertEqual(sample.arities.tolist(), [0, 1])
        self.assertEqual(float(sample.arg_logp[0]), 0.0)
        self.assertLess(float(sample.arg_logp[1]), 0.0)

    def test_a_tactic_beyond_max_args_is_incomplete(self):
        model = actor_critic(len(self.vocab), len(ID_TO_TACTIC), max_args=1)
        model.eval()
        with torch.no_grad():
            sample = model.act(
                Batch.from_data_list(self.datas[:1]),
                id_to_tactic=ID_TO_TACTIC,
                tactic_mask=_only(1),
                greedy=True,
            )
        self.assertEqual(sample.arities.tolist(), [1])
        self.assertEqual(sample.complete.tolist(), [False])

    def test_a_row_without_a_legal_premise_marks_its_step_invalid(self):
        datas, _vocab = _graphs(*STATES)
        datas[0].premise_mask = torch.zeros_like(datas[0].premise_mask)
        sample = self.act(datas, _only(0, 0))
        self.assertEqual([bool(v) for v in sample.arg_step_valid[0]], [False, True])
        self.assertEqual(float(sample.arg_logp[0]), 0.0)


class EncodeOnceTests(unittest.TestCase):
    """One proposal round encodes each state once and draws only what it asks for."""

    def setUp(self):
        torch.manual_seed(0)
        self.datas, vocab = _graphs(*STATES, "x : Nat\n⊢ x = x")
        self.model = actor_critic(len(vocab), len(ID_TO_TACTIC))
        self.model.eval()
        self.batch = Batch.from_data_list(self.datas)

    def propose(self, counts, **kwargs):
        with torch.no_grad():
            return self.model.propose_actions(self.batch, counts, id_to_tactic=ID_TO_TACTIC, **kwargs)

    def test_one_encoder_pass_for_a_multi_draw_round(self):
        calls = []
        encode = self.model.encode_graph
        self.model.encode_graph = lambda data: calls.append(data.num_graphs) or encode(data)
        sample, rows = self.propose([4, 2, 1], generator=torch.Generator().manual_seed(0))
        self.assertEqual(calls, [3])
        self.assertEqual(rows.tolist(), [0, 0, 0, 0, 1, 1, 2])
        self.assertEqual(sample.tactic_action.shape, (7,))

    def test_unequal_counts_give_exactly_the_requested_instances(self):
        sample, rows = self.propose([0, 3, 1], generator=torch.Generator().manual_seed(1))
        self.assertEqual(rows.tolist(), [1, 1, 1, 2])
        self.assertEqual(sample.arities.shape, (4,))
        with self.assertRaises(ValueError):
            self.propose([1, 1])

    def test_each_instance_has_its_own_conditional_argument_probability(self):
        mask = torch.zeros(3, len(ID_TO_TACTIC), dtype=torch.bool)
        mask[:, [0, 1]] = True  # exact (one argument) or have (two)
        sample, rows = self.propose([3, 3, 3], generator=torch.Generator().manual_seed(3), tactic_mask=mask)
        for instance, row in enumerate(rows.tolist()):
            arity = int(sample.arities[instance])
            args = torch.tensor([[int(step[instance]) for step in sample.arg_actions[:arity]]])
            with torch.no_grad():
                tactic_logp, arg_logp, *_ = self.model.evaluate_actions(
                    Batch.from_data_list([self.datas[row]]),
                    sample.tactic_action[instance : instance + 1],
                    args,
                    tactic_mask=mask[row : row + 1],
                )
            # The instance's prior equals the recomputed probability of its own
            # tactic and arguments on its own state.
            self.assertAlmostEqual(float(sample.tactic_logp[instance]), float(tactic_logp[0]), places=5)
            self.assertAlmostEqual(float(sample.arg_logp[instance]), float(arg_logp[0]), places=5)

    def test_a_fixed_seed_repeats_and_leaves_the_global_rng_alone(self):
        before = torch.get_rng_state()
        first, _ = self.propose([2, 2, 2], generator=torch.Generator().manual_seed(7))
        second, _ = self.propose([2, 2, 2], generator=torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(torch.equal(first.tactic_action, second.tactic_action))
        for a, b in zip(first.arg_actions, second.arg_actions):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.allclose(first.arg_logp, second.arg_logp))


class SeedDerivationTests(unittest.TestCase):
    def test_seeds_depend_only_on_request_coordinates(self):
        seed = seed_from_text(goal="True")
        identity = input_identity(seed)
        self.assertEqual(identity, input_identity(seed_from_text(goal="True")))
        a = derive_search_seed("eval", 42, 0, identity)
        self.assertEqual(a, derive_search_seed("eval", 42, 0, identity))
        self.assertNotEqual(a, derive_search_seed("eval", 42, 1, identity))
        self.assertNotEqual(a, derive_search_seed("train", 42, 0, identity))
        self.assertNotEqual(a, derive_search_seed("eval", 43, 0, identity))
        self.assertLess(a, 2**63)

    def test_equal_inputs_are_told_apart_by_occurrence(self):
        seeds = [seed_from_text(goal="True"), seed_from_text(goal="False"), seed_from_text(goal="True")]
        specs = proposal_specs(seeds, mode="sample", namespace="eval", base_seed=1, pass_index=0)
        self.assertEqual(len({spec.seed for spec in specs}), 3)
        again = proposal_specs(seeds, mode="sample", namespace="eval", base_seed=1, pass_index=0)
        self.assertEqual(specs, again)
        # The first occurrence keeps its seed when the list grows.
        longer = proposal_specs([*seeds, seeds[1]], mode="sample", namespace="eval", base_seed=1, pass_index=0)
        self.assertEqual(longer[:3], specs)

    def test_specs_reject_unknown_modes_and_bad_seeds(self):
        with self.assertRaises(ValueError):
            ProposalSpec(mode="beam", seed=0)
        with self.assertRaises(ValueError):
            ProposalSpec(mode="sample", seed=-1)
        with self.assertRaises(ValueError):
            ProposalSpec(mode="sample", seed=True)


if __name__ == "__main__":
    unittest.main()
