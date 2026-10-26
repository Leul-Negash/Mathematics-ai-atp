import itertools
import numpy as np
import pytest
import torch
from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import ArgumentReference, CandidatePoolSnapshot
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import LibraryMemory
from maths_ai.gnn_inference.atp_lean_gnn.source_aware_decoder import SourceAwareDecoder


def inputs():
    torch.manual_seed(31)
    model = SourceAwareDecoder(8, 3, max_args=2, text_dim=4)
    library = LibraryMemory([10, 11], ['Nat.add_comm', 'Nat.add_zero'], np.eye(2, 4, dtype=np.float32), 'fixture')
    pool = CandidatePoolSnapshot((0, 2), (10, 11), (.8, .4), artifact_fingerprint='fixture', library=library)
    return model, torch.randn(8), torch.randn(3, 8), torch.tensor(1), pool


def test_sample_and_forced_probabilities_match_and_mixed_gradients_flow():
    model, state, nodes, tactic, pool = inputs()
    for _ in range(20):
        sampled = model.run(state, nodes, tactic, pool)
        forced = model.run(state, nodes, tactic, pool, forced=sampled.arguments)
        torch.testing.assert_close(sampled.logp, forced.logp)
    with torch.no_grad():
        model.library_residual_scale.fill_(.4)
    args = (ArgumentReference('library', 11), ArgumentReference('local', 2))
    result = model.run(state, nodes, tactic, pool, forced=args)
    (-result.logp).backward()
    for component in (model.arg_count_head, model.source_head, model.argument_selector, model.lemma_adapter):
        assert sum(p.grad.abs().sum() for p in component.parameters() if p.grad is not None) > 0
    assert model.log_inverse_text_temperature.grad.abs() > 0


def test_joint_distribution_normalizes_and_entropy_surrogate_gradient():
    model, state, nodes, tactic, pool = inputs()
    choices = [ArgumentReference('local', x) for x in pool.local_ids] + [ArgumentReference('library', x) for x in pool.lemma_ids]
    logs = torch.stack([model.run(state, nodes, tactic, pool, forced=seq).logp
                        for count in range(3) for seq in itertools.product(choices, repeat=count)])
    probs = logs.exp()
    torch.testing.assert_close(probs.sum(), torch.tensor(1.))
    exact = (probs * logs).sum()
    surrogate = (probs.detach() * (logs.detach() + 1) * logs).sum()
    first = torch.autograd.grad(exact, tuple(model.parameters()), retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(surrogate, tuple(model.parameters()), allow_unused=True)
    for a, b in zip(first, second):
        if a is not None:
            torch.testing.assert_close(a, b)


def test_empty_support_allows_only_zero_arguments_and_rejects_drift():
    model, state, nodes, tactic, pool = inputs()
    result = model.run(state, nodes, tactic, CandidatePoolSnapshot(()))
    assert result.arguments == () and result.logp == 0
    with pytest.raises(ValueError, match='outside'):
        model.run(state, nodes, tactic, pool, forced=(ArgumentReference('library', 99),))


def test_pool_fingerprint_includes_order_query_and_artifacts():
    from dataclasses import replace
    _, _, _, _, pool = inputs()
    assert pool.fingerprint != replace(pool, query_fingerprint='other').fingerprint
    assert pool.fingerprint != replace(pool, local_ids=(2, 0)).fingerprint
    with pytest.raises(ValueError):
        replace(pool, lemma_scores=(float('nan'), .4))
