"""Model identity of goal locals versus their live execution identity."""

from __future__ import annotations

import pytest

from maths_ai.data_models.model_identity import (
    carry_model_identity,
    relabel_free_variables,
    with_model_identity,
)
from maths_ai.data_models.proof_components import Goal, GoalLocal


def local(name, internal, index, sexp="(:c Nat)", **extra) -> GoalLocal:
    return GoalLocal(
        user_name=name,
        internal_name=internal,
        context_index=index,
        binder_role=":explicit",
        type_pp="ℕ",
        type_model_sexp=sexp,
        type_sexp="(:c Nat)",
        **extra,
    )


def goal(locals_, target="(:app (:c Eq) (:fv FV0) (:fv FV1))") -> Goal:
    return Goal(
        expression="x = y",
        goal_model_sexp=target,
        goal_sexp="(:c True)",
        locals=locals_,
        model_sexp_version=1,
    )


# A replayed context is dense; the source had a hidden declaration at index 1.
REPLAYED = goal(
    [
        local("x", "_live.1", 0),
        local("inst_1✝", "_live.2", 1, sexp="(:app (:c Inhabited) (:fv FV0))"),
    ]
)
SOURCE = [("x", 0), ("inst✝¹", 2)]


def test_relabelling_changes_only_free_variable_labels():
    sexp = '(:app (:c "FV1") (:fv FV1) (:fv FV?) (:lit "(:fv FV1)"))'
    assert relabel_free_variables(sexp, lambda k: k + 5) == (
        '(:app (:c "FV1") (:fv FV6) (:fv FV?) (:lit "(:fv FV1)"))'
    )
    assert relabel_free_variables(sexp, lambda k: k) == sexp


def test_source_identity_replaces_live_names_and_indices_for_the_model():
    mapped = with_model_identity(REPLAYED, SOURCE)
    assert [(l.user_name, l.context_index) for l in mapped.locals] == SOURCE
    assert mapped.goal_model_sexp == "(:app (:c Eq) (:fv FV0) (:fv FV2))"
    assert mapped.locals[1].type_model_sexp == "(:app (:c Inhabited) (:fv FV0))"
    # Lean's identity is kept for execution and replay.
    assert [(l.execution_or_live_name, l.execution_or_live_index) for l in mapped.locals] == [
        ("x", 0),
        ("inst_1✝", 1),
    ]
    assert mapped.locals[0].execution_name is None and mapped.locals[0].execution_index is None
    assert mapped.execution_names() == {"inst✝¹": "inst_1✝"}
    assert [l.internal_name for l in mapped.locals] == ["_live.1", "_live.2"]


def test_model_identity_must_be_complete_and_ordered():
    with pytest.raises(ValueError):
        with_model_identity(REPLAYED, SOURCE[:1])
    with pytest.raises(ValueError):
        with_model_identity(REPLAYED, [("x", 2), ("inst✝¹", 0)])


def test_successors_keep_carried_identities_and_number_new_locals_after_them():
    parent = with_model_identity(REPLAYED, SOURCE)
    child = goal(
        [
            local("x", "_live.1", 0),
            local("inst_1✝", "_live.2", 1, sexp="(:app (:c Inhabited) (:fv FV0))"),
            local("h", "_live.9", 2, sexp="(:app (:c Eq) (:fv FV0) (:fv FV1))"),
        ],
        target="(:app (:c P) (:fv FV2))",
    )
    carried = carry_model_identity(parent, child)
    assert [(l.user_name, l.context_index) for l in carried.locals] == [
        ("x", 0),
        ("inst✝¹", 2),
        ("h", 3),
    ]
    assert carried.locals[2].type_model_sexp == "(:app (:c Eq) (:fv FV0) (:fv FV2))"
    assert carried.goal_model_sexp == "(:app (:c P) (:fv FV3))"
    assert carried.locals[2].execution_index == 2


def test_a_goal_without_a_mapping_keeps_leans_own_indices():
    parent = goal([local("x", "_live.1", 0), local("y", "_live.2", 3)])
    child = goal(
        [local("x", "_live.1", 0), local("y", "_live.2", 3), local("z", "_live.5", 7)],
        target="(:fv FV7)",
    )
    carried = carry_model_identity(parent, child)
    assert [l.context_index for l in carried.locals] == [0, 3, 7]
    assert carried.goal_model_sexp == "(:fv FV7)"
    assert all(l.execution_name is None and l.execution_index is None for l in carried.locals)


def test_renamed_and_removed_locals_follow_the_tactic():
    parent = with_model_identity(REPLAYED, SOURCE)
    # `rename_i h` gave the instance a new name; `x` was cleared.
    child = goal(
        [local("h", "_live.2", 0, sexp="(:c Inhabited)")], target="(:fv FV0)"
    )
    carried = carry_model_identity(parent, child)
    assert [(l.user_name, l.context_index) for l in carried.locals] == [("h", 2)]
    assert carried.goal_model_sexp == "(:fv FV2)"


def test_replay_seed_uses_live_identity_and_carries_the_model_one():
    seed = with_model_identity(REPLAYED, SOURCE).replay_seed()
    assert [(l.name, l.context_index) for l in seed.locals] == [("x", 0), ("inst_1✝", 1)]
    assert [l.model_identity for l in seed.locals] == SOURCE


def test_tactic_arguments_resolve_to_live_names():
    from maths_ai.gnn_inference.atp_lean_gnn.graph import model_goal_to_dag
    from maths_ai.gnn_inference.atp_lean_gnn.inference import (
        _local_names_by_fv_label,
        _resolve_local_node_name,
    )

    mapped = with_model_identity(
        goal([local("Nat", "_live.1", 0), local("y", "_live.2", 1)]),
        [("n", 0), ("y", 1)],
    )
    # The model saw `n`; Lean knows the local as `Nat`.
    dag = model_goal_to_dag(mapped)
    hyp = next(
        node for node in dag.nodes if node.label == "Hyp" and dag.nodes[node.children[1]].label == "n"
    )
    names = _local_names_by_fv_label(dag)
    assert _resolve_local_node_name(hyp, dag, names, mapped.execution_names()) == "Nat"
    assert _resolve_local_node_name(hyp, dag, names) == "n"
