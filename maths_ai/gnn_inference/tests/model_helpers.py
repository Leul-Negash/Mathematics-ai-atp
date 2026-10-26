from __future__ import annotations

import hashlib

from maths_ai.data_models.proof_components import Goal, GoalLocal
from pantograph_async import (
    ExecutionGoal,
    ExecutionLocal,
    WireExpression,
)

from maths_ai.gnn_inference.atp_lean_gnn.model_factory import (
    build_actor_critic_model,
    build_pointer_model,
)
from maths_ai.gnn_inference.atp_lean_gnn.model_spec import ModelSpec


def spec(
    *,
    hidden_dim: int = 16,
    num_layers: int = 2,
    dropout: float = 0.1,
    max_args: int = 2,
):
    return ModelSpec.from_dict(
        {
            "architecture": "graphsage",
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "encoder": {"num_layers": num_layers},
            "use_node_type": True,
            "max_args": max_args,
        }
    )


def pointer(num_node_labels: int, num_tactics: int, **kwargs):
    return build_pointer_model(
        model_spec=spec(**kwargs),
        num_node_labels=num_node_labels,
        num_tactics=num_tactics,
    )


def actor_critic(num_node_labels: int, num_tactics: int, **kwargs):
    return build_actor_critic_model(
        model_spec=spec(**kwargs),
        num_node_labels=num_node_labels,
        num_tactics=num_tactics,
    )


def structured_goal(
    expression: str = "p → p", hypotheses: list[str] | None = None
) -> Goal:
    hypotheses = hypotheses or []
    locals_: list[GoalLocal] = []
    for index, hypothesis in enumerate(hypotheses):
        name, separator, type_pp = hypothesis.partition(":")
        if not separator:
            raise ValueError(
                f"Test hypothesis must have 'name : type' form: {hypothesis!r}"
            )
        locals_.append(
            GoalLocal(
                user_name=name.strip(),
                internal_name=f"_test.{index}",
                context_index=index,
                binder_role=":explicit",
                type_pp=type_pp.strip(),
                type_model_sexp=f"(:c Test.Type{index})",
                type_sexp=f"(:c Test.Type{index})",
            )
        )
    digest = hashlib.sha256(expression.encode("utf-8")).hexdigest()[:12]
    return Goal(
        expression=expression,
        goal_model_sexp=f"(:c Test.Goal{digest})",
        goal_sexp=f"(:c Test.Goal{digest})",
        locals=locals_,
        model_sexp_version=1,
    )


def pantograph_goal(goal: Goal) -> ExecutionGoal:
    goal.require_model_state()

    def expr(pp, model, raw=None):
        return WireExpression(
            pp=pp, sexp=raw or model, model_sexp=model, model_sexp_version=1
        )

    locals_ = tuple(
        ExecutionLocal(
            internal_name=local.internal_name,
            user_name=local.user_name,
            context_index=local.context_index,
            binder_role=local.binder_role,
            is_instance=local.is_instance,
            is_let=local.is_let,
            is_inaccessible=False,
            type=expr(local.type_pp, local.type_model_sexp, local.type_sexp),
            value=(
                expr(local.value_pp, local.value_model_sexp, local.value_sexp)
                if local.is_let
                else None
            ),
        )
        for local in goal.locals or []
    )
    return ExecutionGoal(
        goal_name="test-goal",
        case_tag=goal.case_tag,
        is_conversion=False,
        fragment="tactic",
        target=expr(goal.expression, goal.goal_model_sexp, goal.goal_sexp),
        locals=locals_,
    )


class FakeStateScope:
    """Scoped fake for search tests; transport ownership is tested with real children."""

    is_usable = True

    def __init__(self, client):
        self.client = client
        self.allocations = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        self.allocations.clear()
        self.is_usable = False

    async def start_goal(self, expression, *, levels=()):
        self.client.started_levels = tuple(levels)
        state = await self.client._start_goal(expression)
        self.allocations[id(state)] = state
        return state

    async def apply_tactic(self, state, tactic, *, goal_index=0):
        result = await self.client._apply_tactic(state, tactic)
        self.allocations[id(result)] = result
        return result

    async def release(self, state):
        self.allocations.pop(id(state), None)


class ScopedFakeClient:
    is_usable = True

    def state_scope(self):
        return FakeStateScope(self)

    async def aclose(self):
        self.is_usable = False


class LocalCandidateProvider:
    """Controlled local-only retrieval fixture; production uses verified artifacts."""
    async def retrieve(self, goal, local_ids, graph_fingerprint, *, excluded=()):
        from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import CandidatePoolSnapshot
        from maths_ai.gnn_inference.atp_lean_gnn.policy_actions import fingerprint
        from maths_ai.gnn_inference.atp_lean_gnn.retrieval_artifacts import goal_query
        return CandidatePoolSnapshot(tuple(local_ids),graph_fingerprint=graph_fingerprint,query_fingerprint=fingerprint(goal_query(goal)))
