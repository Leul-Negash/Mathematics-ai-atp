"""Shared proof-state parsing, universe analysis, raw rendering and validation."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from maths_ai.lean_replay import (
    ReplayFailure,
    ReplayInputError,
    StructuredLocal,
    render_replay,
    seed_from_records,
    seed_from_text,
    validate_replay,
)
from maths_ai.lean_replay.columns import check_raw_schema, dataset_locals, decode_json_list


def reason(call) -> ReplayFailure:
    with pytest.raises(ReplayInputError) as error:
        call()
    return error.value.reason


def local(
    internal_id: str,
    name: str,
    type_sexp: str,
    index: int,
    role: str = ":explicit",
    value_sexp: str | None = None,
) -> StructuredLocal:
    return StructuredLocal(
        internal_id=internal_id,
        name=name,
        context_index=index,
        binder_role=role,
        is_instance=role == ":instance-implicit",
        is_let=role == ":let",
        type_sexp=type_sexp,
        value_sexp=value_sexp,
    )


# Universes ---------------------------------------------------------------------


def test_named_universes_are_declared_and_kept():
    spec = render_replay(
        seed_from_text("α : Type u_1\nβ : Type u_1\nγ : Sort (max u v)\n⊢ α = α")
    )
    assert spec.universe_levels == ("u_1", "u", "v")
    assert spec.expression.startswith("∀ (α : Type u_1), ∀ (β : Type u_1)")


def test_explicit_constant_universes_count():
    spec = render_replay(seed_from_text(goal="ULift.{u, v} α = ULift.{u, v} α"))
    assert spec.universe_levels == ("u", "v")


def test_a_term_named_like_a_universe_is_not_one():
    spec = render_replay(
        seed_from_text('u_1 : Nat\ns : String\n⊢ u_1 = u_1 ∧ s = "Type u_2" -- Type u_3')
    )
    assert spec.universe_levels == ()


def test_unsafe_universe_names_get_a_consistent_alias():
    spec = render_replay(seed_from_text("α : Type u₁\nβ : Sort (u₁ + 1)\n⊢ True"))
    assert spec.universe_levels == ("replay_u0",)
    assert spec.level_map == (("u₁", "replay_u0"),)
    assert "Type replay_u0" in spec.expression and "(replay_u0 + 1)" in spec.expression


def test_level_and_term_metavariables_are_distinguished():
    level = seed_from_text("α : Type ?u.319125\n⊢ True")
    term = seed_from_text("h : ?m.2235\n⊢ True")
    assert reason(lambda: render_replay(level)) is ReplayFailure.UNRESOLVED_UNIVERSE
    assert reason(lambda: render_replay(term)) is ReplayFailure.UNRESOLVED_TERM


# Inaccessible and shadowed names ------------------------------------------------


def test_inaccessible_locals_are_introduced_anonymously_under_their_base_name():
    spec = render_replay(seed_from_text("inst✝¹ : Inhabited Nat\ninst✝ : Inhabited Nat\n⊢ True"))
    assert spec.expression == "∀ (inst : Inhabited Nat), ∀ (inst : Inhabited Nat), True"
    assert spec.local_names == ("_", "_")
    assert spec.expected_names == ("inst✝¹", "inst✝")
    assert spec.name_map == ()


def test_a_shadowed_local_referenced_later_is_renamed_and_recorded():
    spec = render_replay(seed_from_text("x✝ : Nat\nx : Nat\n⊢ x✝ = x"))
    assert spec.expression == "∀ (x_1 : Nat), ∀ (x : Nat), x_1 = x"
    assert spec.local_names == ("_", "x")
    assert spec.name_map == (("x✝", "x_1✝"),)


# Structured records -------------------------------------------------------------


ALPHA = local("_a", "α", "(:sort (+ u_1 1))", 0, ":implicit")
INST = local("_i", "inst✝", "((:c Inhabited) (:fv _a))", 2, ":instance-implicit")
X = local("_x", "x", "(:fv _a)", 3)
TARGET = "((:c Eq) (:fv _a) (:fv _x) ((:c Inhabited.default) (:fv _a) (:fv _i)))"


def test_structured_seed_renders_explicit_terms_roles_and_levels():
    spec = render_replay(seed_from_records(TARGET, [ALPHA, INST, X]))
    assert spec.expression == (
        "∀ {α : (Sort (u_1+1))}, ∀ [inst : (@Inhabited @α)], ∀ (x : @α), "
        "(@Eq @α @x (@Inhabited.default @α @inst))"
    )
    assert spec.local_names == ("α", "_", "x")
    assert spec.universe_levels == ("u_1",)


def test_structured_validation_compares_by_position_and_reports_name_drift():
    seed = seed_from_records(TARGET, [ALPHA, INST, X])
    spec = render_replay(seed)
    replayed = [
        local("_r1", "α", "(:sort (+ u_1 1))", 0, ":implicit"),
        local("_r2", "inst✝", "((:c Inhabited) (:fv _r1))", 1, ":instance-implicit"),
        local("_r3", "y", "(:fv _r1)", 2),
    ]
    target = "((:c Eq) (:fv _r1) (:fv _r3) ((:c Inhabited.default) (:fv _r1) (:fv _r2)))"
    result = validate_replay(seed, spec, target, replayed)
    assert result.name_drift == (("x", "y"),)
    assert (result.source_indices, result.replay_indices) == ((0, 2, 3), (0, 1, 2))

    wrong = target.replace("Inhabited.default", "Classical.ofNonempty")
    assert reason(lambda: validate_replay(seed, spec, wrong, replayed)) is (
        ReplayFailure.SEMANTIC_MISMATCH
    )
    swapped = [replayed[0], replayed[1], local("_r3", "x", "(:fv _r1)", 2, ":implicit")]
    assert reason(lambda: validate_replay(seed, spec, target, swapped)) is (
        ReplayFailure.SEMANTIC_MISMATCH
    )


def test_shadowed_instances_referenced_later_are_renamed():
    # `[Ring R] [AddCommGroup M] [Module R M]`: every instance is `inst`, and the
    # module instance refers to both earlier ones after they are shadowed.
    seed = seed_from_records(
        "(:c True)",
        [
            local("_R", "R", "(:sort 1)", 0),
            local("_M", "M", "(:sort 1)", 1),
            local("_i2", "inst✝²", "((:c Ring) (:fv _R))", 2, ":instance-implicit"),
            local("_i1", "inst✝¹", "((:c AddCommGroup) (:fv _M))", 3, ":instance-implicit"),
            local(
                "_i0",
                "inst✝",
                "((:c Module) (:fv _R) (:fv _M) (:fv _i2) (:fv _i1))",
                4,
                ":instance-implicit",
            ),
        ],
    )
    spec = render_replay(seed)
    # Only the ring instance is used after `inst` is rebound; the additive
    # group instance is used before the last rebinding and keeps its name.
    assert "∀ [inst_1 : (@Ring @R)], ∀ [inst : (@AddCommGroup @M)]" in spec.expression
    assert "[inst : (@Module @R @M @inst_1 @inst)]" in spec.expression
    assert spec.local_names == ("R", "M", "_", "_", "_")
    assert spec.expected_names == ("R", "M", "inst_1✝", "inst✝¹", "inst✝")
    assert spec.name_map == (("inst✝²", "inst_1✝"),)


def test_a_shadowed_raw_local_referenced_later_is_renamed():
    seed = seed_from_records(
        "((:c Eq) (:c Nat) (:fv _a) (:fv _b))",
        [local("_a", "x✝", "(:c Nat)", 0), local("_b", "x", "(:c Nat)", 1)],
    )
    spec = render_replay(seed)
    assert spec.expression == "∀ (x_1 : @Nat), ∀ (x : @Nat), (@Eq @Nat @x_1 @x)"
    assert spec.name_map == (("x✝", "x_1✝"),)


def test_bound_variables_numerals_and_capture():
    # `x` inside the lambda would capture the local `x` the body refers to.
    seed = seed_from_records(
        "((:c Eq) (:c Nat) (:lambda x (:c Nat) ((:c HAdd.hAdd) 0 (:fv _x))) (:lit 3))",
        [local("_x", "x", "(:c Nat)", 0)],
    )
    expression = render_replay(seed).expression
    assert "(fun (x_1 : @Nat) => (@HAdd.hAdd @x_1 @x))" in expression
    assert expression.endswith("(nat_lit 3))")


def test_sanitized_binders_are_named_only_when_used():
    seed = seed_from_records(
        "((:c And) (:forall _ (:c Nat) (:c True)) (:forall _ (:c Nat) ((:c Eq) (:c Nat) 0 0)))",
        [],
    )
    expression = render_replay(seed).expression
    assert "(∀ (_ : @Nat), @True)" in expression
    assert "(∀ (x_1 : @Nat), (@Eq @Nat @x_1 @x_1))" in expression


def test_projection_placeholders_are_inferred_and_match_anything():
    seed = seed_from_records(
        "((:c Eq) (:c Nat) ((:c Prod.fst) _ _ (:fv _p)) (:lit 0))",
        [local("_p", "p", "((:c Prod) (:c Nat) (:c Nat))", 0)],
    )
    spec = render_replay(seed)
    assert "(@Prod.fst _ _ @p)" in spec.expression
    replayed = [local("_q", "p", "((:c Prod) (:c Nat) (:c Nat))", 0)]
    target = "((:c Eq) (:c Nat) ((:c Prod.fst) (:c Nat) (:c Nat) (:fv _q)) (:lit 0))"
    validate_replay(seed, spec, target, replayed)


def test_a_nested_application_head_is_flattened():
    seed = seed_from_records("(((:c HAdd.hAdd) (:c Nat) (:c Nat)) (:c Nat) (:c inst) (:lit 1) (:lit 2))", [])
    assert render_replay(seed).expression == (
        "(@HAdd.hAdd @Nat @Nat @Nat @inst (nat_lit 1) (nat_lit 2))"
    )


def test_local_definition_value_is_replayed():
    seed = seed_from_records(
        "((:c Eq) (:c Nat) (:fv _y) (:fv _y))",
        [local("_y", "y", "(:c Nat)", 0, ":let", "(:lit 2)")],
    )
    assert render_replay(seed).expression.startswith("let y : @Nat := (nat_lit 2); ")


def test_quoted_names_and_string_literals_are_rendered():
    expression = render_replay(seed_from_records('((:c "Foo.«a b»") (:lit "a\\b c"))', [])).expression
    assert expression == '(@Foo.«a b» "a\\\\b c")'


@pytest.mark.parametrize(
    "target,expected",
    [
        ("((:c Eq) (:mv _uniq.4))", ReplayFailure.UNRESOLVED_TERM),
        ("(:sort (+ (:mv _uniq.9) 1))", ReplayFailure.UNRESOLVED_UNIVERSE),
        ("(:subst (:mv _uniq.1) (:fv _uniq.2))", ReplayFailure.UNSUPPORTED_SYNTAX),
        ("(:c _private.Mathlib.Foo.0.bar)", ReplayFailure.UNSUPPORTED_SYNTAX),
        ("((:lambda x (:c Nat) 0) _)", ReplayFailure.UNSUPPORTED_SYNTAX),
        ('((:c Eq) (:lit "a"b"))', ReplayFailure.MALFORMED_STRUCTURED),
        ("((:c Eq) (:c Nat)", ReplayFailure.MALFORMED_STRUCTURED),
        ("(:fv _uniq.404)", ReplayFailure.MISSING_CONTEXT),
        ("((:c Eq) 0)", ReplayFailure.MALFORMED_STRUCTURED),
    ],
)
def test_unsupported_raw_input_is_rejected_with_its_reason(target, expected):
    assert reason(lambda: render_replay(seed_from_records(target, []))) is expected


@pytest.mark.parametrize(
    "records,expected",
    [
        ([local("_a", "a", "(:c Nat)", 0), local("_a", "b", "(:c Nat)", 1)], ReplayFailure.MALFORMED_STRUCTURED),
        ([local("_a", "a", "(:c Nat)", 1), local("_b", "b", "(:c Nat)", 1)], ReplayFailure.MALFORMED_STRUCTURED),
        ([local("_a", "a", "(:fv _b)", 0), local("_b", "b", "(:c Nat)", 1)], ReplayFailure.MISSING_CONTEXT),
        ([local("_a", "a", "(:c Nat)", 0, ":let")], ReplayFailure.MISSING_CONTEXT),
        ([local("_a", "a", "(:c Nat)", 0, ":unknown")], ReplayFailure.MALFORMED_STRUCTURED),
    ],
)
def test_structured_context_is_checked(records, expected):
    assert reason(lambda: seed_from_records("(:c True)", records)) is expected


def test_dataset_columns_align_by_checked_name_and_keep_index_gaps():
    context = [
        {"binder_role": ":explicit", "context_index": 0, "internal_name": "_a",
         "is_instance": False, "is_let": False, "user_name": "a"},
        {"binder_role": ":explicit", "context_index": 5, "internal_name": "_b",
         "is_instance": False, "is_let": False, "user_name": "b"},
    ]
    raw = [{"name": "a", "sexp": "(:c Nat)"}, {"name": "b", "sexp": "(:c Nat)"}]
    locals_ = dataset_locals(raw, context)
    assert [item.context_index for item in locals_] == [0, 5]

    renamed = [raw[0], {"name": "c", "sexp": "(:c Nat)"}]
    assert reason(lambda: dataset_locals(renamed, context)) is ReplayFailure.MALFORMED_STRUCTURED
    assert reason(lambda: dataset_locals(raw[:1], context)) is ReplayFailure.MALFORMED_STRUCTURED
    bad_model = [{"internal_name": "_a"}, {"internal_name": "_z"}]
    assert reason(lambda: dataset_locals(raw, context, bad_model)) is (
        ReplayFailure.MALFORMED_STRUCTURED
    )
    assert reason(lambda: decode_json_list("{not json", "local_context")) is (
        ReplayFailure.MALFORMED_STRUCTURED
    )
    assert reason(lambda: decode_json_list(json.dumps({"a": 1}), "x")) is (
        ReplayFailure.MALFORMED_STRUCTURED
    )
    assert reason(lambda: check_raw_schema(3)) is ReplayFailure.MALFORMED_STRUCTURED
    check_raw_schema(4)


def test_fingerprint_identifies_the_rendered_replay():
    first = render_replay(seed_from_text(goal="True", source={"row": 1}))
    again = render_replay(seed_from_text(goal="True", source={"row": 1}))
    other = render_replay(seed_from_text(goal="False", source={"row": 1}))
    assert first.fingerprint == again.fingerprint != other.fingerprint


def test_replay_package_imports_without_torch_gnn_or_client():
    code = (
        "import sys, maths_ai.lean_replay\n"
        "heavy = {'torch', 'pantograph_async', 'maths_ai.gnn_inference'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
