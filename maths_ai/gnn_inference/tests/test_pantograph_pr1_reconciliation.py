"""Offline assertions about live PR #1 evidence, not future client acceptance."""

from copy import deepcopy
import json
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures" / "pantograph"
BACKPORT = Path(__file__).parents[1] / "pantograph_backport"
CONTRACT = json.loads((BACKPORT / "contract.json").read_text())
PROVENANCE = json.loads((FIXTURES / "upstream_pr1" / "provenance.json").read_text())
CAPTURE = json.loads((FIXTURES / "upstream_pr1" / "capture.json").read_text())
ENTRIES = {entry["label"]: entry for entry in CAPTURE["interactions"]}
BASELINE = json.loads((FIXTURES / "baseline" / "baseline.json").read_text())
OLD_ENTRIES = {entry["label"]: entry for entry in BASELINE["interactions"]}
MERGE = "0dd5d2225e2d093f66a4b6231c1aa35004f2c8f3"


def test_contract_base_matches_verified_merge_and_unchanged_environment():
    assert CONTRACT["source_pins"]["pantograph"] == MERGE
    assert PROVENANCE["upstream_commit"] == MERGE
    assert PROVENANCE["upstream_branch"] == "gnn-sexpr-v410"
    assert PROVENANCE["merge_parents"][0] == CONTRACT["upstream"]["previous_base"]
    assert CONTRACT["source_pins"]["toolchain"] == PROVENANCE["toolchain"]
    assert (
        CONTRACT["source_pins"]["mathlib"]
        == PROVENANCE["mathlib_pin_expected_not_archive_verified"]
    )
    assert set(PROVENANCE["source_hashes"]) == set(PROVENANCE["tracked_changes"])
    assert len(PROVENANCE["tracked_changes"]) == 6
    assert PROVENANCE["build"]["lake build repl"] == "passed"
    assert PROVENANCE["build"]["lake test"] == "passed"
    assert PROVENANCE["repl_sha256"] != PROVENANCE["original_archive_repl_sha256"]


def test_planning_revision_does_not_add_a_runtime_schema_selector():
    assert CONTRACT["contract_revision"] == 2
    assert CONTRACT["descriptor"]["protocolVersion"] == 1
    assert CONTRACT["complete_theorem_certification"] is False
    assert set(CONTRACT["transition_validation"]) == {"version", "scope", "checked"}
    assert CONTRACT["tactic_responses"]["flag_location"].startswith("Top-level only")
    for variant in ("success", "parse_failure", "tactic_failure"):
        required = CONTRACT["tactic_responses"][variant]["required"]
        assert {"hasSorry", "hasUnsafe", "messages"} <= set(required)
    assert "validation" in CONTRACT["tactic_responses"]["success"]["required"]


@pytest.mark.parametrize("label", sorted(ENTRIES))
def test_capture_preserves_actual_wire_response(label):
    entry = ENTRIES[label]
    assert json.loads(entry["response_line"]) == entry["response"]
    command, request = entry["request_line"].split(" ", 1)
    assert command == entry["command"]
    assert json.loads(request) == entry["request"]


def test_omitted_goal_index_now_selects_zero():
    entry = ENTRIES["omitted_goal_id_defaults_to_zero"]
    assert "goalId" not in entry["request"]
    assert entry["response"]["goals"] == []
    assert "nextStateId" in entry["response"]
    assert CONTRACT["goal_selection"]["server_omitted_goalId"] == 0
    assert CONTRACT["goal_selection"]["owned_client"].startswith("Always serialize")


def test_admission_flag_is_present_but_not_completed_check_metadata():
    result = ENTRIES["admission_flag_present_without_validation"]["response"]
    assert result["goals"] == []
    assert result["hasSorry"] is True
    assert result["hasUnsafe"] is False
    assert "validation" not in result
    assert ENTRIES["unsupported_descriptor"]["response"]["error"] == "command"


def test_logged_error_remains_unreported_success_in_upstream_capture():
    entry = ENTRIES["logged_error_still_succeeds_without_diagnostics"]
    assert entry["request"]["tactic"] == (
        'run_tac Lean.logError "capture error" <;> trivial'
    )
    assert entry["response"]["goals"] == []
    assert entry["response"]["messages"] == []
    assert entry["response"]["hasSorry"] is False
    assert entry["response"]["hasUnsafe"] is False
    assert "validation" not in entry["response"]
    assert (
        "Logged errors must produce tacticErrors"
        in CONTRACT["message_schema"]["success_log_errors"]
    )


@pytest.mark.parametrize("label", ["parse_failure", "tactic_failure"])
def test_wrapped_failures_use_merged_message_shape_and_unknown_position(label):
    result = ENTRIES[label]["response"]
    assert result["hasSorry"] is result["hasUnsafe"] is False
    assert "nextStateId" not in result
    assert result["messages"]
    for message in result["messages"]:
        assert set(message) == set(CONTRACT["message_schema"]["required"])
        assert message["severity"] == "error"
        assert message["kind"] == "[anonymous]"
        assert message["pos"] == CONTRACT["message_schema"]["position"]["unavailable"]


@pytest.mark.parametrize(
    "label",
    [
        "identity_intro",
        "conjunction_split",
        "nonzero_index",
        "focused_all_goals",
        "instance_intro",
        "let_intro",
    ],
)
def test_new_goal_metadata_preserves_historical_canonical_model_fields(label):
    original = OLD_ENTRIES[label]["response"]["goals"]
    current = ENTRIES[label]["response"]["goals"]
    assert len(current) == len(original)
    for old_goal, new_goal in zip(original, current):
        assert new_goal["fragment"] == "tactic"
        assert new_goal["isConversion"] is False
        # Compare diagnostic evidence only; this is not a runtime schema adapter.
        model_bearing_fields = deepcopy(new_goal)
        del model_bearing_fields["fragment"]
        assert model_bearing_fields == old_goal


def test_focus_branch_reuse_and_state_count_are_preserved():
    assert ENTRIES["nonzero_index"]["request"]["goalId"] == 1
    assert len(ENTRIES["focused_all_goals"]["response"]["goals"]) == 1
    assert ENTRIES["identity_after_parent_delete"]["response"]["goals"] == []
    assert ENTRIES["identity_alternative"]["response"]["goals"] == []
    assert CAPTURE["readiness_line"] == "ready."
    assert len(CAPTURE["interactions"]) == 31
    assert ENTRIES["initial_stat"]["response"] == {"nGoals": 0}
    assert ENTRIES["final_stat"]["response"] == {"nGoals": 0}
    assert PROVENANCE["stderr_tail"] == ""
