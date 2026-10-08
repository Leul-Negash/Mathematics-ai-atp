# Repository-owned client: reconciled source and contract

Reconciled on 2026-10-08 after [upstream PR #1](https://github.com/jajos12/Pantograph/pull/1).
Original Stages 0 and 1 of [the client plan](../docs/dev_plans/repository_owned_pantograph_async_client.md)
shipped for contract revision 1. [contract.json](contract.json) now specifies
revision 2. The codec, authored `future_contract.json`, and tests were subsequently
refreshed together to revision 2 on 2026-10-08. The Stage 1 exit gate passed;
Stage 2 transport/lifecycle is also implemented; its exit gate passed on October 8.
Stage 3 may begin. Production factories, dependencies, and Lean
sources remain unchanged. Do not launch the proposed client for RL.

## Current verified source basis

The companion patch base is `https://github.com/jajos12/Pantograph.git`, branch
`gnn-sexpr-v410`, at merge `0dd5d2225e2d093f66a4b6231c1aa35004f2c8f3`.
Its first parent is the original base `73781c2d58456e4bf369dadd4a501e1b78a0b177`.
The branch was fetched into `/tmp/pantograph-pr1-reconciliation`, not the
downloaded archive. The clone was clean before capture. Source/build evidence
is recorded in [upstream_pr1/provenance.json](../tests/fixtures/pantograph/upstream_pr1/provenance.json).

The six changed files are `Pantograph/Goal.lean`, `Pantograph/Protocol.lean`,
`Pantograph/Serial.lean`, `Repl.lean`, `Test/Integration.lean`, and `Test/Proofs.lean`.
Their deliberate differences from the archive are the merged PR, not unidentified
local patches. Preserve the model serializer and merged conversion repairs.

Both projects retain `leanprover/lean4:v4.10.0-rc1`; compiler revision is
`3b58e0649156`. The expected deployment Mathlib pin remains
`29dcec074de168ac2bf835a77ef68bbe069194c5`. The downloaded Mathlib archive
has no Git metadata; that is an expected setup pin, not a locally verified HEAD.

`lake build repl` and `lake test` passed, including the merged protocol/conversion
tests. The archive's `.lake` cache was copied into the temporary clone and changed
targets were rebuilt. This is diagnostic validation, not Stage 7's fresh release
build. The new binary SHA-256 is
`15da3517f2d8b0592a1cefe6fc5e0b18323ee8027ed68d58783a233a43c25c78`.
The original downloaded executable, Mathlib archive, and user configuration were preserved.

## Live PR #1 evidence

[upstream_pr1/capture.json](../tests/fixtures/pantograph/upstream_pr1/capture.json)
contains 31 raw exchanges without PyPantograph, preserving original wire lines.
Initial/final allocated-state counts are both zero; the subprocess was reaped.

- Omitted `goalId` now selects index 0. Explicit nonzero selection and focused
  execution with automatic sibling resumption still work.
- Tactic variants contain top-level `hasSorry`, `hasUnsafe`, and `messages`.
  A live `sorry` sets `hasSorry=true`.
- Goals require `fragment` alongside `isConversion`. Captured canonical/model
  goal fields retain the original graph inputs.
- Parse/thrown-error strings are wrapped as messages with unavailable `(0, 0)`
  positions. Success still reports empty messages.
- `run_tac Lean.logError "capture error" <;> trivial` still succeeds with empty
  goals/messages and both flags false.
- `protocol.describe` and explicit `validation` metadata are absent.

Current flags inspect the instantiated selected-parent expression and directly
referenced unsafe constants; an unavailable parent defaults to false. Positive
unsafe-expression scanning is not established by these raw captures. Broader
transition checks, real message-log rejection, parser/rollback verification,
and completed-check metadata remain Stage 4 work. The unextended merged binary
is not the validated owned-client training contract.

## Revision-2 target contract

`contract_revision=2` tracks an unreleased planning revision, not a runtime mode.
`protocolVersion=1` remains the first intended deployed owned-client protocol.
The codec refresh migrated existing types, fixtures, and tests together to one
schema; no old/new decoding fallback exists.

| Area | Reconciled requirement |
| --- | --- |
| Requests | Minimal commands unchanged; owned tactics always serialize explicit nonnegative integer `goalId`; omission is not unfocused execution |
| Success | `nextStateId`, `goals`, `messages`, top-level `hasSorry`/`hasUnsafe`, and `validation` |
| Validation | Only `version=1`, `scope=transition-expressions`, `checked=true`; flags stay at top level, without duplicates |
| Parse/tactic rejection | `parseError` or `tacticErrors`, plus `messages` and both flags false; failed-operation flags are not proof evidence |
| Messages | Required `kind`, `severity`, `pos`, `data`; optional `endPos`; no `fileName` |
| Positions | Known: positive one-based line, nonnegative zero-based column; exactly `(0, 0)` means unavailable; end requires known ordered start |
| Goals | Required `fragment` in `tactic`/`conv`/`calc`; `isConversion` true exactly for `conv`; normal search accepts tactic fragments only |
| Model data | Canonical strings, local context indices, binder/instance/let metadata, and graph features unchanged |

The required options and command-error categories remain bound by `contract.json`.
Positive successful proof flags retain the decoded allocation for its scope owner
to delete before raising a domain rejection. Missing/indeterminate successful
evidence is fatal. Logged errors must produce tactic rejection, not successful
empty goals. No pretty text is substituted for normalized model expressions.

Immutable execution values and translator projections gain fragment metadata
during the completed refresh. Python's validation value combines top-level flags with
nested completion metadata; canonical graph inputs gain no new mode feature.
Canonical objects, fingerprints, DAG node features/roots/edges remain frozen in
[canonical_graph_expectations.json](../tests/fixtures/pantograph/canonical_graph_expectations.json).
The full caller/event-loop inventory and sequential gates remain in the main plan.

Stage 4 extends the merged scan to changed/coupled/reachable assignments,
introduced dependencies/local values, and retained obligations. It captures
real diagnostics, rolls back failed state mutations, detects hidden holes, and
rejects missing checks. Imported Mathlib remains the trust basis. Graph-based
success remains; complete assembled-theorem certification is a separate feature.

## Historical baseline and capture procedure

October 4 evidence at repository `216e074da75d8d5ca3011fd40146fd200f6a9715`
is retained in [baseline/provenance.json](../tests/fixtures/pantograph/baseline/provenance.json)
and [baseline/baseline.json](../tests/fixtures/pantograph/baseline/baseline.json).
The original verified source matched the downloaded archive except its omitted
`.gitignore`. The original executable hash was
`511e32d79635a4fe55f711f6fc1d42c6d5044203829b57eb37ab72e8227570c3`.
Original admission/logged-error responses had empty goals without flags/messages.
The original authored fixture is preserved in
[historical_revision1/future_contract.json](../tests/fixtures/pantograph/historical_revision1/future_contract.json)
and is rejected by the current codec. The active
[future_contract.json](../tests/fixtures/pantograph/future_contract.json) now holds
authored revision-2 expectations, not live companion-validation evidence.

The existing capture utility deliberately verifies this historical archive/base
pair. Do not point it at PR #1 and disable its source-equality guard. Its original
invocation remains:

```bash
.venv/bin/python maths_ai/gnn_inference/scripts/capture_pantograph_fixtures.py \
  --source-root maths_ai/sexpr_environment/sexpr_environment/mathlib4 \
  --repl-source maths_ai/sexpr_environment/sexpr_environment/Pantograph \
  --verified-base /tmp/pantograph-owned-client-base \
  --repl /tmp/pantograph-owned-client-base/.lake/build/bin/repl \
  --output /tmp/pantograph-baseline-new
```

It refuses existing outputs, dirty/mismatched sources, and toolchain mismatch.
New PR evidence used a temporary diagnostic harness reusing its raw `capture`
function with separate merge provenance and labels. These diagnostics resolve
`LEAN_PATH`, use `stdbuf -oL`, and drain/reap their subprocess; they are not the
Stage 2 production transport. The temporary harness and clone are not tracked delivery.

Keep original baseline, live unextended PR, and future final patched captures
distinct. Neither existing live group supplies the proposed full contract.

## Verification and next gate

Run offline evidence and existing codec tests from the repository root:

```bash
.venv/bin/python -m pytest \
  maths_ai/gnn_inference/tests/test_pantograph_pr1_reconciliation.py \
  maths_ai/gnn_inference/tests/test_pantograph_protocol.py -q
```

These tests freeze observed upstream behavior, source identity, revision-2 codec
acceptance/rejection, and unchanged model-bearing fields. They do not claim
companion validation is implemented. Stage 1's exit gate passed; the next
implementation gate is now Stage 3 scopes/typed operations. Stages 3–8 remain
unimplemented; production
migration requires Stage 4 validation and Stage 5 live verification.

October 8 refresh verification: 243 pure codec tests, 45 upstream evidence
assertions, and 28 existing environment/model tests passed (316 combined).
Canonical graph goldens pass for authored and live PR goal payloads. A fresh
interpreter prohibits importing PyPantograph, Torch, and model/training/search
implementations. Old revision-1 payloads and unextended PR successful transitions
are rejected. Merged parse/tactic failures decode with unavailable positions
preserved; positive checked success flags retain their result IDs for cleanup.
Formatting and `git diff --check` passed. The earlier Lean build/test and 31 live
exchanges remain source evidence, not execution of the future companion checks.

Broader refresh regression run outside the sandbox: 550 passed, 1 skipped,
5 subtests passed, and the same pre-existing worker-count expectation failure
described below. There is no Stage 1 blocker to starting Stage 2. Later gates
are not waived: Stage 4 must implement companion validation, Stage 5 must verify
the actual build, and Stage 6 alone switches production callers. The unrelated
configuration-test expectation remains a full-suite release item.

Original October 4 verification: 159 pure codec tests, 187 codec/environment/model
tests, original Lean build/test, and 31 raw exchanges passed. A broader run outside
the sandbox had 421 passed, 1 skipped, and 1 pre-existing configuration expectation
failure: a test expects `collection_workers=1`, but committed and user configuration
both specify 4. That configuration was preserved. No RL/GPU/checkpoint, Conda,
or remote deployment validation was performed.

## Stage 2 transport and lifecycle handoff

The standalone [transport](../../hybrid_reasoner/pantograph_client.py) and
[launch validator](../../hybrid_reasoner/pantograph_launch.py) are implemented.
The existing environment value supplies launch paths/imports/options, request
timing (`timeout`), startup/cleanup timing (`server_start_timeout_s`,
`worker_shutdown_timeout_s`), and the response/stderr limits. PyPantograph imports
are deferred until the unchanged application factory uses them.

Owned startup requires explicit Lake/REPL paths, toolchain agreement, executable
permissions, valid imports/options, GNU `stdbuf`, and a nonempty resolved
`LEAN_PATH`. It copies child environment settings without changing `os.environ`.
A non-checkout executable requires an explicit build manifest with matching
`toolchain` and `repl_sha256`. Stage 7 supplies fuller build provenance.

The client currently supports contract startup, `stat`, context management, and
awaited close. Generic transactions remain private; typed goal operations and
lexical scopes are Stage 3. Stage 2's READY means transport contract checks
passed, not that the later model-capability probe has run.

There is one stdout reader and one request lock. Request deadlines cover waiting,
write/drain, framing/decoding, and synchronous allocation bookkeeping. A cancelled
lock waiter leaves the active transaction healthy; cancellation or timeout after
a possible write retires/reaps the session. Replies cannot be reused afterward.
Shutdown joins normal readers before handing pipes to cleanup drainers. Helpers
and REPL use shared process ownership, bounded stderr tails, retained cleanup,
and termination/kill of their own process groups. Failure to confirm reaping is
reported without claiming CLOSED.

Verification on October 8: 78 new Stage 2 tests passed; 394 combined focused tests
passed. The full GNN suite had 628 passed, 1 skipped, 5 subtests passed, and the
same pre-existing worker-count expectation failure. Actual child processes cover
split/oversized/malformed/unsolicited output, resolver and spawn failures,
cancellation, ignored termination, lock fairness, independent clients, deadline
overruns, and retained/concurrent cleanup. Formatting and `git diff --check` passed.
The threaded cross-loop check ran outside the sandbox; the sandboxed stalled run
was interrupted and is not counted as a pass.

Stage 2 does not implement the Lean companion, model probe, or production caller
migration. Stage 3 may begin; later release gates remain unchanged. Nothing was
staged or committed, and the user configuration/downloaded environment was preserved.
