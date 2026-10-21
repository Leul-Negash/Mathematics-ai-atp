# Repository-owned client: reconciled source and contract

Reconciled on 2026-10-08 after [upstream PR #1](https://github.com/jajos12/Pantograph/pull/1).
Original Stages 0 and 1 of [the client plan](../docs/dev_plans/repository_owned_pantograph_async_client.md)
shipped for contract revision 1. [contract.json](contract.json) now specifies
revision 2. The codec, authored `future_contract.json`, and tests were subsequently
refreshed together to revision 2 on 2026-10-08. The Stage 1 exit gate passed;
Stage 2 transport/lifecycle is also implemented; its exit gate passed on October 8.
Stage 3 typed operations/state scopes are implemented and its fake-server exit
gate passed on October 8. Stage 4 is implemented and its exit gate passed on 2026-10-10. The companion
patch extends the merged Lean sources. Stages 5 and 6 passed their local/live
gates on 2026-10-17; Python production callers now use the owned client. Dependency
removal/reproducible setup and target deployment remain Stages 7–8. Do not treat
the local CPU checks as approval for a long RL run. Earlier dated handoffs below
retain their historical status; the latest handoff records the current state.

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
implementation gate is Stage 5 actual-client verification. Stage 4 passed its
companion Lean gate on 2026-10-10; Stages 5–8 remain pending. Production
migration still requires Stage 5 live verification.

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

Stage 2 delivered contract startup, `stat`, context management, and
awaited close. Generic transactions remain private; Stage 3 now supplies typed
goal operations and lexical scopes. Stage 2's READY means transport contract checks
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
migration. Its handoff authorized Stage 3, now completed below; later release gates remain unchanged. Nothing was
staged or committed, and the user configuration/downloaded environment was preserved.

## Stage 3 typed operations and state ownership handoff

`PantographClient.state_scope()` returns an async context manager exposing
`start_goal(expression)`, `apply_tactic(state, tactic, goal_index=0)`, and
`release(state)`. `StartedGoal` has one selectable initial goal but no fabricated
goal list. `TacticState` preserves decoded goals, messages, and validation.
The codec/types from Stage 1 required no schema changes for this stage.

Every allocation is registered in its scope synchronously under the transaction
lock. Input validation runs under that same lock before writing: correct loop,
session, active scope, exact issued state object, liveness, and selectable goal
index. Copied/altered execution wrappers are not issued states; canonical goal
copies may still outlive the scope. Empty materialized goals cannot be selected,
and normal tactic operations reject conversion/calc fragments.

Successful branches do not consume their input state. Early same-owner release
is idempotent, including concurrent calls. Scope exit batches remaining live IDs
and waits for one retained cleanup task on normal exit, exception, or repeated
cancellation. The batch includes results of requests already in flight when exit
begins. Closing prevents queued new allocations. Handles are invalidated on
process retirement, and retired scopes join close without issuing deletion.

Admission/unsafe results are registered and deleted before
`PantographTacticRejected` is raised. Deletion failure or cancellation retires and
reaps the process instead; infrastructure errors are not hidden as tactic failure.
Malformed evidence after a server allocation also retires the process. No state
destructor, garbage-collection cleanup, reset command, or in-place restart exists.

Verification on October 8: all 37 tests in
[test_pantograph_state_scopes.py](../tests/test_pantograph_state_scopes.py) passed.
They use actual diagnostic subprocesses, check fake allocation counts returning
to baseline and child reaping, and reject foreign/released/altered handles before
I/O. Coverage includes lock races, in-flight scope exit, duplicate release,
unmaterialized/empty states, rejection cleanup, validation/registration failures,
repeated cancellation, and replacement processes reusing numeric state zero.
The full GNN suite had 665 passed, 1 skipped, 5 subtests passed, and the unchanged
worker-count expectation failure recorded above. Threaded loop checks ran outside
the sandbox. Formatting and `git diff --check` passed.

The fake models independent snapshots. Parent deletion preserving descendants and
sibling alternatives must still be verified against real Lean in Stage 5.
Stage 4 may now implement the companion extension over the verified PR #1 base.
Production still uses PyPantograph; no downloaded REPL, Mathlib, dependency, or
user configuration was changed, and no new RL/deployment validation was performed.


## Stage 4 companion Lean contract handoff — 2026-10-10

Implemented on `client_centric_repo` in `/home/nolawi/pantograph_fork` over the
exact merged PR #1 base. The tracked
[companion patch](repository_client_contract.patch) and
[provenance descriptor](repository_client_contract.provenance.json) contain the
Lean implementation, migrated tests, reusable capture utility and checked-boundary
documentation. No source files were staged or committed.

The descriptor and successful transition metadata now match contract revision 2.
Shared goal operations capture real messages, reject logged errors, and restore
the caller's meta/elaborator/environment/message/trace/info/cache state. Full
selected-mode sequences consume their complete input. Synthetic elaboration runs
inside the transaction; unresolved reachable obligations remain goals. The scanner
follows assignments, delayed pending assignments, local types/values and relevant
introduced constant/projection dependencies. Missing or cyclic evidence produces
a named validation error. Each state preserves its original trust environment,
including across multiple transitions. Serialization restores its caller.

Merged omitted-index, nonzero selection, sibling resumption, calc, structured
operations and conversion repairs remain. The model serializer's normalized
expression strings, local indices, binder metadata and graph schema are preserved.

Verification: `lake build repl Test.ContractFixtures` and `lake test` passed,
with 299 Lean assertions including 37 new Stage 4 assertions. Existing Python
codec/reconciliation/model/environment regressions passed all 316 tests.
[Raw final-contract captures](../tests/fixtures/pantograph/stage4_companion/capture.json)
record 61 Mathlib exchanges accepted by the existing revision-2 decoder. All four
canonical object/graph goldens match exactly, including local internal names.
Initial and final allocated-state counts are zero, and the subprocess was reaped.
The final capture ran outside the sandbox; an earlier sandboxed shutdown timeout
is excluded from passed evidence. Patch application to the clean merged base was
verified, and all applied source hashes match the implementation checkout.

The build reused a separately copied archive dependency/build cache and rebuilt
changed targets. Stage 7 fresh release reproduction remains pending. Imported
Mathlib is the trust basis; its archive lacks Git metadata, so the deployment pin
remains an expected pin rather than a locally verified HEAD.

Stage 4 is complete. Stage 5 must verify the actual owned client, model probe and
state lifetimes against this artifact. Stages 6–8, production migration, fresh
release reproduction and RL/deployment validation remain pending. Transition
checks do not certify an assembled original theorem proof. No downloaded source,
reference executable, production dependency or user RL configuration was changed.

## Stages 5 and 6 handoff — 2026-10-17

The patched fork and its delivery artifacts were readable. The executable hash,
patch hash, and recorded Lean source hashes match the Stage 4 provenance. No
additional resource was required, and neither the fork nor the user's downloaded
environment was modified. The existing untracked Stage 4 exports are preserved.

Stage 5 delivered the CPU-only
[smoke utility](../scripts/pantograph_client_smoke.py),
[shared diagnostics](../../hybrid_reasoner/pantograph_diagnostics.py), and
[marked live tests](../tests/test_pantograph_client_live.py).
[Final evidence](../tests/fixtures/pantograph/stage5_client/final_report.json)
records the actual executable hash and Python source hashes, protocol/options,
model fingerprints and exact graph checks, normalized instance/local-let round
trips, constant/dependent-let replay, computed unsafe/admission rejection,
diagnostics, explicit focus, ancestor deletion, independent alternatives, 100
scoped proofs, zero final allocations, and confirmed process reaping. The initial
`report.json` predates the added dependent-let cases. Raw Stage 4 responses remain
under `stage4_companion/`; historical baseline and unextended PR captures remain
unchanged. A sandboxed early probe stalled and was interrupted; only successful
outside-sandbox runs are counted as live evidence.

Stage 6 replaced the factory/parser patch with the strict owned factory and one
startup deadline including model probes. The environment now requires explicit
paths. Reasoners/executors own search scopes and release starts, cached roots,
inputs, and results; searches on a shared reasoner are sequential. Typed
infrastructure failures propagate without learning records. The pool owns worker
replacement, joins startup/collection rollback, closes clients on their own loop,
and joins collection before shutting down all current clients. Shutdown cannot
miss a concurrently constructed replacement. Generation synchronization is
forbidden while collection is active.

Inference takes canonical goals and performs synchronous, gradient-free tensor
work without creating sessions or event loops. The single-reasoner collect/train
helpers must be awaited. Translator CLI/replay/export callers use named owned
projections, resolve local names explicitly, and validate paths before touching
outputs. Existing nonempty output directories are not deleted. Its package/facade
smoke uses repository-qualified imports. The unrelated stale STV smoke input was
updated to include the `rule-proof` marker already required by the unchanged log
parser, with a negative assertion for unmarked output.

Verification includes two required live tests with actual factory/executor use
and two independent clients sharing one CPU model binding. Missing designated
live-test paths fail explicitly rather than skipping. Fresh-interpreter imports
of the migrated production modules succeed with PyPantograph blocked. Dedicated
tests cover transport replacement, fatal contract/state propagation, constructor
failure, cancelled startup, active-loop inference/gradient suppression, learning
validity, scope lifetimes, and shutdown/replacement races. Translator import smoke
and bytecode compilation pass. See the current
[operating instructions](../docs/owned_pantograph_client.md) for notebook, smoke,
live-test and translator commands.

Final verification: the designated live module passed both tests without skips.
The full GNN suite had **656 passed, 3 skipped, 5 subtests passed**, and the one
pre-existing worker-count expectation failure described below. The three ordinary
run skips include the two separately passed live checks and the unavailable CUDA
case. All migrated-scope/factory/transport tests passed. Whitespace checks pass
with `cr-at-eol` enabled for the translator README's existing CRLF line endings;
the existing line-ending style was retained.

Stage 7 is next: remove the unused dependency, update setup to reproduce the
validated patch/build, and run clean-environment regressions. The pre-existing
worker-count test expects one while both committed/preserved configuration specify
four; it remains a separate full-suite release item. Conda/GPU/checkpoint and
remote RL validation remain Stage 8 work. No dependencies were uninstalled, no
training run was launched, and no files were staged or committed.

## MCTS-AC-branch port handoff — 2026-10-21

The owned client now runs `MCTS-AC-branch` as well. The port starts from MCTS
commit `89a2417e522720d2cfe3f2e6218614e4d828c8de` and reuses files from the
complete AC tree at `e8e4b18e081d6dd84491433b00e8cfa596cdfbee`; it is not a
merge of AC. Shared client, codec, contract, fixture, inference, and translator
files are byte-identical to that tree. `joint_inference.py`, `hypergraph.py`,
`rl_reasoner.py`, `rl_live_collection.py`, `rl_training_driver.py`, and
`pln_rl_training.py` were adapted in place, so PUCT selection, partial-hypertree
simulations, virtual losses, validity-aware backups, node-specific pending
actions, cooperative deadlines, and HTPS replay with its separate optimizer are
unchanged. The server epoch, in-reasoner restart, stale-leaf reproposal, private
close calls, and process inspection are gone; the pool replaces the complete
client/reasoner worker after a transport failure or hard/request timeout.

Baseline at `89a2417` (with PyPantograph's Python sources on the path): 319
passed, 1 skipped, 6 failed, and one collection error. The failures are three
Thompson-sampler tests, two old server-construction tests, and the stdbuf path
test; the collection error is the translator `test_imports.py` script. After
the port the full suite has **740 passed, 5 skipped, 5 subtests passed**, and
the 4 failures that also fail at the baseline: the three Thompson-sampler tests
and the stdbuf path test, which depends on the host's `stdbuf`. The five skips
are the four live cases in an ordinary run and the two-GPU topology case.

The designated live run passed all four live tests without skips against a
companion REPL rebuilt from the pinned fork commit and patch, with Mathlib
`29dcec074de168ac2bf835a77ef68bbe069194c5`. The MCTS evidence is in
`tests/fixtures/pantograph/mcts_owned_client/`; it records this branch's
revision, search-source digests, configuration, executable digest, and
unavailable resources. The AC `stage5_client/` reports remain AC results.

```bash
python -m pytest maths_ai/gnn_inference/tests/test_pantograph_client_live.py \
  maths_ai/gnn_inference/tests/test_mcts_owned_client.py \
  --run-pantograph-live -m pantograph_live -q
python -m maths_ai.gnn_inference.scripts.mcts_client_smoke \
  --source-root /path/to/compiled/mathlib4 \
  --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl \
  --report /new/path/mcts_report.json
```

Two fixes found during the port also apply to AC. The collection pool now
bounds startup and searches with `asyncio.timeout` instead of
`asyncio.wait_for`: on Python 3.11 a second cancellation during shutdown could
stop `wait_for` from waiting on a client whose cleanup was still running. The
spawn-cancellation test also gives its cancellation case a startup deadline it
cannot race.

Stages 7 and 8 remain pending on this branch as on AC: PyPantograph is still a
declared dependency, the setup script still pins the original unextended
Pantograph source and does not produce the validated executable, and no clean
environment install, Conda/GPU/checkpoint deployment, or training run was done.
