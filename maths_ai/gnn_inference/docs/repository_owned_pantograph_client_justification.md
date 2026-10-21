# Why this repository is implementing its own Pantograph client

Date: 2026-10-08. Scope: the custom Lean REPL, existing PyPantograph integration,
and the [repository-owned async client plan](dev_plans/repository_owned_pantograph_async_client.md).

## Summary

Stage 4 is designed to preserve the existing PyPantograph wire interface while
strengthening the REPL's tactic execution and validation. Ordinary supported
operations are expected to remain usable with the inspected PyPantograph client.
The planned response shapes pass offline checks against that client's actual
Python methods. Compatibility with the completed executable is not yet confirmed:
Stage 4 has not been implemented or built.

The repository-owned client is not required merely to communicate with the custom
REPL. It is justified by the stronger execution contract this application needs:
preserving model-specific data, requiring explicit validation evidence, owning
proof-state lifetimes, and controlling asynchronous process cleanup. The recent
custom-fork update removes some original protocol incompatibilities, but it does
not supply this complete Python-side contract.

## Will existing PyPantograph still work after Stage 4?

### Inspected client and limits of the evidence

The local `.venv` contains the `pantograph` distribution version **0.3.15**,
installed from PyPantograph commit
`2c9fad2727c405c9976b766088ca761f14733f95`. The installed package's
`direct_url.json` confirms this source identity. This does not identify the client
installed in the separate JupyterHub/Conda environment.

The assessment uses the installed `Server.goal_tactic_async`, `GoalState.parse`,
`Goal.parse`, `Variable.parse`, and `Message.parse` implementations. It also checks
the repository's [model-expression parser extension](../../hybrid_reasoner/pantograph_model_sexpr.py).
The response inputs are the
[authored revision-2 fixtures](../tests/fixtures/pantograph/future_contract.json),
not output from a completed Stage 4 executable. Offline checks replace only
`Server.run_async` with a response provider and exercise the real tactic-response
handling method. They do not test process launch, Lean execution, or networking.

### Compatibility assessment

| Planned REPL behavior | Existing client's behavior | Assessment |
| --- | --- | --- |
| Preserve top-level `hasSorry` and `hasUnsafe` | Reads both flags; deletes the returned state and raises `TacticFailure` when either is true | Compatible with existing rejection handling |
| Add successful-state `validation` metadata | `GoalState.parse` reads selected fields and ignores this addition | Compatible response parsing, but no enforcement of the new evidence |
| Add `protocol.describe` | Existing startup does not require this command | Additive command; the old client does not gain capability verification |
| Preserve omitted `goalId` selecting goal index 0 | Default `Site()` omits `goalId` | The Stage 4 plan explicitly retains this behavior |
| Preserve `fragment`, goal fields, and model-expression serialization | Goal parser requires `fragment`; repository extension preserves model expressions and local metadata | Compatible planned shapes; custom model metadata still needs the repository extension |
| Preserve string-array `tacticErrors` and structured `messages` | A response without `goals` becomes `TacticFailure`; its messages are parsed | Existing failure path remains usable |
| Preserve message severities and `(0, 0)` positions when unavailable | Parses `information`, `warning`, and `error`, including those positions | Compatible parsing; the old client renders the coordinates without the owned client's unavailable-position interpretation |
| Return a named `validation` command error when inspection cannot complete | Any top-level `error` becomes `ServerError` | Error is surfaced, but not with the owned client's dedicated validation/protocol policy |
| Reject logged errors, trailing tactic input, or previously undetected unsafe/admitted results | May now receive failures where it previously received success | Intentional semantic change, not a guarantee of identical results |

The offline checks passed for all six safe success fixtures, admission and unsafe
rejections with deletion requests, parse/tactic/logged-error failures, and a
synthetic named validation error. All three message severities and unavailable
positions parsed. Repeating safe success checks with the repository parser
extension also passed, and normalized model expressions were retained.

One negative check is important: changing `validation.checked` to `false` still
produced success in PyPantograph. Its parser does not enforce this field. A client
that accepts the wire format is not necessarily a client that verifies the
repository's execution requirements.

These checks support **expected compatibility for the inspected client and
planned core operations**, not unconditional compatibility for every PyPantograph
release, every additional server operation, or the remote Conda installation.
Stage 4 also preserves extra server operations and conversion repairs, but those
operations have not all been exercised in this assessment.

### Verification still required after Stage 4

Run the installed PyPantograph client against the actual patched executable with
the explicit Mathlib source root, custom executable path, and existing launch
configuration. Check startup, goal creation, tactic success, omitted and explicit
goal indices, diagnostics, ordinary failures, admission/unsafe rejection, and
state deletion. Test any additional operations external users depend on.

Repeat the relevant checks in the actual JupyterHub/Conda environment after
recording its installed client version and source. The existing line-buffering
launch repair is separate from response-schema compatibility and remains needed
where the executable buffers interactive stdout.

Maintaining REPL wire compatibility does not require a selectable PyPantograph
backend in this repository. The plan still migrates all production callers to the
owned client together in Stage 6.

## Why own the Python client?

### 1. Preserve the data the model actually consumes

The custom REPL emits normalized model S-expressions, stable local-context
indices, and binder/instance/let metadata. Stock PyPantograph primarily decodes
pretty-printed expressions and its own generic goal structures. This repository
currently replaces upstream `Goal.parse` and `Variable.parse` globally to retain
the custom data, with checks against the upstream dataclass layout.

The owned codec decodes the custom schema directly into repository-owned types.
Changes to an upstream parser's structure no longer determine whether model
inputs can be constructed. Canonical expression strings and graph features must
remain unchanged during migration so existing checkpoint inputs remain valid.

### 2. Require validation evidence rather than assume it

The custom fork's merged PR #1 supplies top-level admission/unsafe flags and
repairs omitted goal selection. It does not yet expose `protocol.describe` or
the successful-state validation metadata required by this plan. The recorded
live logged-error example still returns success with empty goals and false flags.

The companion Lean extension must capture actual diagnostics and inspect the
relevant transition expressions. The owned Python codec requires the declared
validation version and scope, `checked=true`, and valid proof flags before a
successful tactic response can be accepted. Missing evidence is an infrastructure
failure, not a failed theorem and not a successful transition.

Python cannot reconstruct these guarantees from an empty goal list. Replacing
the client does not remove the need to fix the shared Lean execution helpers.

### 3. Make proof-state ownership deterministic

Upstream `GoalState.__del__` appends a state identifier to a deferred deletion
list. Python object collection does not provide an explicit theorem-level release
boundary. The owned design associates state handles with one client session and
one scope, rejects foreign or released handles, and awaits scope cleanup.

This is particularly useful when many theorem attempts create branches, time out,
or are cancelled. Stage 3 will implement typed operations and state scopes; the
completed transport stage alone does not yet provide that full API.

### 4. Control the entire asynchronous request lifetime

The owned transport serializes requests to each REPL process, uses one event-loop
owner, and applies a deadline across request admission, writing, response reading,
decoding, and allocation bookkeeping. If cancellation occurs after a request
might have reached Lean, it retires that session rather than allowing an old
response to be mistaken for a later response.

It also drains stderr with bounded retention and awaits process-group shutdown
and child reaping. These are application requirements during parallel collection,
not claims that PyPantograph cannot be repaired to provide comparable behavior.

### 5. Separate learning failures from infrastructure failures

An invalid tactic is a normal search outcome. An incompatible executable,
malformed response, missing validation evidence, or broken process is not.
Repository-owned exceptions give training and evaluation callers a shared basis
for deciding which outcomes can contribute to learning and which must stop or
retire collection. All callers must migrate to that policy together.

### 6. Make custom executable provenance reproducible

The design launches an explicit executable with an explicit project root and
checks toolchain/build identity. The descriptor provides a runtime contract check;
the build manifest records source, patch, toolchain, and executable identity.
This reduces the risk of testing one custom build and training with another.

## Alternatives and tradeoffs

| Approach | Benefits | Costs and limitations |
| --- | --- | --- |
| Keep PyPantograph and apply shared Python repairs | Lower initial migration cost; retains its broader API; PR #1 resolves important original mismatches | Still needs explicit custom-data decoding, validation enforcement, lifetime/transport repairs, and ongoing upstream integration work |
| Maintain a PyPantograph fork | Can implement those guarantees while retaining the wider API | Owns a larger maintenance surface and upstream merge work; still requires the Lean validation extension |
| Implement the narrow repository-owned client | Direct contract ownership; no global upstream parser replacement; scope and transport policies match this application | Repository maintains Python transport/types/tests and must migrate every caller; does not automatically provide PyPantograph's whole API |
| Update only the custom REPL | Repairs shared Lean behavior for all clients and preserves existing Python integration | Existing Python still ignores validation evidence and retains its current state/transport policies |

The chosen approach is the **narrow owned client plus shared Lean repairs**.
Keeping PyPantograph with comprehensive Python repairs remains technically viable.
The recommendation depends on this repository needing the full execution and
lifetime contract, not on a claim that basic wire communication is impossible.

## Boundaries, costs, and current status

- Stages 0–2 are implemented development components. Typed state scopes, the
  Lean companion extension, actual-executable verification, and production
  migration remain later work. Current production callers still use PyPantograph.
- The unextended PR #1 executable is intentionally rejected by the owned client's
  required capability handshake. There is no validation bypass.
- Success remains graph-based search success with validated Lean transitions.
  Assembling and kernel-certifying a complete theorem proof from independently
  replayed graph nodes is a separate feature.
- Replacing the client does not itself guarantee higher GPU utilization or faster
  training. Inference batching, worker counts, and collection/update boundaries
  remain separate mechanisms. Multiple REPL workers can share one GPU.
- Maintenance includes the narrow Python command API, Lean patch, pinned build
  procedure, captured fixtures, cancellation/process tests, and model-graph
  equivalence tests. Production readiness requires the later deployment gates.

## Repository evidence

- [Staged implementation and Lean requirements](dev_plans/repository_owned_pantograph_async_client.md)
- [Revision-2 execution contract](../pantograph_backport/contract.json)
- [Source provenance and captured live findings](../pantograph_backport/README.md)
- [Pure response codec](../../hybrid_reasoner/pantograph_protocol.py)
- [Owned async transport](../../hybrid_reasoner/pantograph_client.py)
- [Explicit launch validation](../../hybrid_reasoner/pantograph_launch.py)
- [Existing model-expression parser integration](../../hybrid_reasoner/pantograph_model_sexpr.py)
- [Alternative-client assessment](pantograph_client_options_report.md)
- [Custom-REPL update feasibility](pantograph_repl_update_feasibility.md)
