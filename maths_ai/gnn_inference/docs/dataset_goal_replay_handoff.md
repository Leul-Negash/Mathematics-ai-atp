# Dataset goal reconstruction (#58) — handoff, 2026-10-22

Operating details are in [dataset_goal_replay.md](dataset_goal_replay.md). This
note records what was built, the identities it was verified against and what is
still open.

## Inputs

| Input | Identity |
| --- | --- |
| Application base | `AC-branch` at `59148bbf495b0b0dc7fa303169c42c7c9633931e` |
| Client | `pantograph-async-client` 0.1.1, `123nol/Pantograph-Client@d1dac7e8a8ea296192e7a5b2616114b73e0812b7` (merge of PR #2, which adds `start_goal(levels=...)`) |
| Companion REPL | `jajos12/Pantograph@0dd5d222` plus the client's patch, binary sha256 `ffc2a91985137f9183c4cc1571c08ece182996484feecf44b5e5d22d932d7dc0` |
| Lean / Mathlib | `leanprover/lean4:v4.10.0-rc1`, Mathlib `29dcec074de168ac2bf835a77ef68bbe069194c5` |
| Dataset | `jajostrains/Mathlib-Normalized-Sexpr` (now `iCogLabs-Dev/Mathlib-Normalized-Sexpr`), revision `f44841f98193c09b20c6092198a22e98ff2cd7a8`, raw schema 4 |
| Raw serializer | `serializeExpressionSexp` in Pantograph `Serial.lean`; identical in the dataset producer `22ddfaaf` and the companion |

## Stages

| Stage | Done |
| --- | --- |
| 0 Evidence | Dataset revision pinned; compact original rows (0, 5, 7, 27, 41) and one synthetic `π` row frozen in `tests/fixtures/dataset_replay/rows.json`, each marked; producer grammar read from `Serial.lean` |
| 1 Client | `start_goal(expression, *, levels=())`, strict encoding, docs, 0.1.1; client suite 420 offline, 5 live (including a named-universe goal), sdist/wheel check and 427 from the sdist against the wheel |
| 2 Text | Lexical declaration parser shared by every `parse_state` caller; universe analysis by position; typed seeds; `levels` forwarded; sanitizer removed |
| 3 Structured | `DatasetRow.structured_seed`, checked raw decoder and explicit renderer, raw sexps kept on canonical goals, one renderer for initial and intermediate replay |
| 4 Admission | Preflight on a dedicated client, quarantine by reason, reports that are never overwritten, identity-keyed cache, inadmissible seeds counted apart from failed searches, `validate_rl_seeds.py` |
| 5 Verification | Results below |

## Results

Same first 500 train rows at the pinned revision, Mathlib session, request
timeout 60 s. Reports: [admission_train_500.json](../tests/fixtures/dataset_replay/evidence/admission_train_500.json)
(new path, written by `validate_rl_seeds.py`) and
[base_path_train_500.json](../tests/fixtures/dataset_replay/evidence/base_path_train_500.json)
(base path, a one-off comparison script, not part of the branch).

| | Base path (59148bb) | This branch |
| --- | --- | --- |
| Rows reaching Lean | 497 | 320 after quarantine and budget skips |
| Accepted by Lean | 8 | 320 |
| Checked against the source | no | yes, all 320 match |
| Lean rejections | 489 (279 unknown universe level, 197 parse, 11 unknown identifier, 2 other) | 0 |

On this branch the other 180 rows are reported, not searched:

| Outcome | Rows |
| --- | --- |
| `unsupported-syntax`: private Mathlib constants, no source name | 51 |
| `missing-context`: `let` locals, schema 4 stores no value | 46 |
| `unresolved-universe` | 9 |
| `unresolved-term` | 3 |
| skipped `over-size-replay` (> 50,000 characters) | 63 |
| skipped `replay-repl-failure` (REPL printed a heartbeat error) | 8 |

All 320 admitted goals printed the predicted local names (no name drift);
289 have context-index gaps the dense replay closes. Admitted rows span 17
Mathlib areas (most from `Data`, `MeasureTheory`, `RingTheory`, `Algebra`,
`Topology`); every row over 1,600 printed characters was quarantined, and the
report lists coverage by area and size. Offline, on 400 sampled
rows, 115 of 348 renderable seeds rename at least one local (352 of 6,023
locals, almost all earlier `inst` instances).

### Model input

On the live fixture rows (5, 7, 41) the replayed goal's normalized target and
local types are identical to the dataset's `model_goal_sexp` and
`model_hyp_sexps`, and the binder roles match, once free-variable context
indices are renumbered. Two differences reach the GNN, and both need review
before long RL training:

- Context indices: 289 of the 320 admitted rows have gaps in their source
  context (hidden declarations the dataset does not list), and the replay is
  dense, so `FV` labels shift relative to the prepared supervised data. Nothing
  compacts them in code; Lean's rebuilt context simply has no hidden locals.
- Renamed locals: a local renamed to avoid capture (`inst_1✝` instead of
  `inst✝²`) is a new name label. Tactic arguments resolve against the live
  goal, so they use the replayed names consistently.

Existing graph and canonical goldens are unchanged, and no vocabulary or
checkpoint was modified.

### Text parser

Printed text compared with the old parser on the 400 sampled source states:
329 identical, 22 where a multiline type had been split into extra locals,
27 where a `let` value had been folded into the type, 22 whitespace only.

Tests, run with GNU `stdbuf` first on `PATH`:

| Run | Result |
| --- | --- |
| `maths_ai` and `tests` | 388 passed, 4 skipped, 5 failed; the same 5 fail at the base (worker-count config test, `stdbuf` path test on this host, 3 Thompson-sampler tests) |
| Live with Mathlib (`--run-pantograph-live`) | `test_pantograph_client_live.py` 2 passed; `test_dataset_replay.py` 1 passed (structured repairs, text rejections, universes, model-input equality, canonical, branch and cancelled replays, cleanup) |
| `rl_smoke` (live) | one on-policy optimizer step, finite losses |
| Translator `test_imports` | passed |
| Client suite | 425 with live tests (3 skipped without a companion checkout), sdist/wheel check, 427 from the sdist against the wheel |

The evidence run used the PR commit `455f8a3`; the merged pin `d1dac7e` adds only a CI
change, and the suites and live tests above were rerun with it installed from the pin.

## Decisions

- Universe names that the client can send are kept (`u_1` stays `u_1`); only
  names it cannot send get a `replay_uN` alias. The plan's example aliases every
  name, which would change every printed goal without changing what Lean
  checks; the mapping is recorded either way in `level_map`.
- A shadowed local that is still referenced is renamed rather than found by
  type: `‹Ring R›` left tactic metavariables that Mathlib's unifier could not
  resolve within the heartbeat limit on row 41.
- Client callers that start fixed goals without universes (the client's own
  probes and example) keep the default `levels=()`; every application caller
  passes its levels explicitly.

## For the MCTS port

MCTS-AC-branch needs the same cutover, not a separate mechanism:

- the client pin with `levels`, and the `lean_replay` package;
- `prove(seed)` as the only entry point, so `SearchFactory` and the pool pass
  `TheoremItem.seed`;
- MCTS node re-materialization through `materialize_seed(goal.replay_seed())`
  (it replays canonical goals the same way `_materialize_goal` does here);
- pool admission through `seed_admission`, and the `inadmissible` outcome kept
  out of `N`/`W`, HTPS data and the anneal clocks, as it is kept out of the
  solve rate here.

## Known, unrelated

Visible and unchanged: the worker-count config test (expects one worker, the
config has four), the `stdbuf` path test on hosts where `/usr/bin/stdbuf` is
uutils, three Thompson-sampler tests, and the stale `tests/test_protocol.py`
digest in the client's extraction manifest.

## Open

- Raw schema 4 records no `let` values, so rows with local definitions are
  quarantined (about one in ten of the sampled rows). Recording `value_sexp`
  in `local_context` or `raw_hyp_sexps` would admit them.
- About one row in ten refers to a private Mathlib constant
  (`_private.….0.…`), which no source name can reach; those are quarantined
  as `unsupported-syntax`.
- Universe level metavariables (`(:mv …)` in a sort) have no assignment in
  the dataset; those rows stay quarantined.
- When an earlier local's name is rebound and still used, it is renamed
  (`inst✝²` becomes `inst_1✝`); the admission report counts these and the
  replay records them. Those names are new graph labels for the GNN.
- The companion REPL prints an uncaught Lean exception (for example a
  heartbeat timeout) as plain text on stdout instead of a JSON error; the
  client then closes the session as a protocol failure. This is a REPL fix.
- Very large raw goals (row 16 is 408 KB) can exceed the request timeout;
  admission skips them as `replay-timeout`.
- MCTS-AC-branch adopts this in a later port.
