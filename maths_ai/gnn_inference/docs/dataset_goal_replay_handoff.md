# Dataset goal reconstruction (#58) — handoff, 2026-10-22

Operating details are in [dataset_goal_replay.md](dataset_goal_replay.md). This
note records what was built, the identities it was verified against and what is
still open.

## Inputs

| Input | Identity |
| --- | --- |
| Application base | `AC-branch` at `59148bbf495b0b0dc7fa303169c42c7c9633931e` |
| Client | `pantograph-async-client` 0.1.1, `Leul-Negash/Pantograph-Client@455f8a352d9d72a796e5b3e5cedf643c0909f954` (adds `start_goal(levels=...)`; not yet merged into `123nol/Pantograph-Client`) |
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
288 have context-index gaps the dense replay closes. Offline, on 400 sampled
rows, 115 of 348 renderable seeds rename at least one local (352 of 6,023
locals, almost all earlier `inst` instances).

Printed text compared with the old parser on the 400 sampled source states:
329 identical, 22 where a multiline type had been split into extra locals,
27 where a `let` value had been folded into the type, 22 whitespace only.

Tests, run with GNU `stdbuf` first on `PATH`:

| Run | Result |
| --- | --- |
| `maths_ai` and `tests` | 385 passed, 4 skipped, 5 failed; the same 5 fail at the base (worker-count config test, `stdbuf` path test on this host, 3 Thompson-sampler tests) |
| Live with Mathlib (`--run-pantograph-live`) | `test_pantograph_client_live.py` 2 passed, `test_dataset_replay.py` 1 passed |
| `rl_smoke` (live) | one on-policy optimizer step, finite losses |
| Translator `test_imports` | passed |
| Client suite | 420 offline, 5 live, sdist/wheel check, 427 from the sdist against the wheel |

The client used here was installed from a clean checkout of `455f8a3`.

## Open

- The client change is on my fork. After it is merged into
  `123nol/Pantograph-Client`, the pin in `pyproject.toml` and `uv.lock` moves
  to the merged commit.
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
