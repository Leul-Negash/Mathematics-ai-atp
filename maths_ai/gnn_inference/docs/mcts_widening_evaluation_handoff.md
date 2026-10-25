# Complete-action priors, progressive widening and sampled evaluation

This is the handoff for issue #62 on MCTS-AC-branch. The design is in
[the plan](dev_plans/progressive_widening_and_sampled_mcts_evaluation.md);
the mechanisms are described in place in
[the walkthrough](rl_process_walkthrough.md). This file covers what changed,
how to run it and what was verified.

## Identities

| | |
| --- | --- |
| Target base | MCTS-AC-branch `dd6eedc` (merge of PR #61) |
| Client | `123nol/Pantograph-Client@d1dac7e`, version 0.1.1 (unchanged) |
| Mathlib | `29dcec07`, `lake exe cache get` |
| REPL | companion REPL built with `lake build repl` on `leanprover/lean4:v4.10.0-rc1` |

Every changed file is listed with its base and current hash in
[source_manifest.json](../tests/fixtures/pantograph/mcts_widening_evaluation/source_manifest.json).
`test_owned_client_migration.py` checks it as the next link after the
earlier manifests, which are unchanged.

## What changed

- **Proposal mode and seeds.** `prove(seed, *, proposal=ProposalSpec(...))`
  replaces `greedy=`. `greedy` takes the step-wise argmax of the tactic and
  every argument; `sample` draws both. Each search owns a `torch.Generator`
  seeded from its spec; seeds are derived from the input's replay
  fingerprint, never from the worker or the global RNG.
- **Complete-action priors.** A proposal is a `ProposedAction` with the raw
  action's unrounded log-prior: tactic log-prob plus the pointer log-probs of
  exactly the steps it owns. Edges keep the log-priors of the distinct raw
  actions that render their command, and `ProofHypergraph.puct_priors`
  normalizes their mass over the live (non-dead) edges on every selection.
  `EdgeVisitStats.prior_prob` is gone. The generic engine scores its ranked
  arguments with a softmax over the eligible pool, without replacement, and
  no longer proposes actions it could not complete.
- **Execution cache and observations.** A command runs in Lean once per node
  (`ProofNode.command_outcomes`). `RLSearchResult.observations` holds one
  `PolicyObservation` per node and raw action, with its multiplicity across
  rounds, in place of `edge_actions` and `failure_actions`.
- **Progressive widening.** A PUCT node's source stays open across rounds,
  bounded by `WideningPolicy` (distinct commands grow with completed visits;
  raw draws are capped). Selection requests a round where the capacity has
  grown since the node's last round, defers nodes with nothing to propose or
  select, and gives every selected node a visit at backup. A node is dead only
  when its source is closed and no edge survives. Harvest returns unknown for
  an open node whose known edges all failed.
- **Evaluation.** `eval_proposal_mode`, `eval_seed`, `eval_repeats` (and CLI
  overrides) apply to `--eval-only` and periodic evaluation alike. Results
  list every repeat and the searched-seed aggregate, with proposal and
  widening counters; `--eval-report` writes a new report file.
- **Checkpoints.** RL schema 4 stores the evaluation contract next to
  `best_proof_rate`, and the saved search settings include the widening
  bounds, proposal seed derivation and prior contract. Older or differently
  configured runs are refused for exact resume; their model weights remain
  valid warm starts. Model tensors, vocabularies and replay queue rows are
  unchanged.

## Running it

```bash
python -m maths_ai.gnn_inference.scripts.rl_train \
  --config maths_ai/gnn_inference/configs/rl_actor_critic.json \
  --eval-only --checkpoint /path/to/best.pt \
  --eval-proposal-mode sample --eval-seed 42 --eval-repeats 3 \
  --eval-report /path/to/new_eval_report.json
```

Use `--eval-proposal-mode greedy` for the deterministic diagnostic. Without
`--eval-only` the same settings drive periodic evaluation; training collection
always samples. The shipped config is a complete PUCT configuration with the
default widening bounds.

## Results

| Check | Result |
| --- | --- |
| Baseline focused set on `dd6eedc` | 244 passed, 1 skipped, 9 deselected (matches the plan) |
| Offline suite, `maths_ai` and `tests` | 563 passed, 13 skipped, 4 failed; the 4 also fail on `dd6eedc` (the `stdbuf` path test, three Thompson-sampler tests) |
| Live tests (`--run-pantograph-live -m pantograph_live`) | 11 passed, 1 skipped (the prepared-vocabulary/checkpoint gate) |
| Live widening (`verify_widening`) | first root round only a rejected command; a later round reuses its cached rejection (one Lean call) and adds `constructor`; both AND goals close; root priors sum to 1 |
| `rl_smoke --no-pln` | passes in `puct` and `legacy` with `sample` and `greedy` proposals, one on-policy step each |

The offline suite needs GNU `stdbuf` on `PATH`; with uutils `stdbuf` the
client refuses to start and one factory test waits forever. The live runs
used `PATH=$HOME/lean-env/bin:$HOME/.elan/toolchains/leanprover--lean4---v4.10.0-rc1/bin:$PATH`.

All results above are functional. None is a search-quality or proof-rate
measurement.

## Not done here

- Sampled and greedy evaluation, and a bounded training round, with a real
  prepared dataset and checkpoint. Neither is available on this machine, so
  the checkpoint gate was skipped; this is a blocked acceptance item, not a
  pass.
- Tuning `puct_c` for normalized priors and the widening defaults. The
  values are bounded starting points.
