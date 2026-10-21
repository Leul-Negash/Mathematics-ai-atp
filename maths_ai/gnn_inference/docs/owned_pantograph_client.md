# Running the repository-owned Pantograph client

The application uses one async client contract. It no longer imports PyPantograph,
changes its parser methods, selects its bundled executable, or restarts a process
inside a reasoner. Stage 7 dependency removal/reproducible setup and Stage 8 target
deployment validation remain separate work.

Use a Stage 4 companion executable, not the original download or unextended PR #1
build. Keep the existing checkpoint and prepared-data paths. Set the existing
`source_root` and `pantograph_repl` configuration keys to the compiled Mathlib
project and validated executable for the target machine. No binary copy into this
repository is necessary. Paths and Lean toolchains are checked before launch.

## CPU-only verification

From the repository root, in the active Python environment:

```bash
python -m maths_ai.gnn_inference.scripts.pantograph_client_smoke \
  --source-root /path/to/compiled/mathlib4 \
  --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl
```

The diagnostic needs no checkpoint, prepared dataset, GPU, or PeTTa process. It
checks model/local-let/instance round trips, exact graph inputs, ordinary failures,
admission and unsafe rejection, branch lifetimes, diagnostics, and 100 scoped
proofs. A successful report has equal initial/final allocation counts and a
confirmed closed process. `--report /new/path/report.json` saves evidence without
overwriting an existing file. This does not kernel-certify an assembled theorem
proof from independent search nodes.

The designated live test run must not silently skip missing resources:

```bash
export PANTOGRAPH_TEST_SOURCE_ROOT=/path/to/compiled/mathlib4
export PANTOGRAPH_TEST_REPL=/path/to/Pantograph/.lake/build/bin/repl
python -m pytest maths_ai/gnn_inference/tests/test_pantograph_client_live.py \
  --run-pantograph-live -q
```

## MCTS search verification

Best-first and PUCT search use the same scoped execution API. Each theorem owns one
state scope; a PUCT leaf stays allocated until all its candidate tactics finish, and
each successful alternative is converted to canonical goals and released at once. An
overlapping `prove` on the same reasoner is rejected before it touches the running
search's pending actions, greedy mode, or result. A transport failure or timeout ends
the search; the pool, not the reasoner, replaces the complete client/reasoner worker.

Checked tactic rejections become actor failure records. Goal elaboration failures and
session failures close the node without a critic label or failure record, and an
unknown simulation backup increments `N` but not `N_v` or `W`. Cooperative deadlines
still return the completed partial graph.

The search gate runs the client gate above, then both search modes against the actual
executable: a genuine multi-leaf PUCT batch, graph equivalence of search nodes with
the canonical expectations, checked rejection of alternatives, scope closure after a
cooperative cutoff, repeated searches on one session, two pool workers with
independent clients sharing one model device, and one bounded on-policy plus HTPS
update:

```bash
python -m maths_ai.gnn_inference.scripts.mcts_client_smoke \
  --source-root /path/to/compiled/mathlib4 \
  --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl \
  --report /new/path/mcts_report.json
```

The report records this branch's own identity (target base commit, repository
revision, and digests of the search modules) next to the client's executable
identity. The live tests require the same explicit paths:

```bash
python -m pytest maths_ai/gnn_inference/tests/test_pantograph_client_live.py \
  maths_ai/gnn_inference/tests/test_mcts_owned_client.py \
  --run-pantograph-live -m pantograph_live -q
```

The RL smoke uses the same factory and runs PUCT by default; pass
`--selection-policy legacy` for the best-first loop and `--no-pln` to run without
the petta binary:

```bash
python -m maths_ai.gnn_inference.scripts.rl_smoke \
  --source-root /path/to/compiled/mathlib4 \
  --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl --no-pln
```

## Notebook and inference use

Use `await` inside notebooks and one `asyncio.run` at terminal entry points. A live
client and reasoner must stay on their creating event loop. Each theorem or
materialization request owns a state scope:

```python
from pathlib import Path
from maths_ai.hybrid_reasoner.pantograph_env import PantographEnv
from maths_ai.hybrid_reasoner.pantograph_model_sexpr import (
    create_model_sexpr_client, materialize_replay,
)
from maths_ai.data_models.proof_components import LeanGoalSeed, seed_replay_spec

env = PantographEnv(
    source_root=Path('/path/to/compiled/mathlib4'),
    pantograph_repl=Path('/path/to/Pantograph/.lake/build/bin/repl'),
    imports=('Mathlib',),
)
async with await create_model_sexpr_client(env) as client:
    async with client.state_scope() as scope:
        replay = seed_replay_spec(LeanGoalSeed(expression='True'))
        state = await materialize_replay(scope, replay)
        goal = state.goals[0].goal
        # predictor is an already-loaded GNNPredictor; no subprocess or new loop.
        predictions = predictor.predict_tactics_with_arguments(goal)
```

Canonical goals may outlive the scope; live execution states may not. Use the
exact state object issued by its scope. Early release is idempotent, and tactic
results do not consume their input, allowing alternative branches. Tactics always
send an explicit zero-based goal index; unresolved siblings resume automatically.

`collect_and_train` and `collect_and_train_onpolicy` are async and must be awaited.
Single-reasoner searches are sequential. Cross-theorem parallelism belongs in the
pool, with one independent client/reasoner per worker. The pool replaces retired
workers only between theorems and completes collection before model updates.
Protocol/state errors stop collection; they are not negative learning samples.

## Translator

Generation now uses the same strict factory and scoped replay. From the repository
root:

```bash
python -m maths_ai.pln_inference.metta.translator.translator_modules.cli \
  --input tests.json --output /new/output/directory --axioms-path /path/to/axioms \
  --source-root /path/to/compiled/mathlib4 \
  --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl
```

In notebooks, await `run_demo(..., env=env, axioms_path=...)`. Generation refuses a
nonempty output directory; it does not clear existing user files. Ranking mode
does not launch Lean. Raw expression projections explicitly preserve local names,
types, and values rather than relying on an upstream dataclass layout.
