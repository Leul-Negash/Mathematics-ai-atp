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
