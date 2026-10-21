# Running the Pantograph client

The application uses one async client contract. It no longer imports PyPantograph,
changes its parser methods, selects its bundled executable, or restarts a process
inside a reasoner. Stage 7 reproducible setup and Stage 8 target deployment
validation remain separate work.

The client itself is the standalone [`pantograph-async-client`](https://github.com/123nol/Pantograph-Client) package,
installed from the commit pinned in `pyproject.toml`. It owns transport, launch
validation, state scopes, the wire codec, protocol probes, and the companion
patch. This repository keeps only the application side in
[`pantograph_adapter.py`](../../hybrid_reasoner/pantograph_adapter.py)
(execution goals to canonical `Goal`/`GoalLocal`, translator projections, and the
graph-version check) and
[`pantograph_model_sexpr.py`](../../hybrid_reasoner/pantograph_model_sexpr.py)
(Mathlib model-session startup and canonical replay). Dependencies point one way:
`maths_ai` imports `pantograph_async`, never the reverse.

Use a Stage 4 companion executable, not the original download or unextended PR #1
build. Keep the existing checkpoint and prepared-data paths. Set the existing
`source_root` and `pantograph_repl` configuration keys to the compiled Mathlib
project and validated executable for the target machine. No binary copy into this
repository is necessary. Paths and Lean toolchains are checked before launch.

## Installing and upgrading the client

`uv sync` (or `pip install -e .`) installs `pantograph-async-client` from the exact
commit in its `pyproject.toml` dependency (`pantograph-async-client @ git+...@<commit>`),
and `uv.lock` records the same revision. PyPantograph is not installed. To upgrade:

1. Read the client release notes for contract changes. A new descriptor or
   `MODEL_SEXP_VERSION` needs a matching companion build and graph check here.
2. Change the commit in that dependency to the new immutable commit and run
   `uv lock`.
3. Rerun `setup_sexpr_environment` if the companion patch or pins changed, then run
   the adapter, factory, and live tests below.

`require_client_compatibility()` refuses to start a model session when the
installed client decodes a different model-expression version than the graph
representation expects.

## CPU-only verification

From the repository root, in the active Python environment:

```bash
python -m maths_ai.gnn_inference.scripts.pantograph_client_smoke \
  --source-root /path/to/compiled/mathlib4 \
  --pantograph-repl /path/to/Pantograph/.lake/build/bin/repl
```

The diagnostic needs no checkpoint, prepared dataset, GPU, or PeTTa process. It
first runs the client's protocol gate, then this repository's graph gate in a
second session. Together they check model/local-let/instance round trips, exact
graph inputs, ordinary failures,
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
from pantograph_async import PantographEnv
from maths_ai.hybrid_reasoner.pantograph_adapter import execution_goal_to_goal
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
        goal = execution_goal_to_goal(state.goals[0])
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
