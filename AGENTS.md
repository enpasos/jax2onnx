# AGENTS.md

## Project overview

This monorepo now runs entirely on the **IR-only** pipeline:

* `converter2/` – main entrypoint. Uses `onnx_ir` to build an intermediate representation, then serializes to ONNX later.
* `plugins2/` – modular lowering for JAX/Flax primitives into IR (no ONNX proto imports here).
* `tests/` – unit, integration, and policy tests targeting the new world (legacy `extra_tests/`, `examples/`, and `plugins/` are gone).

* Docs: `docs/design.md` covers the core vs. plugin architecture, and `docs/subgraph_input_handling.md` explains ONNX control-flow subgraph wiring.

**Python**: 3.11+ (some users run 3.12 successfully).
**Packaging**: Poetry.
**Style**: Ruff (lint+format), mypy (type hints), Black-ish formatting via Ruff.

### Recent updates (2025-10-02)

- All NNX example modules now construct RNGs via `with_rng_seed(...)` and avoid inline `lambda` layers—mirror that pattern when adding tests so `construct_and_call` stays hashable under JAX 0.7.
- The `jax.nn`/`flax.nnx` dot-product-attention lowering now normalizes masked weights safely (`NaN` only in float64); keep plugin changes in sync with test expectations when touching attention masks.

### Compatibility (2025-10)

- Our supported toolchain target is **JAX ≥0.7.2** and **Flax/NNX ≥0.12.0**. These upgrades introduced a few important behavioural changes that the codebase already accounts for:
  * JAX 0.7.x tightened tracing rules. We **no longer force** `jax_dynamic_shapes` on startup for these versions—doing so produces `jit` staging crashes. The flag is still enabled automatically for older (<0.7.0) JAX releases.
  * JAX 0.7.x requires hashable primitive parameters; the plugin system’s `construct_and_call(...).with_dtype(...)` now builds the module **once per dtype** and reuses the instance instead of constructing inside the traced call. This prevents stray `jit` primitives from appearing in jaxprs.
  * Flax NNX 0.12.0 enforces strict pytree semantics. All plugin/example modules have been updated to wrap array-holding containers with `nnx.List(...)` (or explicit `nnx.data(...)`) so attributes with Arrays are marked as data. When adding new NNX examples, follow the same pattern.

---

## Quick start (build & test)

```bash
# 0) setup
poetry install -E all

# 1) lint/format/type-check
poetry run ruff check .
poetry run ruff format .     # formats in-place
poetry run mypy src          # or package root if different (see pyproject)

# 2) run the full test suite (must be green)
poetry run pytest -q

# 3) run a focused test while iterating
poetry run pytest -q tests/path/test_file.py::TestClass::test_case
```

> ✅ **Rule of thumb**: If tests aren’t green locally, you’re not done.

---

## Repo conventions & constraints

### IR vs ONNX (critical)

* The **converter2** pipeline and **plugins2** plugins must be **IR-only**.
  **Do not** import `onnx` (protobuf) in `converter2/*` or `plugins2/*`. There is a policy test under `tests/extra_tests2/framework/test_no_onnx_in_converter2_plugins2.py` that will fail if you do.

* ONNX **protobuf** shape inference or serialization should live in top-level adapters, not in IR passes or plugins.

### Old world removal (current focus)

* Legacy `converter/`, `plugins/`, `examples/`, and `tests/extra_tests/` have been removed. Clean up any remaining imports, registry shims, or docs that still reference them.
* Update tooling (`scripts/`, `tests/`, docs) to rely solely on `converter2` + `plugins2` resources. If something still depends on the old modules, replace it with the IR-only equivalent or delete it.
* Keep an eye on `MigrationStatus.md`: it now tracks only IR coverage. Regenerate it (`poetry run python scripts/generate_migration_status.py`) after adding or pruning tests so the status stays accurate.
* Deleting unused assets is encouraged, but do it incrementally with green tests. Prefer one plugin/test family per change set to keep diffs understandable.

### Randomness & module construction (critical)

* **Never seed at import time.** Constructors must receive an explicit `jax.random.PRNGKey` (Equinox) or `nnx.Rngs` (Flax NNX). No module-level `PRNGKey(...)` or `nnx.Rngs(...)`.
* **Keys are single-use.** Split once per independent consumer (e.g., params vs. dropout). While iterating locally, enable `jax.config.update("jax_debug_key_reuse", True)` to catch reuse bugs immediately.
* **Expose explicit callables in metadata/tests.** Wrap stochastic objects with `construct_and_call(...)` from `plugins2.plugin_system` and use the placeholders `with_requested_dtype()`, `with_rng_seed(seed)`, or `with_prng_key(seed)` when dtype/seed must track the test harness.
* **Do not use `callable_factory`.** All metadata should provide a `"callable"` entry built via `construct_and_call` so the generator can rebuild modules for f32/f64 variants automatically.

Example:

```python
"callable": construct_and_call(
    nnx.LinearGeneral,
    in_features=(8, 32),
    out_features=(256,),
    axis=(-2, -1),
    dtype=with_requested_dtype(),
    param_dtype=with_requested_dtype(),
    rngs=with_rng_seed(0),
),
```

### Numeric validation parity

* `skip_numeric_validation` is reserved for stochastic cases. A `plugins2` testcase may only set it when the matching legacy testcase does as well—see `tests/extra_tests2/framework/test_do_not_skip_numeric_validation.py`.

### Structure

* `converter2/ir_optimizations.py` contains IR-level graph passes (reshape/transpose pair folding, Dropout cleanup, dead code elimination, etc.).
* `plugins2/_post_check_onnx_graph.py` provides **expect\_graph** – structural assertions helpers for tests.
* `plugins2/flax/nnx/*` contains Flax NNX lowering (e.g., `dropout.py`).

### Coding style

* Keep functions **<100 lines**. Prefer small, pure helpers.
* Use explicit imports, add/extend tests with every change.
* Public APIs: no breaking changes without updating docs, tests, and changelog.

---

## Debugging toolkit

Set these env vars to trace particular subsystems (print to stdout):

* `JAX2ONNX_IROPT_DEBUG=1` – general IR optimizer traces.
* `JAX2ONNX_RSH_DEBUG=1` – reshape pair folding.
* `JAX2ONNX_TRN_DEBUG=1` – transpose pair folding.
* `JAX2ONNX_DCE_DEBUG=1` – dead code elimination (which nodes were dropped).
* `JAX2ONNX_TM_DEBUG=1` – **training\_mode inlining** for Dropout (logs proof, producer indices, readbacks).

Example:

```bash
JAX2ONNX_TM_DEBUG=1 poetry run pytest -q tests/extra_tests2/framework/test_ir_optimizations.py::test_dropout_training_mode_inlined_constant_false_and_not_removed
```

---

## Testing patterns you’ll see (and should follow)

### expect\_graph – readable structural tests

Use `expect_graph` to assert shapes/paths/operators without heavy fixtures:

```python
from jax2onnx.plugins2._post_check_onnx_graph import expect_graph as EG2

check = EG2(
    ["Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10"],
    symbols={"B": None},            # unify symbols
    must_absent=["Not", "Identity"],
    no_unused_inputs=True,
    mode="all",
    search_functions=False,         # set True to scan function bodies too
)
assert check(model)
```

Supported patterns:

* `Op[:shape] -> Op[:shape] -> ...`
* Shapes like `Bx20`, `?x10`, or concrete `7x20`.
* `symbols` unify symbolic dims across the path.
* `must_absent` ensures certain ops do not exist anywhere.
* `no_unused_inputs` fails if graph has dangling inputs.

### Focused test runs

```bash
poetry run pytest -q tests/examples2/test_nnx.py::Test_MLP::test_simple_mlp_dynamic
```

> Prefer adding small “extra\_tests” that isolate new behavior you introduce. They run fast and are great for regression safety.

When a testcase needs dtype-specific instantiation, expose a `construct_and_call(...)` callable (with helpers like `with_requested_dtype()` and `with_rng_seed(...)`) so the harness can materialize consistent f32/f64 variants.

---

## Working on the IR optimizer

File: `converter2/ir_optimizations.py`

### Philosophy

* **IR in, IR out**. Do not rely on ONNX proto behavior here.
* Work with **live** list containers where possible (`graph.nodes` / `graph._nodes` / `graph.node`) to avoid copy-on-write pitfalls.
* After any mutation, ensure updates **persist**:

  * If you must, rebuild nodes (construct a new `ir.Node` with the desired inputs/outputs) and replace the item in the list.
  * When writing back a whole list, update **all mirrors** (`nodes`, `_nodes`, `node`) to avoid sync issues.

### Helpers (commonly used & expected by tests)

* ` _is_elem(op_type)` – safe predicate for “benign elementwise ops”.
* ` _get_perm_attr(node)` / ` _perms_compose_identity(p1, p2)` – transpose utils.
* ` _has_input_name_or_obj(node, name, obj)`, ` _count_consumers(...)`, ` _find_next_consumer_idx(...)` – matching utilities for tests.

### Common passes

* **Transpose pair folding**: fold `Transpose -> (elemwise)* -> Transpose` when composed perm is identity.

* **Reshape pair folding**: fold `Reshape -> (elemwise)* -> Reshape` when shapes unify (with `-1` wildcard).

* **Dropout training\_mode inlining**:

  * If `training_mode` is provably **False** or the pattern is **`Not(True)`**, replace the 3rd input with the “missing input” (empty name), **do not** remove the Dropout node itself.
  * Remove the now-orphan `Not` explicitly.
  * Be robust to `onnx_ir` tensor wrappers (decode via `_to_numpy_from_any`), and graph input vs. Constant producer.

* **DCE**: walk backwards from graph outputs and keep only reachable nodes.

* **Prune unused graph inputs**: top graph only; preserve function signatures.

### Mutation & persistence (pitfall)

Different `onnx_ir` builds expose node containers differently:

* Sometimes `graph.nodes` is a **live list**; mutate it in place.
* Sometimes it’s a proxy/tuple; rebuild or call a setter.
* Always try to persist to **all** containers (`nodes`, `_nodes`, `node`).

If a mutation seems to “not stick”, rebuild the node (fresh `ir.Node`) and replace it in the list, then write back to all mirrors.

---

## Working on plugins (plugins2)

* Plugins must be **IR-only**.
* Keep logic minimal; non-local cleanups should happen in the **optimizer**, not in plugins.
* For NNX Dropout:

  * `call_time=True` → emit `Not(deterministic)` + 3-input Dropout; avoid inserting graph inputs into function bodies.
  * Ensure output shapes are stamped correctly (`_stamp_type_and_shape`) to preserve symbols like `B`.

---

## Policy & safety rails

* **No ONNX proto imports** in `converter2/*` or `plugins2/*` (tests enforce).
* **No large binaries** – link externally if needed.
* If you see flakiness:

  * xfail it with a clear reason and open an issue.
* Don’t change public APIs without docs + tests + changelog.

---

## PR checklist (what you must verify before asking for review)

* [ ] All tests pass (`poetry run pytest -q`).
* [ ] Added tests that cover new/changed behavior.
* [ ] Lint/format clean (`ruff check .`, `ruff format .`).
* [ ] Types pass (mypy).
* [ ] Docs/CHANGELOG updated with a human-readable summary.

---

## Useful commands & tips

* **Focused test**:

  ```bash
  poetry run pytest -q tests/path/test_file.py::TestClass::test_case
  ```
* **Format only files you changed**:

  ```bash
  git diff --name-only | xargs poetry run ruff format
  ```
* **Run with debug**:

  ```bash
  JAX2ONNX_IROPT_DEBUG=1 JAX2ONNX_TM_DEBUG=1 poetry run pytest -q tests/extra_tests2/framework/test_ir_optimizations.py::test_...
  ```

---

## Common pitfalls (please read)

* **Copy-on-write lists**: some `onnx_ir` builds expose proxy containers; mutating a copied list won’t persist. Use live lists, rebuild nodes when necessary, and write back to **all** containers.
* **Implicit shape loss**: Unary ops whose outputs don’t inherit input shapes may break shape-based tests. Use `propagate_unary_shapes_ir` to stamp shapes/dtypes.
* **Function bodies**: Don’t prune function inputs/outputs; keep function signatures stable. Prefer IR passes that work for both top graphs and function graphs, but only **prune** at the top graph.
* **Dropout semantics**: `training_mode` is **3rd input**; “missing input” (empty name) means `False` (inference mode). Don’t remove Dropout during inlining—keep it present with the missing input.

---

## When you’re stuck

1. Add a tiny failing test in `tests/extra_tests2/framework/` that reproduces your issue with a minimal graph.
2. Turn on the relevant debug flags (`JAX2ONNX_*_DEBUG=1`) and capture the logs in CI output.
3. Verify mutations persist (inspect both `graph.nodes` and `graph._nodes` after your pass).
4. If an attribute/value isn’t being read, extend `_to_numpy_from_any` or `_get_attr` defensively (don’t import ONNX proto).

---

## Glossary

* **IR**: Lightweight intermediate representation (`onnx_ir`) used during conversion.
* **ONNX proto**: The protobuf schema and helpers; **not** used in `converter2/plugins2`.
* **Live list**: A Python list that reflects the graph’s actual node container (mutations persist).
* **Missing input**: An ONNX convention where an optional input is given as an empty name `""`.

---

*Thank you for keeping tests green, diffs small, and passes robust across `onnx_ir` variants. The optimizer and test harness are deliberately defensive: favor correctness and clarity over cleverness.*
