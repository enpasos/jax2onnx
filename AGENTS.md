# AGENTS.md

## Project overview

This monorepo implements a **JAX → ONNX** conversion stack with two pipelines:

* `converter/` – legacy, ONNX‐proto heavy path.
* `converter2/` – **IR-only** path (recommended). Uses `onnx_ir` to build an intermediate representation, then serializes to ONNX later.
* `plugins2/` – modular lowering for JAX/Flax primitives into IR (no ONNX proto imports here).
* `tests/` – unit, integration, and policy tests (including “no onnx in converter2/plugins2” guards).

**Python**: 3.11+ (some users run 3.12 successfully).
**Packaging**: Poetry.
**Style**: Ruff (lint+format), mypy (type hints), Black-ish formatting via Ruff.

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
