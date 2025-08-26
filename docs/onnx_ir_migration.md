# jax2onnx → onnx\_ir migration plan (Draft RFC)

*Last updated: 2025-08-26*

## 1) Context & goals

**Goal:** Introduce an alternative converter pipeline based on an ONNX IR builder (“onnx\_ir”), while keeping the existing pipeline fully functional. We migrate incrementally, testcase-by-testcase (or component-by-component), without breaking users and without regressing CI.

**Non-goals (for now):**

* Changing the public user-facing API beyond an optional feature flag.
* Rewriting all plugins at once.
* Forcing all tests to run through the new pipeline immediately.

## 2) High-level approach

* Run two pipelines side-by-side:

    * **Old:** `converter/` + `plugins/` (status quo)
    * **New:** `converter2/` + `plugin2/` using `onnx_ir` builder
* Introduce a **feature flag** (`use_onnx_ir`) that routes calls to either pipeline.
* Migrate the smallest, low-risk testcases first; keep CI green by isolating failures to the new path (xfail/skip where appropriate).
* When parity is reached, flip the default to the new pipeline, then remove the old code.

## 3) Terminology & naming

* **onnx\_ir:** New internal IR builder layer to construct ONNX graphs/models.
* **plugin2:** New plugin API for the `onnx_ir` pipeline.
* **converter2:** New converter stack that emits `onnx_ir` and serializes to ONNX.

> ⚠️ Keep `plugin2` / `converter2` names stable until the legacy stack is removed.

## 4) Repository layout

**Target structure** (additions in *bold*; **single** `tests/` tree; IR tests live under `*2` subfolders):

```
jax2onnx/
  user_interface.py
  converter/
  plugins/
  converter2/
  plugin2/
  plugin/__init__.py         # shared registry (see §5)
  ir/                        # optional: shared onnx_ir helpers
  sandbox/
    onnx_ir_*.py
  tests/
    t_generator.py           # single generator for all test subtrees
    conftest.py              # global config (does NOT force IR)
    primitives/
    examples/
    extra_tests/
    primitives2/             # IR-only tests
      conftest.py            # forces JAX2ONNX_USE_ONNX_IR=1
    examples2/
      conftest.py            # forces JAX2ONNX_USE_ONNX_IR=1
    extra_tests2/
      conftest.py            # forces JAX2ONNX_USE_ONNX_IR=1
  scripts/
    generate_tests.py        # single script (no duplication)
```

> **No duplicated scripts**: one `t_generator.py`, one `generate_tests.py`. Both legacy and `*2` subtrees use the same generator.

## 5) Moving `plugin_system.py`

* **Action:** Move `plugin_system.py` → `jax2onnx/plugin/__init__.py` (or `plugin/registry.py`).
* **Shim:** Keep `jax2onnx/plugin_system.py` that re-exports from `jax2onnx/plugin` and emits a `DeprecationWarning`:

```python
from warnings import warn
warn(
    "jax2onnx.plugin_system is deprecated; import from jax2onnx.plugin instead",
    DeprecationWarning,
    stacklevel=2,
)
from jax2onnx.plugin import *
```

## 6) Feature flag & routing

**Default remains `use_onnx_ir=False`** during migration. We flip later.

### 6.1 Flag surface

* **In tests (legacy subtrees):** each testcase may set `use_onnx_ir: bool` (default `False`).
* **In `tests/*2` subtrees:** the flag is **forced to `True`** via that subtree’s `conftest.py`.
* **In API:** `to_onnx(*, use_onnx_ir: bool | None = None, ...)`.
* **Env toggles:** `JAX2ONNX_USE_ONNX_IR` overrides both. Optional `JAX2ONNX_SHADOW_COMPARE` for dev/CI.

**Subtree conftest sketch (e.g., `tests/primitives2/conftest.py`):**

```python
import os, pytest

@pytest.fixture(autouse=True, scope="session")
def force_onnx_ir():
    os.environ["JAX2ONNX_USE_ONNX_IR"] = "1"
```

### 6.2 Routing in `user_interface.to_onnx`

```python
from .converter import to_onnx as _to_onnx_v1
from .converter2 import to_onnx as _to_onnx_v2

DEFAULT_USE_ONNX_IR = False

def to_onnx(func, inputs, *, use_onnx_ir: bool | None = None, **kw):
    if use_onnx_ir is None:
        use_onnx_ir = os.getenv("JAX2ONNX_USE_ONNX_IR", "").strip().lower() in ("1","true","yes") or DEFAULT_USE_ONNX_IR
    return (_to_onnx_v2 if use_onnx_ir else _to_onnx_v1)(func, inputs, **kw)
```

### 6.3 Optional “shadow mode”

`JAX2ONNX_SHADOW_COMPARE=1` → run **both** pipelines (where cheap), compare structure/metadata/ORT outputs. CI/nightly only.

## 7) Test strategy

* **Single** `tests/` tree with dual subtrees:

    * `tests/primitives|examples|extra_tests/` → legacy path, default `use_onnx_ir=False`.
    * `tests/primitives2|examples2|extra_tests2/` → IR path, **forced `use_onnx_ir=True`**.
* Markers:

    * `@pytest.mark.ir_only` to skip legacy subtrees if shared.
    * `@pytest.mark.legacy_only` to skip under `*2` if shared.
    * `@pytest.mark.ir_xfail(reason=...)` for known WIP in `*2`.
* Generator:

    * `tests/t_generator.py` threads testcase-level `use_onnx_ir` into `user_interface.to_onnx`.
    * Synthesizes inputs for shape-only cases; per-variant tolerances supported.

## 8) Sandbox

* Keep exploratory scripts under `jax2onnx/sandbox/onnx_ir_*.py`.
* Include a tiny `converter2` + ORT example as a reproducible contributor sample.

## 9) `plugin2` design (v2 API)

* Emit `onnx_ir` nodes (prefer IR builder over raw protobuf).
* Explicit shape/dtype contracts.
* Separate: op selection, attributes, shape/dtype inference, name scoping.
* Central registry under `jax2onnx/plugin` with distinct v1/v2 namespaces.

## 10) `converter2` architecture

Stages:

1. **Front-end:** JAX tracing → ClosedJaxpr (same trace entry as legacy).
2. **Lowering:** Walk `jaxpr.eqns`; dispatch to `plugin2` emitters; build `onnx_ir` graph.
3. **IR passes (small/fast):** const fold, DCE, name hygiene.
4. **Serialization:** `onnx_ir` → `ModelProto`, fill opset/imports, run checker.
5. **Debuggability:** Stable names; IR/model dump on demand.

**Compat note:** JAX ≥ 0.6 moved `Literal`/`Var` to `jax.extend.core`. The converter ships a small compat alias so both old/new JAX work.

## 11) Public API & docs

* `to_onnx(..., use_onnx_ir: bool | None = None)` is **experimental**.
* Add doc page: “Adopting the ONNX IR pipeline (experimental)” with examples/caveats.

## 12) Migration phases & milestones

**Phase 0 – Framing**

* [x] `user_interface.to_onnx` routing + env var.
* [x] Initial `converter2/` skeleton with ClosedJaxpr walk + `onnx_ir` emission (MVP path).
* [x] `plugin2` entry for `lax.tanh` and registry entry (keeps legacy `@register_primitive` for testcase kick-off).
* [x] `tests/primitives2` subtree + conftest that forces IR.
* [ ] CI: smoke job for `*2`.
* [x] Single generator/script; no `scripts2/`.

**Phase 1 – Core math & tensor ops**

* [ ] add/mul/sub/div/neg/cast/reshape/transpose/concat/slice/gather/matmul emitters
* [ ] port small curated subset into `*2` subtrees; optional shadow compare

**Phase 2 – Shape/Index ops + NNX basics**

* [ ] arange/where/select/cumsum/reduce\*
* [ ] minimal NNX: linear/conv/batch\_norm (static shapes)

**Phase 3 – Control flow & dynamic shapes**

* [ ] while\_loop / scan / cond; broaden NNX; start Equinox track

**Phase 4 – Parity & flip**

* [ ] parity (see §14), flip default to IR, deprecate legacy, then remove legacy stack and collapse `*2` subtrees

## 13) Risks & mitigations

* **Flag confusion** → keep default `False`; clear release notes.
* **Import churn** → shim + deprecation warning + codemod note.
* **Flaky dual runs** → PR CI light for IR; heavy/nightly for shadow compare.
* **Shape/dtype drift** → shadow compare + central `ShapeEnv`.

## 14) Parity definition (flip DoD)

* **Functional:** All legacy-passing tests pass on IR (no xfails) for covered features.
* **Performance:** Conversion time within ±10% on a representative set.
* **Stability:** No new IR issues for two weeks after enabling on `main`.

## 15) Developer ergonomics

* Logging namespaces: `jax2onnx.ir`, `jax2onnx.converter2`, `jax2onnx.plugin2`.
* `JAX2ONNX_DEBUG_IR_DUMP=1` → dump IR + ModelProto to `./.artifacts/latest/`.
* Errors should include primitive, shapes/dtypes, and “file a repro” hint.

## 16) Example snippets

**Subtree conftest (forces IR in that subtree)**

```python
# tests/primitives2/conftest.py
import os, pytest
@pytest.fixture(autouse=True, scope="session")
def force_onnx_ir():
    os.environ["JAX2ONNX_USE_ONNX_IR"] = "1"
```

**Per-testcase flag in metadata (consumed by t\_generator)**

```python
{
  "testcase": "tanh",
  "callable": lambda x: jax.lax.tanh(x),
  "input_shapes": [(3,)],
  "use_onnx_ir": True
}
```

**Minimal routing**

```python
def to_onnx(..., use_onnx_ir: bool | None = None):
    use_onnx_ir = _resolve_flag(use_onnx_ir)
    return converter2.to_onnx(...) if use_onnx_ir else converter.to_onnx(...)
```

## 17) Documentation & comms

* README section: “Experimental ONNX IR pipeline”.
* Release notes per phase: supported vs. not-yet-supported.
* Invite contributors to migrate specific primitives/components with a checklist.

## 18) Decisions (was “Open questions”)

1. **IR implementation:** base the new stack on **onnx\_ir** → [https://github.com/onnx/ir-py](https://github.com/onnx/ir-py).
2. **Opset strategy:** **same as the old generator world (for now)**.
3. **Scope:** convert **all existing plugins and examples step by step** (publish as we go).
4. **Performance:** measure **after implementation** (or earlier if problematic) for speed & memory.
5. **Trace determinism:** **no expected problems**; monitor and adjust if anything crops up.

---

### Current MVP status

* `converter2` can lower a simple ClosedJaxpr and emit `Tanh` via `plugin2` into `onnx_ir`.
* JAX 0.6 compat handled (`jax.extend.core.Literal/Var`).
* IR version pinned to match onnxruntime’s supported **IR\_VERSION=10** so ORT loads the model cleanly.

### TL;DR

* Legacy stays default (`use_onnx_ir=False`) while we migrate.
* Single `tests/` tree; `primitives2/examples2/extra_tests2` force IR via local `conftest.py`.
* One generator + one script power both sets.
* Incremental migration; flip default after parity; then remove legacy.
