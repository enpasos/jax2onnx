# jax2onnx → onnx\_ir migration plan (Draft RFC)

*Last updated: 2025-08-25*

## 1) Context & goals

**Goal:** Introduce an alternative converter pipeline based on an ONNX IR builder (“onnx\_ir”), while keeping the existing pipeline fully functional. We migrate incrementally, testcase-by-testcase (or component-by-component), without breaking users and without regressing CI.

**Non-goals (for now):**

* Changing the public user-facing API beyond an optional feature flag.
* Rewriting all plugins at once.
* Forcing all tests to run through the new pipeline immediately.

## 2) High-level approach

* Run two pipelines side-by-side:

  * **Old**: `converter/` + `plugins/` (status quo)
  * **New**: `converter2/` + `plugin2/` using `onnx_ir` builder
* Introduce a **feature flag** (`use_onnx_ir`) that routes calls to either pipeline.
* Migrate the smallest, low-risk testcases first; keep CI green by isolating failures to the new path (xfail/skip where appropriate).
* When parity is reached, flip the default to the new pipeline, then remove the old code.

## 3) Terminology & naming

* **onnx\_ir**: The new internal IR builder layer used to construct ONNX graphs/models.
* **plugin2**: New plugin API for the `onnx_ir` pipeline.
* **converter2**: New converter stack that emits `onnx_ir` and serializes to ONNX.

> ⚠️ Keep `plugin2` / `converter2` names stable until the legacy stack is removed.

## 4) Repository layout changes

**Target structure** (additions in *bold*; no duplicate tests folder):

```
jax2onnx/
  user_interface.py
  converter/
  plugins/
  converter2/
  plugin2/
  plugin/__init__.py         # new, shared registry (see §5)
  ir/                        # optional: shared onnx_ir utilities/builders
  sandbox/
    onnx_ir_*.py
  tests/
    t_generator.py           # SINGLE generator used by all subtrees
    conftest.py              # legacy-wide config (does NOT force IR)
    primitives/
    examples/
    extra_tests/
    # onnx_ir-focused subtrees (flag forced via local conftest.py):
    primitives2/
      conftest.py            # forces JAX2ONNX_USE_ONNX_IR=1 for this subtree
    examples2/
      conftest.py            # same
    extra_tests2/
      conftest.py            # same
  scripts/
    generate_tests.py        # SINGLE script to (re)generate tests via t_generator
```

> **No duplicated scripts**: one `t_generator.py`, one `generate_tests.py`. Both legacy and `*2` subtrees use the same generator.

## 5) Moving `plugin_system.py`

* **Action:** Move `plugin_system.py` → `jax2onnx/plugin/__init__.py` (or `plugin/registry.py`).
* **Shim:** Keep `jax2onnx/plugin_system.py` that re-exports from `jax2onnx/plugin` and emits a `DeprecationWarning`.

```python
# jax2onnx/plugin_system.py (shim)
from warnings import warn
warn(
    "jax2onnx.plugin_system is deprecated; import from jax2onnx.plugin instead",
    DeprecationWarning,
    stacklevel=2,
)
from jax2onnx.plugin import *
```

## 6) Feature flag & routing

**Default must remain `use_onnx_ir=False`** during migration. We’ll flip later.

### 6.1 Flag surface

* **In tests (legacy subtrees):** each testcase may set `use_onnx_ir: bool` (default `False`).
* **In `tests/*2` subtrees:** the flag is **forced to `True`** via each subtree’s `conftest.py`.
* **In API:** `to_onnx(*, use_onnx_ir: bool | None = None, ...)`.
* **Env toggles:** `JAX2ONNX_USE_ONNX_IR` overrides both; `JAX2ONNX_SHADOW_COMPARE` optional for dev/CI.

**Subtree conftest sketch (e.g., `tests/examples2/conftest.py`):**

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

### 6.3 Optional "shadow mode"

`JAX2ONNX_SHADOW_COMPARE=1` → run **both** pipelines (where cheap), compare structure, metadata, and ORT outputs. CI/nightly only.

## 7) Test strategy

### 7.1 Single tests/ tree with dual subtrees

* `tests/primitives|examples|extra_tests/` → legacy path, default `use_onnx_ir=False`.
* `tests/primitives2|examples2|extra_tests2/` → IR path, **forced `use_onnx_ir=True`** via local `conftest.py`.

### 7.2 Markers & discipline

* `@pytest.mark.ir_only` to skip in legacy subtrees if shared.
* `@pytest.mark.legacy_only` to skip under `*2` subtrees if shared.
* `@pytest.mark.ir_xfail(reason=...)` for known WIP in `*2`.

### 7.3 Generation flow (single source)

* `tests/t_generator.py` is the single source of truth, capable of emitting tests for both legacy and `*2` subtrees.
* `scripts/generate_tests.py` calls the generator for all targets. No `scripts2/`.

### 7.4 Numeric checks / dtypes

* The generator synthesizes inputs for shape-only cases, respects per-variant tolerances, and threads testcase-level `use_onnx_ir` into `to_onnx`.

### 7.5 CI matrix

* **PR (fast):** full `tests/` legacy; smoke subset from `*2`.
* **Nightly:** full legacy + full `*2`; optional shadow compare on a curated set.

## 8) Sandbox tests

* Keep exploratory scripts under `jax2onnx/sandbox/onnx_ir_*.py`.
* Include a tiny `converter2` + ORT example as a reproducible contributor sample.

## 9) `plugin2` design (v2 API)

* Emit `onnx_ir` nodes (not raw protobuf unless necessary).
* Make shape/dtype contracts explicit.
* Separate op selection, attribute building, shape/dtype inference, and name scoping.
* Central registry lives under `jax2onnx/plugin` with v1/v2 namespaces.

## 10) `converter2` architecture

Stages: (1) JAX tracing → jaxpr; (2) lower via `plugin2` to `onnx_ir`; (3) light IR passes; (4) serialize to ModelProto; (5) debuggability hooks and stable naming.

## 11) Public API & docs

* `to_onnx(..., use_onnx_ir: bool | None = None)` is **experimental**.
* Add doc page: “Adopting the ONNX IR pipeline (experimental)”.

## 12) Migration phases & milestones

**Phase 0 – Framing**

* [x] Move `plugin_system.py` → `jax2onnx/plugin`, add shim & warnings.
* [ ] Add `converter2/` skeleton + `plugin2/` base & registry.
* [x] `user_interface.to_onnx` routing + env var.
* [ ] Scaffold `tests/*2/` subtrees, each with a small `conftest.py` that forces IR.
* [ ] CI: introduce a smoke job for `*2`.
* [ ] **Unify generators/scripts**: keep a *single* `tests/t_generator.py` and a *single* `scripts/generate_tests.py`.

**Phase 1 – Core math & tensor ops**

* [ ] Implement emitters for core ops (add/mul/sub/div/neg/cast/reshape/transpose/concat/slice/gather/matmul).
* [ ] Port a tiny curated subset into `*2` subtrees; add shadow compare where cheap.

**Phase 2 – Shape/Index ops + NNX basics**

* [ ] Add arange/where/select/cumsum/reduce\_\*.
* [ ] Minimal NNX path: linear/conv/batch\_norm happy path with static shapes.

**Phase 3 – Control flow & dynamic shapes**

* [ ] while\_loop/scan/cond emitters; expand NNX coverage; begin Equinox track.

**Phase 4 – Parity & flip**

* [ ] Reach parity (see §14), flip default to IR, deprecate legacy, then remove legacy stack and collapse `*2` subtrees back into main.

## 13) Risks & mitigations

* **Flag confusion** → keep default `False` until parity; loud release notes.
* **Import churn** → shim + deprecation warning + codemod note.
* **Flaky dual runs** → keep PR CI minimal for IR; heavier compare nightly.
* **Shape/dtype drift** → shadow compare + central `ShapeEnv`.

## 14) Parity definition (flip DoD)

* **Functional**: All legacy-passing tests pass on IR (no xfails) for covered features.
* **Performance**: Conversion time within ±10% on a representative set.
* **Stability**: No new IR issues for two weeks after enabling on `main`.

## 15) Developer ergonomics

* Logging categories: `jax2onnx.ir`, `jax2onnx.converter2`, `jax2onnx.plugin2`.
* `JAX2ONNX_DEBUG_IR_DUMP=1` → dump IR + ModelProto to `./.artifacts/latest/`.
* Errors include primitive name, shapes/dtypes, and “file a repro” hint.

## 16) Example snippets

**Subtree conftest (forces IR only in that subtree)**

```python
# tests/primitives2/conftest.py
import os, pytest
@pytest.fixture(autouse=True, scope="session")
def force_onnx_ir():
    os.environ["JAX2ONNX_USE_ONNX_IR"] = "1"
```

**Per-testcase flag in metadata (consumed by t\_generator)**

```python
case = {
    "testcase": "linear",
    "callable": my_linear,
    "input_shapes": [("B", 30)],
    "use_onnx_ir": True,   # opt-in early from legacy subtree if desired
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

## 18) Previously open questions → **Decisions**

1. **IR implementation:** base the new stack on the **onnx\_ir** library
   → [https://github.com/onnx/ir-py](https://github.com/onnx/ir-py)
2. **Opset strategy:** **same as the old generator world (for now)**.
3. **Scope:** convert **all existing plugins and examples step by step** (publish as we go).
4. **Performance:** measure **after implementation** (or earlier if it looks problematic) for speed & memory.
5. **Trace determinism:** **no expected problems**; monitor and adjust if anything crops up.

---

### TL;DR

* Legacy stays default (`use_onnx_ir=False`) while we migrate.
* Single `tests/` tree with `primitives2/examples2/extra_tests2` forcing IR via local `conftest.py`.
* One generator + one script power both sets.
* Incremental migration; flip default after parity; then remove legacy.
