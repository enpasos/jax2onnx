# jax2onnx → onnx\_ir migration plan (Draft RFC)

*Last updated: 2025‑08‑25*

## 1) Context & goals

**Goal:** Introduce an alternative converter pipeline based on an ONNX IR builder ("onnx\_ir"), while keeping the existing pipeline fully functional. We migrate incrementally, testcase‑by‑testcase (or component‑by‑component), without breaking users and without regressing CI.

**Non‑goals (for now):**

* Changing the public user‑facing API beyond an optional feature flag.
* Rewriting all plugins at once.
* Forcing all tests to run through the new pipeline immediately.

## 2) High‑level approach

* Run two pipelines side‑by‑side:

  * **Old**: `converter/` + `plugins/` (status quo)
  * **New**: `converter2/` + `plugin2/` using `onnx_ir` builder
* Introduce a **feature flag** (`use_onnx_ir`) that routes calls to either pipeline.
* Start migrating the smallest, low‑risk testcases first; keep CI green by isolating failures to the new path (xfail/skip where appropriate).
* When parity is reached, flip the default to the new pipeline, then remove the old code.

## 3) Terminology & naming

* **onnx\_ir**: The new internal IR builder layer we use to construct ONNX graphs/models (may wrap onnx‑script or a custom builder; exact implementation is an internal detail of `converter2`).
* **plugin2**: New plugin API for the `onnx_ir` pipeline.
* **converter2**: New converter stack that emits `onnx_ir` and serializes to ONNX.

> ⚠️ **Naming consistency**: keep `plugin2` / `converter2` names stable throughout the migration to avoid churn. We can rename after the old stack is deleted.

## 4) Repository layout changes

**Target structure** (additions in *bold*):

```
jax2onnx/
  user_interface.py
  converter/
  plugins/
  converter2/
  plugin2/
  plugin/__init__.py         # new location of the registry (see §5)
  ir/                        # optional: shared onnx_ir utilities/builders
  sandbox/
    onnx_ir_*.py             # existing/new sandbox cases
  tests/                     # legacy pipeline (default: use_onnx_ir=False)
    ...
  tests2/                    # onnx_ir pipeline (forced: use_onnx_ir=True via conftest)
    conftest.py
    ...
  scripts/
    generate_tests.py        # legacy generator
  scripts2/
    generate_tests2.py       # IR-aware generator emitting tests for tests2/
```

## 5) Moving `plugin_system.py`

* **Action:** Move `plugin_system.py` to `jax2onnx/plugin/__init__.py` (or `jax2onnx/plugin/registry.py`).
* **Back‑compat shim:** Keep a tiny `jax2onnx/plugin_system.py` that re‑exports from the new location and emits a `DeprecationWarning`.

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

* **Why now?** We need a neutral package (`jax2onnx/plugin`) that both old `plugins/` and new `plugin2/` can rely on.

## 6) Feature flag & routing

> **Gap fixed:** The initial proposal had a contradiction. To keep behavior unchanged for users, the default **must be `use_onnx_ir=False`** during migration. Later, we can flip to `True`.

### 6.1 Flag surface

* **In tests (legacy tree `tests/`):** each testcase may set `use_onnx_ir: bool` (default `False`).
* **In tests2:** the flag is **forced to `True` via `tests2/conftest.py`**; individual tests usually don’t specify it.
* **In API:** `to_onnx(*, use_onnx_ir: bool | None = None, ... )` (unchanged).
* **Env toggles:** `JAX2ONNX_USE_ONNX_IR` overrides both trees; `JAX2ONNX_SHADOW_COMPARE` optional for dev/CI.

### 6.2 Routing in `user_interface.to_onnx`

```python
# user_interface.py
from .converter import to_onnx as _to_onnx_v1
from .converter2 import to_onnx as _to_onnx_v2

def to_onnx(func, *inputs, use_onnx_ir: bool | None = None, **kw):
    if use_onnx_ir is None:
        use_onnx_ir = bool(int(os.getenv("JAX2ONNX_USE_ONNX_IR", "0"))) or DEFAULT_USE_ONNX_IR
    return (_to_onnx_v2 if use_onnx_ir else _to_onnx_v1)(func, *inputs, **kw)
```

### 6.3 Optional "shadow mode"

* Add an internal env switch `JAX2ONNX_SHADOW_COMPARE=1` to run **both** pipelines, then compare:

  * graph structure (node counts, op types)
  * model inference equality (ORT run – only when cheap)
  * metadata parity (input/output shapes/dtypes)
* Only used in CI/nightly; does not affect users.

## 7) Test strategy

### 7.1 Dual-tree layout

* **`tests/`** → Legacy pipeline. Default `use_onnx_ir=False`. Keep existing parametrization style where helpful.
* **`tests2/`** → ONNX IR pipeline. **Always uses `use_onnx_ir=True`** enforced via a session fixture.

**tests2/conftest.py (sketch):**

```python
# tests2/conftest.py
import os, pytest

@pytest.fixture(autouse=True, scope="session")
def force_onnx_ir():
    os.environ["JAX2ONNX_USE_ONNX_IR"] = "1"
```

### 7.2 Markers & discipline

* `@pytest.mark.ir_only` → skip on legacy tree if accidentally collected.
* `@pytest.mark.legacy_only` → skip on tests2 if a file is temporarily shared.
* `@pytest.mark.ir_xfail(reason=...)` → convenience alias for `pytest.mark.xfail` used **only** in tests2 where the legacy test passes but the IR path is WIP.

### 7.3 Generators

* **`scripts/generate_tests.py`** remains for legacy.
* **`scripts2/generate_tests2.py`** generates IR-focused tests into `tests2/` (may mirror structure/names from `tests/`). Prefer **one source of truth** for each migrated case to avoid drift; if duplication is unavoidable, add a comment at the top of both files pointing to its counterpart.

### 7.4 CI matrix

* **PR (fast):** run `pytest tests/` fully; run a **smoke subset** of `tests2/` (e.g., via `-k ir_smoke or -m "not slow and not network"`).
* **Nightly:** run full `tests/` + full `tests2/`; optionally enable `JAX2ONNX_SHADOW_COMPARE=1` on a curated subset.

### 7.5 Migration mechanics

* When a testcase/component is migrated, add/port it to `tests2/` first. Keep the legacy copy in `tests/` until parity is reached for that area, then decide whether to:

  * keep both (to guard against regressions in routing), or
  * remove the legacy duplicate to reduce maintenance (recommended once IR is default).

## 8) Sandbox tests

* Keep exploratory notebooks/scripts under `jax2onnx/sandbox/onnx_ir_*.py`.

* Have at least one script that builds a tiny model with `converter2` and runs ORT to validate outputs. This acts as a reproducible example for contributors.

* Keep exploratory notebooks/scripts under `jax2onnx/sandbox/onnx_ir_*.py`.

* Have at least one script that builds a tiny model with `converter2` and runs ORT to validate outputs. This acts as a reproducible example for contributors.

## 9) `plugin2` design (v2 API)

**Goals:**

* Clean, minimal emitters that return `onnx_ir` nodes (not raw protobuf unless necessary).
* Explicit shape/dtype contracts (return types) to facilitate static checks.
* Clear separation of concerns: op selection, attribute building, shape/dtype inference, and name scoping.

**Sketch:**

```python
# jax2onnx/plugin2/base.py
class EmitterCtx:
    def const(self, name, value, dtype, shape=None): ...
    def op(self, op_type: str, *inputs, **attrs) -> "Value": ...  # returns IR Value
    def infer(self, value: "Value"): ...  # shape/dtype

class PrimitivePluginV2(Protocol):
    jax_primitive: str
    def lower(self, ctx: EmitterCtx, *args, **params) -> "Value | tuple[Value,...]": ...
```

* **Registration**: central registry in `jax2onnx/plugin` with separate namespaces for v1 and v2 to avoid collisions.
* **Versioning**: plugins may advertise minimal opset and constraints; `converter2` resolves compatible patterns.

## 10) `converter2` architecture

Pipeline stages:

1. **Front‑end**: JAX tracing → Jaxpr (same as legacy) + capture constants/metadata.
2. **Lowering**: Map JAX primitives to `plugin2` emitters, yielding `onnx_ir` graph fragments.
3. **IR passes** (optional, small and fast): constant folding, dead‑node elimination, attribute canonicalization, name‑hygiene.
4. **Serialization**: `onnx_ir` → ModelProto, fill opset/imports, run ONNX checker.
5. **Debuggability**: Stable names for nodes/values, and hooks to dump IR.

**Shape/dtype management:**

* Central `ShapeEnv` shared across emitters; all outputs register their inferred type.
* Fallback to runtime shape capture is allowed for dynamic dims but must be explicit in logs.

## 11) Public API & docs

* `to_onnx(..., use_onnx_ir: bool | None = None)` documented as **experimental**.
* Add docs page: “Adopting the ONNX IR pipeline (experimental)” with examples and caveats.

## 12) Migration phases & milestones

**Phase 0 – Framing (PR 1–2)**

* [ ] Move `plugin_system.py` → `jax2onnx/plugin`, add shim & warnings.
* [ ] Add `converter2/` skeleton + `plugin2/` base & registry.
* [ ] `user_interface.to_onnx` routing + env var.
* [ ] **Scaffold dual trees**: create `tests2/` with `conftest.py` that forces IR; create `scripts2/` with a minimal `generate_tests2.py`.
* [ ] CI: introduce `ONNX_IR_SMOKE=1` job that runs a tiny subset of `tests2/`.

**Phase 1 – Core math & tensor ops (PRs 3–N)**

* [ ] Implement emitters for core ops (`add`, `mul`, `sub`, `div`, `neg`, `cast`, `reshape`, `transpose`, `concat`, `slice`, `gather`, `matmul`).
* [ ] Port a **tiny** curated subset of primitives tests into `tests2/`.
* [ ] Establish optional shadow compare on a few cases.

**Phase 2 – Shape/Index ops + NNX basics**

* [ ] Add `arange`, `where`, `select`, `cumsum`, `reduce_*`.
* [ ] Minimal NNX path: `linear`, `conv`, `batch_norm` happy path with static shapes.
* [ ] Grow `tests2/` coverage accordingly; trim legacy duplicates where safe.

**Phase 3 – Control flow & dynamic shapes**

* [ ] `while_loop`, `scan`, `cond` emitters (static → dynamic).
* [ ] Broaden NNX coverage; start Equinox subset as separate tracks.

**Phase 4 – Parity & flip**

* [ ] Parity criteria met (see §14).
* [ ] Flip default: `DEFAULT_USE_ONNX_IR=True` in one release.
* [ ] Deprecate legacy via warnings for one minor release; then **remove `converter/`, `plugins/`, and consolidate `tests2/` → `tests/`** with a final rename.

## 13) Risks & mitigations

* **Flag confusion**: Wrong default breaks users. → Keep default `False` until parity; add explicit release notes.
* **Import churn**: Moving `plugin_system.py` breaks contributors. → Provide shim + warnings + codemod note.
* **Test flakiness** with dual runs. → Keep PR CI minimal for `onnx_ir`; run the heavy compare nightly.
* **Shape/dtype drift** between pipelines. → Shadow compare hooks; central `ShapeEnv`.

## 14) Parity definition (Definition of Done for flip)

* **Functional**: All tests that pass on legacy also pass on `onnx_ir` (no xfails) for the supported feature set.
* **Performance**: Within ±10% conversion time for representative models (documented sample set).
* **Stability**: No new issues filed for `onnx_ir` for two weeks after enabling on `main`.

## 15) Developer ergonomics

* Logging categories: `jax2onnx.ir`, `jax2onnx.converter2`, `jax2onnx.plugin2`.
* `JAX2ONNX_DEBUG_IR_DUMP=1` → write IR and final ModelProto to `./.artifacts/latest/`.
* Error messages should include: primitive name, shapes/dtypes, and hint to file a repro.

## 16) Example snippets

**Test parametrization**

```python
import pytest

@pytest.mark.parametrize("use_onnx_ir", [False, True])
def test_add_scalar(use_onnx_ir):
    if use_onnx_ir and not os.getenv("ON_CI", "0"):  # keep local runs fast
        pytest.skip("IR path only on CI smoke by default")
    model = to_onnx(lambda x: x + 1, jnp.ones((3,)), use_onnx_ir=use_onnx_ir)
    assert run_ort(model, [np.ones((3,), np.float32)]) is not None
```

**Per‑testcase flag in generators**

```python
case = {
    "testcase": "linear",
    "callable": my_linear,
    "input_shapes": [("B", 30)],
    "use_onnx_ir": True,  # opt‑in early
}
```

**Minimal routing**

```python
# user_interface.py

def to_onnx(..., use_onnx_ir: bool | None = None):
    use_onnx_ir = _resolve_flag(use_onnx_ir)
    return converter2.to_onnx(...) if use_onnx_ir else converter.to_onnx(...)
```

## 17) Documentation & comms

* Add a README section: “Experimental ONNX IR pipeline”.
* Release notes for each phase with a short status line (what’s supported, what’s not).
* Invite contributors to migrate specific primitives/components with a checklist in the issue tracker.

## 18) Open questions (to resolve early)

1. **IR implementation**: Are we standardizing on onnx‑script as the builder, or a thin custom IR wrapper? (Impacts developer experience.)
2. **Opset strategy**: Do we target a single minimum opset for `converter2`, or allow per‑op emitters to pick higher opsets? (Recommend per‑op with a global floor.)
3. **NNX/Equinox scope**: Which minimal subset do we guarantee first so downstream users feel the benefit quickly?
4. **Performance budgets**: Do we set conversion time/memory budgets for `converter2` now or after parity?
5. **Trace determinism**: Any differences in naming/scoping that could affect reproducibility or cached artifacts?

---

### TL;DR

* Keep legacy as default (`use_onnx_ir=False`).
* Add `converter2`/`plugin2`, route via a flag, keep CI green with targeted xfails.
* Migrate incrementally; flip default after parity; then remove legacy.
