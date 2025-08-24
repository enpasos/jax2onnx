
# jax2onnx — Migration Plan (JAX 0.6.x → 0.7.x)

> **Goals**
>
> * Upgrade to **JAX 0.7.x** with **Equinox 0.13.x** and **Flax 0.11.1**.
> * Avoid monkey-patches; use only public, supported APIs.
> * Keep the surface area stable for external users (they run jax2onnx at **export time**, not as a runtime dependency).
> * Make the codebase easier to maintain for future JAX updates.

---

## 1) Scope & Success Criteria

**In scope**

* Bump and pin dependencies.
* Replace deprecated/removed APIs.
* Clean separation of “import-time” vs “run-time” side effects (especially RNG/initializers).
* Update the exporter to modern JAX lowering APIs.
* Refresh tests, CI matrix, and docs.

**Success looks like**

* `poetry run python scripts/generate_tests.py` completes without import-time crashes or deprecation warnings.
* All ONNX exports load and produce numerically consistent results with JAX reference runs for the tested dtypes/shapes.
* CI green on Linux (CPU) and optional CUDA (if used).

---

## 2) Target Versions & Support Window

* **Python:** 3.11+
* **JAX:** `>=0.7.1, <0.8`
* **jaxlib:** pinned to the matching wheel for your platform (CPU or CUDA).
* **Flax (NNX):** `==0.11.1`
* **Equinox:** `>=0.13.0`

**PyProject snippet (Poetry)**

```toml
[tool.poetry.dependencies]
python = "^3.11"

# JAX CPU (Linux/macOS)
jax = ">=0.7.1,<0.8"
jaxlib = ">=0.7.1,<0.8"

# Or for CUDA, install the appropriate extra outside pyproject:
# pip install "jax[cuda12_local]==0.7.1" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

flax = "==0.11.1"
equinox = ">=0.13.0"
numpy = ">=1.26"
scipy = ">=1.12"
onnx = ">=1.16"
onnxruntime = ">=1.18"
```

---

## 3) High-Impact API Changes to Address (JAX 0.7)

> The themes: **no internals**, **modern tree utilities**, **stable lowering**, **tidy PRNG**, **jit signature hygiene**.

### 3.1 Tree utilities

* Replace legacy `jax.tree_*` calls with **`jax.tree_util`** (or `jax.tree` where available).

  * **Before:** `jax.tree_map(fn, pytree)`
  * **After:** `jax.tree_util.tree_map(fn, pytree)`

### 3.2 JIT signature

* Always pass the **function as the first positional arg**; kwargs for options only.

  * ✅ `jax.jit(f, static_argnums=(0,))`
  * 🚫 `jax.jit(fun=f, static_argnums=(0,))`

### 3.3 Lowering & shape/type introspection

* Prefer `f.lower(*args, **kwargs)` then use **shape/dtype structs** from the lowered object.
* Avoid any reliance on removed `OutInfo`/internal XLA classes.

### 3.4 Avoid JAX internals

* Remove imports from `jax.lib.xla_client`, `jax.interpreters.*`, and deep `jax.core` types where possible.
* If you need extension points, prefer `jax.extend.*` modules (public extension API).

### 3.5 PRNG

* Standardize on public random APIs. Prefer `jax.random.key(seed)` if present, otherwise `jax.random.PRNGKey(seed)`.
* If you need to detect “looks like a key”, check `x.dtype`/shape rather than `isinstance`.

```python
# Minimal compatibility without patching:
new_key = getattr(jax.random, "key", jax.random.PRNGKey)
k = new_key(0)
```

### 3.6 Sharding/parallel bits

* Don’t depend on `experimental` shard APIs. Use `jax.sharding.*` if you must handle sharded arrays.
* In exporter code, treat sharding metadata as optional; ONNX graphs are device-agnostic.

---

## 4) jax2onnx Codebase Refactors

### 4.1 Import-time side effects (RNG & initializers)

Recent stack traces showed **initializers executing at import time** inside example modules (e.g., NNX `Linear` kernel init → `random.truncated_normal` → `random.uniform`). Under JAX 0.7 this can assert during import.

**Action: defer all model construction** using a shared helper:

```python
# jax2onnx/plugin_system.py
from typing import Any, Callable

def construct_and_call(ctor: Callable[..., Any], /, **init_kwargs) -> Callable[..., Any]:
    """Return a callable that constructs `ctor(**init_kwargs)` on each call, then calls it."""
    def _call(*args, **call_kwargs):
        module = ctor(**init_kwargs)
        return module(*args, **call_kwargs)
    _call.__name__ = f"construct_and_call_{getattr(ctor, '__name__', 'callable')}"
    return _call
```

**Usage in `register_example(..., testcases=[...])`:**

```python
from jax2onnx.plugin_system import construct_and_call

{
  "testcase": "transformer_block",
  "callable": construct_and_call(
      TransformerBlock,
      num_hiddens=256, num_heads=8, mlp_dim=512, rngs=nnx.Rngs(0)
  ),
  "input_shapes": [("B", 10, 256)],
  "input_params": {"deterministic": True},
}
```

* Never instantiate `nnx.Module` instances or sample random tensors at import time.
* For tests that need random inputs, create them on the fly inside the test harness.

### 4.2 Dropout & “deterministic”

* Keep **`dropout=0.0`** in export tests where possible and **pass `deterministic=True`** to module calls.
* Ensure exporter treats dropout as **no-op** when deterministic.

### 4.3 JAXpr extraction path

* Use `jax.make_jaxpr(fn)(*sample_inputs)` for op coverage and exporter mapping.
* Avoid parsing primitive *names*; rely on primitive objects and stable attributes.
* For MLIR/StableHLO, access via public lowering (no private XLA types).

### 4.4 Replace removed helpers

* Replace `jax.util.safe_map/safe_zip` with Python `map/zip` + length checks or a tiny local helper.

### 4.5 Testing plugin imports

* Ensure **importing a plugin never constructs variables** or performs RNG ops.
* If a plugin needs shapes for lazy init, the factory should accept shape hints (provided by the test generator).

---

## 5) Exporter: Stable ONNX-facing Behavior

* Derive **inputs/outputs metadata** from `ShapeDtypeStruct`/abstract avals.
* **Random ops**: export only when deterministic or explicitly frozen; otherwise require RNG-dependent tensors as inputs if you ever support that.
* **Device/sharding**: ignore device placement; ONNX graph remains portable.

---

## 6) Tests & CI

### 6.1 Unit tests

* Add **import tests**: iterate all `jax2onnx.plugins.*` modules and ensure clean import (no RNG/initializers).
* For each registered example:

  * Build inputs on the fly (seeded).
  * Compare **JAX reference** vs **ONNX Runtime** outputs.
  * Cover f32; gate mixed precision separately if needed.

### 6.2 Golden/round-trip tests

* For select larger models (GPT block, CNN, MLP), export to ONNX and verify ORT inference vs JAX.

### 6.3 CI matrix

* **Linux / Python 3.11 / JAX CPU** mandatory.
* Optional CUDA job if supported.
* Fail on warnings (`-W error`) to catch deprecations early.

---

## 7) Developer Experience & Debugging

* Allow full JAX traces when needed:

  * `JAX_TRACEBACK_FILTERING=off`
* Document switching backend (CPU/CUDA) for local repros.
* Provide a **“minimal failing example”** template in `CONTRIBUTING.md`.

---

## 8) Rollout Plan

1. **Branch**: `feature/jax-0.7-upgrade`.
2. **Mechanical changes**: tree\_util imports, jit signatures, remove internals, **introduce `construct_and_call` and migrate testcases**.
3. **Bump deps** and update lockfile; add pins.
4. **Green tests on JAX 0.6 (latest)** with warnings→errors.
5. Switch to **JAX 0.7.1**, fix remaining breakages.
6. **CI**: merge once Linux CPU job is green; publish a pre-release.
7. **Docs**: update README + migration notes; note new minimum versions.
8. **Release**: cut `v0.7.x` of jax2onnx with the new matrix.

---

## 9) Concrete To-Do Checklist

* [ ] Replace `jax.tree_*` with `jax.tree_util.*`.
* [ ] Audit `jax.jit` calls → function positional, options as kwargs.
* [ ] Remove imports from `jax.lib.xla_client`, `jax.interpreters.*`, deep `jax.core` types.
* [ ] Switch lowering/shape inspection to public `lower()`/shape-dtype structures.
* [ ] Remove any `jax.util.safe_map/safe_zip` usage.
* [ ] **Add `construct_and_call` helper** in `plugin_system.py`.
* [ ] **Migrate all testcase `"callable"` entries to `construct_and_call(...)`** (no lambdas, no eager instantiation).
* [ ] Ensure dropout paths use `dropout=0.0` and/or `deterministic=True` in tests.
* [ ] Verify exporter ignores device/sharding metadata.
* [ ] Pin versions in `pyproject.toml`; refresh lockfile.
* [ ] CI: add import-safety test, round-trip ONNX tests, warnings→errors.
* [ ] Update docs (supported versions, migration notes).

---

## 10) Appendix

### 10.1 Tiny compatibility shims (kept local, no patching)

```python
# rng_compat.py
import jax
def make_key(seed: int):
    return getattr(jax.random, "key", jax.random.PRNGKey)(seed)
```

```python
# tree_compat.py
from jax import tree_util as jtu

tree_map = jtu.tree_map
tree_leaves = jtu.tree_leaves
tree_flatten = jtu.tree_flatten
tree_unflatten = jtu.tree_unflatten
```

### 10.2 Example factory pattern with `construct_and_call`

```python
from jax2onnx.plugin_system import construct_and_call

register_example(
    component="GPT_MLP",
    description="An MLP block with GELU activation from nanoGPT.",
    context="examples.gpt",
    children=["nnx.Linear", "nnx.gelu", "nnx.Dropout"],
    testcases=[
        {
            "testcase": "gpt_mlp",
            "callable": construct_and_call(
                MLP, n_embd=768, dropout=0.0, rngs=nnx.Rngs(0)
            ),
            "input_shapes": [("B", 1024, 768)],
            "input_params": {"deterministic": True},
            "run_only_f32_variant": True,
        }
    ],
)
```

### 10.3 Quick migration recipe (find/replace friendly)

* **Before** (eager):

  ```python
  "callable": SomeModule(..., rngs=nnx.Rngs(0)),
  ```
* **After** (lazy):

  ```python
  from jax2onnx.plugin_system import construct_and_call
  "callable": construct_and_call(SomeModule, ..., rngs=nnx.Rngs(0)),
  ```

---

## 11) Communication Notes (External Consumers)

* This is a **build-time tool**; no expectations to bundle jax2onnx into downstream runtime images.
* Call out the **minimum versions** in the README.
* Provide a short “Export Cookbook” with 2–3 canonical examples (Equinox MLP, Flax NNX CNN, pure-JAX function) showing model definition, sample inputs, export to ONNX, and ORT verification.
