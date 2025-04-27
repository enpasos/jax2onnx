# Proof-of-Concept: Custom JAX Primitive **with Symbolic Batch Dimension**

This PoC demonstrates **how a custom primitive can participate in JAX’s
*shape-polymorphism* pipeline** (added in JAX 0.6.0) and still behave
like any native operation when:

* tracing with `jax.make_jaxpr` (symbolic shapes preserved),
* exporting with dynamic shapes via `jax.export`,
* running with arbitrary concrete batch sizes **without retracing**.

The primitive we build – `poc_concat` – is a lightweight replacement for
`jax.numpy.concatenate`.  The goal is **not** to re-implement concat,
but to show how to wire:

1. An **implementation** (for eager / jit execution).
2. An **`abstract_eval` rule** that can *reason* about symbolic shapes.
3. A **Python-side patch** so user code can keep calling
   `jax.numpy.concatenate` while JAX internally sees `poc_concat`.

---

## 1  Symbolic dimension

```python
from jax import export
B = export.symbolic_shape("B")[0]      # `_DimExpr` representing a batch dim
```

`B` behaves like an `int` placeholder – you can put it in shape tuples,
add it, compare it, etc.  
When a function is later traced/exported, the symbol name **“B”** is
carried into the HLO / ONNX graph.

---

## 2  The primitive

```python
from jax.extend.core import Primitive
poc_concat_p = Primitive("poc_concat")
poc_concat_p.multiple_results = False
```

### Implementation (`impl`)

We delegate to `lax.concatenate` – **note the keyword
`dimension`**, *not* `axis`.

```python
def _poc_concat_impl(*arrays, axis):
    from jax import lax
    return lax.concatenate(arrays, dimension=axis)

poc_concat_p.def_impl(_poc_concat_impl)
```

### Abstract evaluation (`abstract_eval`)

Key idea: **let JAX do the heavy lifting once** via
`jax.export(jax.jit(f))`, where `f` is the original (unpatched)
`jnp.concatenate`.  We feed it `ShapeDtypeStruct`s that contain our
symbolic shapes.

The exported object tells us the output `aval` directly, so we can
return a correct `core.ShapedArray`.

```python
def _poc_concat_abstract_eval(*avals, axis):
    # 1. Build ShapeDtypeStructs from the incoming avals
    structs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]

    # 2. Call the *original* concatenate under jit+export
    exported = jax.export.export(jax.jit(lambda *xs: _orig_concat(xs, axis)))(*structs)

    # 3. We only need the first (and only) out_aval
    return exported.out_avals[0]

poc_concat_p.def_abstract_eval(_poc_concat_abstract_eval)
```

---

## 3  Wrapper & Monkey-patching

We temporarily replace `jax.numpy.concatenate` with a thin wrapper that
invokes `poc_concat_p.bind`.  The wrapper passes the original function
to the primitive via a param, so `abstract_eval` can still reach it.

```python
def patched_concat(arrays, *, axis=0):
    return poc_concat_p.bind(*arrays, axis=axis)

import jax.numpy as jnp
_orig_concat = jnp.concatenate        # keep a reference
jnp.concatenate = patched_concat      # monkey-patch
```

*(The PoC restores the original symbol at exit if you care.)*

---

## 4  Running the demo

```bash
python poc_symbolic_primitive2.py
```

Expected output (ellipses ≈ irrelevant logs):

```
---- PoC with symbolic batch dimension 'B' ----
symbolic dim object: B
Traced JAXPR with symbolic shape:
{ lambda ; a:f32[B,1,8] b:f32[B,10,8]. let
    c:f32[B,11,8] = poc_concat[axis=1] a b
  in (c,) }
batch=3, out.shape=(3, 11, 8)
batch=5, out.shape=(5, 11, 8)
```

* The JAXPR shows **`poc_concat`** and preserves the symbol **“B”**.
* Two concrete runs with batch 3 and 5 reuse the same compiled
  executable – no retracing.

---

## 5  Why this matters for `jax2onnx`

`jax2onnx` needs every primitive to:

1. Produce a **sound `aval`** during tracing (via `abstract_eval`);
2. Supply an **ONNX node** (handled elsewhere).

By delegating shape reasoning to `jax.export`, you avoid duplicating
shape algebra in each plugin.  The same pattern works for squeeze,
reshape, gather, … as long as you can express the op with existing JAX
APIs inside the helper function.

---

## 6  Cleaning up

```python
# After the PoC (or in a try/finally):
jnp.concatenate = _orig_concat
```

---

### Requirements

* **JAX ≥ 0.6.0** (`pip install jax==0.6.* jaxlib==0.6.*+cpu`)
* No GPU/TPU needed – the PoC runs on CPU.

 