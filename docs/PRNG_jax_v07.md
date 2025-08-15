# Best Practices for PRNG Handling in Flax NNX & Equinox (JAX 0.4.26+ / 0.7)

JAX’s newer releases strengthened the guarantees around randomness: keys are **single-use capabilities** and accidental reuse now surfaces more aggressively (via debug checks and clearer errors). That’s good for correctness—but it means some older examples (global keys, import-time randomness) now break or behave oddly.

This guide shows how to write Flax **NNX** and **Equinox** code that’s clean, reproducible, and compatible with JAX’s tighter PRNG semantics.

---

## Core Principles

1. **No randomness at import time.** Don’t create `PRNGKey` or sample parameters in module/global scope.
2. **Treat keys as single-use.** Split once per independent random need; never call two random ops with the same key.
3. **Make randomness explicit.** Pass keys (or `nnx.Rngs`) into constructors/forward calls; don’t hide randomness in defaults or globals.

---

## 1) Avoid PRNG at Import Time

### ❌ Anti-pattern (import-time randomness)

```python
# equinox (bad): evaluates at import and reuses the same key forever
import jax, equinox as eqx, jax.numpy as jnp

key = jax.random.PRNGKey(0)  # import-time
class MyLayer(eqx.Module):
    w: jax.Array = eqx.field(init=False)
    def __init__(self):
        self.w = jax.random.normal(key, (128, 128))  # key reuse risk
```

### ✅ Equinox pattern (constructor takes a key; split as needed)

```python
import jax
import jax.numpy as jnp
import equinox as eqx

class MyNet(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, in_dim, hidden, out_dim, key):
        k1, k2 = jax.random.split(key, 2)
        self.l1 = eqx.nn.Linear(in_dim, hidden, key=k1)
        self.l2 = eqx.nn.Linear(hidden, out_dim, key=k2)

    def __call__(self, x):
        return self.l2(jax.nn.relu(self.l1(x)))
```

### ✅ Flax NNX pattern (`nnx.Rngs` flows through init)

```python
import jax
import jax.numpy as jnp
from flax import nnx

class Linear(nnx.Module):
    def __init__(self, din, dout, *, rngs: nnx.Rngs):
        k = rngs.params()  # fresh key for params
        self.w = nnx.Param(jax.random.normal(k, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))

    def __call__(self, x):
        return x @ self.w + self.b
```

> Key idea: the constructor **requires** a key/`Rngs` and performs all random init **inside** `__init__`. Nothing random happens at import.

---

## 2) Safe Tests & Examples (no side-effects, no reuse)

* **Seed inside the test/example**, not at import:

```python
def test_forward():
    master = jax.random.PRNGKey(42)
    model_key, data_key = jax.random.split(master)
    model = MyNet(in_dim=32, hidden=64, out_dim=10, key=model_key)
    x = jax.random.normal(data_key, (8, 32))
    y = model(x)
```

* **Split per independent need.** If you need random data *and* dropout masks, split separately.

* **Enable debug checks (recommended while developing):**

  ```python
  import jax
  jax.config.update("jax_debug_key_reuse", True)
  ```

  This raises if you accidentally reuse a key.

* **No randomness in default args or dataclass defaults.**

  ```python
  # bad
  def __init__(self, key=jax.random.PRNGKey(0)): ...
  # good
  def __init__(self, *, key): ...
  ```

---

## 3) Recommended Architectural Patterns

### A) Flax NNX: pass a shared `nnx.Rngs`

```python
from flax import nnx

class MLP(nnx.Module):
    def __init__(self, din, hidden, dout, *, rngs: nnx.Rngs):
        self.l1 = nnx.Linear(din, hidden, rngs=rngs)   # consumes rngs.params()
        self.l2 = nnx.Linear(hidden, dout, rngs=rngs)
        self.drop = nnx.Dropout(0.1)

    def __call__(self, x, deterministic: bool = True):
        x = self.l1(x)
        x = nnx.gelu(x)
        x = self.drop(self.l2(x), deterministic=deterministic)  # uses rngs.dropout()
        return x

rngs = nnx.Rngs(params=0, dropout=1)  # seed named streams
model = MLP(128, 256, 10, rngs=rngs)
```

**Why it’s nice:** You don’t manually split keys for every submodule; `Rngs` tracks streams and hands out fresh keys (`rngs.params()`, `rngs.dropout()`, …).

---

### B) Equinox: split once in `__init__` for submodules

```python
class BigNet(eqx.Module):
    blocks: tuple

    def __init__(self, depth, key):
        keys = jax.random.split(key, depth)
        self.blocks = tuple(eqx.nn.MLP(128, 256, 128, 3, key=k) for k in keys)

    def __call__(self, x):
        for b in self.blocks:
            x = b(x)
        return x
```

**Why it’s nice:** Clear ownership of randomness, no hidden globals, easy to reason about and test.

---

### C) Named factory functions (great for tests/examples)

```python
def make_linear(seed: int):
    key = jax.random.PRNGKey(seed)
    return eqx.nn.Linear(in_features=30, out_features=3, key=key)

# in test/example
model = make_linear(0)
```

Use this when you want to keep call sites simple but still avoid import-time PRNG.

---

### D) Build multiple models at once (vmapped factory)

```python
import equinox as eqx

def make_mlp(key):
    return eqx.nn.MLP(128, 256, 10, depth=3, key=key)

keys = jax.random.split(jax.random.PRNGKey(0), 8)

@eqx.filter_vmap
def make_many(k):
    return make_mlp(k)

ensemble = make_many(keys)  # 8 independent models
```

---

## Migration Tips (quick wins)

* **Search & remove** any `PRNGKey(...)` created at module top-level; move into `__init__`/functions.
* **Constructor signatures:** require `key` (Equinox) or `rngs: nnx.Rngs` (NNX); update call sites accordingly.
* **Split once, pass along:** create a small “wiring” function that splits a master key into subkeys for submodules.
* **Examples/tests:** wrap in `make_*` factories or use lambdas that construct the model **inside** the callable.

---

## Do / Don’t Checklist

* ✅ Pass keys/`Rngs` into constructors and forwards that need randomness.
* ✅ Split keys per independent random use.
* ✅ Keep seeding inside functions/tests; prefer factories for sample models.
* ❌ Don’t create keys or sample params at import time.
* ❌ Don’t reuse the same key for multiple random ops.
* ❌ Don’t hide randomness in default arguments or dataclass defaults.

---

## FAQ

**Why did older code “work” but now fails?**
JAX tightened PRNG checks to catch subtle bugs (e.g., correlated randomness from key reuse). Surfacing these earlier leads to safer, more reproducible code.

**Why not a global key?**
Globals tempt reuse and make code order-dependent. Passing keys explicitly documents intent and guarantees independence.

**What about dropout/noise in forward?**
Make it explicit: Equinox modules can accept a `key` in `__call__` (e.g., `Dropout(key=...)`), while NNX uses the model’s `rngs` stream plus a `deterministic` flag.

---

Adopting these patterns keeps your Flax NNX and Equinox models robust, deterministic when you want them to be, and aligned with JAX’s modern randomness model.
