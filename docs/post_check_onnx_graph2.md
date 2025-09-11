# `expect_graph2` — Shape-aware ONNX/IR graph checks

`expect_graph2` is a small, readable assertion helper for tests.  
It keeps the simplicity of our current `expect_graph`, but adds:

- **Edge shape checks** (with symbols)
- **No-stray-inputs** assertion
- **Absence checks** (e.g., no `Not`)
- Works on **top graph** and **function bodies**
- Format-agnostic: ONNX `ModelProto` or `onnx_ir` models

---

## Import

```python
from jax2onnx.plugins2._post_check_onnx_graph2 import expect_graph2 as EG2
````

## Quick start

### Static shapes

```python
EG2(
  [
    "Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10",
  ],
  symbols={"B": None},   # unify B across the path
  must_absent=["Not"],   # fail if Not is anywhere in the model
  no_unused_inputs=True, # fail if dangling graph inputs remain (e.g., 'deterministic')
)
```

### Dynamic shapes (unknown batch)

Use `?` for unknown dims:

```python
EG2(
  [
    "Gemm:?x20 -> BatchNormalization:?x20 -> Dropout:?x20 -> Gelu:?x20 -> Gemm:?x10",
  ],
  no_unused_inputs=True,
)
```

### Multiple specs & counts/attrs

```python
EG2(
  [
    (
      "Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10",
      {
        "attrs": {
          "Dropout": {"ratio": lambda v: 0.0 < float(v) <= 0.1000001},
        },
        "counts": {"Dropout": 1, "Not": 0},
      },
    ),
  ],
  symbols={"B": None},
  must_absent=["Not"],
)
```

### Searching function bodies

By default `expect_graph2` searches the top graph **and** all function bodies.
(Non-matching top graphs are fine as long as one graph matches.)

---

## Path & shape syntax

* Use `->` to connect nodes by **dataflow**.
* Put a shape after a node name to check the **edge leaving that node**:

  * `Bx20` → two dims: a symbol `B`, then literal `20`
  * `?x10` → unknown dim then `10`
  * `20`   → 1-D tensor with length `20`
* Symbols (e.g., `B`) must be **consistent** across all shapes in a spec.
* Unknown actual dims unify with `?` and **also** with symbols (they won’t bind).

Examples:

```
"Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10"
"Gemm:?x20 -> BatchNormalization:?x20 -> Dropout:?x20 -> Gelu:?x20 -> Gemm:?x10"
```

---

## Extra arguments

* `symbols={"B": None}` — declare symbols in the path; they’ll bind to actual sizes when they are known.
* `must_absent=["Not", "Identity"]` — fail if any of these ops appear.
* `no_unused_inputs=True` — fail if the **top graph** contains dangling inputs.
* `mode="all" | "any"` — all specs must match (default), or at least one.

---

## Return value

`EG2([...])` returns a **callable** that takes a model and returns `True/False`.
In our test harness:

```python
"post_check_onnx_graph": EG2([...])
```

…is used as:

```python
assert post_check_onnx_graph(model)
```

---

## Notes

* Works for both ONNX models and `onnx_ir` models.
* Function body search is enabled by default. If you later need subgraphs (If/Loop bodies),
  we can extend the internal `_function_graphs` traversal similarly.
