# `expect_graph` — Shape-aware ONNX/IR graph checks

`expect_graph` is our structure+shape assertion helper for ONNX graphs and
`onnx_ir` models. It supersedes the legacy matcher with richer semantics:

- **Shape-aware** edges (`Conv:Bx32`, `Reshape:?x10`) including symbol binding.
- **Counts / attribute predicates** per-operator when you need extra guards.
- **Global policies** such as `must_absent=["Not"]` or `no_unused_inputs=True`.
- Works on the **top graph** and, optionally, **function bodies**.
- Format agnostic: ONNX `ModelProto` or `onnx_ir` objects.

---

## Import

```python
from jax2onnx.plugins2._post_check_onnx_graph import expect_graph as EG
```

Many tests alias it to `EG` so expectations stay compact.

---

## Quick start

### Static shapes

```python
EXPECT_DROPOUT_PATH = EG(
    [
        "Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10",
    ],
    symbols={"B": None},   # bind B the first time we see it
    must_absent=["Not"],   # fail if Not is anywhere in the model
    no_unused_inputs=True, # ensure no dangling graph inputs remain
)
```

### Dynamic shapes (unknown batch)

Use `?` to tolerate unknown dimensions:

```python
EXPECT_DYNAMIC = EG(
    [
        "Gemm:?x20 -> BatchNormalization:?x20 -> Dropout:?x20 -> Gelu:?x20 -> Gemm:?x10",
    ]
)
```

### Multiple specs with counts / attrs

```python
EXPECT_DROPOUT_ATTRS = EG(
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
)
```

### Searching function bodies

By default only the **top graph** is inspected. Set `search_functions=True` to
scan function graphs as well:

```python
EXPECT_FN_BODY = EG(
    ["Reshape:?xK -> Gemm:?xN"],
    search_functions=True,
)
```

---

## Path & shape syntax

* `->` connects producer/consumer nodes.
* Append `:shape` to check the edge leaving that node:
  * `Bx20` → two dims: symbol `B`, literal `20`.
  * `?x10` → unknown dim, then `10`.
  * `20`   → 1-D tensor of length `20`.
* Symbols declared in `symbols={...}` unify across all uses in a spec.
* Unknown actual dims unify with `?` and with unbound symbols (they simply
  refuse to bind).

Examples:

```
"Gemm:Bx20 -> BatchNormalization:Bx20 -> Dropout:Bx20 -> Gelu:Bx20 -> Gemm:Bx10"
"Transpose:?xCxH -> Conv:?xKxH -> Transpose:?xHxK"
```

---

## Extra arguments

* `symbols={"B": None}` — declare shape symbols used in paths.
* `must_absent=["Not", "Identity"]` — globally forbid operators.
* `no_unused_inputs=True` — fail if the top graph has dangling inputs.
* `mode="all" | "any"` — require every spec (default) or any single spec.
* `passthrough_ops={...}` — extend the set of ops skipped while walking
  between anchors (`Reshape`, `Identity`, `Cast*`, `Squeeze`, `Unsqueeze`,
  `Flatten` are included by default).
* `explain_on_fail=False` — silence the debug dump when a check fails.

---

## Return value

`EG([...])` returns a predicate that accepts an ONNX/IR model and resolves to
`True`/`False`. In test metadata you typically wire it through
`"post_check_onnx_graph": EG([...])` and then `assert post_check(model)`.

---

## Notes & tips

* Works with both ONNX protobuf models and `onnx_ir` graphs (duck-typed).
* Counts/attrs are validated **after** a path matches; use them to assert that
  required operators appear exactly once (or never).
* When shapes are missing from the model, shape checks fall back to `None`; use
  `?` in the spec or keep symbols unbound to tolerate that.
* The matcher only inspects data-flow edges; control-flow subgraphs can be
  added later by extending `_function_graphs` if needed.
