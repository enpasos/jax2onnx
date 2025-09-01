# Graph Pattern Matcher (tests)

A tiny, readable way to assert the **shape of an ONNX graph** in tests by describing paths like

```
"Transpose(0)->Conv(1)->Transpose(2)"
```

This checks that there is a **direct producer→consumer chain** of nodes with those op types (and—if given—those exact graph indices).

> Import:
> `from jax2onnx.plugins2._post_check_onnx_graph import expect_graph`

---

## Why

* Replace bespoke lambdas like `_expect_transpose_conv_transpose`.
* Make tests self-documenting: the expectation is in the string.
* Control how strict the match is (subpath vs. exact chain) with anchors or a `match` mode.

---

## Quick start

```python
from jax2onnx.plugins2._post_check_onnx_graph import expect_graph

testcase = {
    "testcase": "conv_basic_bias",
    "callable": ...,
    "input_shapes": [("B", 28, 28, 3)],
    "use_onnx_ir": True,
    "run_only_f32_variant": True,
    "post_check_onnx_graph": expect_graph([
        "Transpose->Conv->Transpose",  # op sequence as a direct chain
    ]),
}
```

If you need to pin to **specific node indices** (0-based in `model.graph.node`):

```python
"post_check_onnx_graph": expect_graph([
    "Transpose(0)->Conv(1)->Transpose(2)",
]),
```

Multiple patterns:

```python
# All must match (default)
expect_graph([
    "Transpose->Conv->Transpose",
    "Conv",  # also ensure at least one Conv exists anywhere
])

# Any may match
expect_graph([
    "Transpose->Conv->Transpose",
    "Conv->Relu",
], mode="any")
```

---

## Pattern syntax

A pattern is a single **path**:

```
OPTYPE[(INDEX)]->OPTYPE[(INDEX)]->...
```

* `OPTYPE` is the node’s `op_type` (e.g., `Conv`, `Transpose`).
* `(INDEX)` is **optional**. When present, it must match the node’s absolute index in `model.graph.node` (0-based). When omitted, **any** node of that type can match.
* `->` means **direct adjacency**: at least one output of the left node is consumed as an input of the right node (no intervening nodes).

### Anchors (optional)

You can add anchors to the pattern string itself:

* `^pattern` – the first node must be a **graph source** (no producers).
* `pattern$` – the last node must be a **graph sink** (no consumers).
* `^pattern$` – both must hold.

Examples:

```
"Conv"                       # at least one Conv exists
"Transpose->Conv"            # some Transpose directly feeds some Conv
"Transpose(3)->Conv(4)"      # specifically node 3 → node 4 with those types
"^Transpose->Conv->Transpose$"  # exact chain from source to sink
```

---

## Match strictness

By default, `expect_graph` looks for each pattern **as a subpath** in the graph (i.e., the graph may be larger):

```python
expect_graph(["Transpose->Conv->Transpose"])                 # default: match="contains"
```

To control strictness, use the `match` keyword:

```python
# Require that the chain starts at a graph source
expect_graph(["Transpose->Conv->Transpose"], match="prefix")

# Require that the chain ends at a graph sink
expect_graph(["Transpose->Conv->Transpose"], match="suffix")

# Require the chain to be both source-anchored and sink-anchored
expect_graph(["Transpose->Conv->Transpose"], match="exact")
```

You can also encode anchors inline (equivalent to the `match` modes):

```python
expect_graph(["^Transpose->Conv->Transpose$"])  # same as match="exact"
```

### Common gotcha

If your graph is `Transpose->Conv->Transpose` and you write:

```python
expect_graph(["Transpose->Conv"])   # passes (it's a subpath)
```

…it will **pass** in the default `contains` mode. If you intend this to **fail** because the final `Transpose` is also required, then either:

* specify the full chain, or
* use `match="exact"` (or anchors `^...$`) with the full chain.

---

## What it checks under the hood

* Builds a lightweight adjacency from node **output names → input names**.
* Supports both `onnx_ir` style (`.inputs`/`.outputs`) and ONNX `ModelProto` style (`.input`/`.output`).
* Backtracks through candidate nodes to find a chain that matches the pattern.
* `match="prefix"`/`"suffix"`/`"exact"` are enforced by checking node in-/out-degree.

---

## API

```python
def expect_graph(
    patterns: Iterable[str],
    *,
    mode: str = "all",
    match: str = "contains",
) -> Callable[[Any], bool]
```

* `patterns`: one or more path strings (see syntax above).
* `mode`:

  * `"all"` (default): **every** pattern must be found.
  * `"any"`: **at least one** pattern must be found.
* `match`:

  * `"contains"` (default): pattern may appear as a subpath in a larger chain.
  * `"prefix"`: first node must be a **source**.
  * `"suffix"`: last node must be a **sink**.
  * `"exact"`: both source & sink (equivalent to `^pattern$`).

---

## Tips

* Prefer **no indices** unless you truly want to lock the graph to a specific order; this keeps tests resilient to benign node reordering.
* Patterns assert **direct** edges. If you need “reachable through any number of nodes,” that’s a different matcher; this one only matches immediate producer→consumer chains.
