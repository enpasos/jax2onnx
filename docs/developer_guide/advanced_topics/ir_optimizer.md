# IR Optimizer Passes

The converter runs a lightweight, IR-only optimization sweep after lowering and before serialization. Passes must be structure-only (no op-specific math) and safe across `onnx_ir` variants. This guide documents the current canon and the invariants each pass must respect.

## Pipeline Placement

The optimizer runs as **Step 2** in the conversion pipeline (see [architecture.md](../architecture.md#conversion-pipeline-detailed)):

1. Build raw IR (`to_onnx`)
2. **`optimize_graph`** ← runs here
3. Late attribute overrides
4. Finalize shapes
5. Return from `conversion_api`
6. Post-process (shape loosening, export prep)

This placement ensures:
- Optimization sees the raw, unpatched graph for maximum benefit.
- Late overrides only patch nodes that survived optimization.
- Shape finalization operates on an already-optimized graph.

Optimizer failures are logged and skipped by default so ordinary exports can
continue. Set `JAX2ONNX_STRICT_OPTIMIZER_FAILURES=1` when debugging or in CI to
re-raise the original optimizer exception.

---

## Pass Registry

`optimize_graph` executes the `_OPTIMIZER_PASSES` registry in order. Each entry
declares its name plus whether it runs on the top-level model, the top-level
graph, and/or function body graphs. Function bodies deliberately skip
model-wide passes and `prune_unused_graph_inputs` so function signatures remain
stable.

Current order:

1. `name_fix`
2. `remove_redundant_casts`
3. `remove_redundant_transpose_reduce`
4. `remove_redundant_transpose_add_forests`
5. `remove_redundant_transpose_pairs`
6. `remove_redundant_reshape_pairs`
7. `remove_identity_reshapes`
8. `common_subexpression_elimination`
9. `lift_constants_to_initializers`
10. `fold_conv_transpose_constant_bias`
11. `rewrite_mul_sigmoid_as_swish`
12. `rewrite_mul_rsqrt_as_div`
13. `inline_dropout_training_mode_constants`
14. `propagate_elementwise_shapes`
15. `propagate_unary_shapes`
16. `remove_redundant_casts_after_propagation`
17. `remove_dead_nodes`
18. `remove_orphan_transposes`
19. `prune_unused_graph_inputs`

Only graph-scoped passes with `function_bodies=True` run inside imported ONNX
Function bodies. Model-wide passes (`name_fix`, CSE, constant lifting,
dead-node removal) and `prune_unused_graph_inputs` stay top-level only.

### Precision policy

The shared converter finalizer passes `IRContext.precision_policy` to
`optimize_graph`. Direct calls default to `PrecisionPolicy.LEGACY`, preserving
the existing pass selection and environment-controlled behavior.

Every registered pass explicitly declares the precision policies for which it
is eligible. The optimizer selects that sequence once and applies it to the
existing top-level and Function-body paths. A newly registered pass must make
this choice explicitly.

For the bounded private SOURCE path, the following passes remain excluded
pending a numerical audit:

- `remove_redundant_transpose_reduce`, which can change reduction order;
- `fold_conv_transpose_constant_bias`;
- `rewrite_mul_sigmoid_as_swish`;
- `rewrite_mul_rsqrt_as_div`.

SOURCE retains the eligible structural cleanup passes. A legacy environment
flag does not authorize one of the excluded rewrites under SOURCE. In
particular, changing `x * (1 / sqrt(y))` to `x / sqrt(y)` removes a rounding
boundary even when the numerator is exactly one. A numerator merely close to
one also must not be discarded.

This pass selection protects the audited source representation; it does not
establish runtime kernel behavior or numerical acceptance for arbitrary models.
The private SOURCE policy is not a public export option.

## Graph Scope and Value Ownership

Passes receive an `ir.Graph` and traverse real `ir.Value` objects. Use the
shared functions in `converter/optimizer_graph_utils.py` for discovery:

```python
producer = producer_in_graph(graph, value)
consumers = consumers_in_graph(graph, value)
belongs_here = node_in_graph(graph, node)
```

These functions use `Value.producer()`, `Value.consumers()`, and `Node.graph`.
They match object identity within the supplied graph. A captured outer value
can have consumers in a child graph while its producer belongs to the parent.
Consumer discovery therefore checks each consumer's ownership rather than
assuming that `value.graph` equals the consumer graph.

Keep `list(graph)` snapshots when a pass removes or inserts nodes during
iteration. Pass the live graph to discovery functions: a snapshot records an
iteration order, not current membership. Removed nodes may remain referenced
by their output values, so a producer reference alone does not prove that the
node is still in the graph.

The explicit `legacy_producer_node`, `legacy_consumer_nodes`, and
`legacy_node_in_sequence` adapters accept detached node sequences. The first
two retain historical name matching for raw names and disconnected value
wrappers; sequence membership always tests exact identity. Existing private
imports remain aliases for these adapters. New passes use the graph-scoped
functions so equal names in different scopes cannot create false edges.
An `ir.Graph` also implements the node-sequence interface, so legacy helpers can
receive the live graph as well as a detached list. Their historical name fallback
applies in both cases; a sequence snapshot cannot supply current graph ownership.
This compatibility rule does not change the stricter native graph helpers.

Read shape and dtype metadata directly from `ir.Value.shape` and
`ir.Value.dtype`. Missing metadata and symbolic dimensions still require
semantic checks; alternate object layouts do not. Register new initializers
with `graph.register_initializer(value)`.

## Graph Rewrite Reference

`rewrite_mul_sigmoid_as_swish_ir` in `converter/ir_optimizations.py` is a
complete executable reference. It matches `x * Sigmoid(x)` within the live
graph, requires opset 24+, and replaces the pair with `Swish`. This example
builds typed values, runs that production pass, and checks the output contract:

```python
import onnx_ir as ir
from onnx_ir.tape import Tape

from jax2onnx.converter.ir_optimizations import rewrite_mul_sigmoid_as_swish_ir
from jax2onnx.converter.optimizer_graph_utils import producer_in_graph


def graph_rewrite_reference() -> ir.Graph:
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[3])
    output = ir.val("y", dtype=ir.DataType.FLOAT, shape=[3])
    output.metadata_props["example"] = "output metadata"
    graph = ir.Graph([x], [output], nodes=[], opset_imports={"": 24})
    tape = Tape(graph)
    sigmoid = tape.op(
        "Sigmoid",
        [x],
        output=ir.val("sigmoid", dtype=ir.DataType.FLOAT, shape=[3]),
    )
    tape.op(
        "Mul",
        [x, sigmoid],
        output=output,
        metadata_props={"example": "node metadata"},
    )

    rewrite_mul_sigmoid_as_swish_ir(graph)

    rewritten = graph.outputs[0]
    producer = producer_in_graph(graph, rewritten)
    assert producer is not None and producer.op_type == "Swish"
    assert producer.graph is graph and producer.inputs[0] is x
    assert rewritten.name == "y"
    assert rewritten.dtype == ir.DataType.FLOAT
    assert rewritten.shape == ir.Shape([3])
    assert rewritten.metadata_props["example"] == "output metadata"
    assert producer.metadata_props["example"] == "node metadata"
    assert producer_in_graph(graph, output) is None
    assert [node.op_type for node in graph] == ["Swish"]
    return graph


graph = graph_rewrite_reference()
```

The pass creates a replacement output with the original name, type, shape,
and value metadata, and copies node metadata to the replacement node. It uses
`graph.insert_before`, `ir.convenience.replace_all_uses_with` with
`replace_graph_outputs=True`, and `graph.remove` to keep use-def links and
outputs connected. A shared or observable Sigmoid output retains its node.
Discovery receives the live graph even while mutation iterates over a snapshot.
An IR rewrite operates on graph values directly; the JAX-var binding pattern
belongs to the [canonical plugin example](../plugin_system.md#primitive-plugins).

## Pass Summary

| Pass | Scope | Purpose |
| --- | --- | --- |
| `name_fix` | model | Normalize generated names through the ONNX Script pass stack. |
| `remove_redundant_casts` | graph + functions | Drop casts whose input already has the target dtype, and fold immediate cast pairs with no net dtype change. |
| `remove_redundant_transpose_reduce` | graph + functions | Remove transpose/reduce patterns where axes and shapes prove the transpose has no semantic effect. |
| `remove_redundant_transpose_add_forests` | graph + functions | Collapse broadcast-add forests introduced by layout conversions when every branch can be safely rewired. |
| `remove_redundant_transpose_pairs` | graph + functions | Fold inverse transpose pairs across shape-preserving elementwise chains. |
| `remove_redundant_reshape_pairs` | graph + functions | Fold flatten/unflatten pairs across shape-preserving elementwise chains. |
| `remove_identity_reshapes` | graph + functions | Drop reshapes whose requested static shape exactly matches the input shape. |
| `common_subexpression_elimination` | model | Run the ONNX Script CSE pass on the top-level model. |
| `lift_constants_to_initializers` | model | Lift eligible `Constant` nodes to graph initializers. |
| `fold_conv_transpose_constant_bias` | graph + functions | Attach a direct immutable channel bias as an absent ConvTranspose bias input. |
| `rewrite_mul_sigmoid_as_swish` | graph + functions | Replace an exact input-times-sigmoid pattern with Swish where the opset supports it. |
| `rewrite_mul_rsqrt_as_div` | graph + functions | Rewrite multiplication by reciprocal square root into direct division where the producer chain is exact. |
| `inline_dropout_training_mode_constants` | graph + functions | Inline static Dropout `training_mode` inputs so inference exports stay compact. |
| `propagate_elementwise_shapes` | graph + functions | Propagate known shape metadata through elementwise operators. |
| `propagate_unary_shapes` | graph + functions | Propagate known shape metadata through unary operators. |
| `remove_redundant_casts_after_propagation` | graph + functions | Re-run cast cleanup after shape/dtype propagation may have exposed new no-op casts. |
| `remove_dead_nodes` | model | Remove nodes that are unreachable from graph outputs. |
| `remove_orphan_transposes` | graph + functions | Remove layout transposes whose outputs no longer feed live consumers. |
| `prune_unused_graph_inputs` | graph only | Remove unused top-level graph inputs while preserving ONNX Function signatures. |

---

## ConvTranspose Constant Bias Folding

**Pattern:** `ConvTranspose(x, weights) → Add(channel_bias)`.

The pass attaches an immutable float32 or float64 channel bias as the third
ConvTranspose input. The convolution must have no existing bias, and its raw
output must have exactly one use and must not be a graph output. The Add must
broadcast only across the NCHW channel dimension without changing the output
shape. Constant reshape, squeeze, unsqueeze, identity, and transpose views of
the bias can be unwrapped when their shapes prove that vector order is unchanged.

The equivalent layout pattern
`ConvTranspose → Transpose → Add(channel_bias) → inverse Transpose` is also
folded when the permutations are explicitly inverse, all intermediate shapes
and dtypes agree, and none of the intermediate outputs is shared or observable.
This exposes the native NCHW pair emitted by channels-last NNX models without
moving across an activation or padding operation.

The pass preserves the Add's output value, public name, dtype, and shape. Shared
initializers remain unchanged, and function/subgraph constants remain in their
own scope. It performs no parameter arithmetic, combines no existing bias, and
does not cross a Pad or another intervening operator. FP16/BF16 are excluded
because integrating the bias can change intermediate rounding.

This structure exposes eligible `ConvTranspose → BatchNormalization` pairs to
[ONNX Runtime 1.31's BatchNorm fusion](https://github.com/microsoft/onnxruntime/pull/29542).
The rewrite itself uses standard ONNX inputs and does not depend on the installed
runtime. Runtime results can still differ slightly through floating-point
evaluation order; regression tests check dtype-appropriate numerical parity.

---

## Transpose Pair Folding

**Pattern**
`Transpose → [pure elementwise]* → Transpose`

**Condition**
The composed permutation of the Transpose nodes equals identity.

**Allowed middle ops**
Elementwise operators that do not reorder elements, including:
`Relu`, `Gelu`, `Elu`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Cast`, `CastLike`, `Identity`, `Not`, etc.

**Not folded**
Anything that crosses non-elementwise operators such as `AveragePool`, `Conv`, or similar layout-sensitive ops.

### Matching heuristics

- Follow the consumer chain by value identity within the current `ir.Graph`.
- Skip helper nodes on side branches (`Const`, `Shape`, etc.) that do not consume the current tensor.
- Require **single consumer** at each hop (no branching rewires).
- Read permutations from the `perm` attribute when available.
- When `perm` is missing, treat the pair as cancellable only if the input and output shapes match and the middle segment is strictly elementwise.

### Rewiring and deletion

- Use `Node.replace_input_with(index, value)` to update individual inputs.
- Rewire all uses of the second transpose's output with
  `ir.convenience.replace_all_uses_with(..., replace_graph_outputs=True)`.
- Update graph/model outputs and the var→value map so no reference points at removed nodes.
- Remove replaced nodes through `graph.remove(...)`; IR maintains ownership
  and its internal node containers.

This pass is intentionally conservative, portable across `onnx_ir` variants, and oblivious to specific operator semantics.

---

## Identity Reshape Removal

**Pattern**
`Reshape(x, shape)` where `shape` is a constant that exactly matches `x`’s known dimensions.

**Condition**
- The `shape` input is a constant tensor with no `-1` or `0` entries.
- Every dimension of `x` is statically known and equal to the requested target.
- Output metadata (if present) already reflects the same shape.

**Effect**
Rewire consumers of the Reshape output directly to the input and drop the node.
Any now-unused shape initializers are left for later dead-code removal passes.

This trims redundant layout annotations generated by higher-level conversions
(e.g., Equinox attention blocks) without touching dynamic reshape cases.

---

## Redundant Cast Removal

**Patterns**
1. `Cast(x, to=T)` where `x` already has dtype `T`.
2. `Cast(x, to=T) → Cast(y, to=S)` where `S` equals the original dtype of `x`.

**Effect**
The Cast node(s) are removed and consumers are rewired to the original input `x`, assuming the net effect on dtype is identity.

---

## Reshape Pair Folding

**Pattern**
`Reshape(A) → [elementwise]* → Reshape(B)`

**Condition**
- The allowed elementwise ops are the same as in Transpose folding (shape-preserving).
- The input shape of the first Reshape matches the output shape of the second Reshape.

**Effect**
Both Reshape nodes are removed. The elementwise ops are rewired to consume `A` directly, and the consumers of `B` are rewired to the output of the last elementwise op. This eliminates redundant flatten/unflatten pairs often emitted by high-level frameworks.

---

## Authoring new passes

- Keep logic IR-only—never import ONNX protobuf utilities.
- Prefer `ir.Value` ownership helpers (`producer()`, `consumers()`,
  `ir.convenience.replace_all_uses_with`, `graph.remove(...)`) over private
  mirror mutation.
- Preserve graph outputs and function signatures explicitly when rewiring.
- Use shared optimizer utilities before adding new local graph-walking helpers.
- Add focused regression tests under `tests/extra_tests/framework/`.
- Document the new rule here and reference the guide from `architecture.md`.
