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
10. `rewrite_mul_rsqrt_as_div`
11. `inline_dropout_training_mode_constants`
12. `propagate_elementwise_shapes`
13. `propagate_unary_shapes`
14. `remove_redundant_casts_after_propagation`
15. `remove_dead_nodes`
16. `remove_orphan_transposes`
17. `prune_unused_graph_inputs`

Only graph-scoped passes with `function_bodies=True` run inside imported ONNX
Function bodies. Model-wide passes (`name_fix`, CSE, constant lifting,
dead-node removal) and `prune_unused_graph_inputs` stay top-level only.

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
| `rewrite_mul_rsqrt_as_div` | graph + functions | Rewrite multiplication by reciprocal square root into direct division where the producer chain is exact. |
| `inline_dropout_training_mode_constants` | graph + functions | Inline static Dropout `training_mode` inputs so inference exports stay compact. |
| `propagate_elementwise_shapes` | graph + functions | Propagate known shape metadata through elementwise operators. |
| `propagate_unary_shapes` | graph + functions | Propagate known shape metadata through unary operators. |
| `remove_redundant_casts_after_propagation` | graph + functions | Re-run cast cleanup after shape/dtype propagation may have exposed new no-op casts. |
| `remove_dead_nodes` | model | Remove nodes that are unreachable from graph outputs. |
| `remove_orphan_transposes` | graph + functions | Remove layout transposes whose outputs no longer feed live consumers. |
| `prune_unused_graph_inputs` | graph only | Remove unused top-level graph inputs while preserving ONNX Function signatures. |

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

- Follow the true consumer chain by **name or object identity** (some `onnx_ir` builds wrap/rename `Value` objects).
- Skip helper nodes on side branches (`Const`, `Shape`, etc.) that do not consume the current tensor.
- Require **single consumer** at each hop (no branching rewires).
- Read permutations from the `perm` attribute when available.
- When `perm` is missing, treat the pair as cancellable only if the input and output shapes match and the middle segment is strictly elementwise.

### Rewiring and deletion

- `onnx_ir.Node.inputs` may be immutable; use `Node.replace_input_with(index: int, value: Value)` when provided by the backend.
- Rewire **all** consumers of the second transpose’s output (by name or object) to the kept tensor.
- Update graph/model outputs and the var→value map so no reference points at removed nodes.
- Delete nodes in reverse order (second transpose first), maintaining any live list mirrors (`graph.nodes`, `graph._nodes`, etc.).

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
