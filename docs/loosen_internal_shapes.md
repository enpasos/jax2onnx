
# Loosen Internal Shapes (Loop/Scan/If Bodies)

**Status:** Advanced / opt-in. **Default: off** (backwards compatible)

## Why

Inside `Loop`/`Scan`/`If` bodies, ops like `Reshape`, `(Un)Squeeze`, `Expand`, `Concat`,
`Gather/GatherND`, `Slice`, `Cast`, or constant producers reintroduce **concrete** shapes
into internal `value_info`. ONNX Runtime’s inferencer then *re-tightens* shapes/types in
subgraphs, which can trigger failures in complex models (often nested loops):

- `TypeInferenceError: expected tensor(double) but got tensor(float)`
- `ShapeInferenceError: Incompatible dimensions`

## What it does

When enabled, we:
- **Drop** internal `value_info` for outputs of shape/dtype-sensitive ops:
  `Reshape`, `Squeeze`, `Unsqueeze`, `Expand`, `Concat`, `Range`, `Shape`,
  `NonZero`, `Gather`, `GatherND`, `Slice`, `Cast`, `Constant`, `ConstantOfShape`,
  and a light heuristic for index `Add`.
- For **remaining** internal `value_info`, keep **dtype + rank** but clear all dims
  → **rank-only** (no `dim_value`/`dim_param`).

This prevents subgraph shapes from being over-constrained and lets ORT infer safely.

## How to enable

**Python API**
```python
from jax2onnx import to_onnx

model = to_onnx(
    fn, inputs=[...],
    enable_double_precision=True,      # optional if you use x64
    loosen_internal_shapes=True        # ← enable
)
````

**Environment variable**

```bash
export JAX2ONNX_LOOSEN_INTERNAL_SHAPES=1
```

(Useful if you can’t change call sites. The function argument takes precedence when set to True.)

**CLI**

```bash
jax2onnx my.module fn --out model.onnx --opset 21 --float64 --loosen-internal-shapes
```

## Netron impact

Netron displays shapes from `value_info`. With loosening on, internal tensors in loop bodies
will mostly show as **dtype + rank** with unknown dims (e.g., `double[?, ?, ?]`), or sometimes
no explicit shape if their `value_info` was dropped. Top-level inputs/outputs remain unchanged.

## When to use

* ORT fails with shape/type inference errors in models using Loop/Scan/If (especially **nested loops**).
* You don’t rely on concrete internal dims for inspection/optimization.

Keep it off if you want maximum shape detail in Netron and your model already loads/runs fine.

## Interop with `enable_double_precision`

* `enable_double_precision=True` keeps graph tensors as `tensor(double)` (helpful for JAX x64).
* `loosen_internal_shapes=True` avoids re-tightening inside control-flow bodies.

They solve different problems and can be used together.

## FAQ

* **Numerical changes?** None — we modify metadata, not computation.
* **Could this hide bugs?** ORT will still error at execution time if shapes are truly incompatible.
* **Why not default on?** Backward compatibility and richer Netron visuals by default.

