
# Loosen Internal Shapes (Loop/Scan/If Bodies)

**Status:** Always enabled (since Sept 2025).

## Why

Inside `Loop`/`Scan`/`If` bodies, ops like `Reshape`, `(Un)Squeeze`, `Expand`, `Concat`,
`Gather/GatherND`, `Slice`, `Cast`, or constant producers reintroduce **concrete** shapes
into internal `value_info`. ONNX Runtime’s inferencer then *re-tightens* shapes/types in
subgraphs, which can trigger failures in complex models (often nested loops):

- `TypeInferenceError: expected tensor(double) but got tensor(float)`
- `ShapeInferenceError: Incompatible dimensions`

## What it does

The converter automatically:

* **Drops** internal `value_info` for outputs of shape/dtype-sensitive ops:
  `Reshape`, `Squeeze`, `Unsqueeze`, `Expand`, `Concat`, `Range`, `Shape`,
  `NonZero`, `Gather`, `GatherND`, `Slice`, `Cast`, `Constant`, `ConstantOfShape`,
  plus a light heuristic for index `Add`.
* For the **remaining** internal `value_info`, keeps **dtype + rank** but clears
  all concrete dimensions → “rank-only” (no `dim_value`/`dim_param`).

This prevents subgraph metadata from re-tightening shapes and lets ONNX Runtime
infer safely even in deeply nested control-flow graphs.

## Netron impact

Netron shows whatever `value_info` reports. With loosening baked in, internal
Loop/Scan/If tensors typically appear as dtype + rank with unknown dims (for
example `double[?, ?, ?]`). Top-level inputs/outputs retain their original
shapes.

## Interop with `enable_double_precision`

`enable_double_precision=True` still controls tensor element types (`float`
vs. `double`). Shape loosening operates independently and is always active.

## FAQ

* **Numerical changes?** None — we only adjust metadata.
* **Could this hide bugs?** ORT will still fail at execution time if shapes are
  genuinely incompatible; loosening merely avoids over-constrained metadata.
