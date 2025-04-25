# Supporting Symbolic Dimensions in jax2onnx

## Goal

The primary goal is to enable `jax2onnx` to correctly handle symbolic dimensions (e.g., dynamic batch sizes specified as strings like `"B"`) when converting JAX functions. This includes ensuring that custom primitives and patched JAX functions within the `jax2onnx` plugin system can correctly perform abstract evaluation with these symbolic dimensions.

## Problem Context: `abstract_eval` for Custom Primitives

The main challenge encountered was implementing the `abstract_eval` rule for custom JAX primitives (like the patched `jnp.concat_p`). This rule must compute the output shape and dtype abstractly. When symbolic dimensions are involved, accurately determining the output shape and returning it in a format compatible with JAX's tracing system proved difficult within the existing `jax2onnx` trace setup.

## Initial Investigation & Failed Approaches

Early attempts focused on modifying the custom primitive's `abstract_eval` directly, leading to several dead ends:

1.  **Returning Strings:** Returning shape tuples containing string representations (e.g., `'B'`) resulted in `TypeError: Shapes must be 1D sequences of integer scalars...`.
2.  **Returning Tracers:** Returning the dimension objects received (which were `DynamicJaxprTracer` instances) resulted in `TypeError: unhashable type...` upon `ShapedArray` creation.
3.  **Operating on Tracers:** Attempting arithmetic (`+`) or comparison (`==`/`!=`) on the received Tracer objects resulted in `jax.errors.UnexpectedTracerError` due to violating trace purity rules.
4.  **Using `jax.eval_shape`:**
    * Calling `eval_shape` on the *patched* function led to `RecursionError`.
    * Calling `eval_shape` with the received `avals` (containing Tracers) failed with `TypeError: ... is not a valid JAX type`.

## Root Cause Analysis: Trace Initiation vs. JAX Shape Polymorphism Internals

The core issue was identified by comparing the `jax2onnx` tracing context with standard JAX shape polymorphism usage:

* **JAX Standard Practice:** JAX (v0.4+) handles shape polymorphism using internal objects like `jax.core.DimVar` and `jax.core.SymbolicDimExpr` (`_DimExpr`). Tracing initiated with these objects (e.g., via `jax.export.symbolic_shape` followed by `jax.eval_shape` or `jax.make_jaxpr`) works correctly. Standard JAX primitive `abstract_eval` rules operate successfully on these `DimVar`/`_DimExpr` objects.
* **`jax2onnx` Current Trace:** The trace initiated by `jax2onnx.to_onnx` currently starts with user-provided **string** dimension names (e.g., `"B"`). During the subsequent `jax.make_jaxpr` call, this setup leads to the `abstract_eval` rules for primitives receiving problematic `DynamicJaxprTracer` instances within the `aval.shape` tuples, instead of the expected `DimVar`/`_DimExpr` objects.
* **Conclusion:** The failures within `abstract_eval` were symptoms caused by receiving the wrong *type* of symbolic dimension representation due to how the trace was initiated. `abstract_eval` cannot reliably operate on these Tracers.

## Proof of Concept (PoC) Validation

A self-contained PoC script (`poc_symbolic_primitive.py`) confirmed this analysis:
* It initiated `jax.eval_shape` using `ShapeDtypeStruct` inputs containing symbolic dimensions created via `jax.export.symbolic_shape("B")` (which yielded `_DimExpr`).
* A custom primitive's `abstract_eval` rule successfully received these `_DimExpr` objects in the input `aval.shape`.
* The `abstract_eval` rule could perform manual shape calculations (using `+` and `==`) on these objects without error.
* The final `core.ShapedArray` construction with the resulting symbolic shape tuple succeeded.

This validated that manual `abstract_eval` logic *is correct* when the trace provides the appropriate JAX symbolic dimension objects.

## Recommended Path Forward: Align Trace Initiation with JAX Polymorphism

The robust solution is to modify `jax2onnx` to initiate its trace using JAX's standard shape polymorphism mechanisms:

1.  **Parse User Input Strings:** In the `jax2onnx` frontend (`parameter_validation.py` or `user_interface.py`), identify dimension strings (e.g., `"B"`, `"H"`, potentially expressions like `"2*N"`) provided by the user in input shapes.
2.  **Convert Strings to JAX Symbols:** Use `jax.export.symbolic_shape` to convert these strings into the corresponding JAX internal symbolic dimension objects (`DimVar` or `_DimExpr`). Maintain consistency (e.g., map the same string `"B"` to the same `DimVar` object across different inputs).
3.  **Construct Abstract Inputs:** Create `jax.ShapeDtypeStruct` instances for each input argument, embedding the obtained symbolic dimension objects into their `.shape` tuples.
4.  **Initiate Trace with Symbolic Inputs:** Modify the core tracing routine (`Jaxpr2OnnxConverter.trace_jaxpr` in `jaxpr_converter.py`) to call `jax.make_jaxpr` (and potentially `jax.eval_shape` to get output avals) using these `ShapeDtypeStruct` objects containing symbolic dimensions as the input arguments.
5.  **Propagate Symbols to ONNX:** The resulting `jaxpr` will contain `aval`s with symbolic shapes. The ONNX conversion logic (`onnx_builder.py` / `jaxpr_converter.py`) must map these JAX symbolic objects to ONNX symbolic dimensions (`dim_param`). This can likely be done using `str(symbolic_dim_object)` to get a name (e.g., `"B"`) or expression string (e.g., `"2*B"`) for the `dim_param`. Ensure consistent mapping for shared dimensions.

## Implementation Details & Considerations

* **Code Changes:**
    * `parameter_validation.py` / `user_interface.py`: Update input shape handling to accept strings and trigger conversion to symbolic objects.
    * `jaxpr_converter.py`: Modify `trace_jaxpr` to use `jax.export.symbolic_shape` (or `symbolic_args_specs`) and pass the resulting `ShapeDtypeStruct`s with symbolic dims to `jax.make_jaxpr`.
    * Plugin `abstract_eval` Rules: Audit existing custom primitive `abstract_eval` rules. Most that use standard arithmetic (`+`, `*`, `//`, `==`) on shape elements should work correctly once they receive `DimVar`/`_DimExpr` instead of Tracers. Rules attempting `int(dim)` on a symbolic dimension will need adjustment. The manual calculation logic developed in the PoC (response #22) should be used for `jnp.concat_p`.
    * `onnx_builder.py` / `jaxpr_converter.py`: Implement the mapping from JAX symbolic dimension objects (in `aval.shape`) to ONNX `dim_param` strings when creating `ValueInfoProto`.
* **Benefits:** Enables `jax2onnx` to export models with dynamic dimensions (e.g., batch size), preserving shape relationships symbolically. Allows custom primitive `abstract_eval` rules to perform more complex symbolic shape inference correctly.
* **Risks/Caveats:**
    * **ONNX Limitations:** ONNX `dim_param` is just a name. Complex JAX symbolic expressions (e.g., `2*B+1`) will likely map to opaque names (e.g., `"dim_expr_1"` or `"2*B+1"`), and ONNX won't understand the underlying arithmetic relationship. This is usually acceptable.
    * **Backward Compatibility:** Decide how to handle previous uses of `None` or `-1` for dynamic dimensions. Consider converting them to default symbols (e.g., `"unk0"`, `"unk1"`) or deprecating in favor of named symbols.
    * **Testing:** Requires thorough testing with various symbolic shape patterns and custom primitives.

This approach aligns `jax2onnx` with JAX's native shape polymorphism features, providing a robust and maintainable solution.

## References

* JAX Shape Polymorphism Docs & Examples: [jax/jax/experimental/jax2tf/README.md](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md), [Shape polymorphism — JAX documentation](https://docs.jax.dev/en/latest/export/shape_poly.html)
* JAX2TF Implementation Details: [jax/jax/experimental/jax2tf/jax2tf.py](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/jax2tf.py)
* Related JAX Discussions: [Shape Polymorphism, with an image downscale · Discussion #15995](https://github.com/google/jax/discussions/15995)