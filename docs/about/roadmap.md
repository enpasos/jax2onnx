# Roadmap

## Planned

* Document and test how to export a JAX model to ONNX and run it in a mobile
  application.
* Expand automated compatibility tests for FP16, BF16 and FP64 computations,
  input dimensions that vary between calls, and non-square arrays. Document
  supported operator and runtime combinations and their numerical limits.
* Add a runnable numerical or physical simulation example that exports a
  complete computation to ONNX and executes it in ONNX Runtime without JAX,
  with numerical checks against the original model.
* Publish a reproducible example that loads a trained reinforcement-learning
  policy from an established library and exports its observation-to-action
  computation to ONNX for inference.
* Keep tables of supported JAX, Flax and Equinox APIs current as those libraries
  evolve, and add regression tests for remaining operator and model conversion
  gaps.


## Current Version

### **jax2onnx 0.19.1**

* **Keep bfloat16 exports valid at the target opset:** ONNX admitted bfloat16
  operator by operator, for example `Conv` and the pooling operators from
  opset 22, `TopK` from opset 24 and `Einsum` from opset 28. Exports below
  that opset wrote the operator with bfloat16 operands, which
  `onnx.checker.check_model(..., full_check=True)` and ONNX Runtime reject;
  a bfloat16 matrix product lowered to `Einsum` was affected at the default
  opset 23. Such operators now compute in float32 between bfloat16 values,
  including inside control-flow and `@onnx_function` bodies. Inputs, outputs,
  constants and state inputs keep their bfloat16 type and payload, and the
  operator stays in bfloat16 from the opset that admits it. Cases that casts
  cannot fix raise `NotImplementedError` with the required opset: bfloat16
  `jax.random.normal` and `jnp.eye` before opset 22, and bfloat16 values
  through `If` or `Loop` before opset 16. See
  [bfloat16 and the ONNX operator schemas](dependencies.md#bfloat16-and-the-onnx-operator-schemas).
  Thanks to Clément Poiret for reporting the `Conv`, `Einsum` and `TopK`
  cases in
  [discussion #290](https://github.com/enpasos/jax2onnx/discussions/290).
* **Export bfloat16 window maxima and minima:** `lax.reduce_window` with a
  maximum or minimum reducer rejected bfloat16 operands as non-floating. It
  now lowers to `MaxPool` like the other floating types.
* **Load float16 LayerNorm exports on ONNX Runtime before 1.30:** ONNX Runtime
  1.24.1 to 1.29.0 aborted the process while creating a session with the
  default `ORT_ENABLE_ALL` for a float16 model with the explicit LayerNorm
  graph, for example the Flax NNX and Linen LayerNorm and the transformer
  examples built on them. The float16 scale and bias were widened to float32
  with `CastLike`, which an optimizer of those runtime versions cannot handle
  on a float16 constant. Parameters whose dtype differs from the dtype they
  are used with, and `jnp.linspace`, are now adapted with `Cast`, which names
  the same conversion and is valid at every opset. The export no longer writes
  `CastLike`, which ONNX introduced in opset 15. Float16 state inputs remain
  affected by a second defect of the same optimizer; see
  [Known Limitations](../user_guide/known_limitations.md#fp16-state-before-onnx-runtime-130)
  and
  [the float16 initializer optimizer](dependencies.md#the-float16-initializer-optimizer-before-onnx-runtime-130).
  Thanks to Clément Poiret for reporting the abort in
  [discussion #290](https://github.com/enpasos/jax2onnx/discussions/290).
* **Match `jax.nn.one_hot` outside the class range:** The export used ONNX
  `OneHot`, which counts an index in `[-depth, -1]` from the end and truncates
  a non-integer index. JAX returns a row of zeros for both, so a label such as
  `-1` exported as the last class. `jax.nn.one_hot` and its Flax Linen alias
  now export the comparison that defines the function in JAX,
  `Unsqueeze -> Equal -> Cast`, which also needs no float32 work for bfloat16
  results. The default result dtype follows the x64 setting as in JAX, where
  the export always wrote float32, and boolean indices are rejected as in JAX.
* **Reject `jnp.cumprod` and `lax.cumprod` below opset 26:** Both export as
  ONNX `CumProd`, which ONNX introduced in opset 26. At the default opset 23
  the export wrote the operator anyway, and the checker and ONNX Runtime
  rejected the file. The export now raises `NotImplementedError` with the
  required opset. With `opset=26` the result is checked against JAX in ONNX
  Runtime. See
  [Known Limitations](../user_guide/known_limitations.md#cumulative-products-from-opset-26).
* **Type random bits by their requested width:** `jax.random.bits` with
  `uint8` or `uint16` failed because the bits were always exported as
  `uint32`. This also made float16 `jax.random.uniform` write an invalid
  `BitShift` at opset 26 and newer. `jax.random.uniform` still requires opset
  26 for every dtype, because JAX builds it with `lax.bitcast_convert_type`.
* **Export window minima taken with `lax.min`:** `lax.reduce_window` with
  `lax.min` and its identity `+inf` reaches the converter as the primitive
  `reduce_window_min`, which had no lowering. It now exports as
  `Neg -> MaxPool -> Neg` for floating operands, like the reducer form.
* **Keep `jax.nn.hard_swish` valid at opset 13:** ONNX introduced `HardSwish`
  in opset 14. Below that the export writes the equivalent
  `x * HardSigmoid(x)`.
* **Keep RMSNorm valid below opset 18:** The explicit Flax NNX and Equinox
  RMSNorm graphs passed the `ReduceMean` axes as an input at every opset.
  `ReduceMean` takes them as an attribute before opset 18.
* **Document random operators on ONNX Runtime:** CPU ONNX Runtime 1.32.0 has
  no kernels for the opset-22 versions of the random operators, so exports
  that draw random numbers load only up to opset 21. See
  [Known Limitations](../user_guide/known_limitations.md#random-operators-from-opset-22).
* **Restore the `expect_graph` snippet tool:** `scripts/emit_expect_graph.py`
  failed for every testcase since `enable_double_precision` was removed from
  `to_onnx`. It now exports through the test generator's own input specs and
  options, so a snippet describes the graph that the generated test checks,
  and generated variants such as `<name>_dynamic` and `<name>_f64` are
  selectable by name.
* **Refresh locked dependencies:** Update ONNX Runtime to 1.32.0, NumPy to
  2.5.5 on Python 3.13 and 3.14, Optax to 0.2.9 and Netron to 9.3.6, along
  with patch releases of development and documentation tools. The JAX, Flax,
  Equinox, ONNX and ONNX IR versions are unchanged. The missing kernels listed
  in the [dependency guide](dependencies.md#onnx-runtime-kernel-gaps) were
  re-checked on ONNX Runtime 1.32.0 and are unchanged. The
  [quantized MNIST example](../user_guide/quantized_mnist.md) keeps the pinned
  environment its results were measured with; its CI job now installs those
  pins instead of expecting the lockfile to match them.

## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
