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

### **jax2onnx 0.19.0**

* **Export model state as named inputs:** Add `to_onnx(..., state=...)`,
  `StateInput` and `StateCaptureError` to export chosen weights and buffers as
  named graph inputs with saved defaults and caller-defined tensor metadata.
  The metadata is stored as given and does not encode training or gradient
  rules. Declared arrays must be pytree leaves of the exported callable. Tied
  weights become one input, and each saved default is bit-identical to its
  array. State arrays are traced as inputs, so derived values stay in the
  computation rather than becoming constants. Unread declarations raise by
  default; `allow_unused_state=True` omits them from the file instead. ONNX
  Runtime does not fold overridable defaults, which costs inference speed;
  `overridable=False` keeps name and metadata on an ordinary initializer for
  fixed inference files. See the tested contract in
  [Model State as Inputs](../user_guide/state_inputs.md) and the
  [Equinox RoPE walkthrough](../user_guide/state_inputs_walkthrough.md) for
  saving, reloading, replacing state and taking a JAX optimizer step.
  Thanks to Clément Poiret, who proposed `state=`, `StateInput`, `overridable`
  and the single binding point for captured constants in
  [discussion #290](https://github.com/enpasos/jax2onnx/discussions/290) and
  tested the result on Equimo.
* **Correct `nnx.LinearGeneral` output layouts and bias-free exports:**
  Restore multi-dimensional `out_features` after `Gemm` for rank-2 inputs,
  including symbolic batch sizes, and flatten rank-1 inputs and inputs whose
  axes are all contracted. Layers without a bias now export, including under
  `jax.ensure_compile_time_eval()`. Flax Linen `DenseGeneral` receives the
  same output-layout fix. Thanks to Clément Poiret for reporting the rank-2
  case in
  [discussion #290](https://github.com/enpasos/jax2onnx/discussions/290).
* **Keep call-time Equinox Dropout inference exports loadable:** When
  `eqx.nn.Dropout` receives a constant `inference=True` at call time, omit the
  optional `training_mode` input, which ONNX treats as False, instead of
  referencing an undefined value that ONNX Runtime rejects. This also applies
  inside larger modules and `@onnx_function` bodies, and the unused flag
  initializer is removed.
* **Preserve JAX power promotion:** Promote `jnp.power` and `jnp.pow` operands
  as JAX does. An integer base with a floating-point exponent, such as
  `jnp.power(10000, x)`, now computes in floating point; the exponent was
  previously cast to the base's integer type, so the model returned integer
  results. Concrete integer exponents retain the base dtype, even when the
  exponent is a wider NumPy integer and JAX x64 is enabled. Thanks to Clément
  Poiret, whose [PR #291](https://github.com/enpasos/jax2onnx/pull/291)
  contributed this fix and the compile-time evaluation, `vmap`, integer
  softmax, scatter and `expect_graph` fixes below.
* **Evaluate `jnp.stack` and `jnp.concatenate` at compile time:** Calls on
  concrete arrays under `jax.ensure_compile_time_eval()` no longer raise
  `TypeError`; their results are embedded as constants. Passing a single array
  to `jnp.stack` stacks its leading-axis slices, matching JAX's
  sequence-of-arrays behavior instead of exporting a wrong output shape.
* **Respect per-sample axes under `vmap`:** Interpret `axis` per sample for
  `jax.nn.softmax`, `jax.nn.log_softmax` and `jax.nn.one_hot`, including
  nested `vmap` and non-leading batch axes. Affected exports previously
  normalized over the wrong axis or misplaced the one-hot class axis without
  an error. Eligible vmapped softmax and log-softmax calls emit a single
  native ONNX operator rather than decomposed reductions. Masked softmax keeps
  the native `Softmax` lowering when the mask fits the logits; masks that
  expand singleton logits dimensions or add sample dimensions use JAX's
  decomposition, preserving output shapes and reduction axes.
* **Export integer softmax inputs:** `jax.nn.softmax` and `jax.nn.log_softmax`
  promote integer inputs to floating point as JAX does, through its
  decomposition, instead of emitting an integer `Softmax` or `LogSoftmax` that
  ONNX Runtime rejects.
* **Export indexed updates under `vmap`:** Support the operand and index
  batching dimensions that `vmap` adds to updates such as
  `x.at[idx].set(values)`, in `lax.scatter` and its `scatter_add`,
  `scatter_sub`, `scatter_mul`, `scatter_min` and `scatter_max` variants,
  including nested `vmap` and symbolic batch sizes. These exports previously
  raised `NotImplementedError`. Updates whose window axes are not trailing are
  no longer reordered, and `mode="clip"` clamps out-of-bounds indices in
  direct `ScatterND` lowerings instead of failing in ONNX Runtime.
* **Sort NaNs after positive infinity:** Preserve JAX's stable ordering in
  `jnp.sort`, `jnp.argsort` and `lax.sort`, including associated payloads and
  multiple sort keys. Select original values by their sort indices to retain
  signed zeros and NaN representations. Thanks to jay23606, whose
  [initial fix proposal](https://github.com/enpasos/jax2onnx/pull/295)
  contributed to resolving
  [issue #294](https://github.com/enpasos/jax2onnx/issues/294).
* **Preserve exported reshape intermediates:** Keep reshape pairs when the
  first reshape or a middle elementwise result is also a graph output, retaining
  its public shape, dtype, name and output position.
* **Preserve Max/Min broadcasts during layout optimization:** Retain reshape
  and transpose boundaries when other operands lack a matching layout proof,
  preventing invalid broadcasts and silent changes in elementwise results.
* **Share stored floating weights across singleton broadcast views:** At opset
  13 and newer, insert singleton axes with `Unsqueeze` so an FP16, BF16, FP32
  or FP64 parameter and its views read one stored tensor instead of separate
  copies. Integer and boolean constants, broadcasts that expand dimensions and
  older opsets retain their existing lowering.
* **Reuse constant parameter reshapes:** Repeated uses of one parameter at the
  same shape share a graph-local constant view, avoiding duplicate copies of
  reshaped weights in reused Flax NNX `LinearGeneral` layers while preserving
  dtype, exact payload and metadata. Overridable state inputs retain runtime
  reshapes.
* **Respect Reshape opset and zero dimensions:** Omit `allowzero` before
  opset 14, retain inferred dimensions, and preserve literal zero dimensions
  from opset 14, including mixed symbolic shapes. Older opsets reject literal
  zero placements that would copy a different input dimension. Use opset 14 or
  newer when symbolic reshape dimensions may become zero at runtime.
* **Strengthen plugin and structural regression checks:** Read operator
  attributes and constants correctly in both ONNX IR and serialized models.
  `expect_graph` attribute assertions require explicit values on every
  matching operator; defaults do not satisfy them. Lists and tuples compare
  alike, and floating attributes compare in single precision. See the
  [expect_graph reference](../developer_guide/advanced_topics/expect_graph_reference.md).
  Plugins that lower nested JAXPRs bind captured constants with
  `ctx.bind_captured_arrays(...)`, so declared state cannot enter the graph
  as a constant unnoticed; see the
  [Plugin System guide](../developer_guide/plugin_system.md). Add a discovery
  regression that fails when a plugin module cannot be imported, preventing
  silent gaps in lowering and generated-test coverage.
* **Refresh locked dependencies and documentation tools:** Update `dm-pix` to
  0.5.0, Netron to 9.3.5, MkDocs Material to 9.8.1 and Markdown to 3.11.3.
  Resolve pymdown-extensions to 12.3 on Python 3.11/3.12 and 13.0 on Python
  3.13/3.14. Declared minimum versions are unchanged.

## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
