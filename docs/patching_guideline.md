### Guideline  

1. **Keep JAX pristine**
   The public behaviour of any jax/flax/equinox function like `jax.lax.select` must stay exactly as it is upstream.
   *No permanent or process‑wide monkey‑patching of core JAX primitives* is allowed.

2. **Patching is conversion‑only**
   Temporary monkey‑patches are acceptable *solely* inside the internal context JAX‑to‑ONNX uses while tracing a function.
   Outside that context—e.g., during numerical validation—those patches **must be gone** and JAX must behave normally.

3. **Tests must respect real JAX semantics**
   If a test case feeds shapes or values that original `jax.select` cannot broadcast, *the test must be rewritten*, not the primitive.
   The numeric check must run—and pass—using unmodified JAX.

4. **Plugin responsibility**
   The `SelectPlugin`’s job - as an example - is to:

   * Emit a correct ONNX `Where` node for valid JAX graphs.
   * Provide an *internal* patch (via `patch_info`) only to let the tracer succeed on broadcasting patterns that JAX itself disallows.
   * Never leak that patch outside the tracer context.

5. **No hidden side‑effects**
   After ONNX conversion is finished, the runtime environment must be indistinguishable from a stock JAX installation.

> In short: **patch inside the sandbox, validate in the wild.**
