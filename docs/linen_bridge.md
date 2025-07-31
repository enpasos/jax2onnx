# Proposal: Robust and Maintainable Conversion of Flax Linen Modules via the NNX Bridge in `jax2onnx`

Direct tracing of `flax.nnx.bridge.ToNNX` fails due to a fundamental conflict between stateful wrapper objects and stateless JAX tracing. Rather than fragile monkey-patching `ToNNX`, this document proposes enhancing `jax2onnx` with a specialized handler that explicitly detects the `ToNNX` wrapper, extracts its state into a stateless representation, and traces a pure function using these parameters.

---

## 1. Background

Converting Flax Linen models to ONNX using `jax2onnx` involves symbolic tracing via `jax.make_jaxpr`. Flax's `nnx.bridge.ToNNX` wrapper introduces stateful handling of parameters and RNG streams, causing conflicts during symbolic tracing.

## 2. The Problem: State Pollution During Symbolic Tracing

Tracing stateful modules directly fails because:

* **Stateful initialization** creates concrete parameters (e.g., kernel shapes).
* **Symbolic tracing** later tries to mix symbolic input shapes with previously stored concrete parameters.

This mismatch leads to shape validation errors during tracing:

```text
ScopeParamShapeError: Initializer expected to generate shape (10, 128) but got shape (Var(id=...):int64[], 128) instead...
```

## 3. Why Monkey-Patching `ToNNX` is Insufficient

Experiments with monkey-patching `ToNNX.__call__` show:

* The method remains in a stateful trace context, thus always polluted by concrete state.
* Symbolic tracers inevitably conflict with existing concrete parameters.

This approach cannot fundamentally solve the issue.

## 4. Proposed Solution: Explicit Special Handling in `jax2onnx`

### Concept

`jax2onnx` should directly recognize and handle `nnx.bridge.ToNNX` instances by:

1. Detecting if the provided model is a `ToNNX` wrapper.
2. Extracting internal state (parameters, optionally mutable states) into a clean dictionary.
3. Creating and tracing a pure, stateless function with extracted parameters passed explicitly.

### Detailed Conceptual Implementation

```python
def to_onnx(model, inputs, ...):
    from flax import nnx

    if isinstance(model, nnx.bridge.ToNNX):
        # Step 1: Extract parameters and mutable state
        nnx_attrs = {
            k: v for k, v in vars(model).items()
            if k not in ['module', 'rngs'] and not k.startswith('_object__')
        }
        linen_variables = nnx.bridge.variables.nnx_attrs_to_linen_vars(nnx_attrs)

        params = linen_variables.get('params', {})
        other_vars = {k: v for k, v in linen_variables.items() if k != 'params'}

        # Step 2: Define a pure function wrapper explicitly
        def pure_apply(params, *args):
            variables = {'params': params, **other_vars}
            return model.module.apply(variables, *args)

        # Step 3: Perform stateless symbolic tracing
        return to_onnx_internal(
            pure_apply,
            [params, *inputs],
            ...
        )
    else:
        # Standard tracing logic
        ...
```

### Advantages

* **Robustness**: Avoids tracing stateful objects directly, eliminating shape conflicts.
* **Minimal Maintenance**: Clearly defined, minimal changes to `jax2onnx`.
* **Official Abstraction**: Leverages official Flax bridge APIs, ensuring compatibility.
* **Future-Proof**: Easily extendable to more complex stateful modules (BatchNorm, etc.).

## 5. Next Steps

* **Prototype**: Develop and test this specialized handler in `jax2onnx`.
* **Validate**: Confirm compatibility with standard Linen layers (Dense, Conv, LayerNorm, etc.).
* **Document**: Clearly document this special handling behavior within `jax2onnx`.
* **Review**: Discuss with maintainers for approval and refinement.

## 6. Conclusion

This proposal provides a robust, explicit, and future-proof solution for converting Flax Linen modules via the NNX bridge, significantly simplifying and stabilizing the ONNX conversion process.
