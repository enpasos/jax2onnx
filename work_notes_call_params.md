# Work Notes – Call Param Wiring (2025-10-03)

## Summary
- Added `no_unused_function_inputs` option to `expect_graph`; used in GPT metadata to ensure ONNX function bodies consume inputs such as `deterministic`.
- ONNX function lowering (`FunctionPlugin`) now:
  - Pulls `_call_input_param_names` from the parent context and carries them into nested function scopes.
  - Treats call-time parameters as dynamic inputs instead of constants, forcing them to surface as graph inputs.
  - Safely inspects tracer metadata (`tracer_to_var`, `constvar_to_val`) to remain compatible with newer JAX versions.
- Implemented missing primitives to support RNG-heavy examples: `random_bits`, `random_wrap`, `shift_right_logical`, `bitcast_convert_type`, and `bitwise_or`.
- Allowlisted `random_bits_uint32` for `skip_numeric_validation`; ORT cannot deterministically reproduce JAX RNG outputs.
- Added regression coverage (`tests/extra_tests2/converter/test_onnx_function_call_params.py`) to ensure ONNX function bodies consume forced call-time parameters.

## Follow-ups
- ✅ Re-ran GPT/VisionTransformer suites under Flax 0.12.0 (2025-10-03) — all green.
- ✅ Added focused regression test for ONNX functions with call-time params (2025-10-03).

## File References
- `jax2onnx/plugins2/_post_check_onnx_graph.py`
- `jax2onnx/plugins2/examples2/nnx/gpt.py`
- `jax2onnx/plugins2/plugin_system.py`
- `jax2onnx/plugins2/jax/random/random_bits.py`
- `jax2onnx/plugins2/jax/lax/shift_right_logical.py`
- `jax2onnx/plugins2/jax/lax/bitcast_convert_type.py`
- `jax2onnx/plugins2/framework/test_post_check_onnx_graph.py`
- `tests/extra_tests2/framework/test_do_not_skip_numeric_validation.py`
