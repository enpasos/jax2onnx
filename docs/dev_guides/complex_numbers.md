# Complex Number Support

The `onnx_ir` pipeline now keeps tensors in native `complex64` / `complex128` throughout lowering. This note captures the helpers, guardrails, and current limitations so plugin authors can wire complex primitives without falling back to real/imag packing.

## Native Complex Helpers

- `pack_native_complex(ctx, tensor)` — unwraps a complex tensor into a `[..., 2]` float view (`[..., 0]=real`, `[..., 1]=imag`). Use immediately before ONNX ops that require packed channels (e.g., `DFT`, `STFT`).
- `unpack_to_native_complex(ctx, tensor)` — rebuilds a complex tensor from a packed view by gathering the channel pair and emitting `Complex`.
- `ensure_complex_dtype(ctx, value, target_dtype)` — inserts a `Cast` when the materialized dtype (`COMPLEX64`, `COMPLEX128`) differs from the expected output.

All helpers live in `jax2onnx/plugins/_complex_utils.py`. They stamp shapes/metadata so downstream passes recognise the new values.

## Axis-0 Padding

`maybe_expand_binary_axis0` and `ensure_axis0_extent` now map complex IR dtypes to `np.complex64` / `np.complex128`, ensuring zero-padding retains the original dtype instead of truncating to `float32`.

## Plugin Coverage

- `lax.add` / `lax.mul` metadata include complex64/complex128 testcases. Numeric checks are skipped today because ONNX Runtime CPU builds ship without complex kernels; add the testcase key to `tests/extra_tests/framework/test_do_not_skip_numeric_validation.py` when skip flags are unavoidable.
- `lax.fft` uses the helpers to wrap ONNX `DFT` (1-D complex FFT). Inputs are packed before the operator and unpacked afterwards. Numeric validation is currently disabled for the same ORT gap as above.
- `tests/extra_tests/converter/test_complex_utils.py` exercises helper round-trips, dtype promotion, and error handling.

## Current Limitations

1. ONNX Runtime ≤ 1.19 emits `Real`/`Imag` kernels only in CUDA builds. CPU providers reject the op pair with complex inputs. Keep `skip_numeric_validation` in place until upstream support arrives.
2. The FFT plugin is limited to `FftType.FFT` with a single length (rank-1). Extending to inverse/real transforms will require additional packing rules and output shape handling.
3. No automated expect-graph snippets yet; regenerate once we finalise numeric coverage.

## Contributing Workflow

1. Use native complex tensors everywhere except the boundary to ops that demand packed channels.
2. Call `pack_native_complex` / `unpack_to_native_complex` around those boundaries.
3. Stamp dtype/shape metadata before binding outputs.
4. If you must skip numeric validation, document the upstream blocker and add the testcase triple to the skip allowlist.
5. Re-run `tests/extra_tests/converter/test_complex_utils.py` and any updated primitive suites (`tests/primitives/test_lax.py::Test_*`).
