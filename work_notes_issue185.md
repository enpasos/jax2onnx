# Implementation Plan: Issue #185 (Custom Input/Output Names)

## 1. Define API behavior

1. Add `input_names: Optional[Sequence[str]] = None` and `output_names: Optional[Sequence[str]] = None` to `to_onnx(...)`.
2. Keep default behavior unchanged when these arguments are omitted.
3. Define mapping rules:
   - `input_names` apply only to positional `inputs`, not `input_params`.
   - `output_names` apply by flattened output index.

## 2. Update public interface (`jax2onnx/user_interface.py`)

1. Extend all `to_onnx(...)` overloads with the new parameters.
2. Extend the runtime signature with the new parameters.
3. Update docstring and logging to document/print the new parameters.
4. Forward `input_names` and `output_names` into `to_onnx_impl(...)`.

## 3. Extend converter plumbing (`jax2onnx/converter/conversion_api.py`)

1. Extend converter `to_onnx(...)` signature with `input_names` and `output_names`.
2. Keep internal canonical names (`in_*`, `out_*`) during lowering/optimization.
3. Apply user-facing renames only near export finalization.

## 4. Add validation rules

1. Validate that provided names are sequences of non-empty strings.
2. Validate uniqueness within `input_names` and within `output_names`.
3. Validate expected lengths:
   - `len(input_names) == len(inputs)`
   - `len(output_names) == number_of_outputs`
4. Validate collisions with dynamic input parameter names (`input_params` keys).

## 5. Apply renaming late for safety

1. Perform renaming after optimization/postprocess to avoid interfering with optimization passes.
2. Rename `ir_model.graph.inputs[i].name` and `ir_model.graph.outputs[i].name` according to validated lists.
3. Ensure the same final names are visible in all return modes: `proto`, `ir`, and `file`.

## 6. Add tests (`tests/extra_tests/converter/test_io_names.py`)

1. Positive tests:
   - Custom names in default export.
   - Custom names with `inputs_as_nchw` / `outputs_as_nchw`.
   - Behavior across `return_mode` values (`proto`, `ir`, `file`).
2. Negative tests:
   - Length mismatch.
   - Duplicate names.
   - Empty names.
   - Collisions with `input_params`.
3. Regression tests:
   - Existing default naming unchanged when names are not provided.

## 7. Documentation updates

1. Add argument docs and examples in `docs/user_guide/getting_started.md`.
2. Add a layout-focused naming example in `docs/user_guide/layout_optimization.md`.

## 8. Verification checklist

1. `poetry run pytest -q tests/extra_tests/converter/test_io_names.py`
2. `poetry run pytest -q tests/extra_tests/converter/test_layout_optimization.py`
3. `poetry run pytest -q tests/extra_tests/converter/test_onnx_function_deterministic_param.py`
4. `poetry run ruff check .`
5. `poetry run ruff format .`
6. `poetry run mypy jax2onnx`

## 9. Codebase Analysis & Verification (Added)

* **Codebase State**: Confirmed `jax2onnx/user_interface.py` is the entry point and handles `input_params` materialization *after* calling `to_onnx_impl`. This confirms that renaming inside `to_onnx_impl` will correctly apply only to positional `inputs` as planned, without conflict with `input_params`.
* **Layout Handling**: `inputs_as_nchw` creates specific input nodes in `to_onnx_impl`; renaming the graph inputs at the end of `to_onnx_impl` will correctly rename these NCHW inputs, satisfying the requirement to support renaming in conjunction with layout attributes.
* **Point of Injection**: The proposed "Apply renaming late" strategy (after `optimize_graph` in `conversion_api.py`) is safe and appropriate to avoid breaking internal passes that might rely on specific naming conventions (though most shouldn't, safety is preferred).
