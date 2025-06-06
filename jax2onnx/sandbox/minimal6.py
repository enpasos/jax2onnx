# file: jax2onnx/sandbox/minimal6.py

import os
import jax
import jax.numpy as jnp
from jax import lax  # For lax.cond
import numpy as np
import onnx
from onnx import TensorProto
import onnxruntime
from jax2onnx import to_onnx


def run_minimal_reproducer_v6():
    jax.config.update("jax_enable_x64", True)
    print(f"JAX enable_x64: {jax.config.jax_enable_x64}")
    print("-" * 40)

    def minimal_problem_fn():
        limit_val = float(2 * 4 * 1 * 1)
        original_operand_val = jnp.arange(
            start=0.0, stop=limit_val, step=1.0, dtype=jnp.float64
        ).reshape((2, 4, 1, 1))

        raw_updates_data_val = jnp.ones((1, 4, 1, 1), dtype=jnp.float64) * 100.0
        reshaped_updates_for_slices_val = jnp.reshape(
            raw_updates_data_val, (1, 4, 1, 1)
        )

        indices_for_axis_0_val = jnp.array([1])

        # Introduce a dummy predicate for lax.cond
        predicate = jnp.array(True)

        # Pack operands for the branches into a single tuple
        branch_operands = (
            original_operand_val,
            indices_for_axis_0_val,
            reshaped_updates_for_slices_val,
        )

        # Branch functions now take a single tuple argument
        def true_branch_takes_tuple(operands_tuple):
            op, idx, upd = operands_tuple  # Unpack the tuple
            return op.at[idx].set(upd)

        def false_branch_takes_tuple(operands_tuple):
            op, idx, upd = operands_tuple  # Unpack the tuple
            # This branch should return a value of the same shape and type as the true_branch.
            return op + 1.0  # Example alternative operation

        # Call lax.cond with a single tuple operand for the branches
        scattered_result = lax.cond(
            predicate,
            true_branch_takes_tuple,
            false_branch_takes_tuple,
            branch_operands,  # Pass the packed tuple
        )

        some_int_value = jnp.array(42, dtype=jnp.int64)
        reshaped_int_value = jnp.reshape(some_int_value, ())

        return scattered_result, reshaped_int_value

    print("--- Defining JAX function for ONNX export ---")
    jax_output_scattered, jax_output_int = None, None
    try:
        jax_output_scattered, jax_output_int = minimal_problem_fn()
        print("JAX function executed successfully (pre-conversion).")
        print(
            f"  JAX scattered output dtype: {jax_output_scattered.dtype}, shape: {jax_output_scattered.shape}"
        )
        print(
            f"  JAX int output dtype: {jax_output_int.dtype}, shape: {jax_output_int.shape}"
        )
    except Exception as e:
        print(f"ERROR executing JAX function (pre-conversion): {e}")
        import traceback

        traceback.print_exc()
        return  # Stop if JAX part fails
    print("-" * 40)

    onnx_output_dir = "./onnx_debug_output"
    os.makedirs(onnx_output_dir, exist_ok=True)
    onnx_file_path_f64_minimal = os.path.join(
        onnx_output_dir, "minimal_problem_v6_f64.onnx"
    )

    print(
        f"--- Exporting JAX function to ONNX (enable_double_precision=True) to {onnx_file_path_f64_minimal} ---"
    )
    onnx_export_successful = False
    exported_model_proto = None
    try:
        exported_model_proto = to_onnx(
            minimal_problem_fn,
            inputs=[],  # Function closes over its data
            input_params={},
            model_name="minimal_problem_v6_f64",
            enable_double_precision=True,
            # opset_version=15 # Keep commented unless a specific version is needed and supported
            # record_primitive_calls_file="primitive_calls_minimal_v6_f64.log" # For debugging jax2onnx
        )

        with open(onnx_file_path_f64_minimal, "wb") as f:
            f.write(exported_model_proto.SerializeToString())
        print(
            f"Minimal ONNX model (float64) potentially saved to: {onnx_file_path_f64_minimal}"
        )
        onnx_export_successful = True

    except Exception as e:
        print(
            f"ERROR during minimal ONNX export with enable_double_precision=True: {e}"
        )
        print(
            "   ^^^^ If this is 'Missing value_info for: ['updates_reshaped_X',...]' or similar, ^^^^"
        )
        print(
            "   ^^^^ this indicates the core issue might now be reachable!                      ^^^^"
        )
        import traceback

        traceback.print_exc()  # Print full traceback for detailed debugging
        onnx_export_successful = False
    print("-" * 40)

    if onnx_export_successful and exported_model_proto:
        print(
            "Minimal ONNX export command executed without raising an exception during to_onnx."
        )
        print("Inspecting the ONNX model and attempting runtime validation...")
        try:
            # Load the model for inspection (using onnx library)
            m_f64 = onnx.load(onnx_file_path_f64_minimal)
            # onnx.checker.check_model(m_f64) # Optional: can be verbose or fail on valid models with custom ops

            print("ONNX Graph Inputs (from loaded f64 model):")
            if not m_f64.graph.input:
                print("  Model has no inputs (as expected).")
            else:
                for inp_idx, inp in enumerate(m_f64.graph.input):
                    tensor_type = inp.type.tensor_type
                    shape = [
                        d.dim_value if d.dim_value > 0 else d.dim_param
                        for d in tensor_type.shape.dim
                    ]
                    print(
                        f"  Input {inp_idx}: Name: {inp.name}, DType: {TensorProto.DataType.Name(tensor_type.elem_type)}, Shape: {shape}"
                    )

            print("ONNX Graph Outputs (from loaded f64 model):")
            all_outputs_correct_type = True
            # Assuming the order of outputs from JAX function matches ONNX model outputs
            expected_onnx_types = [
                TensorProto.DataType.DOUBLE,
                TensorProto.DataType.INT64,
            ]

            for i, outp in enumerate(m_f64.graph.output):
                tensor_type = outp.type.tensor_type
                shape = [
                    d.dim_value if d.dim_value > 0 else d.dim_param
                    for d in tensor_type.shape.dim
                ]
                actual_dtype_name = TensorProto.DataType.Name(tensor_type.elem_type)
                print(
                    f"  Output {i} - Name: {outp.name}, DType: {actual_dtype_name}, Shape: {shape}"
                )
                if (
                    i < len(expected_onnx_types)
                    and expected_onnx_types[i] != tensor_type.elem_type
                ):
                    all_outputs_correct_type = False
                    print(
                        f"    WARNING: Output {outp.name} has DType {actual_dtype_name}, expected {TensorProto.DataType.Name(expected_onnx_types[i])}"
                    )

            if all_outputs_correct_type:
                print("ONNX model output types appear as expected (DOUBLE and INT64).")
            else:
                print("ERROR: ONNX model output types mismatch expectations.")

            # Try to run with ONNX Runtime
            print("Attempting to run the exported ONNX model with onnxruntime...")
            ort_session = onnxruntime.InferenceSession(onnx_file_path_f64_minimal)
            ort_inputs = {}  # No inputs for this model as constants are baked in
            ort_outputs = ort_session.run(None, ort_inputs)
            print(
                f"ONNX Runtime execution successful. Number of outputs: {len(ort_outputs)}"
            )

            if jax_output_scattered is not None and jax_output_int is not None:
                jax_results_list = [
                    np.array(jax_output_scattered),
                    np.array(jax_output_int),
                ]
                for i, onnx_res_single in enumerate(ort_outputs):
                    print(
                        f"  ONNX Runtime output {i} shape: {onnx_res_single.shape}, dtype: {onnx_res_single.dtype}"
                    )
                    # Compare with JAX results
                    if (
                        np.array_equal(jax_results_list[i].shape, onnx_res_single.shape)
                        and jax_results_list[i].dtype == onnx_res_single.dtype
                    ):
                        if np.allclose(
                            jax_results_list[i], onnx_res_single, equal_nan=True
                        ):
                            print(f"    SUCCESS: Output {i} matches JAX result.")
                        else:
                            print(
                                f"    FAILURE: Output {i} differs from JAX result. Max diff: {np.max(np.abs(jax_results_list[i] - onnx_res_single))}"
                            )
                    else:
                        print(f"    FAILURE: Shape or DType mismatch for output {i}.")
                        print(
                            f"      JAX: shape={jax_results_list[i].shape}, dtype={jax_results_list[i].dtype}"
                        )
                        print(
                            f"      ONNX: shape={onnx_res_single.shape}, dtype={onnx_res_single.dtype}"
                        )
            else:
                print("Skipped JAX vs ONNX comparison as JAX pre-execution failed.")

        except Exception as e:
            print(f"Error during ONNX model inspection or runtime: {e}")
            print(
                "This could indicate the exported model is problematic (e.g., due to missing value_info internally or other export issues)."
            )
            import traceback

            traceback.print_exc()

    elif not exported_model_proto and onnx_export_successful:
        print(
            "WARN: to_onnx did not return a model proto but also did not raise an error during export."
        )
    else:
        print("Minimal ONNX export FAILED during to_onnx call or model proto was None.")

    print("Script finished.")


if __name__ == "__main__":
    run_minimal_reproducer_v6()
