import jax
import numpy as np
import pytest

# Ensure the main `to_onnx` function is imported
from jax2onnx import to_onnx


# This is the function that will be nested inside the main graph via pjit.
# A buggy converter will leak its two outputs ('sub_res1', 'sub_res2')
# into the final ONNX graph's output list.
@jax.jit
def leaky_sub_function(x, y):
    """A simple sub-function that will be compiled and nested."""
    return x * y, x + y


def leaky_main_function(input_a, input_b, passthrough_c):
    """
    Main function that calls the jax.jit-compiled subgraph.

    A correct conversion should yield an ONNX model with exactly 2 outputs.
    A buggy conversion will yield 4 outputs (the 2 main ones + the 2 from leaky_sub_function).
    """
    sub_res1, _ = leaky_sub_function(input_a, input_b)  # We ignore the second result
    return sub_res1, passthrough_c


class TestSubgraphOutputLeaking:
    def test_pjit_output_leaking_is_accurately_represented(self):
        """
        This test should fail on a buggy converter with an output count mismatch.
        """
        # 1. Define input data and specs for the conversion.
        input_a = np.array([1.0, 2.0], dtype=np.float32)
        input_b = np.array([3.0, 4.0], dtype=np.float32)
        input_c = np.array([100.0], dtype=np.float32)

        input_specs = [
            jax.ShapeDtypeStruct(input_a.shape, input_a.dtype),
            jax.ShapeDtypeStruct(input_b.shape, input_b.dtype),
            jax.ShapeDtypeStruct(input_c.shape, input_c.dtype),
        ]

        # 2. Execute the original JAX function to get the expected number of outputs.
        jax_outputs = jax.tree_util.tree_leaves(
            leaky_main_function(input_a, input_b, input_c)
        )
        expected_num_outputs = len(jax_outputs)

        print(f"JAX function produced {expected_num_outputs} output leaves.")
        assert (
            expected_num_outputs == 2
        ), "Test setup is incorrect; expected 2 JAX outputs."

        # DEBUG: Print the jaxpr to see what JAX is producing
        print("--- DEBUGGING JAXPR ---")
        jaxpr = jax.make_jaxpr(leaky_main_function)(input_a, input_b, input_c)
        print("Jaxpr object:", jaxpr)
        print("Jaxpr out_avals:", jaxpr.out_avals)
        print(f"Number of Jaxpr out_avals: {len(jaxpr.out_avals)}")
        print("--- END DEBUGGING ---")

        # 3. Convert the JAX function to an ONNX model.
        try:
            # FIX: Removed the unsupported `input_names` argument.
            onnx_model = to_onnx(
                leaky_main_function,
                input_specs,
                model_name="pjit_leak_test",
            )
        except Exception as e:
            pytest.fail(f"ONNX conversion failed unexpectedly: {e}")

        # 4. THE CORE ASSERTION:
        # This is where the test should now fail with your original code.
        num_onnx_outputs = len(onnx_model.graph.output)
        print(f"ONNX model has {num_onnx_outputs} outputs.")

        assert num_onnx_outputs == expected_num_outputs, (
            f"Output count mismatch! JAX: {expected_num_outputs}, ONNX: {num_onnx_outputs}. "
            "This indicates that subgraph outputs are leaking into the main graph."
        )

        # 5. If the assertion passes, validate inference for correctness.
        print("Output count is correct. Proceeding to validate ONNX Runtime.")
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(onnx_model.SerializeToString())

            # FIX: Dynamically get input names from the model.
            model_input_names = [inp.name for inp in sess.get_inputs()]
            input_feed = dict(zip(model_input_names, [input_a, input_b, input_c]))

            print(
                f"Running inference with dynamic input feed: {list(input_feed.keys())}"
            )

            onnx_results = sess.run(None, input_feed)

            assert len(onnx_results) == expected_num_outputs
            np.testing.assert_allclose(jax_outputs[0], onnx_results[0], rtol=1e-6)
            np.testing.assert_allclose(jax_outputs[1], onnx_results[1], rtol=1e-6)

        except ImportError:
            print("Skipping ONNX Runtime validation because it is not installed.")
        except Exception as e:
            pytest.fail(f"ONNX Runtime validation failed: {e}")
