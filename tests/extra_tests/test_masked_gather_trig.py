import jax
import numpy as np
import pytest

# Ensure the main `to_onnx` function is imported
from jax2onnx import to_onnx
import jax.numpy as jnp
from jax import jit


# Match the repro: run with double precision enabled
jax.config.update("jax_enable_x64", True)


@jit
def masked_gather_trig(data, indices):
    """
    JIT-compiled pipeline:
      - force data to float64
      - gather by indices
      - simple trig math
      - mask + where
    """
    data = jnp.asarray(data, dtype=jnp.float64)
    gathered = data[indices]

    result = gathered * 2.0
    result = jnp.sin(result) + jnp.cos(result)

    mask = result > 0.5
    filtered_result = jnp.where(mask, result, 0.0)
    return filtered_result


class TestMaskedGatherTrig:
    def test_masked_gather_trig_f64_pipeline(self):
        """
        Standard testcase following the output-count & ORT-validation pattern.
        Ensures export under enable_double_precision=True produces the correct
        single output and numerically matches JAX.
        """
        # 1) Inputs (keep deterministic). Use int64 indices to match common ONNX Gather.
        data = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=np.float64,
        )
        indices = np.array([0, 2], dtype=np.int64)

        input_specs = [
            jax.ShapeDtypeStruct(data.shape, data.dtype),
            jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        ]

        # 2) Execute the original JAX function to determine expected outputs.
        jax_outputs = jax.tree_util.tree_leaves(masked_gather_trig(data, indices))
        expected_num_outputs = len(jax_outputs)

        print(f"JAX function produced {expected_num_outputs} output leaves.")
        assert (
            expected_num_outputs == 1
        ), "Test setup is incorrect; expected 1 JAX output."

        # DEBUG: Print the jaxpr to see what JAX is producing
        print("--- DEBUGGING JAXPR ---")
        jaxpr = jax.make_jaxpr(masked_gather_trig)(data, indices)
        print("Jaxpr object:", jaxpr)
        print("Jaxpr out_avals:", jaxpr.out_avals)
        print(f"Number of Jaxpr out_avals: {len(jaxpr.out_avals)}")
        print("--- END DEBUGGING ---")

        # 3) Convert the JAX function to an ONNX model.
        try:
            onnx_model = to_onnx(
                masked_gather_trig,
                input_specs,
                model_name="masked_gather_trig_f64",
                enable_double_precision=True,
            )
        except Exception as e:
            pytest.fail(f"ONNX conversion failed unexpectedly: {e}")

        # 4) CORE ASSERTION: ONNX output count matches JAX output leaves.
        num_onnx_outputs = len(onnx_model.graph.output)
        print(f"ONNX model has {num_onnx_outputs} outputs.")
        assert (
            num_onnx_outputs == expected_num_outputs
        ), f"Output count mismatch! JAX: {expected_num_outputs}, ONNX: {num_onnx_outputs}."

        # 5) Validate inference numerics with ONNX Runtime (if available).
        print("Output count is correct. Proceeding to validate ONNX Runtime.")
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(onnx_model.SerializeToString())

            # Dynamically get input names from the model
            model_input_names = [inp.name for inp in sess.get_inputs()]
            input_feed = dict(zip(model_input_names, [data, indices]))

            print(
                f"Running inference with dynamic input feed: {list(input_feed.keys())}"
            )
            onnx_results = sess.run(None, input_feed)

            assert len(onnx_results) == expected_num_outputs
            # NOTE: ONNX Runtime and JAX/XLA may use different libm/evaluation orders
            # for trig functions. In double precision we observe ~2.6e-08 relative
            # differences on CPU. Use a slightly looser tolerance here.
            np.testing.assert_allclose(
                jax_outputs[0],
                onnx_results[0],
                rtol=5e-8,
                atol=1e-12,
            )

        except ImportError:
            print("Skipping ONNX Runtime validation because it is not installed.")
        except Exception as e:
            pytest.fail(f"ONNX Runtime validation failed: {e}")
