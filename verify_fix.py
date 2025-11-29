# verify_fix.py

import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
import onnxruntime as ort

# Force usage of local jax2onnx
sys.path.insert(0, os.getcwd())

from jax2onnx import to_onnx

jax.config.update("jax_enable_x64", True)


def test_scatter_add_in_cond_float64():
    print("Testing scatter_add_in_cond_float64 with local jax2onnx...")

    # Define the function causing issues
    def true_branch(x):
        return x

    def false_branch(x):
        operand = jnp.zeros((3,), dtype=jnp.float64)
        indices = jnp.array([[1]], dtype=jnp.int64)
        updates = jnp.array([1.0], dtype=jnp.float64)
        return jax.lax.scatter_add(
            operand,
            indices,
            updates,
            jax.lax.ScatterDimensionNumbers(
                update_window_dims=(),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            ),
        )

    def f(pred, x):
        return jax.lax.cond(pred, true_branch, false_branch, x)

    # Inputs
    pred = jnp.array(False)
    x = jnp.zeros((3,), dtype=jnp.float64)

    # Convert to ONNX
    print("Converting to ONNX...")
    onnx_model = to_onnx(f, [pred, x], enable_double_precision=True)

    # Run with ONNX Runtime
    print("Running with ONNX Runtime...")
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    inputs = {
        sess.get_inputs()[0].name: np.array(False, dtype=bool),
        sess.get_inputs()[1].name: np.zeros((3,), dtype=np.float64),
    }
    outputs = sess.run(None, inputs)
    print("Success! Output:", outputs)


if __name__ == "__main__":
    try:
        test_scatter_add_in_cond_float64()
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback

        traceback.print_exc()
