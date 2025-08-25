# file: tests/extra_tests/test_lax_cond_repro.py

import unittest
import sys
import subprocess
import os
import tempfile

# The user's failing code is now a string constant. This avoids using the sandbox
# and makes the test completely self-contained.
REPRODUCER_SCRIPT_CODE = """
import jax
import jax.numpy as jnp
from jax import lax
import jax2onnx
import onnxruntime

# This is the function that triggers the bug
def model_with_cond_and_scatter():
    limit_val = float(2 * 4 * 1 * 1)
    original_operand_val = jnp.arange(start=0.0, stop=limit_val, step=1.0, dtype=jnp.float64).reshape((2, 4, 1, 1))
    raw_updates_data_val = (jnp.ones((1, 4, 1, 1), dtype=jnp.float64) * 100.0)
    reshaped_updates_for_slices_val = jnp.reshape(raw_updates_data_val, (1, 4, 1, 1))
    indices_for_axis_0_val = jnp.array([1]) 
    predicate = jnp.array(True)
    branch_operands = (original_operand_val, indices_for_axis_0_val, reshaped_updates_for_slices_val)

    def true_branch_takes_tuple(operands_tuple):
        op, idx, upd = operands_tuple
        return op.at[idx].set(upd)

    def false_branch_takes_tuple(operands_tuple):
        op, _, _ = operands_tuple
        return op + 1.0

    scattered_result = lax.cond(predicate, true_branch_takes_tuple, false_branch_takes_tuple, branch_operands)
    some_int_value = jnp.array(42, dtype=jnp.int64)
    reshaped_int_value = jnp.reshape(some_int_value, ())
    return scattered_result, reshaped_int_value

# Main execution logic
def main():
    jax.config.update("jax_enable_x64", True)
    
    model_proto = jax2onnx.to_onnx(
        model_with_cond_and_scatter,
        inputs=[],
        model_name="cond_scatter_repro",
        enable_double_precision=True
    )
    
    # This is the line that fails with InvalidGraph before the fix
    onnxruntime.InferenceSession(model_proto.SerializeToString())
    
    print("SUCCESS") # Add a success marker

if __name__ == "__main__":
    main()
"""


class TestLaxCondReproducer(unittest.TestCase):

    def test_reproducer_succeeds_after_fix(self):
        """
        This test asserts that the reproducer script runs SUCCESSFULLY.

        BEFORE the fix in cond.py is applied, this test will FAIL
        because the subprocess crashes with an InvalidGraph error.

        AFTER the fix is applied, this test will PASS.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_script:
            script_path = temp_script.name
            temp_script.write(REPRODUCER_SCRIPT_CODE)

        try:
            python_executable = sys.executable
            # We expect the process to succeed.
            # check=True will raise CalledProcessError if the script fails.
            subprocess.run(
                [python_executable, script_path],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            # If the script fails, we fail the test and print the script's error output.
            # This is what we expect to happen before the fix is applied.
            self.fail(
                f"The reproducer script failed unexpectedly."
                f"\\n--- STDERR ---\\n{e.stderr}"
            )
        finally:
            # Clean up the temporary file
            os.remove(script_path)


if __name__ == "__main__":
    unittest.main()
