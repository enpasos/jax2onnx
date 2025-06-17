import jax
from jax2onnx import to_onnx


def simulate():
    """Simple scan operation for ONNX export"""

    def step_fn(carry, x):
        # Simple increment operation
        new_carry = carry + 1
        output = carry * 2
        return new_carry, output

    # Initial value and scan length
    init_carry = 0
    n_steps = 10

    # Run the scan
    final_carry, outputs = jax.lax.scan(step_fn, init_carry, xs=None, length=n_steps)

    return outputs


def main():
    """JIT-compiled entry point for ONNX export"""
    return jax.jit(simulate)()


if __name__ == "__main__":
    to_onnx(main, inputs=[], enable_double_precision=True)
