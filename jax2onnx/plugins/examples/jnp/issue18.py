import jax
import jax.numpy as jnp
import numpy as np
from jax2onnx.plugin_system import register_example

# --- Model Functions from issue18_test.py ---


def sign_fn(x):
    return jnp.sign(x)


def abs_fn(x):
    return jnp.abs(x)


def fori_loop_fn(x):
    def body(i, val):
        return val + i

    return jax.lax.fori_loop(0, 5, body, x)


def while_loop_fn(x):
    def cond(state):
        val, i = state
        return i < 5

    def body(state):
        val, i = state
        return val + i, i + 1

    final_val, _ = jax.lax.while_loop(cond, body, (x, 0))
    return final_val


def scan_fn(x):
    def body(carry, _):
        carry = carry + 1
        return carry, carry

    carry, ys = jax.lax.scan(body, x, None, length=5)
    return ys


def where_fn(x, y):
    return jnp.where(x > 0, x, y)


def arange_fn():
    return jnp.arange(5)


def linspace_fn():
    return jnp.linspace(0, 1, 5)


# --- Registering each function as an Example ---

register_example(
    component="issue18_sign",
    description="Test jnp.sign from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "sign_fn",
            "callable": sign_fn,
            "input_values": [np.array([-2.0, 0.0, 3.0], dtype=np.float32)],
        }
    ],
)

register_example(
    component="issue18_abs",
    description="Test jnp.abs from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "abs_fn",
            "callable": abs_fn,
            "input_values": [np.array([-2.0, 0.0, 3.0], dtype=np.float32)],
        }
    ],
)

register_example(
    component="issue18_fori_loop",
    description="Test fori_loop from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "fori_loop_fn",
            "callable": fori_loop_fn,
            "input_values": [np.array(0.0, dtype=np.float32)],
        }
    ],
)

register_example(
    component="issue18_while_loop",
    description="Test while_loop from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "while_loop_fn",
            "callable": while_loop_fn,
            "input_values": [np.array(0.0, dtype=np.float64)],
            "run_only_f64_variant": True,
        }
    ],
)

register_example(
    component="issue18_scan",
    description="Test scan from issue 18 (no xs)",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "scan_fn",
            "callable": scan_fn,
            "input_values": [np.array(0.0, dtype=np.float32)],
        }
    ],
)

register_example(
    component="issue18_where",
    description="Test where from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "where_fn",
            "callable": where_fn,
            "input_values": [
                np.array([-1.0, 1.0, 0.0], dtype=np.float32),
                np.array([10.0, 11.0, 12.0], dtype=np.float32),
            ],
        }
    ],
)

register_example(
    component="issue18_arange",
    description="Test arange from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[{"testcase": "arange_fn", "callable": arange_fn, "input_values": []}],
)

register_example(
    component="issue18_linspace",
    description="Test linspace from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {"testcase": "linspace_fn", "callable": linspace_fn, "input_values": []}
    ],
)