# file: jax2onnx/sandbox/issue18_test.py

import jax
import jax.numpy as jnp
from jax2onnx import to_onnx
import os


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


def multidim_fn(x):
    return jnp.ones((1, 1, 1, 1), dtype=jnp.float32) * x


def export_all():
    dummy_1d = jnp.ones((3,), dtype=jnp.float32)
    dummy_1d_b = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)
    dummy_scalar = jnp.array(0.0, dtype=jnp.float32)

    os.makedirs("onnx_models", exist_ok=True)

    export_map = {
        "sign": (
            sign_fn,
            [dummy_1d],
        ),  # NotImplementedError: No ONNX handler registered for JAX primitive: 'sign'
        "abs": (
            abs_fn,
            [dummy_1d],
        ),  # NotImplementedError: No ONNX handler registered for JAX primitive: 'abs'
        "fori_loop": (
            fori_loop_fn,
            [dummy_scalar],
        ),  # NotImplementedError: No ONNX handler registered for JAX primitive: 'scan'
        "while_loop": (
            while_loop_fn,
            [dummy_scalar],
        ),  # NotImplementedError: No ONNX handler registered for JAX primitive: 'while'
        "scan": (
            scan_fn,
            [dummy_scalar],
        ),  # NotImplementedError: No ONNX handler registered for JAX primitive: 'scan'
        "where": (
            where_fn,
            [dummy_1d, dummy_1d_b],
        ),  # NotImplementedError: No ONNX handler registered for JAX primitive: 'pjit'
        "arange": (
            arange_fn,
            [],
        ),  # NotImplementedError: No ONNX handler registered for JAX primitive: 'iota'
        "linspace": (
            linspace_fn,
            [],
        ),  # NotImplementedError: No ONNX handler registered for JAX primitive: 'pjit'
    }

    for name, (fn, inputs) in export_map.items():
        print(f"Exporting {name}.onnx...")
        onnx_model = to_onnx(fn, inputs=inputs)
        with open(f"onnx_models/{name}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())


if __name__ == "__main__":
    export_all()
