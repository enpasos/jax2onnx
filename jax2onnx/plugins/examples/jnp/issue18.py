# jax2onnx/plugins/examples/jnp/issue18.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


def _sign_fn(x: jax.Array) -> jax.Array:
    return jnp.sign(x)


def _abs_fn(x: jax.Array) -> jax.Array:
    return jnp.abs(x)


def _fori_loop_fn(x: jax.Array) -> jax.Array:
    def body(i: int, val: jax.Array) -> jax.Array:
        return val + i

    return jax.lax.fori_loop(0, 5, body, x)


def _while_loop_fn(x: jax.Array) -> jax.Array:
    def cond(state: tuple[jax.Array, int]) -> bool:
        _, i = state
        return i < 5

    def body(state: tuple[jax.Array, int]) -> tuple[jax.Array, int]:
        val, i = state
        return val + i, i + 1

    final_val, _ = jax.lax.while_loop(cond, body, (x, 0))
    return final_val


def _scan_fn(x: jax.Array) -> jax.Array:
    def body(carry: jax.Array, _) -> tuple[jax.Array, jax.Array]:
        carry = carry + 1
        return carry, carry

    _, ys = jax.lax.scan(body, x, None, length=5)
    return ys


def _where_fn(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.where(x > 0, x, y)


def _arange_fn() -> jax.Array:
    return jnp.arange(5, dtype=jnp.int32)


def _linspace_fn() -> jax.Array:
    return jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32)


register_example(
    component="issue18_sign",
    description="Test jnp.sign from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "sign_fn",
            "callable": _sign_fn,
            "input_values": [np.array([-2.0, 0.0, 3.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Sign:3"],
                no_unused_inputs=True,
            ),
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
            "callable": _abs_fn,
            "input_values": [np.array([-2.0, 0.0, 3.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Abs:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)

register_example(
    component="issue18_fori_loop",
    description="Test jax.lax.fori_loop from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "fori_loop_fn",
            "callable": _fori_loop_fn,
            "input_values": [np.array(0.0, dtype=np.float32)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 5.0},
                            1: {"const_bool": True},
                        },
                        "path": "Loop",
                    }
                ],
                search_functions=True,
                no_unused_inputs=True,
            ),
        }
    ],
)

register_example(
    component="issue18_while_loop",
    description="Test jax.lax.while_loop from issue 18",
    since="v0.9.0",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "while_loop_fn",
            "callable": _while_loop_fn,
            "input_values": [np.array(0.0, dtype=np.float64)],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 9.223372036854776e18},
                            3: {"const": 0.0},
                        },
                        "path": "Less -> Loop",
                    }
                ],
                search_functions=True,
                no_unused_inputs=True,
            ),
        }
    ],
)

register_example(
    component="issue18_scan",
    description="Test jax.lax.scan from issue 18 (no xs)",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "scan_fn",
            "callable": _scan_fn,
            "input_values": [np.array(0.0, dtype=np.float32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Loop"],
                search_functions=True,
                no_unused_inputs=True,
            ),
        }
    ],
)

register_example(
    component="issue18_where",
    description="Test jnp.where from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "where_fn",
            "callable": _where_fn,
            "input_values": [
                np.array([-1.0, 1.0, 0.0], dtype=np.float32),
                np.array([10.0, 11.0, 12.0], dtype=np.float32),
            ],
            "post_check_onnx_graph": EG(
                ["Greater:3 -> Where:3"],
                no_unused_inputs=True,
            ),
        }
    ],
)

register_example(
    component="issue18_arange",
    description="Test jnp.arange from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "arange_fn",
            "callable": _arange_fn,
            "input_values": [],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 0.0},
                            1: {"const": 5.0},
                            2: {"const": 1.0},
                        },
                        "path": "Range:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        }
    ],
)

register_example(
    component="issue18_linspace",
    description="Test jnp.linspace from issue 18",
    since="v0.6.3",
    context="examples.jnp",
    testcases=[
        {
            "testcase": "linspace_fn",
            "callable": _linspace_fn,
            "input_values": [],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "Range:5 -> Cast:5 -> Mul:5 -> Add:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        }
    ],
)
