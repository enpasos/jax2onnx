# tests/extra_tests/examples/test_io_names_examples.py

from __future__ import annotations

import jax.numpy as jnp

from jax2onnx.plugins.examples.jnp.select import _select
from jax2onnx.plugins.examples.lax.two_times_silu import two_times_silu
from jax2onnx.user_interface import to_onnx


def _select_example_callable(x_input):
    return _select(
        x_input,
        jnp.array(2.0, dtype=jnp.float32),
        jnp.array(0.5, dtype=jnp.float32),
        jnp.array([0, 1, 2], dtype=jnp.int32),
    )


def test_custom_io_names_with_jnp_example_callable():
    model = to_onnx(
        _select_example_callable,
        inputs=[(3,)],
        input_names=["signal"],
        output_names=["selected_signal"],
    )

    assert [value.name for value in model.graph.input] == ["signal"]
    assert [value.name for value in model.graph.output] == ["selected_signal"]


def test_custom_io_names_with_lax_example_callable():
    model = to_onnx(
        two_times_silu,
        inputs=[(4,)],
        input_names=["x_in"],
        output_names=["y_out"],
    )

    assert [value.name for value in model.graph.input] == ["x_in"]
    assert [value.name for value in model.graph.output] == ["y_out"]
