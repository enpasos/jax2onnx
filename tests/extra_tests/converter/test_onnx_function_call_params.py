# tests/extra_tests/converter/test_onnx_function_call_params.py

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    with_rng_seed,
)
from jax2onnx.user_interface import to_onnx


@onnx_function
class FunctionDropout(nnx.Module):
    def __init__(self, rate: float, *, rngs: nnx.Rngs):
        super().__init__()
        self.dropout = nnx.Dropout(rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        return self.dropout(x, deterministic=deterministic)


def test_onnx_function_consumes_call_time_param():
    fn = construct_and_call(
        FunctionDropout,
        rate=0.25,
        rngs=with_rng_seed(0),
    )

    model = to_onnx(
        fn=fn,
        inputs=[jax.ShapeDtypeStruct((2, 3), jnp.float32)],
        input_params={"deterministic": True},
        model_name="function_dropout",
        opset=21,
        enable_double_precision=False,
    )

    check = EG(
        ["Dropout"],
        search_functions=True,
        no_unused_inputs=True,
        no_unused_function_inputs=True,
    )
    assert check(model)
