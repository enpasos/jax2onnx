# jax2onnx/plugins/examples/linen/mlp_sequential.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx

from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.flax.test_utils import expect_mlp_precision, linen_to_nnx
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_requested_dtype,
    with_rng_seed,
)


class LinenMLPSequential(nn.Module):
    din: int
    dmid: int
    dout: int
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Sequential(
            [
                nn.Dense(self.dmid, dtype=self.dtype, param_dtype=self.dtype),
                nn.BatchNorm(
                    use_running_average=True,
                    dtype=self.dtype,
                    param_dtype=self.dtype,
                ),
                nn.Dropout(rate=self.dropout_rate, deterministic=True),
                nn.gelu,
                nn.Dense(self.dout, dtype=self.dtype, param_dtype=self.dtype),
            ]
        )(x)


def _prepared_mlp_sequential(
    *, din: int, dmid: int, dout: int, dtype: Any, rngs: nnx.Rngs
) -> Any:
    # Initialize the same FP32 weights in both variants. Changing initialization
    # precision would produce different random weights even with the same seed.
    with jax.enable_x64(False):
        model = linen_to_nnx(
            module_cls=LinenMLPSequential,
            input_shape=(1, din),
            din=din,
            dmid=dmid,
            dout=dout,
            dtype=jnp.float32,
            rngs=rngs,
        )
    if jnp.dtype(dtype) == jnp.dtype(jnp.float64):
        # Configure the actual Linen module, then explicitly widen its existing
        # state. linen_to_nnx's dtype argument controls only its dummy input.
        wrapped = model._model
        wrapped.to_nnx__module = wrapped.to_nnx__module.clone(dtype=dtype)
        state = jax.tree.map(
            lambda value: (
                value.astype(dtype) if value.dtype == jnp.float32 else value
            ),
            nnx.state(wrapped),
        )
        nnx.update(wrapped, state)
    return model


def _mlp_graph_check(batch: str) -> Callable[[Any], bool]:
    common = {"Gemm": 2, "Dropout": 1, "Gelu": 1}
    return expect_mlp_precision(
        expect_graph(
            [
                (
                    f"Gemm:{batch}x20 -> BatchNormalization:{batch}x20 -> "
                    f"Dropout:{batch}x20 -> Gelu:{batch}x20 -> Gemm:{batch}x10",
                    {"counts": {**common, "BatchNormalization": 1}},
                )
            ],
            symbols={"B": None},
            must_absent=["Not"],
            no_unused_inputs=True,
        ),
        expect_graph(
            [
                (
                    f"Gemm:{batch}x20 -> Sub:{batch}x20 -> Mul:{batch}x20 -> "
                    f"Add:{batch}x20 -> Dropout:{batch}x20 -> Gelu:{batch}x20 -> "
                    f"Gemm:{batch}x10",
                    {
                        "counts": {
                            **common,
                            "BatchNormalization": 0,
                            "Sub": 1,
                            "Add": 2,
                            "Sqrt": 1,
                            "Div": 1,
                            "Mul": 2,
                        }
                    },
                ),
                # Source BatchNorm forms rsqrt(var + epsilon) * scale before
                # multiplying the centered input and adding its bias.
                f"Add:1x20 -> Sqrt:1x20 -> Div:1x20 -> Mul:1x20 -> Mul:{batch}x20",
            ],
            symbols={"B": None},
            must_absent=["Not"],
            no_unused_inputs=True,
        ),
    )


register_example(
    component="LinenMLPSequential",
    description="A Linen MLP built from flax.linen.Sequential.",
    source=(
        "https://flax-linen.readthedocs.io/en/latest/api_reference/"
        "flax.linen/layers.html#flax.linen.Sequential"
    ),
    since="0.11.0",
    context="examples.linen",
    children=[
        "flax.linen.Sequential",
        "flax.linen.Dense",
        "flax.linen.BatchNorm",
        "flax.linen.Dropout",
        "flax.linen.gelu",
    ],
    testcases=[
        {
            "testcase": "simple_linen_mlp_sequential_static",
            "callable": construct_and_call(
                _prepared_mlp_sequential,
                din=30,
                dmid=20,
                dout=10,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [(7, 30)],
            "post_check_onnx_graph": _mlp_graph_check("7"),
        },
        {
            "testcase": "simple_linen_mlp_sequential",
            "callable": construct_and_call(
                _prepared_mlp_sequential,
                din=30,
                dmid=20,
                dout=10,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(17),
            ),
            "input_shapes": [("B", 30)],
            "run_only_dynamic": True,
            "post_check_onnx_graph": _mlp_graph_check("B"),
        },
    ],
)
