# jax2onnx/plugins/examples/eqx/mlp.py

from __future__ import annotations

import equinox as eqx
import jax
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_prng_key,
)


class Mlp(eqx.Module):
    linear1: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    norm: eqx.nn.LayerNorm
    linear2: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        linear1_key: jax.Array,
        linear2_key: jax.Array,
    ):
        self.linear1 = eqx.nn.Linear(in_features, hidden_features, key=linear1_key)
        self.dropout = eqx.nn.Dropout(p=0.2, inference=False)
        self.norm = eqx.nn.LayerNorm(hidden_features)
        self.linear2 = eqx.nn.Linear(hidden_features, out_features, key=linear2_key)

    def __call__(self, x: jax.Array, key: jax.Array | None = None) -> jax.Array:
        x = jax.nn.gelu(self.dropout(self.norm(self.linear1(x)), key=key))
        return self.linear2(x)


def _build_model(
    linear1_key: jax.Array,
    linear2_key: jax.Array,
    *,
    inference: bool,
) -> Mlp:
    model = Mlp(
        30,
        20,
        10,
        linear1_key=linear1_key,
        linear2_key=linear2_key,
    )
    return eqx.nn.inference_mode(model, value=True) if inference else model


def _training_ctor(*, batched: bool):
    def ctor(
        linear1_key: jax.Array,
        linear2_key: jax.Array,
        dropout_key: jax.Array,
    ):
        model = _build_model(linear1_key, linear2_key, inference=False)
        if batched:
            mapped = jax.vmap(model, in_axes=(0, None))

            def _batched_run(x: jax.Array) -> jax.Array:
                return mapped(x, dropout_key)

            return _batched_run

        def _single_run(x: jax.Array) -> jax.Array:
            return model(x, dropout_key)

        return _single_run

    return ctor


def _inference_ctor(*, batched: bool):
    def ctor(linear1_key: jax.Array, linear2_key: jax.Array):
        model = _build_model(linear1_key, linear2_key, inference=True)
        if batched:
            mapped = jax.vmap(model, in_axes=(0, None))

            def _batched_run(x: jax.Array) -> jax.Array:
                return mapped(x, key=None)

            return _batched_run

        def _single_run(x: jax.Array) -> jax.Array:
            return model(x, key=None)

        return _single_run

    return ctor


register_example(
    component="MlpExample",
    description="A simple Equinox MLP (converter pipeline).",
    source="https://github.com/patrick-kidger/equinox",
    since="v0.8.0",
    context="examples.eqx",
    children=["eqx.nn.Linear", "eqx.nn.Dropout", "jax.nn.gelu"],
    testcases=[
        {
            "testcase": "mlp_training_mode",
            "callable": construct_and_call(
                _training_ctor(batched=False),
                linear1_key=with_prng_key(0),
                linear2_key=with_prng_key(1),
                dropout_key=with_prng_key(2),
            ),
            "input_shapes": [(30,)],
            "post_check_onnx_graph": expect_graph(
                [
                    (
                        "Reshape:1x30 -> Gemm:1x20 -> Reshape:20 -> LayerNormalization:20 -> Dropout:20 -> Gelu:20 -> Reshape:1x20 -> Gemm:1x10 -> Reshape:10",
                        {"counts": {"Reshape": 4}},
                    ),
                    (
                        "Dropout:20",
                        {
                            "inputs": {
                                1: {"const": 0.2},
                                2: {"const_bool": True},
                            }
                        },
                    ),
                ],
                no_unused_inputs=True,
            ),
            "skip_numeric_validation": True,
        },
        {
            "testcase": "mlp_inference_mode",
            "callable": construct_and_call(
                _inference_ctor(batched=False),
                linear1_key=with_prng_key(3),
                linear2_key=with_prng_key(4),
            ),
            "input_shapes": [(30,)],
            "post_check_onnx_graph": expect_graph(
                [
                    (
                        "Reshape:1x30 -> Gemm:1x20 -> Reshape:20 -> LayerNormalization:20 -> Dropout:20 -> Gelu:20 -> Reshape:1x20 -> Gemm:1x10 -> Reshape:10",
                        {"counts": {"Reshape": 4}},
                    ),
                    (
                        "Dropout:20",
                        {
                            "inputs": {
                                1: {"const": 0.2},
                                2: {"const_bool": False},
                            }
                        },
                    ),
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "mlp_batched_training_mode",
            "callable": construct_and_call(
                _training_ctor(batched=True),
                linear1_key=with_prng_key(5),
                linear2_key=with_prng_key(6),
                dropout_key=with_prng_key(7),
            ),
            "input_shapes": [(8, 30)],
            "post_check_onnx_graph": expect_graph(
                [
                    (
                        "Gemm:8x20 -> LayerNormalization:8x20 -> Dropout:8x20 -> Gelu:8x20 -> Gemm:8x10",
                        {"counts": {"Reshape": 0}},
                    ),
                    (
                        "Dropout:8x20",
                        {
                            "inputs": {
                                1: {"const": 0.2},
                                2: {"const_bool": True},
                            }
                        },
                    ),
                ],
                no_unused_inputs=True,
            ),
            "skip_numeric_validation": True,
        },
    ],
)
