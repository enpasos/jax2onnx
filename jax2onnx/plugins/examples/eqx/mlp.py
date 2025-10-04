# jax2onnx/plugins/examples/eqx/mlp.py

from __future__ import annotations

import equinox as eqx
import jax
import jax.random as jr
import numpy as np

from jax2onnx.plugins.plugin_system import register_example


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
        key: jax.Array,
    ):
        key_1, key_2 = jr.split(key, 2)
        self.linear1 = eqx.nn.Linear(in_features, hidden_features, key=key_1)
        self.dropout = eqx.nn.Dropout(p=0.2, inference=False)
        self.norm = eqx.nn.LayerNorm(hidden_features)
        self.linear2 = eqx.nn.Linear(hidden_features, out_features, key=key_2)

    def __call__(self, x: jax.Array, key: jax.Array | None = None) -> jax.Array:
        x = jax.nn.gelu(self.dropout(self.norm(self.linear1(x)), key=key))
        return self.linear2(x)


_model = Mlp(30, 20, 10, key=jax.random.PRNGKey(0))
_inference_model = eqx.nn.inference_mode(_model, value=True)
# Equinox modules operate on a single batch element; wrap with vmap for batched inputs.
_batched_model = jax.vmap(_model, in_axes=(0, None))


_ONNX_BOOL = 9  # TensorProto.DataType.BOOL without importing onnx


def _tensor_proto_first_bool(tensor_proto) -> bool | None:
    """Return the first bool element stored in ``tensor_proto`` if present."""

    if getattr(tensor_proto, "data_type", None) != _ONNX_BOOL:
        return None

    if getattr(tensor_proto, "raw_data", None):
        arr = np.frombuffer(tensor_proto.raw_data, dtype=np.bool_)
        return bool(arr[0]) if arr.size else None

    for field in ("int32_data", "int64_data", "uint64_data", "float_data"):
        data = getattr(tensor_proto, field, None)
        if data:
            return bool(np.array(data, dtype=np.bool_)[0])

    # Scalar bools can also surface via explicit ``bools`` attribute in some builds.
    data = getattr(tensor_proto, "bool_data", None)
    if data:
        return bool(data[0])

    return None


def _check_dropout_training_mode(model, expected_mode: bool) -> bool:
    try:
        dropout_node = next(n for n in model.graph.node if n.op_type == "Dropout")
        training_mode_input_name = dropout_node.input[2]
        if training_mode_input_name == "":
            # Missing optional input encodes inference (False).
            return expected_mode is False
        training_mode_init = next(
            i for i in model.graph.initializer if i.name == training_mode_input_name
        )
        value = _tensor_proto_first_bool(training_mode_init)
        return value is not None and value == expected_mode
    except StopIteration:
        return False


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
            "callable": (
                lambda x, *, model=_model, _k=jax.random.PRNGKey(0): model(x, _k)
            ),
            "input_shapes": [(30,)],
            "post_check_onnx_graph": lambda m: (
                _check_dropout_training_mode(m, expected_mode=True)
            ),
            "skip_numeric_validation": True,
        },
        {
            "testcase": "mlp_inference_mode",
            "callable": (lambda x, *, model=_inference_model: model(x, key=None)),
            "input_shapes": [(30,)],
            "post_check_onnx_graph": lambda m: (
                _check_dropout_training_mode(m, expected_mode=False)
            ),
        },
        {
            "testcase": "mlp_batched_training_mode",
            "callable": (
                lambda x, *, model=_batched_model, _k=jax.random.PRNGKey(0): model(
                    x, _k
                )
            ),
            "input_shapes": [(8, 30)],
            "skip_numeric_validation": True,
        },
    ],
)
