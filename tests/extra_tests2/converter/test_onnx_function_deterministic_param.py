from __future__ import annotations

import onnx
import pytest

import jax.numpy as jnp
from flax import nnx

from jax2onnx import onnx_function
from jax2onnx.user_interface import to_onnx


@onnx_function
class _MLPBlock(nnx.Module):
    def __init__(self, num_hiddens, mlp_dim, rngs: nnx.Rngs):
        self.layers = nnx.List(
            [
                nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
                lambda x: nnx.gelu(x, approximate=False),
                nnx.Dropout(rate=0.1, rngs=rngs),
                nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
                nnx.Dropout(rate=0.1, rngs=rngs),
            ]
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, nnx.Dropout):
                x = layer(x, deterministic=deterministic)
            else:
                x = layer(x)
        return x


@onnx_function
class _SuperBlock(nnx.Module):
    def __init__(self):
        rngs = nnx.Rngs(0)
        self.layer_norm = nnx.LayerNorm(3, rngs=rngs)
        self.mlp = _MLPBlock(num_hiddens=3, mlp_dim=6, rngs=rngs)

    def __call__(self, x, deterministic: bool = True):
        return self.mlp(self.layer_norm(x), deterministic=deterministic)


def test_onnx_function_deterministic_param_is_input_ir():
    model = to_onnx(
        _SuperBlock(),
        inputs=[(5, 10, 3)],
        input_params={"deterministic": True},
        model_name="test_deterministic_param_ir",
    )

    for init in model.graph.initializer:
        assert (
            init.name != "deterministic"
        ), "deterministic should not be serialized as initializer"

    found_graph_input = False
    for inp in model.graph.input:
        if inp.name == "deterministic":
            found_graph_input = True
            assert inp.type.tensor_type.elem_type == onnx.TensorProto.BOOL
    assert found_graph_input, "Graph input 'deterministic' must be present"

    if getattr(model, "functions", []):
        for function in model.functions:
            if "deterministic" not in function.input:
                continue
            for vi in function.value_info:
                if vi.name == "deterministic":
                    assert (
                        vi.type.tensor_type.elem_type == onnx.TensorProto.BOOL
                    ), "deterministic in function body should be BOOL"
                    break
            else:  # pragma: no cover - defensive
                pytest.fail("Function missing value_info for deterministic")
