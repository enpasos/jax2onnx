# tests/extra_tests/converter/test_duplicate_params.py

from __future__ import annotations

import jax.numpy as jnp
import pytest
from flax import nnx

from jax2onnx import onnx_function
from jax2onnx.user_interface import to_onnx


def _collect_duplicate_function_inputs(model) -> list[tuple[str, list[str]]]:
    duplicates: list[tuple[str, list[str]]] = []
    for function in getattr(model, "functions", []):
        seen: set[str] = set()
        dup_inputs: list[str] = []
        for input_name in function.input:
            if input_name in seen:
                dup_inputs.append(input_name)
            else:
                seen.add(input_name)
        if dup_inputs:
            duplicates.append((function.name, dup_inputs))
    return duplicates


@onnx_function
class _NestedBlock(nnx.Module):
    def __init__(self, num_hiddens, mlp_dim, dropout_rate=0.1, *, rngs: nnx.Rngs):
        self.layers = nnx.List(
            [
                nnx.Linear(num_hiddens, mlp_dim, rngs=rngs),
                lambda x: nnx.gelu(x, approximate=False),
                nnx.Dropout(rate=dropout_rate, rngs=rngs),
                nnx.Linear(mlp_dim, num_hiddens, rngs=rngs),
                nnx.Dropout(rate=dropout_rate, rngs=rngs),
            ]
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
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
        num_hiddens = 256
        self.layer_norm = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp = _NestedBlock(num_hiddens, mlp_dim=512, rngs=rngs)

    def __call__(self, x, deterministic: bool = True):
        return self.mlp(self.layer_norm(x), deterministic=deterministic)


@pytest.mark.parametrize("shape", [(5, 10, 256)])
def test_duplicate_parameters_ir(shape):
    super_block = _SuperBlock()
    model = to_onnx(
        super_block,
        inputs=[shape],
        model_name="duplicate_param_test_ir",
    )
    duplicates = _collect_duplicate_function_inputs(model)
    assert not duplicates, f"Functions contained duplicate inputs: {duplicates}"
