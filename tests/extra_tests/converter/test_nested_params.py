# tests/extra_tests/converter/test_nested_params.py

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx import onnx_function
from jax2onnx.user_interface import to_onnx
from jax2onnx.utils.parameter_validation import validate_onnx_model_parameters


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
def test_parameter_passing_ir(tmp_path, shape):
    model = to_onnx(
        _SuperBlock(),
        inputs=[shape],
        model_name="nested_param_test_ir",
    )

    errors = validate_onnx_model_parameters(model)
    assert not errors, f"Parameter validation returned errors: {errors}"

    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover
        pytest.skip("onnxruntime not available")

    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, shape)
    jax_result = _SuperBlock()(x, deterministic=True)

    model_path = tmp_path / "nested_param_test_ir.onnx"
    model_path.write_bytes(model.SerializeToString())

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_result = session.run(None, {input_name: np.asarray(x, dtype=np.float32)})[0]

    assert jnp.allclose(jax_result, onnx_result, rtol=1e-5, atol=1e-5)
