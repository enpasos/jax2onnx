# tests/extra_tests/test_random_bits_dtype.py

"""Random bits carry the unsigned type JAX requested for their width."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from jax2onnx import to_onnx

_KEY = jax.random.PRNGKey(0)


@pytest.mark.parametrize("dtype", [jnp.uint8, jnp.uint16, jnp.uint32])
def test_random_bits_have_the_requested_width(dtype: type) -> None:
    # RandomUniform has a kernel in ONNX Runtime up to opset 21.
    model = to_onnx(lambda: jax.random.bits(_KEY, (4000,), dtype=dtype), [], opset=21)

    onnx.checker.check_model(model, full_check=True)
    expected = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    assert model.graph.output[0].type.tensor_type.elem_type == expected
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (bits,) = session.run(None, {})
    assert bits.dtype == np.dtype(dtype)
    # The draw covers the upper half of the range, not just a narrower type.
    assert int(bits.max()) > np.iinfo(dtype).max // 2


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_uniform_is_valid_from_the_bitcast_opset(dtype: type) -> None:
    # jax.random.uniform builds floats from random bits of the float's width
    # (float16: uint16) and reinterprets them with BitCast, which needs opset 26.
    def uniform() -> jax.Array:
        return jax.random.uniform(_KEY, (8,), dtype=dtype)

    model = to_onnx(uniform, [], opset=26)

    onnx.checker.check_model(model, full_check=True)
    expected = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    assert model.graph.output[0].type.tensor_type.elem_type == expected

    with pytest.raises(NotImplementedError, match="BitCast.*opset 26"):
        to_onnx(uniform, [], opset=25)
