# tests/extra_tests/test_gather_modes.py

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import lax

from jax2onnx.user_interface import to_onnx


def _gather_one_hot(x: jnp.ndarray) -> jnp.ndarray:
    indices = jnp.array([[0]], dtype=jnp.int32)
    return lax.gather(
        x,
        indices,
        dimension_numbers=lax.GatherDimensionNumbers(
            offset_dims=(1, 2),
            collapsed_slice_dims=(0,),
            start_index_map=(0,),
        ),
        slice_sizes=(1, x.shape[1], x.shape[2]),
        mode=lax.GatherScatterMode.ONE_HOT,
    )[0]


def test_gather_clip_mode_raises_not_implemented():
    x = np.ones((4, 5, 6), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        to_onnx(
            _gather_one_hot,
            [jax.ShapeDtypeStruct(x.shape, x.dtype)],
            model_name="gather_clip",
            enable_double_precision=False,
        )
