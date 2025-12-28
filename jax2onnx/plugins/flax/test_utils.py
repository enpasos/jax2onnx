# jax2onnx/plugins/flax/test_utils.py

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx
from flax.nnx import bridge


class _LinenToNNXCallable:
    def __init__(self, model, rngs):
        self._model = model
        self._rngs = rngs

    def __call__(self, *args, **kwargs):
        return self._model(*args, rngs=self._rngs, **kwargs)


def linen_to_nnx(
    module_cls,
    input_shape=(1, 32),
    dtype=jnp.float32,
    rngs=None,
    **kwargs,
):
    """Wrap a Linen module as NNX and initialize it with a dummy input."""
    module = module_cls(**kwargs)
    model = bridge.ToNNX(module, rngs=None)
    dummy_x = jnp.zeros(input_shape, dtype=dtype)
    if isinstance(rngs, nnx.Rngs):
        # Avoid mutating NNX RNG state during JAX tracing by using a raw key.
        if "params" in rngs:
            rngs = rngs["params"].key.value
        elif "default" in rngs:
            rngs = rngs["default"].key.value
        else:
            raise ValueError("NNX Rngs must define a 'params' or 'default' stream.")
    if rngs is None:
        model.lazy_init(dummy_x)
        return model
    model.lazy_init(dummy_x, rngs=rngs)
    return _LinenToNNXCallable(model, rngs)
