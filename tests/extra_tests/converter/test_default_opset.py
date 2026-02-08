# tests/extra_tests/converter/test_default_opset.py

import jax.numpy as jnp

from jax2onnx.user_interface import to_onnx


def test_to_onnx_uses_opset_23_by_default():
    def fn(x):
        return jnp.sin(x)

    model = to_onnx(fn, [(2,)], return_mode="proto")

    imports = {(entry.domain, entry.version) for entry in model.opset_import}
    assert ("", 23) in imports
