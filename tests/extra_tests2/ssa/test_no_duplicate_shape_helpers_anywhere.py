from __future__ import annotations

import collections
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import onnx
from jax import lax

from jax2onnx.user_interface import to_onnx


def _all_node_outputs_recursive(g):
    outs = []
    for n in g.node:
        outs.extend([o for o in n.output if o])
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH and a.g is not None:
                outs.extend(_all_node_outputs_recursive(a.g))
            elif a.type == onnx.AttributeProto.GRAPHS and a.graphs:
                for sg in a.graphs:
                    outs.extend(_all_node_outputs_recursive(sg))
    return outs


def _loopy_two_adds(y):
    dt = y.dtype
    one = jnp.array(1.0, dt)
    two = jnp.array(2.0, dt)

    def step(c, _):
        h1 = c + jnp.broadcast_to(one, c.shape)
        r1 = jnp.reshape(h1, h1.shape)
        h2 = (c * 1) + jnp.broadcast_to(two, c.shape)
        r2 = jnp.reshape(h2, h2.shape)
        out = r1 + r2 + (h1 * 0) + (h2 * 0)
        return c + jnp.array(0.0, dt), out

    _, ys = lax.scan(step, y, xs=None, length=2)
    return ys


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_model_has_no_duplicate_shape_helpers(dtype):
    ort = pytest.importorskip("onnxruntime")

    spec = jax.ShapeDtypeStruct(("B", 4), dtype)
    model = to_onnx(
        _loopy_two_adds,
        inputs=[spec],
        enable_double_precision=(dtype == jnp.float64),
        opset=21,
        model_name="ssa_no_dupe_shape_helpers",
        use_onnx_ir=True,
    )

    outs = _all_node_outputs_recursive(model.graph)
    dupes = [
        n
        for n, c in collections.Counter(outs).items()
        if c > 1 and n.endswith("__shape")
    ]
    assert not dupes, f"Duplicate '*__shape' helpers detected: {dupes}"

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    B = 3
    x = np.random.randn(B, 4).astype(np.float32 if dtype == jnp.float32 else np.float64)
    sess.run(None, {sess.get_inputs()[0].name: x})
