from __future__ import annotations
import collections
import pytest
import jax
import jax.numpy as jnp
from jax import lax
from onnx import AttributeProto
from jax2onnx import to_onnx

try:
    import onnxruntime as ort

    HAS_ORT = True
except Exception:
    HAS_ORT = False


def _all_node_outputs_recursive(g):
    outs = []
    for n in g.node:
        outs.extend([o for o in n.output if o])
        for a in n.attribute:
            if a.type == AttributeProto.GRAPH and a.g is not None:
                outs.extend(_all_node_outputs_recursive(a.g))
            elif a.type == AttributeProto.GRAPHS and a.graphs:
                for sg in a.graphs:
                    outs.extend(_all_node_outputs_recursive(sg))
    return outs


def _loopy_two_adds(y):
    # Create two independent Add producers; each will feed a path that
    # needs a runtime shape vector (via reshape/broadcast usage).
    dt = y.dtype
    one = jnp.array(1.0, dt)
    two = jnp.array(2.0, dt)

    def step(c, _):
        h1 = c + jnp.broadcast_to(one, c.shape)
        r1 = jnp.reshape(h1, h1.shape)  # requires Shape(h1)
        h2 = (c * 1) + jnp.broadcast_to(two, c.shape)
        r2 = jnp.reshape(h2, h2.shape)  # requires Shape(h2)
        out = r1 + r2 + (h1 * 0) + (h2 * 0)  # keep producers alive
        return c + jnp.array(0.0, dt), out

    _, ys = lax.scan(step, y, xs=None, length=2)
    return ys


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_model_has_no_duplicate_shape_helpers(dtype):
    spec = jax.ShapeDtypeStruct(("B", 4), dtype)
    mdl = to_onnx(
        _loopy_two_adds,
        inputs=[spec],
        enable_double_precision=(dtype == jnp.float64),
        opset=21,
        model_name="ssa_no_dupe_shape_helpers",
    )
    outs = _all_node_outputs_recursive(mdl.graph)
    dupes = [
        n
        for n, c in collections.Counter(outs).items()
        if c > 1 and n.endswith("__shape")
    ]
    assert not dupes, f"Duplicate '*__shape' helpers detected: {dupes}"
    if HAS_ORT:
        ort.InferenceSession(
            mdl.SerializeToString(), providers=["CPUExecutionProvider"]
        )
