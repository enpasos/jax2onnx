import collections
import jax
import jax.numpy as jnp
from jax import lax
import pytest
from onnx import AttributeProto
from jax2onnx.user_interface import to_onnx


def _collect_node_outputs_recursive(g):
    outs = []
    for n in g.node:
        outs.extend([o for o in n.output if o])
        for a in n.attribute:
            if a.type == AttributeProto.GRAPH and a.g is not None:
                outs.extend(_collect_node_outputs_recursive(a.g))
            elif a.type == AttributeProto.GRAPHS and a.graphs:
                for sg in a.graphs:
                    outs.extend(_collect_node_outputs_recursive(sg))
    return outs


def _find_first_loop_body(g):
    for n in g.node:
        if n.op_type == "Loop":
            for a in n.attribute:
                if a.type == AttributeProto.GRAPH and a.g is not None:
                    return a.g
    for n in g.node:
        for a in n.attribute:
            if a.type == AttributeProto.GRAPH and a.g is not None:
                sub = _find_first_loop_body(a.g)
                if sub is not None:
                    return sub
            elif a.type == AttributeProto.GRAPHS and a.graphs:
                for sg in a.graphs:
                    sub = _find_first_loop_body(sg)
                    if sub is not None:
                        return sub
    return None


def _loop_two_adds_two_reshapes(y):
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def step(c, _):
        b1 = jnp.broadcast_to(one, c.shape)
        a1 = c + b1
        r1 = jnp.reshape(a1, a1.shape)
        b2 = jnp.broadcast_to(one, c.shape)
        a2 = c + b2
        r2 = jnp.reshape(a2, a2.shape)
        return c, r1 + r2

    _, ys = lax.scan(step, y, xs=None, length=2)
    return ys


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_loop_body_has_no_duplicate_shape_helpers(dtype):
    spec = jax.ShapeDtypeStruct(("B", 3, 4), dtype)
    model = to_onnx(
        _loop_two_adds_two_reshapes,
        inputs=[spec],
        enable_double_precision=True,
        opset=21,
        model_name="shape_helper_uniqueness",
    )
    body = _find_first_loop_body(model.graph)
    assert body is not None, "Expected a Loop body."
    outs = _collect_node_outputs_recursive(body)
    helpers = [n for n in outs if n.endswith("__shape")]
    dupes = [n for n, c in collections.Counter(helpers).items() if c > 1]
    assert not dupes, f"Duplicate shape helpers in Loop body: {dupes}"
