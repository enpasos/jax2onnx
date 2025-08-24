"""
Regression: ensure SSA-unique helper names when *both* broadcast_in_dim(..., h.shape)
and reshape(h, h.shape) are used in the same Loop/Scan body for the SAME producer `h`.

Historically this could yield two separate Shape nodes with the exact same output name
(e.g. 'Add__shape'), violating SSA and causing ORT to reject the model:

  INVALID_GRAPH : Graph must be in single static assignment (SSA) form,
  however 'Add__shape' has been used as output names multiple times.
"""

from __future__ import annotations

import collections
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import lax
from onnx import AttributeProto

from jax2onnx import to_onnx

try:
    import onnxruntime as ort  # type: ignore

    HAS_ORT = True
except Exception:
    HAS_ORT = False


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


def _body_broadcast_and_reshape_on_same_h(y):
    """
    In each step:
      h = c + broadcast_to(1, c.shape)      # -> Add (producer)
      b1 = broadcast_to(1, h.shape)         # -> Shape(h) via broadcast_in_dim
      r1 = jnp.reshape(h, h.shape)          # -> Shape(h) via reshape path
    Without a shared 'shape-of' cache, both helpers mint an output like 'Add__shape'
    inside the same Loop body, violating SSA.
    """
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def step(c, _):
        h = c + jnp.broadcast_to(one, c.shape)  # Add producer
        b1 = jnp.broadcast_to(one, h.shape)  # uses Shape(h) in broadcast_in_dim
        # Keep `h` live (avoid fusion away) and request another Shape(h) via reshape:
        h2 = h + jnp.array(0.0, dt)
        r1 = jnp.reshape(h2, h.shape)  # uses Shape(h) in reshape path
        out = h + b1 + r1
        return c + jnp.array(0.0, dt), out

    _, ys = lax.scan(step, y, xs=None, length=2)  # xs=None -> lowered as Loop
    return ys


@pytest.mark.parametrize("dtype", [jnp.float64])  # stress the double-precision path
def test_loop_body_shared_shape_helpers_are_ssa_unique(tmp_path, dtype):
    # Symbolic shape encourages runtime Shape() materialization
    spec = jax.ShapeDtypeStruct(("B", 4), dtype)
    model = to_onnx(
        _body_broadcast_and_reshape_on_same_h,
        inputs=[spec],
        enable_double_precision=True,
        loosen_internal_shapes=True,
        opset=21,
        model_name="broadcast_and_reshape_shared_shape_ssa",
    )

    # Inspect the first Loop body (where the previous duplication occurred)
    body = _find_first_loop_body(model.graph)
    assert body is not None, "Expected a Loop body subgraph."
    outs = _collect_node_outputs_recursive(body)
    dupes = [n for n, c in collections.Counter(outs).items() if c > 1]

    # This assertion FAILS on buggy code (duplicate '...__shape' outputs from two helpers),
    # and PASSES once helper names are properly cached/unique across both paths.
    assert not dupes, f"Duplicate outputs in Loop body (SSA violation): {dupes}"

    # ORT load smoke test — used to raise INVALID_GRAPH on buggy code.
    if HAS_ORT:
        p = tmp_path / "broadcast_and_reshape_shared_shape_ssa.onnx"
        p.write_bytes(model.SerializeToString())
        ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])


def test_numeric_executes(dtype=jnp.float64):
    if not HAS_ORT:
        pytest.skip("onnxruntime not available")
    spec = jax.ShapeDtypeStruct((7, 4), dtype)
    model = to_onnx(
        _body_broadcast_and_reshape_on_same_h,
        inputs=[spec],
        enable_double_precision=True,
        loosen_internal_shapes=True,
        opset=21,
        model_name="broadcast_and_reshape_shared_shape_numeric",
    )
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = np.zeros((7, 4), np.float64)
    feeds = {sess.get_inputs()[0].name: x}
    (y,) = sess.run(None, feeds)
    assert y.shape == (2, 7, 4)
