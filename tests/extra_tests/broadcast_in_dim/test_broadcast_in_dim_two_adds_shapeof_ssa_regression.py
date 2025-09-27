"""
Regression: ensure SSA-unique helper names when two *different* Add producers
each require runtime Shape(h) via broadcast_in_dim inside the SAME Loop body.

On buggy code paths, helper names were derived from the op type only (e.g. 'Add__shape'),
so two distinct Add nodes would each mint an output named 'Add__shape', violating SSA:

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
    # Prefer a top-level Loop body; else search subgraphs depth-first.
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


def _body_two_adds_each_needs_shape(y):
    """
    Per step:
      h1 = c + broadcast_to(1, c.shape)     # Add #1
      b1 = broadcast_to(3, h1.shape)        # Shape(h1) via broadcast_in_dim

      h2 = (c * 1) + broadcast_to(2, c.shape)  # Add #2 (different producer than h1)
      b2 = broadcast_to(4, h2.shape)        # Shape(h2) via broadcast_in_dim

    Buggy naming ('Add__shape') collides for the two distinct Add producers.
    """
    dt = y.dtype
    one = jnp.array(1.0, dt)
    two = jnp.array(2.0, dt)
    three = jnp.array(3.0, dt)
    four = jnp.array(4.0, dt)

    def step(c, _):
        h1 = c + jnp.broadcast_to(one, c.shape)  # Add producer #1
        b1 = jnp.broadcast_to(three, h1.shape)  # needs Shape(h1)

        c_scaled = c * jnp.array(1.0, dt)  # keep this Add distinct in jaxpr
        h2 = c_scaled + jnp.broadcast_to(two, c.shape)  # Add producer #2
        b2 = jnp.broadcast_to(four, h2.shape)  # needs Shape(h2)

        out = b1 + b2 + (h1 * 0) + (h2 * 0)  # keep producers alive
        return c + jnp.array(0.0, dt), out

    _, ys = lax.scan(step, y, xs=None, length=2)  # xs=None → lowered as Loop
    return ys


@pytest.mark.parametrize(
    "dtype", [jnp.float64]
)  # exercise double-precision & rank-only paths
def test_two_adds_shape_helpers_are_ssa_unique(tmp_path, dtype):
    # Symbolic shape promotes runtime Shape() creation in the body
    spec = jax.ShapeDtypeStruct(("B", 4), dtype)
    model = to_onnx(
        _body_two_adds_each_needs_shape,
        inputs=[spec],
        enable_double_precision=True,
        opset=21,
        model_name="two_adds_shape_helpers_ssa",
    )

    # Focus on the first Loop body (where duplicates used to appear)
    body = _find_first_loop_body(model.graph)
    assert body is not None, "Expected a Loop body subgraph."
    outs = _collect_node_outputs_recursive(body)
    dupes = [n for n, c in collections.Counter(outs).items() if c > 1]
    assert not dupes, f"Duplicate outputs in Loop body (SSA violation): {dupes}"

    # ORT load smoke test — used to raise INVALID_GRAPH on buggy code.
    if HAS_ORT:
        p = tmp_path / "two_adds_shape_helpers_ssa.onnx"
        p.write_bytes(model.SerializeToString())
        ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])


def test_numeric_executes(dtype=jnp.float64):
    if not HAS_ORT:
        pytest.skip("onnxruntime not available")
    spec = jax.ShapeDtypeStruct((7, 4), dtype)
    model = to_onnx(
        _body_two_adds_each_needs_shape,
        inputs=[spec],
        enable_double_precision=True,
        opset=21,
        model_name="two_adds_shape_helpers_numeric",
    )
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = np.zeros((7, 4), np.float64)
    feeds = {sess.get_inputs()[0].name: x}
    (y,) = sess.run(None, feeds)
    assert y.shape == (2, 7, 4)
