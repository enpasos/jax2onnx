"""
Regression: ensure SSA-unique helper names (e.g. '*__shape') when
`broadcast_in_dim(..., h.shape)` is used multiple times in the same Loop/Scan body,
where `h` comes from different Add nodes. This targets the failure mode reported by ORT:

  INVALID_GRAPH : ... Graph must be in single static assignment (SSA) form,
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


# ---------- graph helpers ----------
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
    # Depth-first: prefer top-level Loop; else descend into subgraphs
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


# ---------- body that provokes duplicate '*__shape' without the fix ----------
def _two_broadcasts_to_add_shapes(y):
    """
    In each step:
      hA = c + broadcast_to(1, c.shape)      # -> Add node A
      hB = c + broadcast_to(1, c.shape)      # -> Add node B
      b1 = broadcast_to(1, hA.shape)         # needs Shape(hA)
      b2 = broadcast_to(1, hB.shape)         # needs Shape(hB)
    If 'Shape' helper names are minted as f"{producer}__shape" without caching/uniqueness,
    both become 'Add__shape' and violate SSA inside the Loop body.
    """
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def step(c, _):
        # two independent Add producers
        hA = c + jnp.broadcast_to(one, c.shape)
        hB = c + jnp.broadcast_to(one, c.shape)
        # each forces a separate Shape-of(Add) path
        b1 = jnp.broadcast_to(one, hA.shape)
        b2 = jnp.broadcast_to(one, hB.shape)
        out = (hA + b1) + (hB + b2)
        # keep carry live but unchanged
        return c + jnp.array(0.0, dt), out

    _, ys = lax.scan(step, y, xs=None, length=2)  # xs=None => lowered via Loop
    return ys


@pytest.mark.parametrize("dtype", [jnp.float64])  # stress double-precision path
def test_loop_body_shape_helpers_are_ssa_unique(tmp_path, dtype):
    # Symbolic trip axis encourages runtime Shape() use
    spec = jax.ShapeDtypeStruct(("B", 4), dtype)
    model = to_onnx(
        _two_broadcasts_to_add_shapes,
        inputs=[spec],
        enable_double_precision=True,
        loosen_internal_shapes=True,
        opset=21,
        model_name="broadcast_in_dim_shapeof_ssa_regression_strict",
    )

    # Focus on the first Loop body (where the issue occurs)
    body = _find_first_loop_body(model.graph)
    assert body is not None, "Expected a Loop body subgraph."
    outs = _collect_node_outputs_recursive(body)
    dupes = [n for n, c in collections.Counter(outs).items() if c > 1]

    # This assertion FAILS on the buggy implementation (duplicate '*__shape' outputs),
    # and PASSES once helper names are properly cached/unique.
    assert not dupes, f"Duplicate outputs in Loop body (SSA violation): {dupes}"

    # ORT smoke test: would throw INVALID_GRAPH pre-fix
    if HAS_ORT:
        p = tmp_path / "broadcast_in_dim_shapeof_ssa_regression_strict.onnx"
        p.write_bytes(model.SerializeToString())
        ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])


def test_numeric_executes(dtype=jnp.float64):
    """Not a strict numeric test; just check the graph runs."""
    if not HAS_ORT:
        pytest.skip("onnxruntime not available")
    spec = jax.ShapeDtypeStruct((7, 4), dtype)
    model = to_onnx(
        _two_broadcasts_to_add_shapes,
        inputs=[spec],
        enable_double_precision=True,
        loosen_internal_shapes=True,
        opset=21,
        model_name="broadcast_in_dim_shapeof_numeric",
    )
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = np.zeros((7, 4), np.float64)
    feeds = {sess.get_inputs()[0].name: x}
    (y,) = sess.run(None, feeds)
    assert y.shape == (2, 7, 4)
