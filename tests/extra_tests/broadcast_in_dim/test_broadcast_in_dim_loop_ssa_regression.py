# file: tests/extra_tests/broadcast_in_dim/test_broadcast_in_dim_loop_ssa_regression.py
"""
Regression: ensure SSA-unique helper names (e.g. '*__shape') when broadcast_in_dim
and reshape(..., x.shape) are used inside a Loop body. This targets the failure mode
seen as 'INVALID_GRAPH ... Graph must be in SSA form, however "Add__shape" has been
used as output names multiple times.' in complex feed-forward graphs.
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
    import onnxruntime as ort

    HAS_ORT = True
except Exception:
    HAS_ORT = False


# ---------- helpers ----------
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


# ---------- bodies designed to provoke duplicate '*__shape' ----------
def _body_same_symbol_two_reshapes(y):
    """Same Add result 'h' used in TWO reshape(..., h.shape)."""
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def step(c, _):
        b = jnp.broadcast_to(one, c.shape)  # broadcast_in_dim path
        h = c + b  # <- Add (producer of interest)
        r0 = jnp.reshape(h, h.shape)  # first Shape-of(h)
        # keep 'h' "alive" to avoid fusion away
        h = h + jnp.array(0.0, dt)
        r1 = jnp.reshape(h, h.shape)  # second Shape-of(h) (same symbol)
        out = r0 + r1
        return c + jnp.array(0.0, dt), out

    _, ys = lax.scan(step, y, xs=None, length=2)
    return ys


def _body_two_blocks_dup_pattern(y):
    """Two independent, identical blocks each doing Add→reshape(x,x.shape)."""
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def step(c, _):
        # block A
        bA = jnp.broadcast_to(one, c.shape)
        hA = c + bA
        rA0 = jnp.reshape(hA, hA.shape)
        # block B
        bB = jnp.broadcast_to(one, c.shape)
        hB = c + bB
        rB0 = jnp.reshape(hB, hB.shape)
        out = rA0 + rB0
        return c + jnp.array(0.0, dt), out

    _, ys = lax.scan(step, y, xs=None, length=2)
    return ys


def _body_nested_loop_same_symbol_two_reshapes(y):
    """Outer Loop with inner subgraph also reading Shape twice from same symbol."""
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def inner(c, _):
        b = jnp.broadcast_to(one, c.shape)
        h = c + b
        r0 = jnp.reshape(h, h.shape)
        h = h + jnp.array(0.0, dt)
        r1 = jnp.reshape(h, h.shape)
        out = r0 + r1
        return c + jnp.array(0.0, dt), out

    def outer_step(c, _):
        _, ys_inner = lax.scan(inner, c, xs=None, length=2)
        # force a second site in the outer body as well
        b = jnp.broadcast_to(one, c.shape)
        h = c + b
        r = jnp.reshape(h, h.shape)
        out = ys_inner[-1] + r
        return c + jnp.array(0.0, dt), out

    _, ys = lax.scan(outer_step, y, xs=None, length=2)
    return ys


@pytest.mark.parametrize(
    "fn",
    [
        _body_same_symbol_two_reshapes,
        _body_two_blocks_dup_pattern,
        _body_nested_loop_same_symbol_two_reshapes,
    ],
)
@pytest.mark.parametrize("dtype", [jnp.float64])  # stress double-precision path
def test_loop_body_shapeof_helpers_are_ssa_unique(tmp_path, fn, dtype):
    # symbolically shaped input to encourage Shape usage
    spec = jax.ShapeDtypeStruct(("B", 4), dtype)
    model = to_onnx(
        fn,
        inputs=[spec],
        enable_double_precision=True,
        loosen_internal_shapes=True,
        opset=21,
        model_name=f"{fn.__name__}",
    )

    # global SSA check
    all_outs = _collect_node_outputs_recursive(model.graph)
    dupes = [n for n, c in collections.Counter(all_outs).items() if c > 1]
    assert not dupes, f"Duplicate node outputs detected (SSA violation): {dupes}"

    # Loop-body focused SSA check (this is where previous failures occurred)
    body = _find_first_loop_body(model.graph)
    assert body is not None, "Expected a Loop body subgraph."
    body_outs = _collect_node_outputs_recursive(body)
    body_dupes = [n for n, c in collections.Counter(body_outs).items() if c > 1]
    assert not body_dupes, f"Duplicate outputs in Loop body (SSA): {body_dupes}"

    # extra: shape helper names are typically suffixed by '__shape'
    shape_helpers = [n for n in body_outs if n.endswith("__shape")]
    shape_dupes = [n for n, c in collections.Counter(shape_helpers).items() if c > 1]
    assert not shape_dupes, f"Duplicate '__shape' helpers in Loop body: {shape_dupes}"

    # ORT smoke test (the historical error showed up here)
    if HAS_ORT:
        p = tmp_path / f"{fn.__name__}.onnx"
        p.write_bytes(model.SerializeToString())
        ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])


def test_numeric_sanity_executes(dtype=jnp.float64):
    """Ensure the graph is executable; not a strict numeric equivalence test."""
    spec = jax.ShapeDtypeStruct((7, 4), dtype)
    model = to_onnx(
        _body_same_symbol_two_reshapes,
        inputs=[spec],
        enable_double_precision=True,
        loosen_internal_shapes=True,
        opset=21,
        model_name="loop_shapeof_numeric",
    )
    if not HAS_ORT:
        pytest.skip("onnxruntime not available")
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    x = np.zeros((7, 4), np.float64)
    feeds = {sess.get_inputs()[0].name: x}
    (y,) = sess.run(None, feeds)
    assert y.shape == (2, 7, 4)
    (y,) = sess.run(None, feeds)
    assert y.shape == (2, 7, 4)
