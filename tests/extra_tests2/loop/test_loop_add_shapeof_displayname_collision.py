# file: tests/extra_tests/loop/test_loop_add_shapeof_displayname_collision.py
"""
Regression guard: ensure shape-of helpers (names ending with '__shape') are SSA-unique
inside a Loop body, even when two independent Add→reshape(h, h.shape) sites exist.

If a buggy converter reuses display names (e.g. both helpers named 'Add__shape'),
this test will detect duplicates and (optionally) observe ORT INVALID_GRAPH.
On fixed converters, no duplicates should be found and ORT should load (and run).
"""
from __future__ import annotations

import collections
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import lax
from onnx import AttributeProto

from jax2onnx.user_interface import to_onnx

try:
    import onnxruntime as ort

    HAS_ORT = True
except Exception:
    HAS_ORT = False


# ---------- graph utils ----------
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
    # prefer top-level Loop if present
    for n in g.node:
        if n.op_type == "Loop":
            for a in n.attribute:
                if a.type == AttributeProto.GRAPH and a.g is not None:
                    return a.g
    # else search subgraphs depth-first
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


# ---------- tiny repro body ----------
def _loop_body_with_two_add_shapeof_sites(y):
    """Within a scan step, create TWO independent Add→reshape(h, h.shape) sites."""
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def step(c, _):
        # force broadcast_in_dim path to mirror real models
        ones = jnp.broadcast_to(one, c.shape)

        # Block A
        hA = c + ones
        rA = jnp.reshape(hA, hA.shape)
        # keep alive so nothing fuses away
        hA = hA + jnp.array(0.0, dt)

        # Block B (independent Add producing a 2nd Shape-of the Add result)
        hB = c + ones
        rB = jnp.reshape(hB, hB.shape)

        out = rA + rB
        return c, out  # carry unchanged, scan output = out

    _, ys = lax.scan(step, y, xs=None, length=2)  # any length ≥1 creates a Loop
    return ys  # (len, *y.shape)


def _shape_dupes_in_loop_body(model):
    body = _find_first_loop_body(model.graph)
    assert body is not None, "Expected a Loop body subgraph."
    outs = _collect_node_outputs_recursive(body)
    shape_helpers = [n for n in outs if n.endswith("__shape")]
    dupes = {n: c for n, c in collections.Counter(shape_helpers).items() if c > 1}
    return dupes, shape_helpers


@pytest.mark.parametrize("dtype", [jnp.float64])  # mirror feed-forward (double)
def test_loop_body_shape_helpers_are_ssa_unique(dtype):
    # Symbolic leading dim to exercise Shape-of paths
    spec = jax.ShapeDtypeStruct(("B", 3, 4), dtype)
    model = to_onnx(
        _loop_body_with_two_add_shapeof_sites,
        inputs=[spec],
        enable_double_precision=True,
        opset=21,
        model_name="loop_shapeof_guard_unique",
        use_onnx_ir=True,
    )

    dupes, shape_helpers = _shape_dupes_in_loop_body(model)

    # If this ever fails, we reintroduced the bug. The message shows the helpers we saw.
    assert (
        not dupes
    ), f"Duplicate '__shape' helpers in Loop body: {dupes}\nAll shape helpers: {shape_helpers}"


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_onnxruntime_behaviour_matches_ssa_status(dtype):
    """
    If duplicates exist, ORT should reject the model.
    If not, ORT should load and a trivial run should succeed.
    """
    # Use a fully concrete shape here so we can run ORT when the model is valid.
    spec = jax.ShapeDtypeStruct((2, 3, 4), dtype)
    model = to_onnx(
        _loop_body_with_two_add_shapeof_sites,
        inputs=[spec],
        enable_double_precision=True,
        opset=21,
        model_name="loop_shapeof_guard_ort",
        use_onnx_ir=True,
    )

    dupes, _ = _shape_dupes_in_loop_body(model)

    if not HAS_ORT:
        pytest.skip("onnxruntime not available")

    if dupes:
        # Buggy converters should trip INVALID_GRAPH here.
        with pytest.raises(Exception):
            ort.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
    else:
        # Fixed converters: ORT loads and runs
        sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        # one dummy input
        x = np.zeros((2, 3, 4), np.float64)
        feeds = {sess.get_inputs()[0].name: x}
        (y,) = sess.run(None, feeds)
        assert y.shape == (2, 2, 3, 4)
