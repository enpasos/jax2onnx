"""
Regression guard for SSA naming when broadcast_in_dim is used inside a Loop body.
We build a scan with xs=None (→ lowered to ONNX Loop) whose body uses multiple
broadcast_in_dim sites over a symbolic shape (B, 4). With loosen_internal_shapes=True,
the lowering emits Shape/Gather/Unsqueeze helpers at runtime. Historically, helper
names could collide (e.g. 'Add__shape'), violating SSA inside the Loop body.
This test verifies all node outputs in the (sub)graph are unique and that ORT
can load the model without SSA complaints.
"""

from __future__ import annotations

import collections
import pytest
import jax
import jax.numpy as jnp
from jax import lax
from onnx import AttributeProto

from jax2onnx import to_onnx

try:
    import onnxruntime as ort  # optional load check

    HAS_ORT = True
except Exception:
    HAS_ORT = False


def _fn_loop_body_has_two_broadcasts(y):
    """
    Body creates multiple broadcast_in_dim uses and multiple adds so that
    any buggy helper naming (like '<op>__shape') would appear more than once.
    """

    def body(carry, _):
        # Two separate broadcast_in_dim sites to the same symbolic shape
        one = jnp.broadcast_to(jnp.array(1.0, dtype=carry.dtype), carry.shape)
        zero = jnp.broadcast_to(jnp.array(0.0, dtype=carry.dtype), carry.shape)

        # A couple of ops that tend to trigger runtime shape reads when shapes are loosened
        e = carry + one
        g = e + zero

        # Reshape referencing e.shape forces additional shape plumbing
        e_reshaped = jnp.reshape(e, e.shape)
        out = g + e_reshaped

        # Keep carry alive with a no-op add to simulate real workloads
        return carry + jnp.array(0.0, dtype=carry.dtype), out

    # xs=None → lowered to ONNX Loop
    _, ys = lax.scan(body, y, xs=None, length=2)
    return ys


def _collect_node_outputs_recursive(g):
    """Collect all node output names from graph g and any nested subgraphs."""
    outs = []
    for n in g.node:
        for o in n.output:
            if o:
                outs.append(o)
        for a in n.attribute:
            if a.type == AttributeProto.GRAPH and a.g is not None:
                outs.extend(_collect_node_outputs_recursive(a.g))
            elif a.type == AttributeProto.GRAPHS and a.graphs:
                for sg in a.graphs:
                    outs.extend(_collect_node_outputs_recursive(sg))
    return outs


def _find_first_loop_body(g):
    """Return the first Loop subgraph body in g (depth-first), or None."""
    for n in g.node:
        for a in n.attribute:
            if (
                a.type == AttributeProto.GRAPH
                and a.g is not None
                and n.op_type == "Loop"
            ):
                return a.g
    # Recurse into subgraphs
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


def test_broadcast_in_dim_inside_loop_ssa_unique(tmp_path):
    # Symbolic input with (B, 4)
    spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)
    model = to_onnx(
        _fn_loop_body_has_two_broadcasts,
        inputs=[spec],
        loosen_internal_shapes=True,  # promotes runtime Shape helpers
        opset=21,
        model_name="broadcast_in_dim_loop_ssa_unique",
    )

    # 1) Ensure every node.output name across the whole model is SSA-unique
    all_outs = _collect_node_outputs_recursive(model.graph)
    dupes = [name for name, c in collections.Counter(all_outs).items() if c > 1]
    assert not dupes, f"Duplicate node outputs detected (SSA violation): {dupes}"

    # 2) Focused check: dive into the Loop body and ensure it's also SSA-clean
    loop_body = _find_first_loop_body(model.graph)
    assert loop_body is not None, "Expected at least one Loop subgraph in the model."
    body_outs = _collect_node_outputs_recursive(loop_body)
    body_dupes = [name for name, c in collections.Counter(body_outs).items() if c > 1]
    assert not body_dupes, f"Duplicate outputs in Loop body: {body_dupes}"

    # 3) Heuristic: helpers often end with '__shape' when computing runtime shapes.
    #    Make sure those are unique too (catches the classic 'Add__shape' collision).
    shape_helpers = [n for n in body_outs if n.endswith("__shape")]
    shape_dupes = [n for n, c in collections.Counter(shape_helpers).items() if c > 1]
    assert not shape_dupes, f"Duplicate '__shape' helpers in Loop body: {shape_dupes}"

    # Optional: ORT load = extra guard that SSA is really satisfied
    if HAS_ORT:
        path = tmp_path / "broadcast_in_dim_loop_ssa_unique.onnx"
        path.write_bytes(model.SerializeToString())
        try:
            ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as e:
            pytest.fail(f"onnxruntime failed to load model (SSA regression?): {e}")


def test_broadcast_in_dim_loop_emits_shape_nodes_against_body_inputs():
    """
    Sanity check that the runtime Shape helpers inside the Loop body take
    LOOP BODY INPUT symbols as their source (not stale/global names),
    which protects against 'Could not find shape info for tensor <name>' errors.
    """
    spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)
    model = to_onnx(
        _fn_loop_body_has_two_broadcasts,
        inputs=[spec],
        loosen_internal_shapes=True,
        opset=21,
        model_name="broadcast_in_dim_loop_shape_inputs",
    )
    loop_body = _find_first_loop_body(model.graph)
    assert loop_body is not None
    body_inputs = {vi.name for vi in loop_body.input}

    # Gather Shape nodes and verify their inputs are present in body inputs or produced in-body.
    produced = set()
    for n in loop_body.node:
        produced.update(o for o in n.output if o)
    known = body_inputs | produced

    shape_nodes = [n for n in loop_body.node if n.op_type == "Shape"]
    # There should be at least one runtime Shape node if internal shapes were loosened
    assert len(shape_nodes) >= 1
    for n in shape_nodes:
        assert n.input, "Shape node without inputs?"
        assert (
            n.input[0] in known
        ), f"Shape input {n.input[0]!r} not found among body inputs/defs."
