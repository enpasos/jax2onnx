# tests/extra_tests/loop/test_loop_add_shapeof_displayname_collision_dev.py
import jax
import jax.numpy as jnp
from jax import lax
from jax2onnx.user_interface import to_onnx
import onnxruntime as ort
from onnx import AttributeProto


def _body(y):
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def step(c, _):
        ones = jnp.broadcast_to(one, c.shape)
        hA = c + ones
        rA = jnp.reshape(hA, hA.shape)
        hB = c + ones
        rB = jnp.reshape(hB, hB.shape)
        return c, rA + rB

    _, ys = lax.scan(step, y, xs=None, length=2)
    return ys


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


def _rename_symbol_everywhere(g, old, new):
    for n in g.node:
        n.input[:] = [new if x == old else x for x in n.input]
        n.output[:] = [new if x == old else x for x in n.output]
        for a in n.attribute:
            if a.type == AttributeProto.GRAPH and a.g is not None:
                _rename_symbol_everywhere(a.g, old, new)
            elif a.type == AttributeProto.GRAPHS and a.graphs:
                for sg in a.graphs:
                    _rename_symbol_everywhere(sg, old, new)
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.name == old:
            vi.name = new
    for init in g.initializer:
        if init.name == old:
            init.name = new


def test_loop_body_forced_Add__shape_collision_rejected_by_ort():
    # Export a *valid* model first
    spec = jax.ShapeDtypeStruct((2, 3, 4), jnp.float64)
    model = to_onnx(
        _body,
        inputs=[spec],
        enable_double_precision=True,
        opset=21,
        model_name="loop_add_shapeof_displayname_collision_dev",
    )

    # Find the first Loop body subgraph
    body = _find_first_loop_body(model.graph)
    assert body is not None, "Expected a Loop body subgraph."

    # Pick two distinct outputs inside the body to collide
    all_outs = []
    for n in body.node:
        all_outs.extend([o for o in n.output if o])

    shape_like = [o for o in all_outs if o.endswith("__shape")]
    candidates = shape_like[:2] if len(shape_like) >= 2 else all_outs[:2]
    assert len(candidates) >= 2, "Need at least two outputs to force a collision."
    a, b = candidates[0], candidates[1]

    # Force a duplicate output name *inside the Loop body*
    target = "Add__shape"
    if a != target:
        _rename_symbol_everywhere(body, a, target)
    _rename_symbol_everywhere(body, b, target)

    # Sanity: ensure the duplicate is present now
    dup_count = sum(
        1 for n in (o for nn in body.node for o in nn.output if o) if n == target
    )
    assert (
        dup_count >= 2
    ), f"didn't create the duplicate we expected (found {dup_count})"

    # ORT must reject the model with INVALID_GRAPH (SSA violation)
    try:
        ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        raise AssertionError("ORT accepted an SSA-invalid Loop body (unexpected).")
    except Exception:
        # Expected: INVALID_GRAPH complaining about duplicate 'Add__shape'
        pass
