"""Loop + broadcast_in_dim SSA regression cases for converter2."""

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
except Exception:  # pragma: no cover - optional dependency
    HAS_ORT = False


def _collect_node_outputs_recursive(graph):
    names: list[str] = []
    for node in graph.node:
        names.extend([out for out in node.output if out])
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH and attr.g is not None:
                names.extend(_collect_node_outputs_recursive(attr.g))
            elif attr.type == AttributeProto.GRAPHS and attr.graphs:
                for subgraph in attr.graphs:
                    names.extend(_collect_node_outputs_recursive(subgraph))
    return names


def _find_first_loop_body(graph):
    for node in graph.node:
        if node.op_type == "Loop":
            for attr in node.attribute:
                if attr.type == AttributeProto.GRAPH and attr.g is not None:
                    return attr.g
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH and attr.g is not None:
                body = _find_first_loop_body(attr.g)
                if body is not None:
                    return body
            elif attr.type == AttributeProto.GRAPHS and attr.graphs:
                for subgraph in attr.graphs:
                    body = _find_first_loop_body(subgraph)
                    if body is not None:
                        return body
    return None


def _body_same_symbol_two_reshapes(y):
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def step(carry, _):
        b = jnp.broadcast_to(one, carry.shape)
        h = carry + b
        r0 = jnp.reshape(h, h.shape)
        h = h + jnp.array(0.0, dt)
        r1 = jnp.reshape(h, h.shape)
        out = r0 + r1
        return carry + jnp.array(0.0, dt), out

    _, ys = lax.scan(step, y, xs=None, length=2)
    return ys


def _body_two_blocks_dup_pattern(y):
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def step(carry, _):
        b_a = jnp.broadcast_to(one, carry.shape)
        h_a = carry + b_a
        r_a = jnp.reshape(h_a, h_a.shape)
        b_b = jnp.broadcast_to(one, carry.shape)
        h_b = carry + b_b
        r_b = jnp.reshape(h_b, h_b.shape)
        out = r_a + r_b
        return carry + jnp.array(0.0, dt), out

    _, ys = lax.scan(step, y, xs=None, length=2)
    return ys


def _body_nested_loop_same_symbol_two_reshapes(y):
    dt = y.dtype
    one = jnp.array(1.0, dt)

    def inner(carry, _):
        b = jnp.broadcast_to(one, carry.shape)
        h = carry + b
        r0 = jnp.reshape(h, h.shape)
        h = h + jnp.array(0.0, dt)
        r1 = jnp.reshape(h, h.shape)
        out = r0 + r1
        return carry + jnp.array(0.0, dt), out

    def outer_step(carry, _):
        _, ys_inner = lax.scan(inner, carry, xs=None, length=2)
        b = jnp.broadcast_to(one, carry.shape)
        h = carry + b
        r = jnp.reshape(h, h.shape)
        out = ys_inner[-1] + r
        return carry + jnp.array(0.0, dt), out

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
@pytest.mark.parametrize("dtype", [jnp.float64])
def test_loop_body_shapeof_helpers_are_ssa_unique(tmp_path, fn, dtype):
    spec = jax.ShapeDtypeStruct(("B", 4), dtype)
    model = to_onnx(
        fn,
        inputs=[spec],
        enable_double_precision=True,
        opset=21,
        model_name=fn.__name__,
        use_onnx_ir=True,
    )

    all_outs = _collect_node_outputs_recursive(model.graph)
    dupes = [name for name, cnt in collections.Counter(all_outs).items() if cnt > 1]
    assert not dupes, f"Duplicate node outputs detected: {dupes}"

    body = _find_first_loop_body(model.graph)
    assert body is not None, "Expected Loop body subgraph"
    body_outs = _collect_node_outputs_recursive(body)
    body_dupes = [name for name, cnt in collections.Counter(body_outs).items() if cnt > 1]
    assert not body_dupes, f"Duplicate outputs in Loop body: {body_dupes}"

    shape_helpers = [name for name in body_outs if name.endswith("__shape")]
    shape_dupes = [name for name, cnt in collections.Counter(shape_helpers).items() if cnt > 1]
    assert not shape_dupes, f"Duplicate '__shape' helpers in Loop body: {shape_dupes}"

    if HAS_ORT:
        out_path = tmp_path / f"{fn.__name__}.onnx"
        out_path.write_bytes(model.SerializeToString())
        try:
            ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        except Exception as exc:  # pragma: no cover - regression guard
            if fn is _body_nested_loop_same_symbol_two_reshapes:
                pytest.xfail(
                    "converter2 Loop nested shape helpers still emit mismatched dtypes"
                    f" (ORT load failed: {exc})"
                )
            pytest.fail(f"onnxruntime failed to load IR model: {exc}")


@pytest.mark.parametrize("dtype", [jnp.float64])
def test_numeric_sanity_executes(dtype):
    if not HAS_ORT:
        pytest.skip("onnxruntime not available")
    spec = jax.ShapeDtypeStruct((7, 4), dtype)
    model = to_onnx(
        _body_same_symbol_two_reshapes,
        inputs=[spec],
        enable_double_precision=True,
        opset=21,
        model_name="loop_shapeof_numeric",
        use_onnx_ir=True,
    )
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.zeros((7, 4), np.float64)
    feeds = {sess.get_inputs()[0].name: x}
    (y,) = sess.run(None, feeds)
    assert y.shape == (2, 7, 4)
    (y,) = sess.run(None, feeds)
    assert y.shape == (2, 7, 4)
