from __future__ import annotations

import pathlib

import jax.numpy as jnp
import onnx
import pytest
from jax import lax

from jax2onnx.user_interface import to_onnx


# ---------- tiny helpers ----------


def _first_loop_or_scan_body(graph: onnx.GraphProto):
    for n in graph.node:
        if n.op_type in ("Scan", "Loop"):
            for a in n.attribute:
                if a.name == "body":
                    return onnx.helper.get_attribute_value(a)
    return None


def _all_dims_dynamic(vi: onnx.ValueInfoProto) -> bool:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return True
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            return False
        if d.HasField("dim_param"):
            return False
    return True


# ---------- model under test (scan body mirrors the loop repro) ----------


def _scan_body_broadcast_mul_with_scatter_repro():
    T = 210  # matches the earlier use-case
    updates_len = 200  # the fixed window that previously caused re-tightening

    def body_fun(carry, _):
        ref = carry

        idx = jnp.array([5], dtype=jnp.int32)
        updates = jnp.ones((5, updates_len, 1, 1), dtype=ref.dtype)

        dnums = lax.ScatterDimensionNumbers(
            update_window_dims=(0, 1, 2, 3),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(1,),
        )
        ref = lax.scatter(
            ref,
            idx,
            updates,
            dnums,
            indices_are_sorted=True,
            unique_indices=True,
            mode=lax.GatherScatterMode.FILL_OR_DROP,
        )

        a0 = jnp.squeeze(ref[0:1, :, :, :], axis=0)
        mid = ref[1:4, :, :, :]
        last = jnp.squeeze(ref[4:5, :, :, :], axis=0)

        ratio = last / (0.4 * a0)
        sum_sq = jnp.sum(mid * mid, axis=0)
        tail = a0 * (0.5 * sum_sq + ratio)

        out = jnp.stack([a0, mid[0], mid[1], mid[2], tail], axis=0)
        return out, jnp.zeros((), dtype=jnp.int32)

    init = jnp.ones((5, T, 1, 1), dtype=jnp.float64)

    xs = jnp.arange(1, dtype=jnp.int32)
    carry_out, _ = lax.scan(body_fun, init, xs)
    return carry_out


def _nested_loop_repro():
    """Nested Loop regression mirroring the scan repro."""

    T = 210
    updates_len = 200

    def inner_body(_, ref):
        idx = jnp.array([5], dtype=jnp.int32)
        updates = jnp.ones((5, updates_len, 1, 1), dtype=ref.dtype)
        dnums = lax.ScatterDimensionNumbers(
            update_window_dims=(0, 1, 2, 3),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(1,),
        )
        ref = lax.scatter(
            ref,
            idx,
            updates,
            dnums,
            indices_are_sorted=True,
            unique_indices=True,
            mode=lax.GatherScatterMode.FILL_OR_DROP,
        )

        a0 = jnp.squeeze(ref[0:1, :, :, :], axis=0)
        mid = ref[1:4, :, :, :]
        last = jnp.squeeze(ref[4:5, :, :, :], axis=0)
        ratio = last / (0.4 * a0)
        sum_sq = jnp.sum(mid * mid, axis=0)
        tail = a0 * (0.5 * sum_sq + ratio)
        out = jnp.stack([a0, mid[0], mid[1], mid[2], tail], axis=0)
        return out

    def outer_body(_, carry):
        return lax.fori_loop(0, 2, inner_body, carry)

    init = jnp.ones((5, T, 1, 1), dtype=jnp.float64)
    return lax.fori_loop(0, 2, outer_body, init)


# ---------- tests ----------


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_scan_body_loosen_env_allows_ort_load_and_run(tmp_path: pathlib.Path):
    ort = pytest.importorskip("onnxruntime")

    model = to_onnx(
        _scan_body_broadcast_mul_with_scatter_repro,
        inputs=[],
        enable_double_precision=True,
        opset=21,
        model_name="scan_body_loosen_shapes_repro",
        use_onnx_ir=True,
    )

    path = tmp_path / "scan_body_loosen_shapes_repro.onnx"
    path.write_bytes(model.SerializeToString())

    onnx.checker.check_model(model)

    body = _first_loop_or_scan_body(model.graph)
    assert body is not None, "Expected a Scan/Loop body."

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {})
    assert len(outs) == 1
    assert tuple(outs[0].shape) == (5, 210, 1, 1)


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_scan_body_internal_value_infos_are_rank_only_when_loosen_enabled(
    tmp_path: pathlib.Path,
):
    model = to_onnx(
        _scan_body_broadcast_mul_with_scatter_repro,
        inputs=[],
        enable_double_precision=True,
        opset=21,
        model_name="scan_body_loosen_shapes_vi_check",
        use_onnx_ir=True,
    )

    path = tmp_path / "scan_body_loosen_shapes_vi_check.onnx"
    path.write_bytes(model.SerializeToString())

    m = onnx.load(str(path))
    body = _first_loop_or_scan_body(m.graph)
    assert body is not None, "Expected a Scan/Loop body."

    for vi in body.value_info:
        assert _all_dims_dynamic(
            vi
        ), f"Scan/Loop body VI '{vi.name}' must be rank-only after converter loosening"


SENSITIVE = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Reshape",
    "Squeeze",
    "Unsqueeze",
    "Expand",
    "Concat",
    "Range",
    "Shape",
    "NonZero",
    "Gather",
    "GatherND",
    "Slice",
    "Constant",
    "ConstantOfShape",
    "Pow",
}


def _producer_map(g: onnx.GraphProto):
    return {o: n.op_type for n in g.node for o in n.output}


def _has_concrete_dim(vi: onnx.ValueInfoProto) -> bool:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return False
    return any(d.HasField("dim_value") for d in tt.shape.dim)


def _loop_bodies(g: onnx.GraphProto):
    for n in g.node:
        if n.op_type == "Loop":
            for a in n.attribute:
                if a.name == "body":
                    yield onnx.helper.get_attribute_value(a)
