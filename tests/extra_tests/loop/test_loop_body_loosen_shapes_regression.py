# tests/extra_tests/loop/test_loop_body_loosen_shapes_regression.py

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import pytest

import jax.numpy as jnp
from jax import lax

import onnx
import onnxruntime as ort

from jax2onnx.user_interface import to_onnx


def _first_loop_body(graph: onnx.GraphProto):
    for node in graph.node:
        if node.op_type == "Loop":
            for attr in node.attribute:
                if attr.name == "body":
                    return onnx.helper.get_attribute_value(attr)
    return None


def _all_dims_dynamic(value_info: onnx.ValueInfoProto) -> bool:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return True
    return all(not dim.HasField("dim_value") for dim in tensor_type.shape.dim)


def _loop_body_broadcast_mul_with_scatter_repro():
    T = 210

    def body_fun(i, carry):
        ref = carry
        idx = jnp.array([5], dtype=jnp.int32)
        updates = jnp.ones((5, 200, 1, 1), dtype=ref.dtype)
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

    init = jnp.ones((5, T, 1, 1), dtype=jnp.float64)
    return lax.fori_loop(0, 1, body_fun, init)


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_loop_body_loosen_env_allows_ort_load_and_run(tmp_path):
    model = to_onnx(
        _loop_body_broadcast_mul_with_scatter_repro,
        inputs=[],
        enable_double_precision=True,
        opset=21,
        model_name="loop_body_loosen_shapes_repro_ir",
    )

    out_path = tmp_path / "loop_body_loosen_shapes_repro_ir.onnx"
    out_path.write_bytes(model.SerializeToString())

    loaded = onnx.load(str(out_path))
    onnx.checker.check_model(loaded)
    assert _first_loop_body(loaded.graph) is not None

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {})
    assert len(outputs) == 1
    assert tuple(outputs[0].shape) == (5, 210, 1, 1)


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_loop_body_internal_value_infos_are_rank_only_when_loosen_enabled(tmp_path):
    model = to_onnx(
        _loop_body_broadcast_mul_with_scatter_repro,
        inputs=[],
        enable_double_precision=True,
        opset=21,
        model_name="loop_body_loosen_shapes_vi_check_ir",
    )

    out_path = tmp_path / "loop_body_loosen_shapes_vi_check_ir.onnx"
    out_path.write_bytes(model.SerializeToString())

    loaded = onnx.load(str(out_path))
    body = _first_loop_body(loaded.graph)
    assert body is not None

    assert body.value_info, "Expected value_info entries in Loop body"
    for value_info in body.value_info:
        assert _all_dims_dynamic(
            value_info
        ), f"Loop body VI '{value_info.name}' must be rank-only"
