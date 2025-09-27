import pytest
import onnx
import onnxruntime as ort
import jax.numpy as jnp
from jax import lax
from jax2onnx.converter.conversion_api import to_onnx

# ---------- tiny helpers ----------


def _first_loop_body(graph: onnx.GraphProto):
    for n in graph.node:
        if n.op_type == "Loop":
            for a in n.attribute:
                if a.name == "body":
                    return onnx.helper.get_attribute_value(a)
    return None


def _all_dims_dynamic(vi: onnx.ValueInfoProto) -> bool:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return True
    return all(not d.HasField("dim_value") for d in tt.shape.dim)


# ---------- model under test ----------


def _loop_body_broadcast_mul_with_scatter_repro():
    T = 210  # matches the use-case

    def body_fun(i, carry):
        ref = carry  # (5, T, 1, 1)

        # Scatter into the time dimension (dim=1), like in your trace.
        idx = jnp.array([5], dtype=jnp.int32)  # shape (1,)
        updates = jnp.ones((5, 200, 1, 1), dtype=ref.dtype)  # (5,200,1,1)
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
        )  # (5, T, 1, 1)

        a0 = jnp.squeeze(ref[0:1, :, :, :], axis=0)  # (T,1,1)
        mid = ref[1:4, :, :, :]  # (3,T,1,1)
        last = jnp.squeeze(ref[4:5, :, :, :], axis=0)  # (T,1,1)

        # Broadcasted mul/div with fixed 1-dims, mirroring the logs.
        ratio = last / (0.4 * a0)  # (T,1,1)
        sum_sq = jnp.sum(mid * mid, axis=0)  # (T,1,1)
        tail = a0 * (0.5 * sum_sq + ratio)  # (T,1,1)

        out = jnp.stack([a0, mid[0], mid[1], mid[2], tail], axis=0)  # (5,T,1,1)
        return out

    init = jnp.ones((5, T, 1, 1), dtype=jnp.float64)
    out = lax.fori_loop(
        0, 1, body_fun, init
    )  # one iteration is enough to create a Loop
    return out


# ---------- tests ----------


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_loop_body_loosen_env_allows_ort_load_and_run(tmp_path, monkeypatch):

    model = to_onnx(
        _loop_body_broadcast_mul_with_scatter_repro,
        inputs=[],  # nullary function
        enable_double_precision=True,  # matches failing use-case dtype
        opset=21,
        model_name="loop_body_loosen_shapes_repro",
    )

    p = tmp_path / "loop_body_loosen_shapes_repro.onnx"
    p.write_bytes(model.SerializeToString())

    # Structural sanity
    m = onnx.load(str(p))
    onnx.checker.check_model(m)

    # Must contain a Loop
    body = _first_loop_body(m.graph)
    assert body is not None, "Expected a Loop body subgraph."

    # ORT must be able to load and run
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {})
    assert len(outs) == 1
    assert tuple(outs[0].shape) == (5, 210, 1, 1)


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_loop_body_internal_value_infos_are_rank_only_when_loosen_enabled(
    tmp_path, monkeypatch
):

    model = to_onnx(
        _loop_body_broadcast_mul_with_scatter_repro,
        inputs=[],
        enable_double_precision=True,
        opset=21,
        model_name="loop_body_loosen_shapes_vi_check",
    )

    p = tmp_path / "loop_body_loosen_shapes_vi_check.onnx"
    p.write_bytes(model.SerializeToString())

    m = onnx.load(str(p))
    body = _first_loop_body(m.graph)
    assert body is not None, "Expected a Loop body subgraph."

    # Every internal value_info in the Loop body should be rank-only (no fixed dims).
    assert (
        len(body.value_info) >= 1
    ), "Expected internal value_info entries in Loop body."
    for vi in body.value_info:
        assert _all_dims_dynamic(
            vi
        ), f"Loop body VI '{vi.name}' must be rank-only after converter loosening."
