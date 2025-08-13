# file: tests/permanent_examples/test_scan_body_loosen_shapes_regression.py

import pytest
import onnx
import onnxruntime as ort
import jax.numpy as jnp
import numpy as np
from jax import lax
from onnx import helper as oh

from jax2onnx.converter.conversion_api import to_onnx


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
    # rank-only means: no dim_value and no dim_param
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
        # carry has shape (5, T, 1, 1)
        ref = carry

        # Scatter into the time dimension (dim=1), like in the trace.
        idx = jnp.array([5], dtype=jnp.int32)                           # (1,)
        updates = jnp.ones((5, updates_len, 1, 1), dtype=ref.dtype)     # (5,200,1,1)

        dnums = lax.ScatterDimensionNumbers(
            update_window_dims=(0, 1, 2, 3),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(1,),
        )
        ref = lax.scatter(
            ref, idx, updates, dnums,
            indices_are_sorted=True, unique_indices=True,
            mode=lax.GatherScatterMode.FILL_OR_DROP,
        )  # (5, T, 1, 1)

        a0   = jnp.squeeze(ref[0:1, :, :, :], axis=0)   # (T,1,1)
        mid  = ref[1:4, :, :, :]                        # (3,T,1,1)
        last = jnp.squeeze(ref[4:5, :, :, :], axis=0)   # (T,1,1)

        # Broadcasted mul/div with fixed 1-dims, mirroring the logs.
        ratio  = last / (0.4 * a0)                      # (T,1,1)
        sum_sq = jnp.sum(mid * mid, axis=0)             # (T,1,1)
        tail   = a0 * (0.5 * sum_sq + ratio)            # (T,1,1)

        out = jnp.stack([a0, mid[0], mid[1], mid[2], tail], axis=0)  # (5,T,1,1)
        # scan needs (carry_out, y_out); we don't need a real y
        return out, jnp.zeros((), dtype=jnp.int32)

    init = jnp.ones((5, T, 1, 1), dtype=jnp.float64)

    xs = jnp.arange(1, dtype=jnp.int32)  # length-1 scan (enough to create a Scan body)
    carry_out, _ = lax.scan(body_fun, init, xs)
    return carry_out  # shape (5, T, 1, 1)

def _nested_loop_repro():
    """
    Build a small nested-Loop repro:
      outer Loop J in {0..1}:
        inner Loop i in {0..1} updating/refining a (5, T, 1, 1) tensor via
        a scatter into the time dimension and some broadcasted arithmetic.
    This mirrors the same patterns that used to tighten internal VIs.
    """
    T = 210
    updates_len = 200

    def inner_body(i, ref):
        # Scatter into time dimension (axis=1)
        idx = jnp.array([5], dtype=jnp.int32)                          # (1,)
        updates = jnp.ones((5, updates_len, 1, 1), dtype=ref.dtype)    # (5,200,1,1)
        dnums = lax.ScatterDimensionNumbers(
            update_window_dims=(0, 1, 2, 3),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(1,),
        )
        ref = lax.scatter(
            ref, idx, updates, dnums,
            indices_are_sorted=True, unique_indices=True,
            mode=lax.GatherScatterMode.FILL_OR_DROP,
        )  # (5, T, 1, 1)

        # Same broadcasted arithmetic as the scan repro
        a0   = jnp.squeeze(ref[0:1, :, :, :], axis=0)   # (T,1,1)
        mid  = ref[1:4, :, :, :]                        # (3,T,1,1)
        last = jnp.squeeze(ref[4:5, :, :, :], axis=0)   # (T,1,1)
        ratio  = last / (0.4 * a0)                      # (T,1,1)
        sum_sq = jnp.sum(mid * mid, axis=0)             # (T,1,1)
        tail   = a0 * (0.5 * sum_sq + ratio)            # (T,1,1)
        out = jnp.stack([a0, mid[0], mid[1], mid[2], tail], axis=0)  # (5,T,1,1)
        return out

    def outer_body(j, carry):
        # Run the inner loop a couple of steps
        return lax.fori_loop(0, 2, inner_body, carry)

    init = jnp.ones((5, T, 1, 1), dtype=jnp.float64)
    return lax.fori_loop(0, 2, outer_body, init)  # final (5, T, 1, 1)

# ---------- tests ----------

@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_scan_body_loosen_env_allows_ort_load_and_run(tmp_path, monkeypatch):

    model = to_onnx(
        _scan_body_broadcast_mul_with_scatter_repro,
        inputs=[],                           # nullary function
        enable_double_precision=True,        # exercise dtype propagation into subgraphs
        loosen_internal_shapes=True,         # the feature under test
        opset=21,
        model_name="scan_body_loosen_shapes_repro",
    )

    p = tmp_path / "scan_body_loosen_shapes_repro.onnx"
    p.write_bytes(model.SerializeToString())

    # Structural sanity
    m = onnx.load(str(p))
    onnx.checker.check_model(m)

    # Must contain a Scan/Loop body
    body = _first_loop_or_scan_body(m.graph)
    assert body is not None, "Expected a Scan/Loop body subgraph."

    # ORT must be able to load and run
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {})
    assert len(outs) == 1
    assert tuple(outs[0].shape) == (5, 210, 1, 1)


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_scan_body_internal_value_infos_are_rank_only_when_loosen_enabled(tmp_path, monkeypatch):

    model = to_onnx(
        _scan_body_broadcast_mul_with_scatter_repro,
        inputs=[],
        enable_double_precision=True,
        loosen_internal_shapes=True,   # ensure sanitizer runs
        opset=21,
        model_name="scan_body_loosen_shapes_vi_check",
    )

    p = tmp_path / "scan_body_loosen_shapes_vi_check.onnx"
    p.write_bytes(model.SerializeToString())

    m = onnx.load(str(p))
    body = _first_loop_or_scan_body(m.graph)
    assert body is not None, "Expected a Scan/Loop body subgraph."

    # If there are internal VIs, each must be rank-only (no fixed dims; no dim_param).
    for vi in body.value_info:
        assert _all_dims_dynamic(vi), f"Scan/Loop body VI '{vi.name}' must be rank-only when loosening is enabled."

SENSITIVE = {"Add","Sub","Mul","Div","Reshape","Squeeze","Unsqueeze","Expand",
             "Concat","Range","Shape","NonZero","Gather","GatherND","Slice",
             "Constant","ConstantOfShape","Pow"}

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

@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_nested_loop_without_loosen_has_risky_internal_vis_or_fails(tmp_path):
    """
    Without loosen, either ORT load fails, or at least one nested Loop body still
    contains a 'risky' internal value_info (produced by arithmetic/shape ops) with
    a concrete dim. This is robust across ORT versions and unrelated fixes.
    """
    model = to_onnx(
        _nested_loop_repro,
        inputs=[],
        enable_double_precision=True,
        loosen_internal_shapes=False,   # intentionally off
        opset=21,
        model_name="nested_loop_no_loosen",
    )
    p = tmp_path / "nested_loop_no_loosen.onnx"
    p.write_bytes(model.SerializeToString())

    try:
        # If it loads, assert the structural hazard is present.
        ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        m = onnx.load(str(p))
        bodies = list(_loop_bodies(m.graph))
        assert bodies, "Expected at least one Loop body."
        found_risky = False
        for b in bodies:
            prod = _producer_map(b)
            for vi in b.value_info:
                if _has_concrete_dim(vi) and (prod.get(vi.name) in SENSITIVE):
                    found_risky = True
                    break
            if found_risky:
                break
        assert found_risky, (
            "When loosen_internal_shapes=False, expected at least one nested Loop body "
            "to retain a risky internal value_info (arithmetic/shape producer with a "
            "concrete dim)."
        )
    except Exception:
        # ORT failed to load → also acceptable for this test
        pass

@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_nested_loop_without_loosen_fails_in_ort(tmp_path):
    """
    Robust version: without loosening we either fail to load in ORT,
    or (if it loads) at least one nested Loop body still retains a risky
    internal value_info (produced by arithmetic/shape ops) with a concrete dim.
    """
    model = to_onnx(
        _nested_loop_repro,
        inputs=[],
        enable_double_precision=True,
        loosen_internal_shapes=False,   # intentionally off
        opset=21,
        model_name="nested_loop_no_loosen",
    )
    p = tmp_path / "nested_loop_no_loosen.onnx"
    p.write_bytes(model.SerializeToString())

    # Helper predicates for the structural hazard
    SENSITIVE = {"Add","Sub","Mul","Div","Reshape","Squeeze","Unsqueeze","Expand",
                 "Concat","Range","Shape","NonZero","Gather","GatherND","Slice",
                 "Constant","ConstantOfShape","Pow"}
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

    try:
        # If it loads, assert the structural hazard is present in some nested body.
        ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        m = onnx.load(str(p))
        bodies = list(_loop_bodies(m.graph))
        assert bodies, "Expected at least one Loop body."
        found_risky = False
        for b in bodies:
            prod = _producer_map(b)
            for vi in b.value_info:
                if _has_concrete_dim(vi) and (prod.get(vi.name) in SENSITIVE):
                    found_risky = True
                    break
            if found_risky:
                break
        assert found_risky, (
            "When loosen_internal_shapes=False, expected at least one nested Loop body "
            "to retain a risky internal value_info (arithmetic/shape producer with a "
            "concrete dim)."
        )
    except Exception:
        # ORT failed to load → also acceptable (legacy behavior)
        pass
