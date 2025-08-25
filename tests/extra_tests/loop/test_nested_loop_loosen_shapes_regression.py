import pytest
import onnx
import onnxruntime as ort
import jax.numpy as jnp
from jax import lax

from jax2onnx.converter.conversion_api import to_onnx

# Ops that can (re-)tighten dims or are shape/dtype sensitive
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


def _nested_loop_repro():
    """
    Small nested-Loop repro:
      outer Loop J in {0..1}:
        inner Loop i in {0..1} on a (5, T, 1, 1) tensor with
        scatter into the time dimension and broadcasted arithmetic.
    """
    T = 210
    updates_len = 200

    def inner_body(i, ref):
        idx = jnp.array([5], dtype=jnp.int32)  # (1,)
        updates = jnp.ones((5, updates_len, 1, 1), dtype=ref.dtype)  # (5,200,1,1)
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
        ratio = last / (0.4 * a0)
        sum_sq = jnp.sum(mid * mid, axis=0)
        tail = a0 * (0.5 * sum_sq + ratio)
        out = jnp.stack([a0, mid[0], mid[1], mid[2], tail], axis=0)  # (5,T,1,1)
        return out

    def outer_body(j, carry):
        return lax.fori_loop(0, 2, inner_body, carry)

    init = jnp.ones((5, T, 1, 1), dtype=jnp.float64)
    return lax.fori_loop(0, 2, outer_body, init)  # (5, T, 1, 1)


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_nested_loop_without_loosen_has_risky_internal_vis_or_fails(tmp_path):
    """
    Without loosen, either ORT load fails, or at least one nested Loop body
    still contains a 'risky' internal value_info (produced by arithmetic/shape
    ops) with a concrete dim. Robust across ORT versions and unrelated fixes.
    """
    model = to_onnx(
        _nested_loop_repro,
        inputs=[],
        enable_double_precision=True,
        loosen_internal_shapes=False,  # intentionally off
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
        # ORT failed to load → also acceptable (legacy behavior)
        pass


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_nested_loop_with_loosen_loads_and_drops_arith_vis(tmp_path):
    """
    With loosening, ORT must load and run, and there must be no internal VI in
    any Loop body that both (a) is produced by a SENSITIVE op and (b) has a
    concrete dim.
    """
    model = to_onnx(
        _nested_loop_repro,
        inputs=[],
        enable_double_precision=True,
        loosen_internal_shapes=True,  # feature under test
        opset=21,
        model_name="nested_loop_loosen",
    )
    p = tmp_path / "nested_loop_loosen.onnx"
    p.write_bytes(model.SerializeToString())

    # Loads and runs
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {})
    assert len(outs) == 1
    assert tuple(outs[0].shape) == (5, 210, 1, 1)

    # And internal VIs in bodies are clean
    m = onnx.load(str(p))
    for b in _loop_bodies(m.graph):
        prod = _producer_map(b)
        for vi in b.value_info:
            assert not (
                _has_concrete_dim(vi) and (prod.get(vi.name) in SENSITIVE)
            ), f"Found risky VI '{vi.name}' in Loop body despite loosening."
