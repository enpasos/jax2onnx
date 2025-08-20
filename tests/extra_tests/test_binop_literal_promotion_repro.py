# Repro for mixed-dtype elementwise binop when a weak f64 scalar meets a f32 tensor.
# Current (pre-fix) behavior: ORT fails to load the model because Mul has mismatched inputs.
# After the fix, flip the assertions (or remove the xfail) so we assert that ORT loads and runs.

import numpy as np
import pytest
import onnx
import onnx.checker
from jax2onnx import to_onnx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # ensure 1.5 is weak f64


def _broken():
    # f32 ⊕ i32 -> f32
    float_arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    int_arr = jnp.array([3, 4], dtype=jnp.int32)
    concat_result = jnp.concatenate([float_arr, int_arr])

    # just to exercise some indexing with ints as in the original repro
    lookup = jnp.array([100, 200, 300, 400, 500], dtype=jnp.int32)
    indices = jnp.clip(concat_result.astype(jnp.int32), 0, len(lookup) - 1)
    indexed_vals = jnp.take(lookup, indices)

    # ← problematic: f32 * weak f64 → exporter currently emits Mul(f32, f64)
    float_vals = concat_result * 1.5
    return concat_result, indexed_vals, float_vals


def _save_and_check(tmp_path):
    model = to_onnx(
        _broken,
        inputs=[],
        enable_double_precision=True,
        opset=21,
        model_name="mixed_mul_literal_repro",
    )
    p = tmp_path / "mixed_mul_literal_repro.onnx"
    p.write_bytes(model.SerializeToString())
    m = onnx.load(str(p))
    onnx.checker.check_model(m)
    return p, m


def _type_map(m):
    # Build a name->elem_type map for quick inspection
    types = {}
    for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
        tt = vi.type.tensor_type
        if tt.elem_type:
            types[vi.name] = tt.elem_type
    for init in m.graph.initializer:
        types[init.name] = init.data_type
    return types


def test_repro_has_mixed_mul_inputs_and_ort_fails(tmp_path):
    """Documents the current bug (pre-fix): model contains Mul with mixed dtypes and ORT refuses to load."""
    p, m = _save_and_check(tmp_path)

    # Graph-level assertion: at least one Mul has inputs with different elem types.
    tmap = _type_map(m)
    mul_nodes = [n for n in m.graph.node if n.op_type == "Mul"]
    assert mul_nodes, "Expected at least one Mul node in the graph."

    mismatched = [
        n for n in mul_nodes
        if tmap.get(n.input[0]) is not None
        and tmap.get(n.input[1]) is not None
        and tmap[n.input[0]] != tmap[n.input[1]]
    ]
    assert mismatched, "Expected a Mul with mixed input dtypes (the bug) to be present."

    ort = pytest.importorskip("onnxruntime")
    with pytest.raises(Exception) as excinfo:
        ort.InferenceSession(str(p))
    # The exact message can vary a bit across ORT builds; keep the check loose but targeted.
    msg = str(excinfo.value)
    assert "Mul" in msg and ("bound to different types" in msg or "Type parameter (T)" in msg)


# @pytest.mark.xfail(reason="Will pass after dtype harmonization for elementwise binops is implemented")
def test_expected_future_behavior_ort_loads_and_runs(tmp_path):
    """What we want after the fix: ORT loads and produces the same values as JAX (within tolerance)."""
    p, _ = _save_and_check(tmp_path)

    import onnxruntime as ort  # noqa: F401
    sess = ort.InferenceSession(str(p))

    # run ORT
    ort_outs = [np.asarray(x) for x in sess.run(None, {})]

    # reference from JAX
    jax_outs = [np.asarray(x) for x in _broken()]

    # Compare numerically; ignore dtype differences by upcasting floats for comparison.
    for j, o in zip(jax_outs, ort_outs):
        if np.issubdtype(j.dtype, np.floating) or np.issubdtype(o.dtype, np.floating):
            np.testing.assert_allclose(j.astype(np.float64), o.astype(np.float64), rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_equal(j, o.astype(j.dtype))
