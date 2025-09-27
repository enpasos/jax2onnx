from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import lax

from jax._src.export.shape_poly import InconclusiveDimensionOperation

import onnx

from jax2onnx.user_interface import to_onnx

try:
    import onnxruntime as ort

    HAS_ORT = True
except Exception:  # pragma: no cover
    HAS_ORT = False


def _model_with_dup_shape_of(x):
    def body(_, a):
        add = a + 1.0
        b1 = jnp.broadcast_to(0.0, (add.shape[0], add.shape[1]))
        z1 = add + b1
        b2 = jnp.broadcast_to(1.0, (add.shape[0], add.shape[1]))
        z2 = add + b2
        return z1 + z2

    return lax.fori_loop(0, 2, body, x)


def _first_loop_body(model: onnx.ModelProto):
    for node in model.graph.node:
        if node.op_type == "Loop":
            for attr in node.attribute:
                if attr.name == "body":
                    return attr.g
    return None


@pytest.mark.parametrize("shape", [("B", 4)])
def test_loop_body_shapeof_ssa_and_ort_load(tmp_path, shape):
    spec = jax.ShapeDtypeStruct(shape, jnp.float32)
    try:
        model = to_onnx(
            _model_with_dup_shape_of,
            inputs=[spec],
            opset=21,
            model_name="loop_body_shapeof_ssa_strict_ir",
            use_onnx_ir=True,
        )
    except InconclusiveDimensionOperation as exc:
        pytest.xfail(f"converter2 cannot export loops with symbolic dims: {exc}")
    out_path = tmp_path / "loop_body_shapeof_ssa_strict_ir.onnx"
    out_path.write_bytes(model.SerializeToString())
    loaded = onnx.load(str(out_path))
    onnx.checker.check_model(loaded)

    loop_body = _first_loop_body(loaded)
    assert loop_body is not None

    outputs_seen: set[str] = set()
    duplicates: list[str] = []
    for node in loop_body.node:
        for output in node.output:
            if output in outputs_seen:
                duplicates.append(output)
            outputs_seen.add(output)
    assert not duplicates, f"Loop body not in SSA; duplicate outputs: {duplicates}"

    if not HAS_ORT:
        pytest.skip("onnxruntime not available")
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    B = 3
    x = np.random.randn(B, 4).astype(np.float32)
    y_ref = np.asarray(_model_with_dup_shape_of(x))
    (y_onnx,) = sess.run(None, {sess.get_inputs()[0].name: x})
    np.testing.assert_allclose(y_ref, y_onnx, rtol=1e-5, atol=1e-6)
