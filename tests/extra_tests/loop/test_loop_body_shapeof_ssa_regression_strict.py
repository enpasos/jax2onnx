import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax2onnx import to_onnx


def _model_with_dup_shape_of(x):
    """
    Body computes `add = a + 1.0` and then performs TWO independent
    broadcast_to operations that both consume `add.shape`.
    This forces the converter to request `Shape(add)` twice.
    Pre-fix this produced duplicated output names (e.g. 'Add__shape').
    """

    def body(i, a):
        add = a + 1.0
        # two dynamic broadcasts using add.shape
        b1 = jnp.broadcast_to(0.0, (add.shape[0], add.shape[1]))
        z1 = add + b1
        b2 = jnp.broadcast_to(1.0, (add.shape[0], add.shape[1]))
        z2 = add + b2
        return z1 + z2

    return lax.fori_loop(0, 2, body, x)


class TestLoopBodyShapeOfSSARegressionStrict:
    def test_loop_body_shapeof_ssa_and_ort_load(self, tmp_path):
        # dynamic first dim to enforce runtime Shape()
        spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)

        model = to_onnx(
            _model_with_dup_shape_of,
            inputs=[spec],
            opset=21,
            model_name="loop_body_shapeof_ssa_strict",
        )
        onnx_path = tmp_path / "m.onnx"
        onnx_path.write_bytes(model.SerializeToString())

        # Parse ONNX and check Loop body SSA: no duplicate output names.
        import onnx

        m = onnx.load_model(str(onnx_path))

        loop_body = None
        for n in m.graph.node:
            if n.op_type == "Loop":
                for attr in n.attribute:
                    if attr.name == "body":
                        loop_body = attr.g
                        break
        assert loop_body is not None, "Loop body not found."

        outputs_seen = set()
        dups = []
        for n in loop_body.node:
            for out in n.output:
                if out in outputs_seen:
                    dups.append(out)
                outputs_seen.add(out)
        assert not dups, f"Loop body not in SSA; duplicate outputs: {dups}"

        # ORT should load (pre-fix it raised INVALID_GRAPH about 'Add__shape').
        try:
            import onnxruntime as ort
        except Exception:
            pytest.skip("onnxruntime not installed")

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        # numeric sanity
        B = 3
        x = np.random.randn(B, 4).astype(np.float32)
        y_ref = np.asarray(_model_with_dup_shape_of(x))
        y_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        np.testing.assert_allclose(y_ref, y_onnx, rtol=1e-5, atol=1e-6)
