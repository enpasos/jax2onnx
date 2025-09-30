import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jax import lax
from jax._src.export.shape_poly import InconclusiveDimensionOperation
from jax2onnx.user_interface import to_onnx


def _loop_model(x):
    """
    A tiny loop that:
      * broadcasts using a runtime dimension (a.shape[0])
      * does two adds (creating multiple places that may request Shape(Add))
      * reshapes using (a.shape[0], -1) and back to (a.shape[0], 4)
    This triggers repeated shape-of requests within the body.
    """

    def body(i, a):
        b = jnp.broadcast_to(1.0, (a.shape[0], 4))  # uses runtime shape
        c = a + b  # "Add" #1
        f = jnp.reshape(c, (a.shape[0], -1))  # uses runtime shape again
        g = jnp.reshape(f, (a.shape[0], 4))
        h = jnp.broadcast_to(0.0, (a.shape[0], 4))
        y = g + h  # "Add" #2
        return y

    return lax.fori_loop(0, 2, body, x)


class TestLoopBodySSADedup:
    def test_loop_body_is_ssa_and_loads(self, tmp_path):
        # Dynamic first dim to force runtime Shape() use
        spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)

        try:
            model = to_onnx(
                _loop_model,
                inputs=[spec],
                opset=21,
                model_name="loop_body_ssa_dedup",
            )
        except InconclusiveDimensionOperation as exc:
            pytest.xfail(f"converter2 cannot export loops with symbolic dims: {exc}")
        onnx_path = tmp_path / "m.onnx"
        onnx_path.write_bytes(model.SerializeToString())

        # --- Check SSA inside the Loop body (no duplicate output names)
        import onnx

        m = onnx.load_model(str(onnx_path))

        loop_body = None
        for n in m.graph.node:
            if n.op_type == "Loop":
                for attr in n.attribute:
                    if attr.name == "body":
                        loop_body = attr.g
                        break
        assert loop_body is not None, "Loop body not found in exported model."

        outputs_seen = set()
        duplicates = []
        for n in loop_body.node:
            for out in n.output:
                if out in outputs_seen:
                    duplicates.append(out)
                outputs_seen.add(out)
        assert not duplicates, f"Loop body not in SSA; duplicate outputs: {duplicates}"

        # --- Optional: ONNX Runtime load & numeric check
        try:
            import onnxruntime as ort
        except Exception:
            pytest.skip("onnxruntime not installed")

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        B = 3
        x = np.random.randn(B, 4).astype(np.float32)

        y_ref = np.asarray(_loop_model(x))
        y_onnx = sess.run(None, {sess.get_inputs()[0].name: x})[0]

        np.testing.assert_allclose(y_ref, y_onnx, rtol=1e-5, atol=1e-6)
