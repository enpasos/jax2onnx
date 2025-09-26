import onnx
import onnxruntime as ort
import jax.numpy as jnp
from jax import lax

from jax2onnx.user_interface import to_onnx


def _fn_subset_outputs():
    dt1 = jnp.asarray(0.1, dtype=jnp.float64)
    dt2 = jnp.asarray(0.2, dtype=jnp.float64)

    def simulate():
        def step(carry, _):
            carry = carry + dt1 + dt2
            y1 = carry
            y2 = 2.0 * carry
            return carry, (y1, y2)

        _, ys = lax.scan(step, jnp.array(0.0, jnp.float64), xs=None, length=3)
        y1, _ = ys
        return y1

    return simulate


def test_loop_consts_and_subset_loads(tmp_path):
    model = to_onnx(
        _fn_subset_outputs(),
        inputs=[],
        enable_double_precision=True,
        model_name="loop_subset_ir",
        use_onnx_ir=True,
    )
    out_path = tmp_path / "loop_subset_ir.onnx"
    onnx.save_model(model, out_path)
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {})
    assert out.shape == (3,)
