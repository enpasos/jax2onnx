import onnx
import onnxruntime as ort
import jax.numpy as jnp
from jax import lax
from jax2onnx import to_onnx


def _fn_subset_outputs():
    # captured const(s)
    dt1 = jnp.asarray(0.1, dtype=jnp.float64)
    dt2 = jnp.asarray(0.2, dtype=jnp.float64)

    def simulate():
        def step(c, _):
            c = c + dt1 + dt2
            y1 = c
            y2 = 2.0 * c
            return c, (y1, y2)  # carry + 2 scan outs

        # xs=None → lowered to Loop
        _, ys = lax.scan(step, jnp.array(0.0, jnp.float64), xs=None, length=3)
        y1, y2 = ys
        return y1  # return only a subset

    return simulate


def test_loop_consts_and_subset_loads(tmp_path):
    model = to_onnx(
        _fn_subset_outputs(),
        inputs=[],
        enable_double_precision=True,
        model_name="loop_subset",
    )
    p = tmp_path / "loop_subset.onnx"
    onnx.save_model(model, p)

    # should not throw shape-inference / load errors
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {})
    assert out.shape == (3,)
