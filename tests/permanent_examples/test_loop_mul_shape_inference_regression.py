import jax
import jax.numpy as jnp
import onnx
import onnxruntime as ort
import pytest

from jax2onnx.converter.conversion_api import to_onnx


def _ff_like_loop_no_xs_mul():
    """
    Minimal reproduction skeleton for the ORT error:
      Node (Loop_0) ... Graph attribute inferencing failed:
      Node (mul_*) Op (Mul) [ShapeInferenceError] Incompatible dimensions

    Adjust this function to mirror your use case (length=201, 6x6 grids, etc.).
    Keep xs=None so we lower to ONNX Loop.
    """
    L = 201

    # pretend "grid weights" etc. Shape choices mirror your logs.
    # tune these constants to your real case if needed:
    table_f32 = jnp.ones((6, 6, 1, 1), dtype=jnp.float32)
    table_f64 = table_f32.astype(jnp.float64)  # mixed dtypes in body

    # Initial carry – pick something with 6x6x1x1 so broadcasts engage
    carry0 = jnp.zeros((6, 6, 1, 1), dtype=jnp.float64)

    def body(c, _):
        # Shape/path that stresses ONNX shape inference.
        # The gather/reshape/expand chain emulates indexing & broadcast
        # you see before the failing Mul in the external model.
        #
        # NOTE: If your real model uses Scatter/GatherND, swap them in here.
        idx = jnp.array([0], dtype=jnp.int64)  # stand-in index
        picked = jnp.take(table_f64, idx, axis=0)  # (1, 6, 1, 1) after take
        picked = jnp.reshape(picked, (1, 6, 1, 1))

        # Another branch that keeps a fixed 6x6 table so broadcasting must align
        other = table_f64  # (6, 6, 1, 1)

        # Broadcast picked to something that *should* broadcast with other
        # (introduce a leading dim to emulate batch/step mismatches)
        picked_b = jnp.broadcast_to(picked, (1, 6, 1, 1))

        # This Mul is the one we want ORT to trip on (pre-fix).
        y = picked_b * other

        # keep the carry flowing in double
        return c + 0.0, y  # carry, per-iter output

    # xs=None → lowers to ONNX Loop
    _, ys = jax.lax.scan(body, carry0, xs=None, length=L)
    return ys  # stacked output of shape (L, 1, 6, 1, 1) in eager JAX


def test_loop_mul_broadcast_in_loop_ort_fails(tmp_path):
    model = to_onnx(
        _ff_like_loop_no_xs_mul,
        inputs=[],
        enable_double_precision=True,  # match the failing use case
        opset=21,
        model_name="loop_mul_inference_repro",
    )

    p = tmp_path / "loop_mul_inference_repro.onnx"
    p.write_bytes(model.SerializeToString())

    # Ensure the ONNX model is structurally sound first
    m = onnx.load(str(p))
    onnx.checker.check_model(m)

    # ORT must fail to load this model (broadcasted Mul inside Loop with mixed dtypes).
    # Assert the failure rather than marking the test xfail.
    with pytest.raises(Exception) as excinfo:
        ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    msg = str(excinfo.value)
    # Accept either a type mismatch or (older) inference errors.
    assert (
        "Type Error" in msg
        or "TypeInferenceError" in msg
        or "ShapeInferenceError" in msg
    ), f"Unexpected ORT error message:\n{msg}"
