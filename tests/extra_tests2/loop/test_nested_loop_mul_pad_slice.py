# tests/regression/test_nested_loop_mul_pad_slice.py
import jax
import jax.numpy as jnp
import onnx
import pathlib
import pytest
from jax2onnx.user_interface import to_onnx
import onnxruntime as ort


def pad_then_center(x):
    # Typical halo: pad then crop back to original shape; ORT will care about the
    # value_info shape you assign to these in the Loop body.
    xh = jnp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))  # B,C,H+2,W+2
    xc = xh[:, :, 1:-1, 1:-1]  # B,C,H,W
    return xc


def make_fn():
    def inner(carry, _):
        a = carry
        b = pad_then_center(a)  # same runtime shape as a
        z = a * b  # elementwise in inner loop
        return a, z

    def outer(carry, _):
        carry, z = jax.lax.scan(inner, carry, None, length=2)
        return carry, z

    def fn(a):
        _, z = jax.lax.scan(outer, a, None, length=1)  # nested Loop
        return z

    return fn


@pytest.mark.xfail
def test_nested_loop_mul_pad_slice_loads(tmp_path: pathlib.Path):
    x = jnp.ones((1, 3, 8, 8), jnp.float64)
    fn = make_fn()
    model = to_onnx(
        fn,
        inputs=[x],
        enable_double_precision=True,
        model_name="nested_pad_slice_mul",
        use_onnx_ir=True,
    )
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    p = tmp_path / "m.onnx"
    p.write_bytes(model.SerializeToString())
    # Must load in ORT (no TypeInferenceError inside Loop body)

    ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
