# tests/permanent_examples/test_scan_shared_scalar_two_lengths.py
import pathlib
import jax.numpy as jnp
from jax import lax
from jax2onnx import to_onnx
import onnxruntime as ort
import onnx


def _two_scans_shared_scalar():
    xs5 = jnp.arange(5, dtype=jnp.float32)
    xs100 = jnp.arange(100, dtype=jnp.float32)
    dt = jnp.asarray(0.1, dtype=jnp.float32)  # rank-0 scalar, *captured* in the body

    def body(c, x):
        return (c + x + dt, c)  # dt is closed over – no second xs needed

    _, y5 = lax.scan(body, 0.0, xs5)
    _, y100 = lax.scan(body, 0.0, xs100)
    return y5, y100


def test_shared_scalar_two_lengths(tmp_path: pathlib.Path):
    onnx_path = tmp_path / "two_scans_shared_scalar.onnx"

    model = to_onnx(
        _two_scans_shared_scalar,
        inputs=[],  # no run-time inputs
        model_name="two_scans_shared_scalar",
        opset=21,
    )
    onnx.save_model(model, onnx_path)

    # model must load after the Scan-plugin fix
    ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
