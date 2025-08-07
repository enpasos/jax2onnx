# tests/permanent_examples/test_scan_dynamic_len.py
import onnxruntime as ort
from jax2onnx import to_onnx
from jax2onnx.plugins.jax.lax.scan import _two_scans_diff_len_f32

def test_scan_with_two_trip_counts_loads_in_ort():
    model = to_onnx(
        _two_scans_diff_len_f32,
        [],          # fully static graph
        model_name="two_scans_diff",
        opset=21,
    )
    # Must not raise ShapeInferenceError anymore
    ort.InferenceSession(model.SerializeToString())  # noqa: B018
