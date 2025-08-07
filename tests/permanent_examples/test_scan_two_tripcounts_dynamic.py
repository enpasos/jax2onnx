# tests/regressions/test_scan_two_tripcounts_dynamic.py
from jax2onnx import to_onnx
from jax2onnx.plugins.jax.lax.scan import _two_scans_diff_len_f32
import onnxruntime as ort

def _assert_ort_loads(model):
    """
    Try to build an ORT inference session for the given in-memory model.
    Raises an AssertionError with the ORT message if it fails.
    """
    try:
        ort.InferenceSession(model.SerializeToString())   # noqa: B018
    except Exception as exc:                              # pragma: no cover
        raise AssertionError(f"ORT refused to load model: {exc}") from None




def test_two_scans_with_different_lengths_can_load_in_ort():
    """
    There are two lax.scan calls that share a *scalar* sequence input
    but have different trip-counts (5 and 100).
    The model must load in ORT – i.e. all leading dimensions of Scan
    outputs / Expand results have to stay **symbolic**.
    """
    model = to_onnx(
        _two_scans_diff_len_f32,
        [],          
        model_name="two_scans_diff" 
    )
    _assert_ort_loads(model)          # ← this is what used to blow up
