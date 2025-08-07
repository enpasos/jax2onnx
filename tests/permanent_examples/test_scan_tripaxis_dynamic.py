# tests/regression/test_scan_tripaxis_dynamic.py
import numpy as np
import jax.numpy as jnp
from jax import lax
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import to_onnx   # public user-facing helper

import onnxruntime as ort
from jax2onnx import to_onnx
from jax2onnx.plugins.jax.lax.scan import _two_scans_len_mismatch_broadcast_f32

import onnxruntime as ort
from jax2onnx import to_onnx
from jax2onnx.plugins.jax.lax.scan import _two_scans_len_mismatch_broadcast_f32

def test_tripaxis_dynamic_loads():
    model = to_onnx(_two_scans_len_mismatch_broadcast_f32,
                    [],
                    model_name="two_scans_broadcast")
    # must not raise
    ort.InferenceSession(model.SerializeToString())


def _two_scans_len_mismatch():
    """
    First Scan runs 5 steps, second Scan 100 steps.
    The stacked outputs are *not* returned by the function
    (they become “scan_unused_output_*” value-infos in ONNX).
    """
    xs_small = jnp.arange(5,   dtype=jnp.float32)
    xs_big   = jnp.arange(100, dtype=jnp.float32)

    # body returns (carry, stacked_out)  – stacked_out is ignored later
    body = lambda c, x: (c + x, c)   # noqa: E731

    carry, _ = lax.scan(body, 0.0, xs_small)   # trip-count 5
    carry, _ = lax.scan(body, carry, xs_big)   # trip-count 100
    return carry                                # scalar

@pytest.mark.order(-1)  # run *after* the models have been produced
@pytest.mark.parametrize("enable_double_precision", [False])  # one variant is enough
def test_tripaxis_is_dynamic(tmp_path, enable_double_precision):
    """Model must load in ORT – i.e. no dim-value clash across Scan nodes."""
    onnx_path = tmp_path / "two_scans_dynamic.onnx"

    # --- export ---
    model = to_onnx(
        _two_scans_len_mismatch,
        [],                # no runtime inputs 
        model_name="two_scans_dynamic",
        enable_double_precision=enable_double_precision,
    )
    onnx.save_model(model, onnx_path)

    # --- sanity: checker must accept it ---
    onnx.checker.check_model(model)   # would fail with the old, “fixed-dim” code

    # --- real regression guard: ORT session must be creatable ---
    sess = ort.InferenceSession(str(onnx_path))          # noqa: F841

    # additionally assert that every “scan_unused_output” has a *dynamic* dim-param
    for vi in model.graph.value_info:
        if vi.name.startswith("scan_unused_output"):
            first_dim = vi.type.tensor_type.shape.dim[0]
            assert not first_dim.HasField("dim_value"), (
                f"{vi.name} trip-axis is still static ({first_dim.dim_value})"
            )



