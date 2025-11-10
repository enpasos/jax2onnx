# tests/extra_tests/loop/test_loop_concat_extent_regression.py

from __future__ import annotations

import numpy as np
import pytest

from ..helpers import issue52_loop_concat_fixture as repro


@pytest.mark.filterwarnings("ignore:.*Removing initializer.*:UserWarning")
def test_loop_concat_preserves_stack_extent():
    model = repro.export_model()

    loop_dims = repro.dims_for("loop_out_0", model)
    squeeze_dims = repro.dims_for("squeeze_out_0", model)

    assert squeeze_dims, "Expected squeeze_out_0 metadata."
    assert (
        squeeze_dims[0] == repro.STACK_WIDTH
    ), f"squeeze axis-0 should remain {repro.STACK_WIDTH}, got {squeeze_dims}"

    override = repro.loop_axis_override()
    assert (
        override is not None and override.extent == repro.STACK_WIDTH
    ), "IR loop metadata lost the 5-wide extent."

    if loop_dims:
        dim0 = loop_dims[0]
        if isinstance(dim0, int):
            assert dim0 == repro.STACK_WIDTH
        else:
            assert isinstance(dim0, str) and dim0 != "1"

    assert repro.metadata_ok(model)


def test_loop_concat_embeddings_roundtrip(tmp_path):
    """
    Basic smoke test that the ONNX model still round-trips through ORT once the
    metadata stays truthful.
    """

    onnx_model = repro.export_model()
    model_path = tmp_path / "issue52_loop_concat.onnx"
    model_path.write_bytes(onnx_model.SerializeToString())

    # Ensure the exported model loads under ORT without the forced mismatch.
    ort = pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required to reproduce issue #52"
    )
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    feeds = {}
    for inp in session.get_inputs():
        shape = tuple(
            repro.STACK_WIDTH if (not isinstance(dim, int) or dim <= 0) else dim
            for dim in inp.shape
        )
        dtype = np.float64 if inp.type == "tensor(double)" else np.float32
        feeds[inp.name] = np.ones(shape, dtype=dtype)
    session.run(None, feeds)
