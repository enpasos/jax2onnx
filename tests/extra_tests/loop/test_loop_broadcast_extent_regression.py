# tests/extra_tests/loop/test_loop_broadcast_extent_regression.py

from __future__ import annotations

import numpy as np
import pytest

from ..helpers import issue52_broadcast_fixture as repro


@pytest.mark.filterwarnings("ignore:.*Removing initializer.*:UserWarning")
def test_broadcast_preserves_stack_extent():
    model = repro.export_model()

    bcast_dims = repro.dims_for("bcast_out_0", model)
    concat_dims = repro.dims_for("jnp_concat_out_0", model)
    loop_dims = repro.dims_for("loop_out_0", model)
    override = repro.loop_axis_override()

    assert bcast_dims, "Expected broadcast metadata."
    assert (
        bcast_dims[0] == repro.STACK_WIDTH
    ), f"Broadcast axis-0 should remain {repro.STACK_WIDTH}, got {bcast_dims}"

    assert concat_dims, "Expected concat metadata."
    assert (
        concat_dims[0] == repro.STACK_WIDTH * 2
    ), f"Concat axis-0 should be {repro.STACK_WIDTH * 2}, got {concat_dims}"

    assert (
        override is not None and override.extent == repro.STACK_WIDTH
    ), "loop_axis0_override lost stack extent."

    if loop_dims and isinstance(loop_dims[0], int):
        assert (
            loop_dims[0] == repro.STACK_WIDTH
        ), f"Loop metadata should not declare 1 in axis-0 (got {loop_dims})."

    assert repro.metadata_ok(model)


@pytest.mark.filterwarnings("ignore:.*Removing initializer.*:UserWarning")
def test_broadcast_model_runs_under_ort(tmp_path):
    ort = pytest.importorskip(
        "onnxruntime", reason="onnxruntime is required to reproduce issue #52"
    )

    model = repro.export_model()
    model_path = tmp_path / "issue52_broadcast.onnx"
    model_path.write_bytes(model.SerializeToString())

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
