from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import onnx
import onnxruntime as ort
import pathlib
import pytest

from jax import lax
from jax2onnx import to_onnx


def _assert_all_loops_well_formed(model: onnx.ModelProto):
    """
    ONNX invariant for Loop: number of node outputs must equal
    (len(body.output) - 1). The extra -1 is for the `cond` output of the body.
    """
    for n in model.graph.node:
        if n.op_type != "Loop":
            continue
        body = None
        for a in n.attribute:
            if a.name == "body":
                body = a.g
                break
        assert body is not None, "Loop node missing body graph"

        # Body inputs: [iter_count, cond_in, <carried...>]
        # Body outputs: [cond_out, <carried...>, <scan outputs...>]
        expected_num_node_outputs = len(body.output) - 1
        assert (
            len(n.output) == expected_num_node_outputs
        ), f"Loop outputs ({len(n.output)}) != body outputs - 1 ({expected_num_node_outputs})"


@pytest.mark.timeout(60)
def test_scan_no_xs_subset_outputs_and_pruned_carry(tmp_path: pathlib.Path):
    """
    Regression: num_scan == 0 → we lower to ONNX Loop.
    The body produces many 'scan outputs', but at the call-site we drop
    the carry and keep only a subset of those outputs. Older code would
    create the Loop node with too-few outputs, tripping ONNX shape inference.
    """

    @jax.jit  # encourage JAX to prune unused results from the primitive call
    def ff_subset():
        def body(c, _):
            # produce several per-iter outputs so Loop has multiple scan outputs
            y1 = c
            y2 = c + 1.0
            y3 = c * 2.0
            y4 = c - 3.0
            y5 = c / 4.0
            y6 = c**2
            return c + 1.0, (y1, y2, y3, y4, y5, y6)

        c0 = jnp.array(0.0, dtype=jnp.float64)
        # IMPORTANT: do NOT bind the carry result at all (makes it dead)
        ys = lax.scan(body, c0, xs=None, length=4)[1]
        # keep only a subset of the scan outputs
        return ys[1], ys[4]  # (y2, y5)

    # Reference JAX
    y2_ref, y5_ref = ff_subset()
    y2_ref = np.asarray(jax.device_get(y2_ref))
    y5_ref = np.asarray(jax.device_get(y5_ref))

    # Convert to ONNX (shape inference runs during our post-pass)
    model = to_onnx(
        ff_subset,
        inputs=[],  # no runtime inputs → hits Loop path
        opset=21,
        enable_double_precision=True,
        model_name="scan_loop_pruned",
    )

    # Structural invariant: Loop node outputs must match body arity
    _assert_all_loops_well_formed(model)

    # Save + load in ORT (this would crash before the fix)
    onnx_path = tmp_path / "scan_loop_pruned.onnx"
    onnx.save_model(model, onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    y2_onnx, y5_onnx = sess.run(None, {})
    y2_onnx, y5_onnx = np.asarray(y2_onnx), np.asarray(y5_onnx)

    assert np.allclose(y2_onnx, y2_ref, rtol=1e-7, atol=1e-12)
    assert np.allclose(y5_onnx, y5_ref, rtol=1e-7, atol=1e-12)
