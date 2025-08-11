import os
import numpy as np
import onnx
import onnxruntime as ort
import pytest

import jax
import jax.numpy as jnp
from jax import lax

from jax2onnx.converter.user_interface import to_onnx


def _loop_body_broadcast_mul_minimal():
    """
    Minimal scan(xs=None) that ensures the emitted ONNX Loop body contains
    a broadcasted Mul. This mirrors the shape-inference-sensitive pattern
    from the original use case but keeps the body simple and deterministic.
    """
    W = jnp.arange(36.0, dtype=jnp.float64).reshape(6, 6, 1, 1)  # (6,6,1,1)

    def body(carry, _):
        x = jnp.full((6, 1, 1), 0.1, dtype=jnp.float64)  # (6,1,1)
        y = W * x                                        # (6,6,1,1) broadcast
        out = jnp.squeeze(y, axis=(2, 3))                # (6,6)
        return carry + 1, out

    _, ys = lax.scan(body, 0, xs=None, length=5)  # → ONNX Loop
    return ys  # (5,6,6)


def _get_first_loop_body(g: onnx.GraphProto) -> onnx.GraphProto | None:
    for n in g.node:
        if n.op_type == "Loop":
            for a in n.attribute:
                if a.name == "body" and a.g is not None:
                    return a.g
    return None


def _is_rank_only_tensor(vi: onnx.ValueInfoProto) -> bool:
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return True
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            return False
    return True


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_loop_body_broadcast_mul_loads_and_runs_in_ort(tmp_path):
    model = to_onnx(
        _loop_body_broadcast_mul_minimal,
        inputs=[],                    # xs=None pattern
        enable_double_precision=True, # matches use case precision
        opset=21,
        model_name="loop_body_broadcast_mul_minimal",
    )

    p = tmp_path / "loop_body_broadcast_mul_minimal.onnx"
    p.write_bytes(model.SerializeToString())

    # Structural sanity
    m = onnx.load(str(p))
    onnx.checker.check_model(m)

    # ORT must load and run
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    [out_name] = [o.name for o in sess.get_outputs()]
    ort_out = sess.run([out_name], {})[0]

    # Compare numerics against JAX reference
    ref = np.asarray(_loop_body_broadcast_mul_minimal(), dtype=np.float64)
    np.testing.assert_allclose(ort_out, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_loop_body_state_ios_are_rank_only(tmp_path):
    model = to_onnx(
        _loop_body_broadcast_mul_minimal,
        inputs=[],
        enable_double_precision=True,
        opset=21,
        model_name="loop_body_state_rank_only_check",
    )

    p = tmp_path / "loop_body_state_rank_only_check.onnx"
    p.write_bytes(model.SerializeToString())

    m = onnx.load(str(p))
    body = _get_first_loop_body(m.graph)
    assert body is not None, "Expected a Loop body subgraph."

    # Loop body inputs: [iter, cond] + M state
    assert len(body.input) >= 2
    M = len(body.input) - 2

    # Check ONLY the state inputs and their corresponding state outputs are rank-only
    # Body outputs are [cond_out] + M state outs + K scan outs
    assert len(body.output) >= 1 + M

    # State inputs rank-only
    for vi in body.input[2:]:
        assert _is_rank_only_tensor(vi), f"Loop body state input '{vi.name}' must be rank-only."

    # State outputs rank-only (skip cond_out at index 0)
    for vi in body.output[1 : 1 + M]:
        assert _is_rank_only_tensor(vi), f"Loop body state output '{vi.name}' must be rank-only."
