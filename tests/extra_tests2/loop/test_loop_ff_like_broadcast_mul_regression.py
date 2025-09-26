from pathlib import Path

import jax.numpy as jnp
from jax import lax
import onnx
import onnxruntime as ort
import pytest

from jax2onnx.user_interface import to_onnx


def _ff_like_loop_broadcast_mul_repro():
    """
    Feed-forward-ish repro:
      - xs=None → lowers to ONNX Loop
      - inside the body we multiply tensors that carry distinct fixed axes (201 and 6)
        so ORT's Loop-body shape inferencer will try to broadcast them.
    """
    const_big = jnp.ones((1, 201, 6, 1, 1), dtype=jnp.float64)  # (1,201,6,1,1)
    vec6 = jnp.full((6,), 0.1, dtype=jnp.float64)  # (6,)

    def body(carry, _):
        b = jnp.reshape(vec6, (1, 6, 1, 1))  # (1,6,1,1) → broadcast against const_big
        y = const_big * b  # broadcast → (1,201,6,1,1)
        return carry, y

    # No scanned inputs: Loop path
    _, ys = lax.scan(body, jnp.asarray(0.0, dtype=jnp.float64), xs=None, length=3)
    return ys


def _first_loop_body(g: onnx.GraphProto):
    for n in g.node:
        if n.op_type == "Loop":
            for a in n.attribute:
                if a.name == "body" and a.g is not None:
                    return a.g
    return None


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_loop_ff_like_broadcast_mul_loads_and_runs(tmp_path: Path):
    # Build ONNX with double precision to match real use-case
    model = to_onnx(
        _ff_like_loop_broadcast_mul_repro,
        inputs=[],  # xs=None pattern
        enable_double_precision=True,
        opset=21,
        model_name="loop_ff_like_broadcast_mul", use_onnx_ir=True
    )
    p = tmp_path / "loop_ff_like_broadcast_mul.onnx"
    p.write_bytes(model.SerializeToString())

    # Structural sanity
    m = onnx.load(str(p))
    onnx.checker.check_model(m)

    # Loop body should contain at least one Mul
    body = _first_loop_body(m.graph)
    assert body is not None, "Expected a Loop body subgraph."
    assert any(
        n.op_type == "Mul" for n in body.node
    ), "Expected a Mul inside Loop body."

    # ORT must be able to load and execute
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {})
    assert isinstance(outs, list) and len(outs) >= 1
    # result is a scan output stacked over Loop length (3). We don't pin exact values.
    assert outs[0] is not None
