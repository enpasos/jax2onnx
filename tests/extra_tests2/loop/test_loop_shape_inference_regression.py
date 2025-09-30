from typing import Optional
import onnx
import jax.numpy as jnp
from jax import lax
import pytest

from jax2onnx.user_interface import to_onnx


def _fn_loop_body_indexing_mul():
    """
    Construct a scan with xs=None → lowered to ONNX Loop.
    Body does:
        row = big_table[idx0]   # indexing/gather → loses the '201' dim
        y   = row * kernel      # elementwise op that used to trip ORT
    Shapes (conceptually):
        big_table: (201, 6, 6, 1, 1)
        row:       (6,   6, 1, 1)
        kernel:    (6,   6, 1, 1)
    """
    big_table = jnp.zeros((201, 6, 6, 1, 1), dtype=jnp.float64)
    kernel = jnp.ones((6, 6, 1, 1), dtype=jnp.float64)

    def body(carry, _):
        row = big_table[0]  # Gather/Slice-like path in the body
        z = row * kernel  # Elementwise op — ORT used to see incompatible dims
        return carry + 1.0, z

    # xs=None → Loop lowering path
    _, ys = lax.scan(body, 0.0, xs=None, length=2)
    return ys


def _all_dims_dynamic(vi: onnx.ValueInfoProto) -> bool:
    """True if every dimension is dynamic (no fixed dim_value)"""
    tt = vi.type.tensor_type
    if not tt.HasField("shape"):
        return True
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            return False
    return True


def _assert_loop_body_state_rank_only(g: onnx.GraphProto) -> None:
    """Find Loop nodes and assert their body state inputs are rank-only."""
    for n in g.node:
        if n.op_type != "Loop":
            continue
        body: Optional[onnx.GraphProto] = None
        for a in n.attribute:
            if a.name == "body" and a.g is not None:
                body = a.g
                break
        assert body is not None, "Loop without a body subgraph?"

        # Loop body inputs: [iter, cond] + M state
        assert len(body.input) >= 2
        for vi in body.input[2:]:
            assert _all_dims_dynamic(
                vi
            ), f"Loop body state input '{vi.name}' should be rank-only."


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_loop_body_rank_only_and_ort_loads(tmp_path):
    # Build ONNX; keep double precision path on to match original use case.
    model = to_onnx(
        _fn_loop_body_indexing_mul,
        inputs=[],  # xs=None case
        enable_double_precision=True,  # matches failing scenario
        opset=21,
        model_name="loop_rank_only_shapes",
        use_onnx_ir=True,
    )
    p = tmp_path / "loop_rank_only_shapes.onnx"
    p.write_bytes(model.SerializeToString())

    # Structural check: body state I/O are rank-only (dynamic dims)
    m = onnx.load(str(p))
    _assert_loop_body_state_rank_only(m.graph)

    # Runtime check: ORT should accept the model (no shape/type inference error).
    ort = pytest.importorskip("onnxruntime")
    _ = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
