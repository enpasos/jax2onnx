import jax
import jax.numpy as jnp
import onnx
from jax2onnx import to_onnx


def _simple_loop_fn(y0):
    # Tiny loop whose body works on a non-scalar carry (so the body has array I/Os).
    def body(i, carry):
        # any elementwise op is fine; we only care about the presence of the array carry
        return carry * 1.0

    return jax.lax.fori_loop(0, 2, body, y0)


def _is_rank_only_tensor_type(tp: onnx.onnx_ml_pb2.TypeProto) -> bool:
    if not tp.HasField("tensor_type"):
        return False
    tt = tp.tensor_type
    if not tt.HasField("shape"):
        return False
    # rank-only means dims exist but all dims lack both dim_value and dim_param
    if len(tt.shape.dim) == 0:
        return False  # scalar
    for d in tt.shape.dim:
        if d.HasField("dim_value") or d.HasField("dim_param"):
            return False
    return True


def _collect_loop_body_graph(model: onnx.ModelProto):
    for n in model.graph.node:
        if n.op_type == "Loop":
            for a in n.attribute:
                if a.name == "body":
                    return a.g
    return None


def test_loop_body_io_are_relaxed_when_requested():
    # concrete input (no symbolic dims) to mirror integration more closely
    y0 = jnp.ones((2, 3, 4), dtype=jnp.float64)
    model = to_onnx(
        _simple_loop_fn,
        inputs=[y0],
        enable_double_precision=True,
        opset=21,
        model_name="loop_body_io_relaxation",
    )

    body = _collect_loop_body_graph(model)
    assert body is not None, "Loop body graph not found"

    # According to ONNX Loop: inputs = [iter_num, cond_in, carried...]
    # outputs = [cond_out, carried...]
    # We expect all non-scalar array inputs/outputs (beyond control scalars) to be rank-only.
    assert len(body.input) >= 3
    assert len(body.output) >= 2

    # inputs: idx 0 = iter_num (scalar), idx 1 = cond_in (scalar), idx>=2 are arrays
    for idx, vi in enumerate(body.input):
        if idx < 2:
            # control scalars must be left intact (scalars, not rank-only)
            assert (
                vi.type.tensor_type.HasField("shape")
                and len(vi.type.tensor_type.shape.dim) == 0
            )
        else:
            assert _is_rank_only_tensor_type(
                vi.type
            ), f"body.input[{idx}] should be rank-only"

    # outputs: idx 0 = cond_out (scalar), idx>=1 are arrays
    for idx, vi in enumerate(body.output):
        if idx == 0:
            assert (
                vi.type.tensor_type.HasField("shape")
                and len(vi.type.tensor_type.shape.dim) == 0
            )
        else:
            assert _is_rank_only_tensor_type(
                vi.type
            ), f"body.output[{idx}] should be rank-only"
