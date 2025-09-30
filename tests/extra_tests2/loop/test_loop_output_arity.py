import pathlib
import onnx
import onnxruntime as ort
import jax.numpy as jnp
from jax import lax

from jax2onnx.user_interface import to_onnx


def _assert_loops_ok(model: onnx.ModelProto) -> None:
    """
    Body has outputs: [cond_out, state..., scan...]
    Loop node must expose exactly len(body.output) - 1 outputs.
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
        required = len(body.output) - 1
        actual = len(n.output)
        assert (
            actual == required
        ), f"Loop/body output arity mismatch: node has {actual}, body produces {required}"


def _make_fn_only_uses_subset_of_loop_outputs():
    # Body returns (carry, (y1, y2)), but we only return y1 from the top-level fn.
    def simulate():
        def step(c, _):
            c = c + 1.0
            y1 = c
            y2 = 2.0 * c
            return c, (y1, y2)

        # xs=None → lowered to Loop
        _, ys = lax.scan(step, jnp.array(0.0, dtype=jnp.float32), xs=None, length=3)
        y1, y2 = ys
        return y1  # drop carry and y2

    return simulate


def test_loop_output_arity_ok(tmp_path: pathlib.Path):
    fn = _make_fn_only_uses_subset_of_loop_outputs()
    model = to_onnx(
        fn, inputs=[], model_name="loop_out_arity", opset=21, use_onnx_ir=True
    )

    # Structural check
    _assert_loops_ok(model)

    # Round-trip through ORT (this used to fail with "Output N is out of bounds")
    p = tmp_path / "loop_out_arity.onnx"
    onnx.save_model(model, p)
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    # quick numeric: expect [1., 2., 3.]
    (got,) = sess.run(None, {})
    assert got.shape == (3,)
