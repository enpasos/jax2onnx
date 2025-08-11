import numpy as _np
import onnx
import onnxruntime as ort
import jax.numpy as jnp
from jax import lax
import pytest

from jax2onnx import to_onnx


def _fn_feed_forward_like_nested():
    """
    Minimal nested-loop reproducer:
      * inner Loop indexes an f32 table (Slice/GatherND/Gather),
      * immediately adds it to an f64 accumulator (mixed-dtype Add),
      * outer Loop selects the last inner sum (still inside Loop bodies).
    This matches the external failure where ORT expected DOUBLE on an
    indexing-produced tensor inside a Loop body due to mixed-dtype Add.
    """
    table_f32 = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)

    def inner(carry_i: jnp.int32, _):
        # index into the f32 table (yields f32 scalar)
        idx = jnp.mod(carry_i, 4).astype(jnp.int64)
        # Use explicit dynamic_slice to prefer ONNX Slice lowering
        val_f32 = lax.dynamic_slice(table_f32, (idx,), (1,)).squeeze()  # f32
        # build an f64 accumulator (forces mixed-dtype math mode)
        acc_f64 = carry_i.astype(jnp.float64) + jnp.array(0.0, dtype=jnp.float64)  # f64
        # MIXED-DTYPE ADD: f64 + f32 -> needs a Cast(to=DOUBLE) for val_f32
        sum64 = acc_f64 + val_f32
        return carry_i + 1, sum64  # stacked f64 sequence

    def outer(carry_o: jnp.int32, _):
        # inner Loop length 3, no xs → ONNX Loop
        _, ys64 = lax.scan(inner, carry_o, xs=None, length=3)  # f64[3]
        last_idx = jnp.array(2, dtype=jnp.int64)
        # pick last element from the inner sequence inside the outer body
        pick64 = lax.dynamic_slice(ys64, (last_idx,), (1,)).squeeze()  # f64
        return carry_o + 1, pick64  # stacked f64 sequence

    # outer Loop length 2, no xs → ONNX Loop
    _, out64 = lax.scan(outer, jnp.array(0, dtype=jnp.int32), xs=None, length=2)
    return out64  # f64[]


@pytest.mark.filterwarnings("ignore:.*appears in graph inputs.*:UserWarning")
def test_feed_forward_like_nested_loops_mixed_dtypes_loads_and_runs(tmp_path):
    # 1) Export with double precision enabled (as in the external use-case)
    model = to_onnx(
        _fn_feed_forward_like_nested,
        inputs=[],  # everything is captured
        enable_double_precision=True,
        opset=21,
        model_name="feed_forward_like_nested",
    )
    p = tmp_path / "feed_forward_like_nested.onnx"
    p.write_bytes(model.SerializeToString())

    # 2) Must load in ORT (this is where the regression originally failed)
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {})
    # sanity: one scalar, dtype must be f64
    assert len(outs) == 1
    assert _np.asarray(outs[0]).dtype == _np.float64

    # 3) Structural guardrail: in SOME Loop body, the indexing result (Slice/Gather/GatherND)
    #    must flow through a Cast(to=DOUBLE) into an Add. This ensures mixed-dtype harmonization
    #    is applied where it matters (exact pre-fix failure).
    m = onnx.load(str(p))
    from onnx import TensorProto as _TP

    def iter_loop_bodies(graph):
        for n in graph.node:
            if n.op_type == "Loop":
                for a in n.attribute:
                    if a.name == "body" and a.g is not None:
                        yield a.g
                        # nested loops
                        yield from iter_loop_bodies(a.g)

    def _has_index_cast_add(loop_graph):
        # tolerant path search: (indexing … Cast(to=DOUBLE)) feeding an Add input
        out2node = {o: n for n in loop_graph.node for o in n.output}
        index_ops = {"Slice", "GatherND", "Gather"}
        passthrough = {"Squeeze", "Unsqueeze", "Identity", "Reshape", "Transpose"}

        def walk_back(name, seen_cast=False, depth=0, limit=16):
            if not name or depth > limit:
                return False
            node = out2node.get(name)
            if node is None:
                return False
            if node.op_type == "Cast":
                to_double = any(a.name == "to" and a.i == _TP.DOUBLE for a in node.attribute)
                # continue through all inputs; remember we saw Cast(to=DOUBLE)
                return any(walk_back(inp, seen_cast or to_double, depth + 1, limit) for inp in node.input)
            if node.op_type in index_ops:
                # indexing reached; succeed only if a Cast(to=DOUBLE) was seen on the path
                if seen_cast:
                    return True
                return any(walk_back(inp, seen_cast, depth + 1, limit) for inp in node.input)
            if node.op_type in passthrough:
                return any(walk_back(inp, seen_cast, depth + 1, limit) for inp in node.input)
            # stop on other ops
            return False

        for n in loop_graph.node:
            if n.op_type != "Add":
                continue
            # any Add input backed by the pattern?
            if any(walk_back(inp) for inp in n.input):
                return True
        return False

    assert any(_has_index_cast_add(g) for g in iter_loop_bodies(m.graph)), (
        "Expected inside at least one Loop body: indexing (Slice/Gather/GatherND) "
        "→ Cast(to=DOUBLE) → consumed by Add. This guarantees mixed-dtype harmonization."
    )
