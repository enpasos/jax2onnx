import collections
import pytest
import jax
import jax.numpy as jnp
from jax import lax
from onnx import AttributeProto

from jax2onnx import to_onnx

try:
    import onnxruntime as ort  # optional, used for an extra load check

    HAS_ORT = True
except Exception:
    HAS_ORT = False


def _fn_scan_two_broadcasts(y):
    """Scan body uses TWO broadcast_in_dim sites to stress SSA name generation."""

    def body(carry, _):
        # Two separate broadcasts to the same symbolic shape
        one = jnp.broadcast_to(jnp.array(1.0, dtype=carry.dtype), carry.shape)
        zero = jnp.broadcast_to(jnp.array(0.0, dtype=carry.dtype), carry.shape)
        out = carry + one + zero + jnp.reshape(carry, carry.shape)
        # keep carry flowing to make body non-trivial
        return carry + 0.0, out

    _, ys = lax.scan(body, y, xs=None, length=2)
    return ys


def _collect_node_output_names(graph):
    """Recursively collect *all* node output names from `graph` and subgraphs."""
    names = []
    for n in graph.node:
        for o in n.output:
            if o:
                names.append(o)
        for a in n.attribute:
            # Single subgraph (Loop/Scan body, If then_branch/else_branch, etc.)
            if a.type == AttributeProto.GRAPH and a.g is not None:
                names.extend(_collect_node_output_names(a.g))
            # Multiple subgraphs (rare, but part of ONNX attribute spec)
            if a.type == AttributeProto.GRAPHS and a.graphs:
                for sg in a.graphs:
                    names.extend(_collect_node_output_names(sg))
    return names


def test_broadcast_in_dim_helpers_unique_ssa(tmp_path):
    # Symbolic input: (B, 4)
    spec = jax.ShapeDtypeStruct(("B", 4), jnp.float32)
    model = to_onnx(
        _fn_scan_two_broadcasts,
        inputs=[spec],
        opset=21,
        model_name="broadcast_in_dim_ssa_unique",
    )

    # SSA check: no node output name may appear more than once
    all_outs = _collect_node_output_names(model.graph)
    dupes = [name for name, cnt in collections.Counter(all_outs).items() if cnt > 1]
    assert not dupes, f"Found duplicate node outputs in graph/subgraphs: {dupes}"

    # Optional: ORT should load the model without SSA complaints
    if HAS_ORT:
        path = tmp_path / "broadcast_in_dim_ssa_unique.onnx"
        path.write_bytes(model.SerializeToString())
        try:
            ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as e:
            pytest.fail(f"onnxruntime failed to load model (SSA regression?): {e}")
