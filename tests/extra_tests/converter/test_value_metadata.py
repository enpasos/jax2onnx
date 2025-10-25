# tests/extra_tests/converter/test_value_metadata.py

import jax
import jax.numpy as jnp
import onnx_ir as ir

from jax2onnx.converter.conversion_api import to_onnx
from jax2onnx.converter.ir_builder import (
    JAX_CALLSITE_METADATA_KEY,
    PLUGIN_METADATA_KEY,
    STACKTRACE_METADATA_KEY,
)


def _make_symbolic_model():
    return to_onnx(
        fn=lambda x: (x + 1,),
        inputs=[jax.ShapeDtypeStruct(("B", 4), jnp.float32)],
        input_params=None,
        model_name="value_metadata_check",
        opset=21,
        enable_double_precision=False,
        record_primitive_calls_file=None,
    )


def test_value_shapes_and_types_preserved_without_value_info_registry():
    model = _make_symbolic_model()
    graph = model.graph

    assert len(graph.inputs) == 1
    assert len(graph.outputs) == 1

    out_val = graph.outputs[0]
    assert isinstance(out_val.shape, ir.Shape)
    dims = out_val.shape.dims
    assert len(dims) == 2

    batch_dim = dims[0]
    assert isinstance(batch_dim, ir.SymbolicDim)
    assert batch_dim.value == "B"

    feature_dim = dims[1]
    assert isinstance(feature_dim, int)
    assert feature_dim == 4

    assert out_val.type is not None
    assert out_val.type.dtype == ir.DataType.FLOAT

    add_nodes = [node for node in graph.all_nodes() if node.op_type == "Add"]
    assert add_nodes, "Expected to find Add node in generated graph"

    add_out = add_nodes[0].outputs[0]
    assert isinstance(add_out.shape, ir.Shape)
    assert add_out.shape.dims == dims
    assert add_out.type is not None
    assert add_out.type.dtype == ir.DataType.FLOAT


def test_stacktrace_metadata_includes_jax_traceback(monkeypatch) -> None:
    monkeypatch.setenv("JAX2ONNX_ENABLE_STACKTRACE_METADATA", "1")

    def fn(x):
        return x * jnp.sin(x)

    model = to_onnx(
        fn=fn,
        inputs=[jax.ShapeDtypeStruct((3,), jnp.float32)],
        model_name="stacktrace_sample",
        opset=21,
        enable_double_precision=False,
        input_params=None,
        record_primitive_calls_file=None,
    )
    for node in model.graph.all_nodes():
        meta = getattr(node, "metadata_props", {})
        callsite = meta.get(JAX_CALLSITE_METADATA_KEY)
        plugin_id = meta.get(PLUGIN_METADATA_KEY)
        assert isinstance(callsite, str)
        assert callsite, "Expected callsite metadata"
        assert isinstance(plugin_id, str)
        assert plugin_id.count(":") == 1
        assert STACKTRACE_METADATA_KEY not in meta
    # ensure environment cleaned for other tests
    monkeypatch.delenv("JAX2ONNX_ENABLE_STACKTRACE_METADATA", raising=False)
