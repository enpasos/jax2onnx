# tests/extra_tests/converter/test_ir_builder.py

import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx.converter.ir_builder import (
    IRBuilder,
    JAX_CALLSITE_METADATA_KEY,
    JAX_TRACE_METADATA_KEY,
    PLUGIN_METADATA_KEY,
    STACKTRACE_METADATA_KEY,
)

try:  # pragma: no cover - guarded import for environments without tape builder
    from onnx_ir._tape import Builder as _TapeBuilder
except Exception:  # pragma: no cover
    _TapeBuilder = None


@pytest.mark.skipif(_TapeBuilder is None, reason="onnx_ir tape.Builder unavailable")
def test_ir_builder_forwards_tape_ops() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1])
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[1])

    out = builder.Add(x, y, _outputs=["sum"], _version=18)

    node = out.producer()
    assert node.op_type == "Add"
    assert builder.nodes[-1] is node
    assert out.name == "sum"


@pytest.mark.skipif(_TapeBuilder is None, reason="onnx_ir tape.Builder unavailable")
def test_ir_builder_stacktrace_metadata_disabled_by_default() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1])
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[1])

    out = builder.Add(x, y, _outputs=["sum"], _version=18)
    node = out.producer()

    assert STACKTRACE_METADATA_KEY not in node.metadata_props
    assert JAX_CALLSITE_METADATA_KEY not in node.metadata_props
    assert PLUGIN_METADATA_KEY not in node.metadata_props


@pytest.mark.skipif(_TapeBuilder is None, reason="onnx_ir tape.Builder unavailable")
def test_ir_builder_stacktrace_metadata_enabled() -> None:
    builder = IRBuilder(
        opset=18, enable_double_precision=False, enable_stacktrace_metadata=True
    )
    builder.set_stacktrace_mode("minimal")
    builder.set_current_jax_traceback(
        "user_code.py:123 (my_fun)\n/home/env/site-packages/jax/_src/foo.py:1 (bar)"
    )
    builder.set_current_plugin_identifier("test.module.Plugin.lower")
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1])
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[1])

    out = builder.Add(x, y, _outputs=["sum"], _version=18)
    node = out.producer()

    assert node.metadata_props.get(JAX_CALLSITE_METADATA_KEY) == "my_fun:123"
    assert node.metadata_props.get(PLUGIN_METADATA_KEY) == "Plugin.lower:123"
    assert STACKTRACE_METADATA_KEY not in node.metadata_props
    assert JAX_TRACE_METADATA_KEY not in node.metadata_props


@pytest.mark.skipif(_TapeBuilder is None, reason="onnx_ir tape.Builder unavailable")
def test_ir_builder_stacktrace_full_mode_includes_detailed_fields() -> None:
    builder = IRBuilder(
        opset=18, enable_double_precision=False, enable_stacktrace_metadata=True
    )
    builder.set_stacktrace_mode("full")
    builder.set_current_jax_traceback(
        "frame_a.py:10 (call)\n/home/env/site-packages/jax/_src/run.py:1 (run)"
    )
    builder.set_current_plugin_identifier("mod.Plugin.lower")
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1])
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[1])

    out = builder.Add(x, y, _outputs=["sum"], _version=18)
    node = out.producer()

    assert node.metadata_props.get(JAX_CALLSITE_METADATA_KEY) == "call:10"
    assert node.metadata_props.get(PLUGIN_METADATA_KEY) == "Plugin.lower:10"
    assert isinstance(node.metadata_props.get(JAX_TRACE_METADATA_KEY), str)
    assert isinstance(node.metadata_props.get(STACKTRACE_METADATA_KEY), str)


def test_ir_builder_initializer_registration() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)

    weight = builder.add_initializer_from_scalar(
        name="weight", value=np.array([1.0], dtype=np.float32)
    )

    assert builder.graph.initializers["weight"] is weight
    assert weight in builder.initializers
    assert getattr(weight, "const_value", None) is not None


def test_ir_builder_initializer_view_assignment_roundtrip() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)
    weight = builder.add_initializer_from_scalar(name="weight", value=1.0)

    assert weight in builder.initializers

    builder.initializers = []
    assert not builder.initializers
    builder.initializers.append(weight)

    assert builder.graph.initializers["weight"] is weight


def _tensor_value(name: str, array: np.ndarray) -> ir.Value:
    tensor = ir.tensor(array)
    return ir.Value(
        name=name,
        shape=ir.Shape(array.shape if array.shape else ()),
        type=ir.TensorType(ir.DataType.from_numpy(array.dtype)),
        const_value=tensor,
    )


def test_ir_builder_initializer_append_duplicate_same_reuses_existing() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)

    v1 = _tensor_value("w", np.array([1.0], dtype=np.float32))
    v2 = _tensor_value("w", np.array([1.0], dtype=np.float32))

    builder.initializers.append(v1)
    before = builder.graph.initializers["w"]
    builder.initializers.append(v2)

    # The existing initializer remains canonical; duplicates do not overwrite
    assert builder.graph.initializers["w"] is before
    assert len(builder.graph.initializers) == 1


def test_ir_builder_initializer_append_duplicate_different_raises() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)

    v1 = _tensor_value("w", np.array([1.0], dtype=np.float32))
    v2 = _tensor_value("w", np.array([2.0], dtype=np.float32))

    builder.initializers.append(v1)

    with pytest.raises(ValueError):
        builder.initializers.append(v2)


def test_ir_builder_add_initializer_from_scalar_duplicate_same_reuses() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)

    v1 = builder.add_initializer_from_scalar(
        name="alpha", value=np.array([3.14], dtype=np.float32)
    )
    v2 = builder.add_initializer_from_scalar(
        name="alpha", value=np.array([3.14], dtype=np.float32)
    )

    assert v1 is v2
    assert builder.graph.initializers["alpha"] is v1


def test_ir_builder_add_initializer_from_scalar_duplicate_different_raises() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)

    _ = builder.add_initializer_from_scalar(
        name="beta", value=np.array([1.0], dtype=np.float32)
    )
    with pytest.raises(ValueError):
        _ = builder.add_initializer_from_scalar(
            name="beta", value=np.array([2.0], dtype=np.float32)
        )


def test_ir_builder_model_roundtrip_preserves_initializer_connections() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)

    # Create an input and an initializer, then add a node that consumes both.
    x = ir.Value(name="x", shape=ir.Shape((1,)), type=ir.TensorType(ir.DataType.FLOAT))
    w = builder.add_initializer_from_scalar(
        name="w", value=np.array([1.0], dtype=np.float32)
    )
    y = ir.Value(name="y", shape=ir.Shape((1,)), type=ir.TensorType(ir.DataType.FLOAT))

    builder.inputs.append(x)
    builder.add_node("Add", inputs=[x, w], outputs=[y])
    builder.outputs.append(y)

    model = builder.to_ir_model(name="m", ir_version=11)
    g = model.graph
    # There should be exactly one initializer named 'w', and the node should
    # reference the same Value instance as stored in the graph's initializers.
    w2 = g.initializers["w"]
    node = list(g)[0]
    assert node.inputs[1] is w2
