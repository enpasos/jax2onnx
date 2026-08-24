# tests/extra_tests/converter/test_ir_builder.py

import importlib.metadata
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from onnx_ir.tape import Tape

from jax2onnx.converter import ir_builder as ir_builder_module
from jax2onnx.converter.ir_builder import (
    IRBuilder,
    JAX_CALLSITE_METADATA_KEY,
    JAX_TRACE_METADATA_KEY,
    PLUGIN_METADATA_KEY,
    STACKTRACE_METADATA_KEY,
)


def test_ir_builder_uses_public_tape_api() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)

    assert isinstance(builder._tape_builder, Tape)


def test_ir_builder_forwards_tape_ops() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1])
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[1])

    out = builder.Add(x, y, _outputs=["sum"], _version=18)

    node = out.producer()
    assert node.op_type == "Add"
    assert builder.nodes[-1] is node
    assert out.name == "sum"


def test_ir_builder_forwards_multi_output_tape_ops() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[4])
    k = ir.val("k", dtype=ir.DataType.INT64, shape=[1])

    values, indices = builder.TopK(
        x,
        k,
        axis=-1,
        _outputs=["values", "indices"],
        _version=18,
    )

    node = values.producer()
    assert node is indices.producer()
    assert node.op_type == "TopK"
    assert node.attributes["axis"].as_int() == -1
    assert [value.name for value in node.outputs] == ["values", "indices"]
    assert ("", 18) in builder.used_opsets


def test_ir_builder_accepts_integer_output_count() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2])

    outputs = builder.Split(x, _outputs=2, _version=18)

    assert len(outputs) == 2
    assert outputs[0].producer() is outputs[1].producer()
    assert len(builder.nodes) == 1


@pytest.mark.parametrize(
    "invalid_outputs",
    ["result", b"result", ["valid", 1]],
    ids=["string", "bytes", "non-string-element"],
)
def test_ir_builder_rejects_invalid_outputs_before_graph_mutation(
    invalid_outputs: object,
) -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1])

    with pytest.raises(TypeError, match="int or a non-text sequence of strings"):
        builder.Identity(x, _outputs=invalid_outputs, _version=18)

    assert len(builder.nodes) == 0
    assert builder.used_opsets == set()


def test_ir_builder_stacktrace_metadata_disabled_by_default() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[1])
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[1])

    out = builder.Add(x, y, _outputs=["sum"], _version=18)
    node = out.producer()

    assert STACKTRACE_METADATA_KEY not in node.metadata_props
    assert JAX_CALLSITE_METADATA_KEY not in node.metadata_props
    assert PLUGIN_METADATA_KEY not in node.metadata_props


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
    stacktrace = node.metadata_props.get(STACKTRACE_METADATA_KEY)
    assert isinstance(stacktrace, str)
    assert "onnx_ir/_tape.py" not in stacktrace
    assert "onnx_ir/tape.py" not in stacktrace


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


def test_resolve_producer_version_prefers_source_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        '[project]\nname = "jax2onnx"\nversion = "9.8.7"\n', encoding="utf-8"
    )
    monkeypatch.setattr(ir_builder_module, "_SOURCE_TREE_PYPROJECT", pyproject_path)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.5.2")

    assert ir_builder_module._resolve_producer_version() == "9.8.7"


def test_resolve_producer_version_uses_installed_distribution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ir_builder_module, "_SOURCE_TREE_PYPROJECT", tmp_path / "missing.toml"
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "1.2.3")

    assert ir_builder_module._resolve_producer_version() == "1.2.3"


def test_resolve_producer_version_without_package_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ir_builder_module, "_SOURCE_TREE_PYPROJECT", tmp_path / "missing.toml"
    )

    def _missing_distribution(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", _missing_distribution)

    assert ir_builder_module._resolve_producer_version() is None


def test_ir_builder_serializes_producer_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exported models previously left producer_version unset (empty string),
    # even though onnx_ir.Model accepts it directly -- see issue about
    # metadata_props/doc_string/producer_version being empty on every
    # jax2onnx-produced model in a public ONNX-on-the-Hub census.
    monkeypatch.setattr(ir_builder_module, "_PRODUCER_VERSION", "9.8.7")
    builder = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.Value(name="x", shape=ir.Shape((1,)), type=ir.TensorType(ir.DataType.FLOAT))
    builder.inputs.append(x)
    builder.outputs.append(x)

    model = builder.to_ir_model(name="m", ir_version=11)

    assert model.producer_name == "jax2onnx"
    assert model.producer_version == "9.8.7"

    model_proto = ir.serde.serialize_model(model)
    assert model_proto.producer_name == "jax2onnx"
    assert model_proto.producer_version == "9.8.7"


def test_ir_builder_add_node_converts_mapping_attributes() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)
    x = ir.Value(name="x", shape=ir.Shape((2,)), type=ir.TensorType(ir.DataType.FLOAT))
    y = ir.Value(name="y", shape=ir.Shape((2,)), type=ir.TensorType(ir.DataType.FLOAT))

    node = builder.add_node(
        "ReduceMean",
        inputs=[x],
        outputs=[y],
        attributes={"keepdims": 1, "axes": [0]},
    )

    assert node.attributes["keepdims"].as_int() == 1
    assert node.attributes["axes"].as_ints() == (0,)
