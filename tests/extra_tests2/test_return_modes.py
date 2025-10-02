from pathlib import Path

import jax.numpy as jnp
import onnx
import onnx_ir as ir

from jax2onnx.user_interface import to_onnx


def _simple(x):
    return jnp.add(x, 1.0)


def test_return_mode_proto():
    model = to_onnx(_simple, inputs=[(2,)], model_name="rt_proto")
    assert isinstance(model, onnx.ModelProto)
    assert any(output.name for output in model.graph.output)


def test_return_mode_ir():
    ir_model = to_onnx(
        _simple,
        inputs=[(2,)],
        model_name="rt_ir",
        return_mode="ir",
    )
    assert isinstance(ir_model, ir.Model)
    assert getattr(ir_model, "graph", None) is not None


def test_return_mode_file(tmp_path: Path):
    output = tmp_path / "model.onnx"
    result = to_onnx(
        _simple,
        inputs=[(2,)],
        model_name="rt_file",
        return_mode="file",
        output_path=output,
    )
    assert isinstance(result, str)
    assert Path(result) == output
    assert output.is_file()
    loaded = onnx.load_model(output)
    assert any(node.op_type == "Add" for node in loaded.graph.node)

