# tests/extra_tests/converter/test_ir_builder.py

import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx.converter.ir_builder import IRBuilder

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


def test_ir_builder_initializer_registration() -> None:
    builder = IRBuilder(opset=18, enable_double_precision=False)

    weight = builder.add_initializer_from_scalar(
        name="weight", value=np.array([1.0], dtype=np.float32)
    )

    assert builder.initializers_by_name["weight"] is weight
    assert weight in builder.initializers
    assert getattr(weight, "const_value", None) is not None
