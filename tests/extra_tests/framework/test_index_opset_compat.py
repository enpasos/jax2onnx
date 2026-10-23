# tests/extra_tests/framework/test_index_opset_compat.py

from __future__ import annotations

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest

from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins.jax.lax._index_utils import _builder_op, _const_i64


@pytest.mark.parametrize("opset", [11, 12, 13, 23])
def test_index_helpers_use_requested_squeeze_and_unsqueeze_axes_schema(
    opset: int,
) -> None:
    context = IRContext(opset=opset, enable_double_precision=False)
    data = ir.Value(
        name="data", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2, 1))
    )
    context.builder.inputs.append(data)
    squeeze_axes = _const_i64(context, [2, 0], "squeeze_axes")
    squeezed = _builder_op(
        context,
        "Squeeze",
        [data, squeeze_axes],
        name_hint="squeezed",
        dtype=ir.DataType.FLOAT,
        shape=(2,),
    )
    unsqueeze_axes = _const_i64(context, [0, 2], "unsqueeze_axes")
    expanded = _builder_op(
        context,
        "Unsqueeze",
        [squeezed, unsqueeze_axes],
        name_hint="expanded",
        dtype=ir.DataType.FLOAT,
        shape=(1, 2, 1),
    )
    context.builder.outputs.extend([squeezed, expanded])
    for value, expected_axes in ((squeezed, (2, 0)), (expanded, (0, 2))):
        producer = value.producer()
        assert producer is not None
        if opset < 13:
            assert len(producer.inputs) == 1
            assert tuple(producer.attributes["axes"].as_ints()) == expected_axes
        else:
            assert len(producer.inputs) == 2
            assert "axes" not in producer.attributes
    model = context.builder.to_ir_model(name="index_axes")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        ir.to_proto(model).SerializeToString(),
        options,
        providers=["CPUExecutionProvider"],
    )
    source = np.asarray([1.0, -2.0], np.float32).reshape(1, 2, 1)
    actual = session.run(None, {"data": source})
    np.testing.assert_array_equal(actual[0], source.reshape(2))
    np.testing.assert_array_equal(actual[1], source)


@pytest.mark.parametrize("fault", ["dynamic", "wrong-dtype", "wrong-rank"])
def test_legacy_index_axes_reject_unrepresentable_input_before_emission(
    fault: str,
) -> None:
    context = IRContext(opset=11, enable_double_precision=False)
    data = ir.Value(
        name="data", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2))
    )
    if fault == "dynamic":
        axes = ir.Value(
            name="axes", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape((1,))
        )
    else:
        array = (
            np.asarray([0.0], np.float32)
            if fault == "wrong-dtype"
            else np.asarray([[0]], np.int64)
        )
        axes = context.bind_const_for_var(object(), array)
    before = len(context.builder.graph)
    with pytest.raises(ValueError, match="constant INT64 axes"):
        _builder_op(
            context,
            "Squeeze",
            [data, axes],
            name_hint="invalid",
            dtype=ir.DataType.FLOAT,
        )
    assert len(context.builder.graph) == before


def test_legacy_squeeze_retains_omitted_axes_form() -> None:
    context = IRContext(opset=11, enable_double_precision=False)
    data = ir.Value(
        name="data", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape((1, 2, 1))
    )
    squeezed = _builder_op(
        context,
        "Squeeze",
        [data],
        name_hint="all_singletons",
        dtype=ir.DataType.FLOAT,
        shape=(2,),
    )
    producer = squeezed.producer()
    assert producer is not None and len(producer.inputs) == 1
    assert "axes" not in producer.attributes
