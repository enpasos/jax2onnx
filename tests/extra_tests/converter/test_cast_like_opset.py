# tests/extra_tests/converter/test_cast_like_opset.py

"""A conversion to a known dtype is a Cast at every opset.

CastLike exists from opset 15 and is only needed where the target dtype is
not known at export time. ONNX Runtime before 1.30 aborts while creating a
session for a CastLike that widens a float16 constant to float32.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
from flax import linen
from onnx.reference import ReferenceEvaluator

from jax2onnx import to_onnx
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins._utils import cast_param_like


def _op_types(graph: onnx.GraphProto) -> list[str]:
    ops: list[str] = []
    for node in graph.node:
        ops.append(node.op_type)
        for attribute in node.attribute:
            if attribute.HasField("g"):
                ops.extend(_op_types(attribute.g))
    return ops


def _model(ctx: IRContext, name: str) -> onnx.ModelProto:
    model = ir.serde.serialize_model(ctx.builder.to_ir_model(name=name))
    onnx.checker.check_model(model, full_check=True)
    return model


@pytest.mark.parametrize("opset", [14, 15, 23])
def test_parameter_adaptation_is_a_cast_at_every_opset(opset: int) -> None:
    payload = np.asarray([1.0 + 2**-30, -0.0, 3.5], dtype=np.float64)
    ctx = IRContext(opset=opset)
    source = ctx.builder.add_initializer_from_array(name="parameter", array=payload)
    like = ir.val("like", dtype=ir.DataType.FLOAT, shape=[3])
    ctx.builder.inputs.append(like)

    adapted = cast_param_like(ctx, source, like)

    assert adapted is not source and source.dtype == ir.DataType.DOUBLE
    assert adapted.dtype == ir.DataType.FLOAT and adapted.shape == source.shape
    ctx.builder.outputs.append(adapted)
    model = _model(ctx, "parameter_adaptation")
    assert _op_types(model.graph) == ["Cast"]
    (converted,) = ReferenceEvaluator(model).run(
        None, {"like": np.zeros((3,), dtype=np.float32)}
    )
    expected = payload.astype(np.float32)
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, expected)
    np.testing.assert_array_equal(np.signbit(converted), np.signbit(expected))


@pytest.mark.parametrize("opset", [14, 15, 23])
def test_context_cast_like_to_a_known_dtype_is_a_cast(opset: int) -> None:
    ctx = IRContext(opset=opset)
    tensor = ir.val("tensor", dtype=ir.DataType.INT64, shape=[2])
    exemplar = ir.val("exemplar", dtype=ir.DataType.FLOAT, shape=[2])
    ctx.builder.inputs.extend([tensor, exemplar])

    result = ctx.cast_like(tensor, exemplar, name_hint="converted")

    assert result.dtype == ir.DataType.FLOAT and result.shape == tensor.shape
    ctx.builder.outputs.append(result)
    assert _op_types(_model(ctx, "context_cast_like").graph) == ["Cast"]


def test_context_cast_like_to_an_unknown_dtype_is_a_cast_like() -> None:
    ctx = IRContext(opset=15)
    tensor = ir.val("tensor", dtype=ir.DataType.INT64, shape=[2])
    exemplar = ir.Value(name="exemplar")
    ctx.builder.inputs.append(tensor)

    result = ctx.cast_like(tensor, exemplar, name_hint="converted")

    (node,) = ctx.builder.graph
    assert node.op_type == "CastLike" and node.outputs[0] is result
    assert list(node.inputs) == [tensor, exemplar]


def test_context_cast_like_needs_a_known_dtype_below_opset_15() -> None:
    ctx = IRContext(opset=14)
    tensor = ir.val("tensor", dtype=ir.DataType.INT64, shape=[2])
    exemplar = ir.Value(name="exemplar")

    with pytest.raises(NotImplementedError, match="opset >= 15"):
        ctx.cast_like(tensor, exemplar)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("opset", [13, 14])
def test_layer_norm_with_float32_parameters_is_valid_below_opset_15(
    opset: int, dtype: type
) -> None:
    # The float32 scale and bias are adapted to the half-precision input.
    module = linen.LayerNorm()
    variables = module.init(jax.random.PRNGKey(0), jnp.zeros((2, 6), jnp.float32))

    model = to_onnx(
        lambda x: module.apply(variables, x), [jnp.zeros((2, 6), dtype)], opset=opset
    )

    onnx.checker.check_model(model, full_check=True)
    assert "CastLike" not in _op_types(model.graph)
    assert model.graph.output[0].type.tensor_type.elem_type == (
        onnx.TensorProto.BFLOAT16 if dtype is jnp.bfloat16 else onnx.TensorProto.FLOAT16
    )


@pytest.mark.parametrize("opset", [13, 15])
def test_linspace_matches_jax_on_both_sides_of_opset_15(opset: int) -> None:
    def shifted(x: jax.Array) -> jax.Array:
        return x + jnp.linspace(0.0, 1.0, 6)

    x = np.random.default_rng(0).normal(size=(2, 6)).astype(np.float32)
    model = to_onnx(shifted, [jnp.asarray(x)], opset=opset)

    onnx.checker.check_model(model, full_check=True)
    assert "CastLike" not in _op_types(model.graph)
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (got,) = session.run(None, {model.graph.input[0].name: x})
    np.testing.assert_array_equal(got, np.asarray(shifted(jnp.asarray(x))))
