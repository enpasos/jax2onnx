# tests/extra_tests/converter/test_no_cast_like.py

"""The export converts dtypes with Cast and never writes CastLike.

ONNX Runtime before 1.30 aborts while creating a session for a CastLike that
widens a float16 constant to float32; see "The float16 initializer optimizer
before ONNX Runtime 1.30" in the dependency guide. A JAX program is statically
typed, so the target dtype is known at export time and Cast names the same
conversion. Cast is also valid at every opset, while CastLike only exists from
opset 15.
"""

from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
from flax import linen
from onnx.reference import ReferenceEvaluator

import jax2onnx
from jax2onnx import to_onnx
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.plugins._utils import cast_param_like

_PACKAGE = Path(jax2onnx.__file__).resolve().parent


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


def _cast_like_emissions(source: str) -> list[int]:
    """Lines of the calls in ``source`` that would write a CastLike node.

    Covers the ways this package builds nodes: the builder shorthand
    ``builder.CastLike(...)``, and the operator named as a string, either
    positionally as in ``builder.op("CastLike", ...)`` and
    ``ir.Node("", "CastLike", ...)`` or as ``op_type="CastLike"``.
    """
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        shorthand = (
            isinstance(node.func, ast.Attribute) and node.func.attr == "CastLike"
        )
        named = [
            *node.args,
            *(keyword.value for keyword in node.keywords if keyword.arg == "op_type"),
        ]
        by_name = any(
            isinstance(argument, ast.Constant) and argument.value == "CastLike"
            for argument in named
        )
        if shorthand or by_name:
            lines.append(node.lineno)
    return lines


def test_package_does_not_emit_cast_like() -> None:
    offenders = [
        f"{path.relative_to(_PACKAGE.parent)}:{line}"
        for path in sorted(_PACKAGE.rglob("*.py"))
        for line in _cast_like_emissions(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "Convert dtypes with Cast, through cast_value_to_dtype, cast_param_like "
        "or ctx.cast_like. ONNX Runtime before 1.30 aborts on a CastLike that "
        "widens a float16 constant: " + ", ".join(offenders)
    )


def test_cast_like_emissions_are_recognized() -> None:
    source = "\n".join(
        [
            "a = ctx.builder.CastLike(x, like, _outputs=[name])",
            'b = ctx.builder.op("CastLike", [x, like], output=out)',
            'c = ir.Node("", "CastLike", inputs=[x, like], outputs=[out])',
            'd = ir.Node(op_type="CastLike", domain="", inputs=[x, like])',
            # Reading or expecting the operator does not write it.
            'e = node.op_type == "CastLike"',
            'f = node.op_type in {"Cast", "CastLike"}',
            'g = expect_graph(["Cast -> Gemm"], must_absent=["CastLike"])',
            'h = ctx.builder.op("Cast", [x], {"to": 1}, output=out)',
        ]
    )

    assert _cast_like_emissions(source) == [1, 2, 3, 4]


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


@pytest.mark.parametrize("opset", [14, 23])
def test_context_cast_like_needs_a_known_dtype(opset: int) -> None:
    ctx = IRContext(opset=opset)
    tensor = ir.val("tensor", dtype=ir.DataType.INT64, shape=[2])
    exemplar = ir.Value(name="exemplar")

    with pytest.raises(NotImplementedError, match="dtype is unknown.*'exemplar'"):
        ctx.cast_like(tensor, exemplar)

    assert not list(ctx.builder.graph)


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
