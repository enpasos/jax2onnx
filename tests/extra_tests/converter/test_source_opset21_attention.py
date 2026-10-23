# tests/extra_tests/converter/test_source_opset21_attention.py

"""Shared real opset21 routes; finite target samples do not admit whole families."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import user_interface as ui
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_representation import normalized_shape_dimensions
from jax2onnx.precision import CheckStatus, SampleCriteria, get_precision_report
from tests.extra_tests.converter import test_source_matrix_admission as admission
from tests.extra_tests.converter import test_source_ordinary_matrix as ordinary
from tests.extra_tests.converter.test_source_native_contract import _forged_aval

_CANONICAL = (((1,), (0,)), ((), ()))
_GEOMETRIES = {
    "projection": (((4, 4), ("B", 3, 4)), (((1,), (2,)), ((), ()))),
    "scores": ((("B", 3, 2, 2), ("B", 3, 2, 2)), (((3,), (3,)), ((0, 2), (0, 2)))),
    "values": ((("B", 2, 3, 3), ("B", 3, 2, 2)), (((3,), (1,)), ((0, 1), (0, 2)))),
}
_DTYPES = [
    pytest.param(np.float32, np.float32, np.float32, id="same32"),
    pytest.param(np.float64, np.float64, np.float64, id="same64"),
    pytest.param(np.float64, np.float32, np.float64, id="mixed-to64"),
    pytest.param(np.float64, np.float32, np.float32, id="mixed-to32"),
]
# Reuse the previously declared finite matrix budgets, not a universal error bound.
_CRITERIA = ordinary._LONG_CRITERIA


@pytest.fixture(autouse=True)
def _caller_configuration() -> Iterator[None]:
    hidden, effective = admission._globals(), admission._effective()
    with (
        jax.enable_x64(True),
        jax.default_matmul_precision(None),
        jax.numpy_dtype_promotion("standard"),
        jax.numpy_rank_promotion("allow"),
    ):
        yield
    assert (admission._globals(), admission._effective()) == (hidden, effective)


def _values(shapes: Any, dtypes: Any, batch: int) -> tuple[Any, ...]:
    values = []
    for index, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True)):
        concrete = tuple(batch if dim == "B" else dim for dim in shape)
        size = int(np.prod(concrete))
        values.append(((np.arange(size).reshape(concrete) % 7 - 3) / 8).astype(dtype))
        assert values[index].dtype == np.dtype(dtype)
    return tuple(values)


def _specs(shapes: Any, dtypes: Any, symbolic: bool) -> list[Any]:
    batch = export.symbolic_shape("B")[0] if symbolic else 2
    return [
        jax.ShapeDtypeStruct(
            tuple(batch if dim == "B" else dim for dim in shape), dtype
        )
        for shape, dtype in zip(shapes, dtypes, strict=True)
    ]


def _artifact(fn: Any, specs: list[Any], path: Path, *, opset: int = 21) -> Any:
    # Both calls use native graphs. There is no diagnostic validator or bypass.
    trace = jax.make_jaxpr(fn)(*specs)
    validate_source_jaxpr(trace, opset)
    before = admission._globals(), admission._effective()
    actual = ui._to_onnx(
        fn,
        specs,
        precision_policy=PrecisionPolicy.SOURCE,
        opset=opset,
        return_mode="file",
        output_path=path,
    )
    assert (admission._globals(), admission._effective()) == before
    assert Path(actual) == path and path.is_file()
    model = onnx.load(path)
    assert (
        next(item.version for item in model.opset_import if item.domain == "") == opset
    )
    onnx.checker.check_model(model, full_check=True)
    report = get_precision_report(model)
    assert report is not None and report.profile == "source-static-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    assert tuple(
        (v.dtype, v.shape, v.weak_type) for v in report.traced_outputs
    ) == tuple(
        (
            np.dtype(v.aval.dtype).name,
            normalized_shape_dimensions(v.aval.shape),
            v.aval.weak_type,
        )
        for v in ordinary._body(trace).outvars
    )
    return model


def _accept(model: Any, fn: Any, values: Any, criteria: SampleCriteria) -> None:
    _, eager, compiled = ordinary._reference(fn, values)
    for reference in (eager, compiled):
        for optimization in ("disabled", "all"):
            ordinary._check_samples(
                model, values, reference, criteria=criteria, optimize=optimization
            )


@pytest.mark.parametrize("geometry", tuple(_GEOMETRIES))
@pytest.mark.parametrize("left_dtype,right_dtype,result_dtype", _DTYPES)
@pytest.mark.parametrize("symbolic", [False, True], ids=["static", "symbolic"])
def test_opset21_native_attention_contraction_geometries(
    geometry: str,
    left_dtype: Any,
    right_dtype: Any,
    result_dtype: Any,
    symbolic: bool,
    tmp_path: Path,
) -> None:
    shapes, dimensions = _GEOMETRIES[geometry]
    dtypes = left_dtype, right_dtype

    def model(left: Any, right: Any) -> Any:
        return jax.lax.dot_general(
            left, right, dimensions, preferred_element_type=result_dtype
        )

    artifact = _artifact(
        model, _specs(shapes, dtypes, symbolic), tmp_path / "contraction.onnx"
    )
    assert sum(node.op_type == "MatMul" for node in artifact.graph.node) == 1
    assert any(node.op_type == "Transpose" for node in artifact.graph.node)
    for batch in (2, 5) if symbolic else (2,):
        _accept(
            artifact, model, _values(shapes, dtypes, batch), SampleCriteria("bitwise")
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("branch", ["outer", "zero-contraction", "zero-output"])
def test_opset21_existing_mul_and_empty_contraction_routes(
    dtype: Any,
    branch: str,
    tmp_path: Path,
) -> None:
    shapes, dimensions, operator = {
        "outer": (((2,), (3,)), (((), ()), ((), ())), "Mul"),
        "zero-contraction": (((2, 0), (0, 3)), _CANONICAL, "ConstantOfShape"),
        "zero-output": (((0, 2), (2, 3)), _CANONICAL, "ConstantOfShape"),
    }[branch]

    def model(left: Any, right: Any) -> Any:
        return jax.lax.dot_general(left, right, dimensions)

    artifact = _artifact(
        model, _specs(shapes, (dtype, dtype), False), tmp_path / "branch.onnx"
    )
    assert any(node.op_type == operator for node in artifact.graph.node)
    _accept(
        artifact, model, _values(shapes, (dtype, dtype), 2), SampleCriteria("bitwise")
    )


def _attention(
    x: Any, q_weight: Any, k_weight: Any, v_weight: Any, out_weight: Any
) -> Any:
    dtype = q_weight.dtype
    projected = []
    for weight in (q_weight, k_weight, v_weight):
        value = jax.lax.dot_general(
            weight, x, (((1,), (2,)), ((), ())), preferred_element_type=dtype
        )
        value = jax.lax.transpose(value, (1, 2, 0))
        projected.append(jax.lax.reshape(value, (x.shape[0], 3, 2, 2)))
    query, key, value = projected
    scores = jax.lax.dot_general(query, key, (((3,), (3,)), ((0, 2), (0, 2))))
    scores = scores / np.asarray(2, dtype=dtype)
    maximum = jnp.max(scores, axis=-1, keepdims=True)
    maximum = jax.lax.stop_gradient(
        jnp.maximum(np.asarray(-np.inf, dtype=dtype), maximum)
    )
    weights = jax.lax.exp(scores - maximum)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    result = jax.lax.dot_general(weights, value, (((3,), (1,)), ((0, 1), (0, 2))))
    result = jax.lax.reshape(
        jax.lax.transpose(result, (0, 2, 1, 3)), (x.shape[0], 3, 4)
    )
    result = jax.lax.dot_general(out_weight, result, (((1,), (2,)), ((), ())))
    return jax.lax.transpose(result, (1, 2, 0))


@pytest.mark.parametrize(
    "weight_dtype,input_dtype",
    [(np.float32, np.float32), (np.float64, np.float64), (np.float64, np.float32)],
)
@pytest.mark.parametrize("symbolic", [False, True], ids=["static", "symbolic"])
def test_opset21_complete_small_attention_expression(
    weight_dtype: Any,
    input_dtype: Any,
    symbolic: bool,
    tmp_path: Path,
) -> None:
    shapes = (("B", 3, 4),) + ((4, 4),) * 4
    dtypes = (input_dtype,) + (weight_dtype,) * 4
    artifact = _artifact(
        _attention, _specs(shapes, dtypes, symbolic), tmp_path / "attention.onnx"
    )
    operations = [node.op_type for node in artifact.graph.node]
    assert operations.count("MatMul") == 6
    assert {
        "Transpose",
        "ReduceMax",
        "Max",
        "Exp",
        "ReduceSum",
        "Div",
        "GatherElements",
    } <= set(operations)
    for batch in (2, 5) if symbolic else (2,):
        _accept(
            artifact,
            _attention,
            _values(shapes, dtypes, batch),
            _CRITERIA[np.dtype(weight_dtype).name],
        )


@pytest.mark.parametrize(
    "dtype,operation", [(np.float32, "max"), (np.float64, "max"), (np.float64, "min")]
)
@pytest.mark.parametrize("specials", [False, True], ids=["zero-words", "nan-infinity"])
@pytest.mark.parametrize("broadcast", [False, True])
def test_opset21_finite_min_max_selection_words_and_specials(
    dtype: Any,
    operation: str,
    specials: bool,
    broadcast: bool,
    tmp_path: Path,
) -> None:
    # No subnormals: their independently recorded target failures remain open.
    x = np.asarray(
        [np.nan, -np.inf, np.inf, -3] if specials else [-0.0, 0.0, -0.0, 0.0], dtype
    )
    y = np.asarray(
        [1, np.inf, -np.inf, 2] if specials else [0.0, -0.0, -0.0, 0.0], dtype
    )
    values = (x[:, None], y[None, :]) if broadcast else (x, y)

    def model(left: Any, right: Any) -> Any:
        fn = jax.lax.max if operation == "max" else jax.lax.min
        return fn(left, right), fn(right, left)

    artifact = _artifact(
        model, list(map(ordinary._spec, values)), tmp_path / "selection.onnx"
    )
    assert sum(node.op_type == "GatherElements" for node in artifact.graph.node) >= 1
    criteria = (
        SampleCriteria("exact", equal_nan=True)
        if specials
        else SampleCriteria("bitwise")
    )
    _accept(artifact, model, values, criteria)


@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.int32, np.complex64])
def test_opset21_does_not_admit_other_dot_dtypes(dtype: Any) -> None:
    def model(left: Any, right: Any) -> Any:
        return jax.lax.dot_general(left, right, _CANONICAL)

    with pytest.raises(SourcePrecisionError) as caught:
        ui._to_onnx(
            model,
            [jax.ShapeDtypeStruct((2, 3), dtype), jax.ShapeDtypeStruct((3, 2), dtype)],
            opset=21,
            precision_policy=PrecisionPolicy.SOURCE,
        )
    error = caught.value
    assert error.category == "profile-exclusion" and error.opset == 21
    assert "intersection" in error.detail
    dtype_name = np.dtype(dtype).name
    if dtype is np.int32:
        assert error.primitive == "dot_general" and error.location == "equation 0"
        assert error.requested_dtypes == (dtype_name,) * 3
    else:
        assert error.primitive is None and error.location == "input 0"
        assert error.requested_dtypes == (dtype_name,)


@pytest.mark.parametrize(
    "precision",
    [
        jax.lax.Precision.HIGH,
        jax.lax.Precision.HIGHEST,
        jax.lax.DotAlgorithmPreset.F32_F32_F32,
        jax.lax.DotAlgorithm(np.float32, np.float32, np.float32),
    ],
)
def test_opset21_fp32_precision_and_algorithms_keep_owning_gate(precision: Any) -> None:
    def model(left: Any, right: Any) -> Any:
        return jax.lax.dot_general(left, right, _CANONICAL, precision=precision)

    with pytest.raises(SourcePrecisionError) as caught:
        ui._to_onnx(
            model,
            [
                jax.ShapeDtypeStruct((2, 3), np.float32),
                jax.ShapeDtypeStruct((3, 2), np.float32),
            ],
            opset=21,
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert caught.value.primitive == "dot_general"
    assert "precision/algorithm" in caught.value.detail


@pytest.mark.parametrize(
    "defect",
    [
        "dimensions",
        "duplicate-axis",
        "result-dtype",
        "result-shape",
        "native-result",
        "placement",
        "unknown-parameter",
        "preferred-width",
    ],
)
def test_opset21_real_gate_keeps_native_equation_and_placement_checks(
    defect: str,
) -> None:
    graph = admission._graph()
    validate_source_jaxpr(graph, 21)
    equation = graph.eqns[0]
    output = graph.outvars[0]
    if defect == "dimensions":
        equation.params["dimension_numbers"] = (((1,), ()), ((), ()))
    elif defect == "duplicate-axis":
        equation.params["dimension_numbers"] = (((1, 1), (0, 0)), ((), ()))
    elif defect == "result-dtype":
        output.aval = _forged_aval(output.aval, dtype=np.dtype(np.float64))
    elif defect == "result-shape":
        output.aval = _forged_aval(output.aval, shape=(3, 3))
    elif defect == "native-result":
        output.aval = _forged_aval(output.aval, weak_type=True)
    elif defect == "placement":
        equation.params["out_sharding"] = object()
    elif defect == "unknown-parameter":
        equation.params["source_opset21_override"] = True
    else:
        equation.params["preferred_element_type"] = np.float64
        output.aval = _forged_aval(output.aval, dtype=np.dtype(np.float64))
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 21)
    assert (
        caught.value.primitive == "dot_general"
        and caught.value.location == "equation 0"
    )


def test_opset18_dot_and_opset21_fp32_min_remain_excluded() -> None:
    graph = admission._graph()
    with pytest.raises(SourcePrecisionError, match="original opset18/21"):
        validate_source_jaxpr(graph, 18)
    with pytest.raises(SourcePrecisionError) as caught:
        ui._to_onnx(
            jax.lax.min,
            [jax.ShapeDtypeStruct((2,), np.float32)] * 2,
            opset=21,
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert caught.value.primitive == "min" and "intersection" in caught.value.detail
