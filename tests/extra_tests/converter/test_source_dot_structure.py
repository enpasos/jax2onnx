# tests/extra_tests/converter/test_source_dot_structure.py

"""Real SOURCE contractions and independent final-graph counterexamples."""

from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from typing import Any
from unittest.mock import patch
import warnings

import jax
from jax import export as jax_export
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat.jax import Jaxpr, JaxprEqn, Literal, Primitive, Var
from jax2onnx.converter import conversion_api
from jax2onnx.converter import source_structure as structure
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_representation import normalized_shape_dimensions
from jax2onnx.plugins.jax.lax._source_dot_structure import freeze_dot_structure_rule
from jax2onnx.precision import CheckStatus, get_precision_report


def _export(
    fn: Any, shapes: Any, dtypes: Any = None, *, optimized: bool = True
) -> ir.Model:
    dtypes = dtypes or (np.float32,) * len(shapes)
    specs = [
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in zip(shapes, dtypes, strict=True)
    ]
    optimizer = (
        nullcontext()
        if optimized
        else patch.object(
            conversion_api, "optimize_graph", lambda model, **kwargs: None
        )
    )
    with jax.enable_x64(True), optimizer:
        model = ui._to_onnx(
            fn, specs, precision_policy=PrecisionPolicy.SOURCE, return_mode="ir"
        )
    assert isinstance(model, ir.Model)
    return model


def _checked(model: ir.Model) -> None:
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "validated", result.reasons
    assert result.expected_equations == result.checked_equations
    assert result.runtime == result.numerics == "unverified"
    report = get_precision_report(model)
    assert report is not None
    assert all(
        check.status is CheckStatus.UNVERIFIED
        for check in report.checks
        if check.category in {"runtime-execution", "numerical-acceptance"}
    )
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def _reverse_weight(
    lhs: Any, rhs: Any, *, preferred: Any = None, precision: Any = None
) -> Any:
    return jax.lax.dot_general(
        lhs,
        rhs,
        (((1,), (2,)), ((), ())),
        preferred_element_type=preferred,
        precision=precision,
    )


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    ("left_dtype", "right_dtype", "preferred", "output_dtype"),
    [
        (np.float32, np.float32, None, ir.DataType.FLOAT),
        (np.float64, np.float64, None, ir.DataType.DOUBLE),
        (np.float64, np.float32, None, ir.DataType.DOUBLE),
        (np.float32, np.float64, None, ir.DataType.DOUBLE),
        (np.float64, np.float32, np.float32, ir.DataType.FLOAT),
    ],
)
def test_actual_free_symbol_contraction_preserves_native_axes_and_widths(
    optimized: bool,
    left_dtype: Any,
    right_dtype: Any,
    preferred: Any,
    output_dtype: ir.DataType,
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    model = _export(
        lambda left, right: _reverse_weight(left, right, preferred=preferred),
        ((4, 3), (batch, 5, 3)),
        (left_dtype, right_dtype),
        optimized=optimized,
    )
    _checked(model)
    assert normalized_shape_dimensions(tuple(model.graph.outputs[0].shape)) == (
        4,
        "B",
        5,
    )
    assert model.graph.outputs[0].dtype == output_dtype


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "route", ["static-tensor", "leading-tensor", "paired-batch", "general-static"]
)
def test_existing_direct_and_general_contraction_routes_remain_checked(
    optimized: bool, route: str
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    if route == "static-tensor":
        shapes, axes = ((2, 3, 4), (4, 5)), (((2,), (0,)), ((), ()))
    elif route == "leading-tensor":
        shapes, axes = ((batch, 3, 4), (4, 5)), (((2,), (0,)), ((), ()))
    elif route == "paired-batch":
        shapes, axes = ((batch, 3, 4), (batch, 4, 5)), (((2,), (1,)), ((0,), (0,)))
    else:
        shapes, axes = ((2, 3, 4), (3, 5, 4)), (((2,), (2,)), ((1,), (0,)))
    model = _export(
        lambda left, right: jax.lax.dot_general(left, right, axes),
        shapes,
        optimized=optimized,
    )
    _checked(model)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("route", ["general", "tensor", "batch"])
def test_two_independent_contractions_compose_with_subtraction(
    optimized: bool, route: str
) -> None:
    if route == "general":
        shape, axes = (3, 3), (((0,), (0,)), ((), ()))
    elif route == "tensor":
        shape, axes = (3, 3), (((1,), (0,)), ((), ()))
    else:
        (batch,) = jax_export.symbolic_shape("B")
        shape, axes = (batch, 3, 3), (((2,), (1,)), ((0,), (0,)))

    def compose(left: Any, right: Any, other_left: Any, other_right: Any) -> Any:
        return jax.lax.sub(
            jax.lax.dot_general(left, right, axes),
            jax.lax.dot_general(other_left, other_right, axes),
        )

    model = _export(compose, (shape,) * 4, optimized=optimized)
    _checked(model)
    result = model.meta[structure.RESULT_KEY]
    assert result.expected_equations == result.checked_equations == 3


def _grouping_model(*, optimized: bool = True, extra_input: bool = False) -> ir.Model:
    (batch,) = jax_export.symbolic_shape("B")
    shapes = ((batch, 2, 3, 4), (4, 5, 6))
    axes = (((3,), (0,)), ((), ()))
    if extra_input:
        return _export(
            lambda left, right, other: (jax.lax.dot_general(left, right, axes), other),
            (*shapes, shapes[0]),
            optimized=optimized,
        )
    return _export(
        lambda left, right: jax.lax.dot_general(left, right, axes),
        shapes,
        optimized=optimized,
    )


@pytest.mark.parametrize("optimized", [False, True])
def test_dynamic_shape_vectors_follow_original_native_symbol_origin(
    optimized: bool,
) -> None:
    model = _grouping_model(optimized=optimized)
    _checked(model)
    assert normalized_shape_dimensions(tuple(model.graph.outputs[0].shape)) == (
        "B",
        2,
        3,
        5,
        6,
    )


def _square_general_model() -> ir.Model:
    return _export(
        lambda left, right: jax.lax.dot_general(left, right, (((0,), (0,)), ((), ()))),
        ((3, 3), (3, 3)),
    )


@pytest.mark.parametrize("fault", ["source-edge", "permutation", "operator"])
def test_same_typed_contraction_mutations_are_failed(fault: str) -> None:
    model = _square_general_model()
    _checked(model)
    matrix = next(node for node in model.graph if node.op_type == "MatMul")
    transpose = next(node for node in model.graph if node.op_type == "Transpose")
    if fault == "source-edge":
        transpose.replace_input_with(0, model.graph.inputs[1])
    elif fault == "permutation":
        transpose.attributes["perm"] = ir.AttrInt64s("perm", [0, 1])
    else:
        matrix.op_type = "Sub"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    assert structure.validate_source_structure(model).status == "failed"


def test_unimplemented_valid_gemm_form_is_a_structural_gap() -> None:
    model = _export(
        lambda left, right: jax.lax.dot_general(left, right, (((1,), (0,)), ((), ()))),
        ((3, 3), (3, 3)),
    )
    _checked(model)
    next(node for node in model.graph if node.op_type == "MatMul").op_type = "Gemm"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    result = structure.validate_source_structure(model)
    assert result.status == "unverified"
    assert result.reasons == ("dot-lowering-form-unavailable",)


@pytest.mark.parametrize("fault", ["allowzero", "shape-storage", "local-cast"])
def test_grouping_attributes_storage_and_local_rounding_are_checked(fault: str) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    dtypes = (
        (np.float64, np.float32) if fault == "local-cast" else (np.float32, np.float32)
    )
    model = _export(_reverse_weight, ((4, 3), (batch, 5, 3)), dtypes)
    _checked(model)
    if fault == "local-cast":
        node = next(node for node in model.graph if node.op_type == "Cast")
        node.attributes["to"] = ir.AttrInt64("to", ir.DataType.FLOAT.value)
    else:
        node = next(node for node in model.graph if node.op_type == "Reshape")
        if fault == "allowzero":
            node.attributes["allowzero"] = ir.AttrInt64("allowzero", 0)
        else:
            leaf = node.inputs[1]
            assert leaf is not None
            leaf.const_value = ir.tensor(np.asarray([1, 3, 4], dtype=np.int64))
    assert structure.validate_source_structure(model).status == "failed"


@pytest.mark.parametrize("fault", ["symbol-source", "symbol-axis", "piece-order"])
def test_shape_program_never_uses_same_named_or_same_shaped_substitutes(
    fault: str,
) -> None:
    model = _grouping_model(extra_input=True)
    _checked(model)
    if fault == "symbol-source":
        node = next(node for node in model.graph if node.op_type == "Shape")
        node.replace_input_with(0, model.graph.inputs[2])
    elif fault == "symbol-axis":
        gather = next(node for node in model.graph if node.op_type == "Gather")
        index = gather.inputs[1]
        assert index is not None
        index.const_value = ir.tensor(np.asarray([1], dtype=np.int64))
    else:
        concat = next(node for node in model.graph if node.op_type == "Concat")
        first, second = concat.inputs[1], concat.inputs[2]
        concat.replace_input_with(1, second)
        concat.replace_input_with(2, first)
    assert structure.validate_source_structure(model).status == "failed"


@pytest.mark.parametrize(
    "route", ["outer", "empty-output", "empty-contraction", "fp16-work"]
)
def test_unimplemented_legal_dot_forms_remain_exports_with_unverified_structure(
    route: str,
) -> None:
    dtypes = None
    preferred = None
    if route == "outer":
        shapes, axes = ((2,), (3,)), (((), ()), ((), ()))
    elif route == "empty-output":
        shapes, axes = ((0, 3), (3, 2)), (((1,), (0,)), ((), ()))
    elif route == "empty-contraction":
        shapes, axes = ((2, 0), (0, 3)), (((1,), (0,)), ((), ()))
    else:
        shapes, axes = ((2, 3), (3, 4)), (((1,), (0,)), ((), ()))
        dtypes, preferred = (np.float16, np.float16), np.float32
    model = _export(
        lambda left, right: jax.lax.dot_general(
            left, right, axes, preferred_element_type=preferred
        ),
        shapes,
        dtypes,
    )
    assert model.meta[structure.RESULT_KEY].status == "unverified"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def test_native_fp64_enum_precision_is_frozen_without_runtime_acceptance() -> None:
    model = _export(
        lambda left, right: _reverse_weight(
            left, right, precision=jax.lax.Precision.HIGH
        ),
        ((4, 3), (2, 5, 3)),
        (np.float64, np.float64),
    )
    _checked(model)
    rule = model.meta[structure._LEDGER].scopes[0].equations[0].rule
    assert rule.precision == ("HIGH", "HIGH")


@pytest.mark.parametrize(
    "parameter", ["precision", "preferred", "placement", "unknown", "identity"]
)
def test_unreviewed_native_dot_semantics_do_not_receive_a_rule(parameter: str) -> None:
    spec = jax.ShapeDtypeStruct((3, 3), np.float32)
    equation = jax.make_jaxpr(
        lambda left, right: jax.lax.dot_general(left, right, (((0,), (0,)), ((), ())))
    )(spec, spec).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.outvars
    )
    assert freeze_dot_structure_rule(equation, inputs, outputs) is not None
    params = dict(equation.params)
    if parameter == "precision":
        params["precision"] = (jax.lax.Precision.HIGH,) * 2
    elif parameter == "preferred":
        params["preferred_element_type"] = np.dtype(np.float64)
    elif parameter == "placement":
        params["out_sharding"] = object()
    elif parameter == "unknown":
        params["unreviewed_source_parameter"] = object()
    changed = equation.replace(params=params)
    if parameter == "identity":
        changed = changed.replace(primitive=Primitive("dot_general"))
    assert freeze_dot_structure_rule(changed, inputs, outputs) is None


def test_frozen_dot_plan_retains_no_native_trace_scope_or_array() -> None:
    (native,) = jax_export.symbolic_shape("B")
    model = _grouping_model()
    _checked(model)

    def inspect(value: Any) -> None:
        assert not isinstance(
            value,
            (
                Jaxpr,
                JaxprEqn,
                Var,
                Literal,
                np.ndarray,
                type(native),
                type(native.scope),
            ),
        )
        if is_dataclass(value) and not isinstance(value, type):
            for member in fields(value):
                inspect(getattr(value, member.name))
        elif isinstance(value, dict):
            for key, child in value.items():
                inspect(key)
                inspect(child)
        elif isinstance(value, (tuple, list, set)):
            for child in value:
                inspect(child)

    inspect(vars(model.meta[structure._LEDGER]))


def test_dot_observation_preserves_serialized_artifact_report_and_warnings() -> None:
    (batch,) = jax_export.symbolic_shape("B")
    fn = _reverse_weight
    original = IRContext.__init__

    def disabled(self: IRContext, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        self._source_structure = None

    results = []
    for enabled in (False, True):
        observer = (
            nullcontext() if enabled else patch.object(IRContext, "__init__", disabled)
        )
        with observer, warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            model = _export(fn, ((4, 3), (batch, 5, 3)), (np.float64, np.float32))
        results.append(
            (
                ir.to_proto(model).SerializeToString(deterministic=True),
                get_precision_report(model),
                tuple((entry.category, str(entry.message)) for entry in observed),
            )
        )
        if enabled:
            _checked(model)
    assert results[0] == results[1]
