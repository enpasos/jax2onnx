# tests/extra_tests/converter/test_source_shape_structure.py

"""Actual SOURCE shape compositions and independent provenance mutations."""

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
from jax2onnx.plugins.jax.lax._source_shape_structure import freeze_shape_structure_rule
from jax2onnx.plugins.jax.lax.reshape import _propagate_shape_helper_dtype
from jax2onnx.precision import CheckStatus, get_precision_report


def _export(
    fn: Any, shapes: Any, *, dtypes: Any = None, optimized: bool = True
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "layout", ["static", "scalar", "permuted", "symbol", "symbol-permuted"]
)
def test_real_reshape_preserves_native_order_and_exact_shape(
    dtype: Any, optimized: bool, layout: str
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    if layout == "static":
        source, target, permutation = (2, 3, 4), (6, 4), None
    elif layout == "scalar":
        source, target, permutation = (1, 1), (), None
    elif layout == "permuted":
        source, target, permutation = (2, 3, 4), (8, 3), (2, 0, 1)
    else:
        source, target = (batch, 2, 3), (batch, 6)
        permutation = (0, 2, 1) if layout == "symbol-permuted" else None
    model = _export(
        lambda x: jax.lax.reshape(x, target, dimensions=permutation),
        (source,),
        dtypes=(dtype,),
        optimized=optimized,
    )
    _checked(model)
    assert normalized_shape_dimensions(tuple(model.graph.outputs[0].shape)) == tuple(
        "B" if dim is batch else dim for dim in target
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "layout",
    [
        "scalar",
        "middle",
        "same-rank",
        "right-aligned",
        "leading-symbol",
        "nonleading-symbol",
    ],
)
def test_real_broadcast_keeps_the_supplied_mapped_axes(
    dtype: Any, optimized: bool, layout: str
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    if layout == "scalar":
        source, target, dimensions = (), (2, 3), ()
    elif layout == "middle":
        source, target, dimensions = (2, 3), (2, 4, 3), (0, 2)
    elif layout == "same-rank":
        source, target, dimensions = (1, 3), (2, 3), (0, 1)
    elif layout == "right-aligned":
        source, target, dimensions = (3,), (2, 3), (1,)
    elif layout == "leading-symbol":
        source, target, dimensions = (batch, 3), (batch, 2, 3), (0, 2)
    else:
        source, target, dimensions = (batch, 3), (2, 3, batch, 4), (1, 2)

    def broadcast(x: Any) -> Any:
        # Keep formal admission unchanged while exercising an internal B axis.
        if layout == "nonleading-symbol":
            x = jax.lax.transpose(x, (1, 0))
        return jax.lax.broadcast_in_dim(x, target, dimensions)

    model = _export(
        broadcast,
        (source,),
        dtypes=(dtype,),
        optimized=optimized,
    )
    _checked(model)
    assert normalized_shape_dimensions(tuple(model.graph.outputs[0].shape)) == tuple(
        "B" if dim is batch else dim for dim in target
    )


def _composition(x: Any) -> Any:
    rounded = jax.lax.convert_element_type(x, np.float64)
    flat = jax.lax.reshape(rounded, (x.shape[0], 6))
    expanded = jax.lax.broadcast_in_dim(flat, (x.shape[0], 4, 6), (0, 2))
    moved = jax.lax.transpose(expanded, (1, 0, 2))
    return jax.lax.sub(moved, moved)


@pytest.mark.parametrize("optimized", [False, True])
def test_normal_shape_composition_preserves_nonleading_atomic_symbol(
    optimized: bool,
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    model = _export(_composition, ((batch, 2, 3),), optimized=optimized)
    _checked(model)
    assert normalized_shape_dimensions(tuple(model.graph.outputs[0].shape)) == (
        4,
        "B",
        6,
    )
    assert model.graph.outputs[0].dtype == ir.DataType.DOUBLE


def _static_model(operation: str, *, other: bool = False) -> ir.Model:
    def shape(x: Any) -> Any:
        return (
            jax.lax.reshape(x, (3, 2))
            if operation == "reshape"
            else jax.lax.broadcast_in_dim(x, (2, 4, 3), (0, 2))
        )

    fn = (lambda x, y: (shape(x), y)) if other else shape
    return _export(fn, ((2, 3),) * (2 if other else 1))


def _storage(value: ir.Value, array: Any) -> None:
    producer = value.producer()
    tensor = ir.tensor(array)
    if producer is not None and producer.op_type == "Constant":
        producer.attributes["value"] = ir.AttrTensor("value", tensor)
    else:
        value.const_value = tensor


@pytest.mark.parametrize("operation", ["reshape", "broadcast"])
@pytest.mark.parametrize(
    "fault", ["source-edge", "operator", "dtype", "shape-vector", "attribute"]
)
def test_static_shape_contradictions_are_failed(operation: str, fault: str) -> None:
    model = _static_model(operation, other=True)
    _checked(model)
    layout = next(
        node
        for node in model.graph
        if node.op_type == ("Reshape" if operation == "reshape" else "Expand")
    )
    if fault == "source-edge":
        data = next(node for node in model.graph if node.op_type == "Reshape")
        data.replace_input_with(0, model.graph.inputs[1])
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
    elif fault == "operator":
        layout.op_type = "Add"
    elif fault == "dtype":
        layout.outputs[0].dtype = ir.DataType.DOUBLE
    elif fault == "attribute":
        layout.attributes["allowzero"] = ir.AttrInt64("allowzero", 1)
    else:
        vector = layout.inputs[1]
        assert vector is not None
        producer = vector.producer()
        if producer is not None and producer.op_type == "Concat":
            vector = producer.inputs[0]
            assert vector is not None
            _storage(vector, np.asarray([9], dtype=np.int64))
        else:
            replacement = [2, 3] if operation == "reshape" else [4, 2, 3]
            _storage(vector, np.asarray(replacement, dtype=np.int64))
    assert structure.validate_source_structure(model).status == "failed"


@pytest.mark.parametrize("operation", ["reshape", "broadcast"])
@pytest.mark.parametrize("fault", ["index", "concat-axis", "static-piece"])
def test_dynamic_shape_program_checks_indices_attributes_and_storage(
    operation: str, fault: str
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    if operation == "reshape":
        fn, source = lambda x: jax.lax.reshape(x, (batch, 6)), (batch, 2, 3)
    else:

        def fn(x: Any) -> Any:
            return jax.lax.broadcast_in_dim(
                jax.lax.transpose(x, (1, 0)), (2, 3, batch, 4), (1, 2)
            )

        source = (batch, 3)
    model = _export(fn, (source,), optimized=False)
    _checked(model)
    if fault == "index":
        gather = next(node for node in model.graph if node.op_type == "Gather")
        index = gather.inputs[1]
        assert index is not None
        _storage(
            index, np.asarray([1 if operation == "reshape" else 0], dtype=np.int64)
        )
    else:
        concat = next(node for node in model.graph if node.op_type == "Concat")
        if fault == "concat-axis":
            concat.attributes["axis"] = ir.AttrInt64("axis", 1)
        else:
            piece = next(
                value
                for value in concat.inputs
                if value is not None and value.producer() is None
            )
            _storage(piece, np.asarray([9], dtype=np.int64))
    assert structure.validate_source_structure(model).status == "failed"


def test_reshape_scalar_bridge_axes_are_checked_before_cancellation() -> None:
    (batch,) = jax_export.symbolic_shape("B")
    model = _export(
        lambda x: jax.lax.reshape(x, (batch, 6)), ((batch, 2, 3),), optimized=False
    )
    _checked(model)
    unsqueeze = next(node for node in model.graph if node.op_type == "Unsqueeze")
    axes = unsqueeze.inputs[1]
    assert axes is not None
    _storage(axes, np.asarray([1], dtype=np.int64))
    assert structure.validate_source_structure(model).status == "failed"


@pytest.mark.parametrize("optimized", [False, True])
def test_actual_reshape_shape_helpers_have_intrinsic_integer_metadata(
    optimized: bool,
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    model = _export(
        lambda x: jax.lax.reshape(x, (batch, 6)), ((batch, 2, 3),), optimized=optimized
    )
    _checked(model)
    expected = {
        "Shape": (3,),
        "Gather": (1,),
        "Squeeze": (),
        "Unsqueeze": (1,),
        "Concat": (2,),
    }
    seen = set()
    for node in model.graph:
        if node.op_type not in expected:
            continue
        output = node.outputs[0]
        assert isinstance(output.type, ir.TensorType)
        assert output.dtype == ir.DataType.INT64
        assert (
            output.shape is not None and tuple(output.shape) == expected[node.op_type]
        )
        seen.add(node.op_type)
    assert {"Shape", "Gather", "Concat"} <= seen
    if not optimized:
        assert {"Squeeze", "Unsqueeze"} <= seen


@pytest.mark.parametrize(
    ("input_dtypes", "declared_dtype", "expected_dtype"),
    [
        ((ir.DataType.INT64, None), None, None),
        ((ir.DataType.INT64, ir.DataType.INT32), None, None),
        ((), None, None),
        ((ir.DataType.INT32,), None, ir.DataType.INT32),
        ((ir.DataType.INT64, ir.DataType.INT64), ir.DataType.FLOAT, ir.DataType.FLOAT),
    ],
)
def test_shape_dtype_propagation_requires_known_actual_agreement(
    input_dtypes: tuple[ir.DataType | None, ...],
    declared_dtype: ir.DataType | None,
    expected_dtype: ir.DataType | None,
) -> None:
    operands = [
        ir.Value(
            name=f"part{index}",
            type=None if dtype is None else ir.TensorType(dtype),
            shape=ir.Shape((1,)),
        )
        for index, dtype in enumerate(input_dtypes)
    ]
    output = ir.Value(
        name="helper",
        shape=ir.Shape((len(operands),)),
        type=None if declared_dtype is None else ir.TensorType(declared_dtype),
    )
    before = [value.type for value in operands]
    _propagate_shape_helper_dtype(output, operands)
    assert output.dtype == expected_dtype
    assert [value.type for value in operands] == before


@pytest.mark.parametrize("operation", ["Unsqueeze", "Concat"])
@pytest.mark.parametrize(
    "fault", ["missing-type", "wrong-type", "missing-shape", "wrong-shape"]
)
def test_shape_helper_metadata_is_neither_assumed_nor_silently_repaired(
    operation: str,
    fault: str,
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    model = _export(
        lambda x: jax.lax.reshape(x, (batch, 6)), ((batch, 2, 3),), optimized=False
    )
    _checked(model)
    output = next(node.outputs[0] for node in model.graph if node.op_type == operation)
    if fault == "missing-type":
        output.type = None
    elif fault == "wrong-type":
        output.type = ir.TensorType(ir.DataType.FLOAT)
    elif fault == "missing-shape":
        output.shape = None
    else:
        output.shape = ir.Shape((9,))
    result = structure.validate_source_structure(model)
    assert result.status == ("unverified" if fault.startswith("missing") else "failed")
    assert result.reasons == (
        ("shape-value-type-unavailable",)
        if fault.startswith("missing")
        else ("shape-value-type-or-shape-differs",)
    )


def test_native_reshape_permutation_cannot_be_replaced_by_same_typed_order() -> None:
    model = _export(lambda x: jax.lax.reshape(x, (4,), dimensions=(1, 0)), ((2, 2),))
    _checked(model)
    transpose = next(node for node in model.graph if node.op_type == "Transpose")
    transpose.attributes["perm"] = ir.AttrInt64s("perm", [0, 1])
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    assert structure.validate_source_structure(model).status == "failed"


def test_shape_constant_uses_authoritative_attribute_and_covers_its_node() -> None:
    model = _static_model("reshape")
    _checked(model)
    reshape = next(node for node in model.graph if node.op_type == "Reshape")
    leaf = ir.val("shape_constant", ir.DataType.INT64, [2])
    leaf.const_value = ir.tensor(np.asarray([3, 2], dtype=np.int64))
    constant = ir.Node(
        "",
        "Constant",
        [],
        outputs=[leaf],
        attributes=[
            ir.AttrTensor("value", ir.tensor(np.asarray([3, 2], dtype=np.int64)))
        ],
    )
    model.graph.insert_before(reshape, constant)
    reshape.replace_input_with(1, leaf)
    assert structure.validate_source_structure(model).status == "validated"
    constant.attributes["value"] = ir.AttrTensor(
        "value", ir.tensor(np.asarray([2, 3], dtype=np.int64))
    )
    assert structure.validate_source_structure(model).reasons == (
        "literal-storage-differs",
    )


@pytest.mark.parametrize("overridable_default", [False, True])
def test_runtime_input_cannot_replace_an_immutable_shape_vector(
    overridable_default: bool,
) -> None:
    model = _export(
        lambda x, vector: (jax.lax.reshape(x, (3, 2)), vector),
        ((2, 3), (2,)),
        dtypes=(np.float32, np.int64),
    )
    _checked(model)
    runtime = model.graph.inputs[1]
    if overridable_default:
        runtime.const_value = ir.tensor(np.asarray([3, 2], dtype=np.int64))
        model.graph.register_initializer(runtime)
    reshape = next(node for node in model.graph if node.op_type == "Reshape")
    reshape.replace_input_with(1, runtime)
    assert structure.validate_source_structure(model).reasons == (
        "literal-runtime-input-substitute",
    )


@pytest.mark.parametrize("fault", ["none", "static-tail", "data-edge"])
def test_external_native_symbol_origin_is_a_gap_without_hiding_known_faults(
    fault: str,
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    weight = np.ones((3,), dtype=np.float32)
    model = _export(
        lambda x, other: (jax.lax.broadcast_in_dim(weight, (batch, 3), (1,)), x, other),
        ((batch, 3), (3,)),
        optimized=False,
    )
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "unverified"
    assert result.reasons == ("shape-symbol-origin-unavailable",)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    if fault == "static-tail":
        expand = next(node for node in model.graph if node.op_type == "Expand")
        target = expand.inputs[1]
        assert target is not None and target.producer() is not None
        concat = target.producer()
        assert concat is not None and concat.op_type == "Concat"
        piece = concat.inputs[1]
        assert piece is not None
        _storage(piece, np.asarray([4], dtype=np.int64))
        assert structure.validate_source_structure(model).status == "failed"
    elif fault == "data-edge":
        reshape = next(node for node in model.graph if node.op_type == "Reshape")
        reshape.replace_input_with(0, model.graph.inputs[1])
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
        assert structure.validate_source_structure(model).reasons == (
            "shape-source-input-edge-differs",
        )


def test_last_bound_same_named_other_operand_is_not_symbolic_authority() -> None:
    (batch,) = jax_export.symbolic_shape("B")
    model = _export(
        lambda x, other: (jax.lax.broadcast_in_dim(x, (batch, 2, 3), (0, 2)), other),
        ((batch, 3), (batch, 3)),
        optimized=False,
    )
    assert model.meta[structure.RESULT_KEY].status == "unverified"
    assert model.meta[structure.RESULT_KEY].reasons == (
        "shape-symbol-origin-unavailable",
    )
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize(
    "route", ["empty-reshape", "empty-broadcast", "constant-broadcast", "fp16"]
)
def test_unimplemented_legal_shape_forms_still_export_with_gaps(route: str) -> None:
    if route == "empty-reshape":
        fn, shapes, dtypes = lambda x: jax.lax.reshape(x, (0, 6)), ((2, 0, 3),), None
    elif route == "empty-broadcast":
        fn, shapes, dtypes = (
            lambda x: jax.lax.broadcast_in_dim(x, (0, 3), (0, 1)),
            ((1, 3),),
            None,
        )
    elif route == "constant-broadcast":
        fn, shapes, dtypes = (
            lambda: jax.lax.broadcast_in_dim(
                np.ones((3,), dtype=np.float32), (2, 3), (1,)
            ),
            (),
            None,
        )
    else:
        fn, shapes, dtypes = (
            lambda x: jax.lax.reshape(x, (3, 2)),
            ((2, 3),),
            (np.float16,),
        )
    model = _export(fn, shapes, dtypes=dtypes)
    assert model.meta[structure.RESULT_KEY].status == "unverified"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize(
    "fault",
    ["native-scope", "string-shape", "permutation", "placement", "unknown", "identity"],
)
def test_source_parameters_need_actual_native_shape_authority(fault: str) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    equation = jax.make_jaxpr(lambda x: jax.lax.reshape(x, (batch, 6)))(
        jax.ShapeDtypeStruct((batch, 2, 3), np.float32)
    ).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.outvars
    )
    assert freeze_shape_structure_rule(equation, inputs, outputs) is not None
    params = dict(equation.params)
    if fault == "native-scope":
        (foreign,) = jax_export.symbolic_shape("B")
        params["new_sizes"] = (foreign, 6)
    elif fault == "string-shape":
        params["new_sizes"] = ("B", 6)
    elif fault == "permutation":
        params["dimensions"] = (0, 0, 2)
    elif fault == "placement":
        params["out_sharding"] = object()
    elif fault == "unknown":
        params["unreviewed_parameter"] = object()
    changed = equation.replace(params=params)
    if fault == "identity":
        changed = changed.replace(primitive=Primitive("reshape"))
    assert freeze_shape_structure_rule(changed, inputs, outputs) is None


def test_broadcast_parameter_axis_mapping_is_frozen_from_source() -> None:
    (batch,) = jax_export.symbolic_shape("B")
    equation = jax.make_jaxpr(
        lambda x: jax.lax.broadcast_in_dim(x, (batch, 2, 3), (0, 2))
    )(jax.ShapeDtypeStruct((batch, 3), np.float32)).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.outvars
    )
    assert freeze_shape_structure_rule(equation, inputs, outputs) is not None
    params = {**equation.params, "broadcast_dimensions": (0, 0)}
    assert (
        freeze_shape_structure_rule(equation.replace(params=params), inputs, outputs)
        is None
    )


@pytest.mark.parametrize("optimized", [False, True])
def test_shape_observation_preserves_allocation_artifact_report_and_warnings(
    optimized: bool,
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
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
            model = _export(_composition, ((batch, 2, 3),), optimized=optimized)
        results.append(
            (
                len(list(model.graph)),
                len(model.graph.initializers),
                ir.to_proto(model).SerializeToString(deterministic=True),
                get_precision_report(model),
                tuple((entry.category, str(entry.message)) for entry in observed),
            )
        )
        if enabled:
            _checked(model)
    assert results[0] == results[1]


def test_shape_rules_retain_no_native_trace_scope_or_array() -> None:
    (native,) = jax_export.symbolic_shape("B")
    model = _export(_composition, ((native, 2, 3),))
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
