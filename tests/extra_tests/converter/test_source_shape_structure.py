# tests/extra_tests/converter/test_source_shape_structure.py

"""Actual SOURCE shape compositions and independent provenance mutations."""

from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace
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


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "shape,starts,limits,steps,dtype,original_empty",
    [
        ((3,), (0,), (0,), None, np.int32, True),
        ((5,), (1,), (4,), None, np.bool_, False),
        ((6,), (1,), (6,), (2,), np.uint32, False),
        ((2, 4), (0, 1), (2, 4), (1, 2), np.float32, False),
        ((0, 3), (0, 1), (0, 3), None, np.float64, False),
        ((3,), (0,), (3,), None, np.int32, False),
    ],
)
def test_static_slice_original_geometry_has_complete_structure(
    optimized, shape, starts, limits, steps, dtype, original_empty
) -> None:
    def original(x: Any) -> Any:
        if original_empty:
            return x[:0]
        return jax.lax.slice(x, starts, limits, steps)

    model = _export(original, (shape,), dtypes=(dtype,), optimized=optimized)
    _checked(model)
    strides = steps or (1,) * len(shape)
    expected = tuple(
        (b - a + s - 1) // s for a, b, s in zip(starts, limits, strides, strict=True)
    )
    assert tuple(model.graph.outputs[0].shape) == expected
    if starts != (0,) * len(shape) or limits != shape or strides != (1,) * len(shape):
        assert sum(n.op_type == "Slice" for n in model.graph) == 1


@pytest.mark.parametrize(
    "fault",
    ["starts", "limits", "axes", "steps", "source-edge", "dtype", "shape", "attribute"],
)
def test_static_slice_storage_edges_and_output_mutations_fail(fault: str) -> None:
    def original(x: Any, other: Any) -> Any:
        return jax.lax.slice(x, (1,), (5,), (2,)), other

    model = _export(original, ((6,), (6,)), dtypes=(np.int32, np.int32))
    _checked(model)
    node = next(n for n in model.graph if n.op_type == "Slice")
    if fault in {"starts", "limits", "axes", "steps"}:
        index = {"starts": 1, "limits": 2, "axes": 3, "steps": 4}[fault]
        _storage(node.inputs[index], np.asarray([7], dtype=np.int64))
    elif fault == "source-edge":
        assert model.graph.inputs[1] is not node.inputs[0]
        node.replace_input_with(0, model.graph.inputs[1])
    elif fault == "dtype":
        node.outputs[0].dtype = ir.DataType.UINT32
    elif fault == "shape":
        node.outputs[0].shape = ir.Shape((3,))
    else:
        node.attributes["unreviewed"] = ir.AttrInt64("unreviewed", 1)
    assert structure.validate_source_structure(model).status == "failed"


@pytest.mark.parametrize(
    "fault",
    [
        "primitive",
        "effects",
        "params",
        "params-type",
        "input-arity",
        "output-arity",
        "dtype",
        "encoding",
        "logical-shape",
        "output-shape",
        "weakness",
        "symbol",
        "symbol-axis",
        "logical-symbol-axis",
        "negative-start",
        "past-input-limit",
        "negative-step",
        "zero-step",
        "bounds-overflow",
        "steps-overflow",
        "bool-bound",
        "list-bound",
    ],
)
def test_static_slice_unowned_or_incompatible_parameters_do_not_freeze(
    fault: str,
) -> None:
    eqn = jax.make_jaxpr(lambda x: jax.lax.slice(x, (1,), (5,), (2,)))(
        jax.ShapeDtypeStruct((6,), np.int32)
    ).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in eqn.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in eqn.outvars
    )
    assert freeze_shape_structure_rule(eqn, inputs, outputs) is not None
    params = dict(eqn.params)
    if fault == "primitive":
        eqn = eqn.replace(primitive=Primitive("slice"))
    elif fault == "effects":
        eqn = eqn.replace(effects=frozenset({object()}))
    elif fault == "params":
        params["unreviewed"] = None
    elif fault == "params-type":

        class Params(dict):
            pass

        eqn = eqn.replace(params=Params(params))
    elif fault == "input-arity":
        inputs = ()
    elif fault == "output-arity":
        outputs = outputs * 2
    elif fault in {
        "dtype",
        "encoding",
        "logical-shape",
        "output-shape",
        "weakness",
        "symbol",
        "symbol-axis",
        "logical-symbol-axis",
    }:
        changes = {
            "dtype": {"logical_dtype": "uint32", "dtype": ir.DataType.UINT32},
            "encoding": {"encoding": "complex-real-pair"},
            "logical-shape": {"logical_shape": (1, 2)},
            "output-shape": {"shape": (3,), "logical_shape": (3,)},
            "weakness": {"weak_type": True},
            "symbol": {"symbol_token": object()},
            "symbol-axis": {"symbol_axis": 0},
            "logical-symbol-axis": {"logical_symbol_axis": 0},
        }[fault]
        outputs = (replace(outputs[0], **changes),)
    else:
        field, value = {
            "negative-start": ("start_indices", (-1,)),
            "past-input-limit": ("limit_indices", (7,)),
            "negative-step": ("strides", (-1,)),
            "zero-step": ("strides", (0,)),
            "bounds-overflow": ("limit_indices", (2**63,)),
            "steps-overflow": ("strides", (2**63,)),
            "bool-bound": ("start_indices", (True,)),
            "list-bound": ("start_indices", [1]),
        }[fault]
        params[field] = value
    if fault != "params-type":
        eqn = eqn.replace(params=params)
    if fault == "bool-bound":
        assert type(eqn.params["start_indices"][0]) is bool
    assert freeze_shape_structure_rule(eqn, inputs, outputs) is None


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


@pytest.mark.parametrize(
    "kind,shape", [("reshape", (2, 3)), ("broadcast", (3,)), ("scalar", ())]
)
@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.uint32])
def test_static_discrete_shape_contracts_reuse_exact_geometry(
    tmp_path, kind, shape, dtype
):
    def original(x):
        if kind == "reshape":
            return jax.lax.reshape(x, (3, 2), dimensions=(1, 0))
        if kind == "broadcast":
            return jax.lax.broadcast_in_dim(x, (2, 3), (1,))
        assert kind == "scalar"
        return jax.lax.broadcast_in_dim(x, (2, 3), ())

    model = _export(original, (shape,), dtypes=(dtype,))
    _retain_discrete_shape_evidence(tmp_path, model)
    _checked(model)
    assert model.graph.outputs[0].dtype == ir.DataType.from_numpy(np.dtype(dtype))


def _retain_discrete_shape_evidence(tmp_path, model, result=None):
    import json
    from dataclasses import asdict

    (tmp_path / "actual-source-shape.onnx").write_bytes(
        ir.to_proto(model).SerializeToString()
    )
    actual = model.meta[structure.RESULT_KEY] if result is None else result
    (tmp_path / "actual-source-structure.json").write_text(
        json.dumps(asdict(actual), sort_keys=True) + "\n"
    )


def _raw_discrete_identity_reshape(dtype):
    return _export(
        lambda x: jax.lax.reshape(x, (3, 2), dimensions=(1, 0)),
        ((2, 3),),
        dtypes=(dtype,),
        optimized=False,
    )


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.uint32])
def test_source_identity_reshape_preserves_marked_result_and_removes_unmarked_internal(
    tmp_path, dtype
):
    from jax2onnx.converter import ir_optimizations as optimizer

    model = _raw_discrete_identity_reshape(dtype)
    original = model.graph.outputs[0]
    marked = original.producer()
    assert marked is not None and marked.op_type == "Reshape"
    assert structure._VALUES in original.meta
    marker = original.meta[structure._VALUES]
    data, shape = marked.inputs
    intermediate = ir.Value(
        name="unmarked_identity_result", type=data.type, shape=data.shape
    )
    unmarked = ir.Node("", "Reshape", [data, shape], outputs=[intermediate])
    model.graph.insert_before(marked, unmarked)
    marked.replace_input_with(0, intermediate)
    assert structure._VALUES not in intermediate.meta
    optimizer.remove_identity_reshapes_ir(model.graph)
    result = structure.validate_source_structure(model)
    _retain_discrete_shape_evidence(tmp_path, model, result)
    assert model.graph.outputs[0] is original and original.producer() is marked
    assert original.meta[structure._VALUES] is marker
    assert all(node is not unmarked for node in model.graph)
    assert marked.inputs[0] is data
    assert result.status == "validated", result.reasons
    _checked(model)


@pytest.mark.parametrize("marker", [None, (), "malformed"])
def test_source_identity_reshape_keeps_even_empty_or_malformed_marker(tmp_path, marker):
    from jax2onnx.converter import ir_optimizations as optimizer

    model = _raw_discrete_identity_reshape(np.bool_)
    original = model.graph.outputs[0]
    node = original.producer()
    assert structure._VALUES in original.meta
    original.meta[structure._VALUES] = marker
    optimizer.remove_identity_reshapes_ir(model.graph)
    result = structure.validate_source_structure(model)
    _retain_discrete_shape_evidence(tmp_path, model, result)
    assert model.graph.outputs[0] is original and original.producer() is node
    assert any(candidate is node for candidate in model.graph)
    assert result.status == "unverified"
    assert any("final-value-" in reason for reason in result.reasons)


@pytest.mark.parametrize("replacement", ["upstream", "fresh-same-form"])
def test_source_identity_reshape_unproved_same_named_replacement_is_rejected(
    tmp_path, replacement
):
    from jax2onnx.converter import ir_optimizations as optimizer

    model = _raw_discrete_identity_reshape(np.int32)
    original = model.graph.outputs[0]
    node = original.producer()
    assert node is not None and node.op_type == "Reshape"
    assert structure._VALUES in original.meta
    optimizer.remove_identity_reshapes_ir(model.graph)
    assert model.graph.outputs[0] is original and original.producer() is node
    if replacement == "upstream":
        substitute = node.inputs[0]
        substitute.name = original.name
    else:
        substitute = ir.Value(
            name=original.name, type=original.type, shape=original.shape
        )
        model.graph.insert_before(
            node, ir.Node("", "Reshape", list(node.inputs), outputs=[substitute])
        )
    assert substitute is not original and substitute.name == original.name
    assert substitute.dtype == original.dtype and tuple(substitute.shape) == tuple(
        original.shape
    )
    assert structure._VALUES not in substitute.meta
    ir.convenience.replace_all_uses_with(
        original, substitute, replace_graph_outputs=True
    )
    model.graph.remove(node)
    result = structure.validate_source_structure(model)
    _retain_discrete_shape_evidence(tmp_path, model, result)
    assert result.status == "unverified"
    assert "final-value-witness-lost" in result.reasons


@pytest.mark.parametrize("form", ["missing", "zero", "one", "direct"])
def test_empty_discrete_broadcast_alignment_requires_literal_zero_semantics(form):
    import json

    from scripts import source_precision_inventory as inventory_owner
    from tests.extra_tests.converter.source_precision_diagnostics import (
        typed_trace_manifest,
    )

    closed = jax.make_jaxpr(lambda x: jax.lax.broadcast_in_dim(x, (2, 0, 3), (1, 2)))(
        jax.ShapeDtypeStruct((0, 3), np.int32)
    )
    equation = closed.jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in equation.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in equation.outvars
    )
    rule = freeze_shape_structure_rule(equation, inputs, outputs)
    assert rule is not None
    value_info = onnx.helper.make_tensor_value_info
    initializers = [
        onnx.helper.make_tensor(
            "expanded-shape", onnx.TensorProto.INT64, [3], [2, 0, 3]
        )
    ]
    nodes = []
    intermediates = []
    edge = "x"
    if form != "direct":
        initializers.append(
            onnx.helper.make_tensor(
                "aligned-shape", onnx.TensorProto.INT64, [3], [1, 0, 3]
            )
        )
        attrs = {} if form == "missing" else {"allowzero": 0 if form == "zero" else 1}
        nodes.append(
            onnx.helper.make_node(
                "Reshape", ["x", "aligned-shape"], ["aligned"], **attrs
            )
        )
        # Stamp the expected form even for the counterexamples: metadata alone
        # must not turn default-zero-copy semantics into literal-zero semantics.
        intermediates.append(value_info("aligned", onnx.TensorProto.INT32, [1, 0, 3]))
        edge = "aligned"
    nodes.append(onnx.helper.make_node("Expand", [edge, "expanded-shape"], ["out"]))
    proto = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes,
            "empty-broadcast-alignment",
            [value_info("x", onnx.TensorProto.INT32, [0, 3])],
            [value_info("out", onnx.TensorProto.INT32, [2, 0, 3])],
            initializer=initializers,
            value_info=intermediates,
        ),
        opset_imports=[onnx.helper.make_opsetid("", 23)],
    )
    model = ir.from_proto(proto)

    def match_literal(actual, expected):
        assert actual is model.graph.initializers[actual.name]
        assert actual.producer() is None and actual.const_value is not None
        assert actual.dtype == expected.tensor.dtype
        assert tuple(actual.shape) == expected.tensor.shape
        assert (
            structure.ConstantContract.from_array(actual.const_value.numpy())
            == expected.storage
        )

    valid = form in {"one", "direct"}
    if valid:
        consumed = rule.check(
            model.graph,
            tuple(model.graph.inputs),
            tuple(model.graph.outputs),
            match_literal,
        )
        assert {node.op_type for node in consumed} == (
            {"Expand"} if form == "direct" else {"Reshape", "Expand"}
        )
    else:
        with pytest.raises(
            structure.StructureMismatch, match="shape-attribute-(set-)?differs"
        ):
            rule.check(
                model.graph,
                tuple(model.graph.inputs),
                tuple(model.graph.outputs),
                match_literal,
            )
    manifest = json.loads(json.dumps(typed_trace_manifest(closed)))
    closure = inventory_owner._source_bitwise_closure(manifest, proto)
    assert (closure is not None) is valid


@pytest.mark.parametrize(
    "fault",
    ["shape", "weakness", "encoding", "dtype", "symbol", "geometry", "bool-dimension"],
)
def test_static_discrete_shape_contracts_reject_forged_geometry(fault):
    native = jax.make_jaxpr(lambda x: jax.lax.broadcast_in_dim(x, (2, 3), (1,)))(
        jax.ShapeDtypeStruct((3,), np.int32)
    ).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.outvars
    )
    assert freeze_shape_structure_rule(native, inputs, outputs) is not None
    params = dict(native.params)
    if fault in {"geometry", "bool-dimension"}:
        params["broadcast_dimensions"] = (0,) if fault == "geometry" else (True,)
        eqn = native.replace(params=params)
    else:
        changes = {
            "shape": {"shape": (2, 1), "logical_shape": (2, 1)},
            "weakness": {"weak_type": True},
            "encoding": {"encoding": "packed-key"},
            "dtype": {"logical_dtype": "float32", "dtype": ir.DataType.FLOAT},
            "symbol": {"symbol_token": object()},
        }[fault]
        outputs = (replace(outputs[0], **changes),)
        eqn = native
    assert freeze_shape_structure_rule(eqn, inputs, outputs) is None
