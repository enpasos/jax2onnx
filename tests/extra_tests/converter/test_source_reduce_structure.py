# tests/extra_tests/converter/test_source_reduce_structure.py

"""Real SOURCE reductions and independent signature/geometry counterexamples."""

from contextlib import nullcontext
from dataclasses import replace
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
from jax2onnx._compat.jax import Primitive
from jax2onnx.converter import conversion_api
from jax2onnx.converter import source_structure as structure
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.converter.source_representation import normalized_shape_dimensions
from jax2onnx.plugins.jax.lax._source_reduce_structure import (
    _ReductionRule,
    freeze_reduction_structure_rule,
)
from jax2onnx.precision import CheckStatus, get_precision_report


def _operation(name: str) -> Any:
    return jax.lax.reduce_sum if name == "sum" else jax.lax.reduce_max


def _export(
    fn: Any,
    shapes: Any,
    *,
    dtype: Any = np.float32,
    opset: int = 21,
    optimized: bool = True,
) -> ir.Model:
    specs = [jax.ShapeDtypeStruct(shape, dtype) for shape in shapes]
    optimizer = (
        nullcontext()
        if optimized
        else patch.object(
            conversion_api, "optimize_graph", lambda model, **kwargs: None
        )
    )
    with jax.enable_x64(True), optimizer:
        model = ui._to_onnx(
            fn,
            specs,
            precision_policy=PrecisionPolicy.SOURCE,
            opset=opset,
            return_mode="ir",
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


@pytest.mark.parametrize("operation", ["sum", "max"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "layout", ["static", "retain-leading", "reindex", "remove-symbol", "all-axes"]
)
def test_real_reductions_remove_only_the_frozen_native_axes(
    operation: str,
    dtype: Any,
    optimized: bool,
    layout: str,
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    if layout == "static":
        shape, axes, expected = (2, 3, 4), (2, 0), (3,)
    elif layout == "retain-leading":
        shape, axes, expected = (batch, 3, 4), (2,), ("B", 3)
    elif layout == "reindex":
        shape, axes, expected = (batch, 3, 4), (0, 2), ("B",)
    elif layout == "remove-symbol":
        shape, axes, expected = (batch, 3, 4), (1,), (3, 4)
    else:
        shape, axes, expected = (batch, 3, 4), (0, 1, 2), ()

    def reduce(x: Any) -> Any:
        # The facade admits a leading symbol; exercise nonleading geometry
        # inside the ordinary native graph without changing that admission.
        if layout in {"reindex", "remove-symbol", "all-axes"}:
            x = jax.lax.transpose(x, (1, 0, 2))
        return _operation(operation)(x, axes=axes)

    model = _export(
        reduce,
        (shape,),
        dtype=dtype,
        optimized=optimized,
    )
    _checked(model)
    assert normalized_shape_dimensions(tuple(model.graph.outputs[0].shape)) == expected


@pytest.mark.parametrize(
    ("operation", "opset"),
    [("sum", 16), ("sum", 18), ("sum", 21), ("max", 21), ("max", 23)],
)
@pytest.mark.parametrize("optimized", [False, True])
def test_supported_source_opsets_use_their_actual_reduction_signature(
    operation: str, opset: int, optimized: bool
) -> None:
    model = _export(
        lambda x: _operation(operation)(x, axes=(1,)),
        ((2, 3),),
        opset=opset,
        optimized=optimized,
    )
    _checked(model)
    node = next(node for node in model.graph if node.op_type.startswith("Reduce"))
    assert len(node.inputs) == 2 and "axes" not in node.attributes


@pytest.mark.parametrize("operation", ["sum", "max"])
@pytest.mark.parametrize("optimized", [False, True])
def test_no_axis_reductions_keep_identity_and_exact_source_edges(
    operation: str, optimized: bool
) -> None:
    model = _export(
        lambda x, other: (_operation(operation)(x, axes=()), other),
        ((2, 3), (2, 3)),
        optimized=optimized,
    )
    _checked(model)
    assert not any(node.op_type.startswith("Reduce") for node in model.graph)
    model.graph.outputs[0] = model.graph.inputs[1]
    assert structure.validate_source_structure(model).status == "failed"


@pytest.mark.parametrize("operation", ["sum", "max"])
@pytest.mark.parametrize(
    "fault",
    [
        "source-edge",
        "operator",
        "axis",
        "keepdims",
        "dtype",
        "empty-axes",
        "axes-dtype",
        "noop-attribute",
        "opset",
    ],
)
def test_final_reduction_contradictions_are_failed(operation: str, fault: str) -> None:
    model = _export(
        lambda x, other: (_operation(operation)(x, axes=(1,)), other), ((3, 3), (3, 3))
    )
    _checked(model)
    node = next(node for node in model.graph if node.op_type.startswith("Reduce"))
    if fault == "source-edge":
        node.replace_input_with(0, model.graph.inputs[1])
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
    elif fault == "operator":
        node.op_type = "ReduceMean"
        onnx.checker.check_model(ir.to_proto(model), full_check=True)
    elif fault == "keepdims":
        node.attributes["keepdims"] = ir.AttrInt64("keepdims", 1)
    elif fault == "dtype":
        node.outputs[0].dtype = ir.DataType.DOUBLE
    elif fault == "noop-attribute":
        node.attributes["noop_with_empty_axes"] = ir.AttrInt64(
            "noop_with_empty_axes", 1
        )
    elif fault == "opset":
        model.graph.opset_imports[""] = 12 if operation == "sum" else 17
    else:
        axes = node.inputs[1]
        assert axes is not None
        if fault == "axis":
            axes.const_value = ir.tensor(np.asarray([0], dtype=np.int64))
            onnx.checker.check_model(ir.to_proto(model), full_check=True)
        elif fault == "empty-axes":
            axes.shape = ir.Shape((0,))
            axes.const_value = ir.tensor(np.asarray([], dtype=np.int64))
        else:
            axes.dtype = ir.DataType.INT32
            axes.const_value = ir.tensor(np.asarray([1], dtype=np.int32))
    assert structure.validate_source_structure(model).status == "failed"


def test_axes_constant_node_uses_authoritative_tensor_not_cached_annotation() -> None:
    model = _export(lambda x: jax.lax.reduce_sum(x, axes=(1,)), ((3, 3),))
    _checked(model)
    node = next(node for node in model.graph if node.op_type == "ReduceSum")
    expected = np.asarray([1], dtype=np.int64)
    leaf = ir.val("axes_constant", ir.DataType.INT64, [1])
    leaf.const_value = ir.tensor(expected.copy())
    constant = ir.Node(
        "",
        "Constant",
        [],
        outputs=[leaf],
        attributes=[ir.AttrTensor("value", ir.tensor(expected))],
    )
    model.graph.insert_before(node, constant)
    node.replace_input_with(1, leaf)
    assert structure.validate_source_structure(model).status == "validated"
    constant.attributes["value"] = ir.AttrTensor(
        "value", ir.tensor(np.asarray([0], dtype=np.int64))
    )
    assert structure.validate_source_structure(model).reasons == (
        "literal-storage-differs",
    )


def test_unimplemented_identity_decomposition_remains_an_exported_structural_gap() -> (
    None
):
    model = _export(lambda x: jax.lax.reduce_sum(x, axes=(1,)), ((3, 3),))
    _checked(model)
    node = next(node for node in model.graph if node.op_type == "ReduceSum")
    original = node.inputs[0]
    assert original is not None
    copied = ir.val("unreviewed_identity", ir.DataType.FLOAT, [3, 3])
    model.graph.insert_before(
        node, ir.Node("", "Identity", [original], outputs=[copied])
    )
    node.replace_input_with(0, copied)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    result = structure.validate_source_structure(model)
    assert result.status == "unverified"
    assert result.reasons == ("reduce-input-form-unavailable",)


@pytest.mark.parametrize("operation", ["sum", "max"])
@pytest.mark.parametrize("route", ["empty", "fp16"])
def test_legal_unimplemented_reduction_branches_still_export(
    operation: str, route: str
) -> None:
    shape, dtype = ((2, 0, 3), np.float32) if route == "empty" else ((2, 3), np.float16)
    model = _export(
        lambda x: _operation(operation)(x, axes=(1,)),
        (shape,),
        dtype=dtype,
        # FP16 is admitted at23; the unchanged scientific21 gate excludes it.
        opset=23 if route == "fp16" else 21,
    )
    assert model.meta[structure.RESULT_KEY].status == "unverified"
    assert "equation-0:rule-unavailable" in model.meta[structure.RESULT_KEY].reasons
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize(
    ("operation", "dtype", "opset", "reason"),
    [
        ("max", np.float32, 18, "original opset18/21 elementary intersection"),
        ("sum", np.float16, 21, "scientific opset21 operation/type intersection"),
        ("max", np.float16, 21, "scientific opset21 operation/type intersection"),
    ],
)
def test_current_profile_gaps_are_not_changed_by_structure_auditing(
    operation: str, dtype: Any, opset: int, reason: str
) -> None:
    # These are current migration gaps, not claims of an ONNX limitation.
    with pytest.raises(SourcePrecisionError, match=reason):
        _export(
            lambda x: _operation(operation)(x, axes=(1,)),
            ((2, 3),),
            dtype=dtype,
            opset=opset,
        )


def _native_rule(operation: str) -> _ReductionRule:
    spec = jax.ShapeDtypeStruct((3, 3), np.float32)
    equation = jax.make_jaxpr(lambda x: _operation(operation)(x, axes=(1,)))(
        spec
    ).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.outvars
    )
    rule = freeze_reduction_structure_rule(equation, inputs, outputs)
    assert isinstance(rule, _ReductionRule)
    return rule


@pytest.mark.parametrize(
    ("operation", "opset", "input_axes"),
    [("sum", 12, False), ("sum", 13, True), ("max", 17, False), ("max", 18, True)],
)
def test_schema_boundaries_are_checked_on_final_graph_imports(
    operation: str, opset: int, input_axes: bool
) -> None:
    # Operation-specific SOURCE admission is separate (including Max18's
    # current exclusion); this checks rule/schema boundaries without widening it.
    # This hand-built fixture isolates the reduction signature. Real floating
    # extrema also require the separately checked zero/NaN correction graph.
    rule = replace(_native_rule(operation), float_correction=False)
    source = ir.val("source", ir.DataType.FLOAT, [3, 3])
    output = ir.val("result", ir.DataType.FLOAT, [3])
    axes = ir.val("axes", ir.DataType.INT64, [1])
    axes.const_value = ir.tensor(np.asarray([1], dtype=np.int64))
    attributes = [ir.AttrInt64("keepdims", 0)]
    if not input_axes:
        attributes.append(ir.AttrInt64s("axes", [1]))
    node = ir.Node(
        "",
        rule.operation,
        [source, axes] if input_axes else [source],
        outputs=[output],
        attributes=attributes,
    )
    graph = ir.Graph(
        [source],
        [output],
        nodes=[node],
        name="reduction_fixture",
        opset_imports={"": opset},
    )
    if input_axes:
        graph.register_initializer(axes)
    seen = []

    def match(value: ir.Value | None, expected: structure.LiteralOperand) -> None:
        assert value is axes
        assert expected.tensor.dtype == ir.DataType.INT64 and expected.tensor.shape == (
            1,
        )
        assert expected.storage == structure.ConstantContract.from_array(
            np.asarray([1], dtype=np.int64)
        )
        seen.append(value)

    assert rule.check(graph, (source,), (output,), match) == (node,)
    assert len(seen) == int(input_axes)
    onnx.checker.check_model(
        ir.to_proto(ir.Model(graph, ir_version=10)), full_check=True
    )
    graph.opset_imports[""] = (
        rule.axes_input_since - 1 if input_axes else rule.axes_input_since
    )
    with pytest.raises(structure.StructureMismatch):
        rule.check(graph, (source,), (output,), match)
    del graph.opset_imports[""]
    with pytest.raises(
        structure.StructureUnavailable, match="reduce-opset-unavailable"
    ):
        rule.check(graph, (source,), (output,), match)


def test_no_axis_rule_accepts_only_alias_or_exact_identity() -> None:
    base = _native_rule("sum")
    rule = replace(base, axes=(), output=base.source)
    source = ir.val("source", ir.DataType.FLOAT, [3, 3])
    other = ir.val("other", ir.DataType.FLOAT, [3, 3])
    output = ir.val("result", ir.DataType.FLOAT, [3, 3])
    node = ir.Node("", "Identity", [source], outputs=[output])
    graph = ir.Graph([source, other], [output], nodes=[node], opset_imports={"": 21})

    def no_literal(value: ir.Value | None, expected: structure.LiteralOperand) -> None:
        pytest.fail("nonliteral fixture must not invoke literal matching")

    assert rule.check(graph, (source,), (source,), no_literal) == ()
    assert rule.check(graph, (source,), (output,), no_literal) == (node,)
    node.replace_input_with(0, other)
    with pytest.raises(
        structure.StructureMismatch, match="reduce-source-input-edge-differs"
    ):
        rule.check(graph, (source,), (output,), no_literal)


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "negative",
        "out-of-range",
        "dtype",
        "keepdims",
        "placement",
        "identity",
        "symbol-token",
    ],
)
def test_freezer_does_not_certify_unreviewed_native_semantics(fault: str) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    spec = jax.ShapeDtypeStruct((3, batch, 4), np.float32)
    equation = jax.make_jaxpr(lambda x: jax.lax.reduce_sum(x, axes=(0,)))(
        spec
    ).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.outvars
    )
    assert freeze_reduction_structure_rule(equation, inputs, outputs) is not None
    params = dict(equation.params)
    if fault == "duplicate":
        params["axes"] = (0, 0)
    elif fault == "negative":
        params["axes"] = (-1,)
    elif fault == "out-of-range":
        params["axes"] = (3,)
    elif fault == "dtype":
        params["dtype"] = np.dtype(np.float64)
    elif fault == "keepdims":
        params["keepdims"] = True
    elif fault == "placement":
        params["out_sharding"] = object()
    elif fault == "symbol-token":
        outputs = (replace(outputs[0], symbol_token=object()),)
    changed = equation.replace(params=params)
    if fault == "identity":
        changed = changed.replace(primitive=Primitive("reduce_sum"))
    assert freeze_reduction_structure_rule(changed, inputs, outputs) is None


@pytest.mark.parametrize("optimized", [False, True])
def test_reduction_observation_preserves_artifact_report_and_warnings(
    optimized: bool,
) -> None:
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
            model = _export(
                lambda x: (
                    jax.lax.reduce_sum(x, axes=(1,)),
                    jax.lax.reduce_max(x, axes=(1,)),
                ),
                ((3, 4),),
                dtype=np.float64,
                optimized=optimized,
            )
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


@pytest.mark.parametrize("operation", ["min", "max"])
def test_float_extrema_structure_checks_correction_sources_and_selection(
    operation: str,
) -> None:
    function = jax.lax.reduce_min if operation == "min" else jax.lax.reduce_max
    model = _export(
        lambda x, other: (function(x, axes=(1,)), other),
        ((3, 3), (3, 3)),
        dtype=np.float64,
        opset=23,
    )
    _checked(model)
    nodes = {node.op_type: node for node in model.graph}
    candidate = next(
        node.outputs[0]
        for node in model.graph
        if node.op_type == ("ReduceMin" if operation == "min" else "ReduceMax")
        and node.inputs[0] is model.graph.inputs[0]
    )
    zero_test = next(
        node.outputs[0]
        for node in model.graph
        if node.op_type == "Equal" and node.inputs[0] is candidate
    )
    # Reuse one artifact. Every corruption is undone before the next check.
    for node, index, replacement in (
        (nodes["IsNaN"], 0, model.graph.inputs[1]),
        (nodes["Reciprocal"], 0, model.graph.inputs[1]),
        (nodes["Where"], 0, zero_test),
        (nodes["Or"], 1, zero_test),
    ):
        original = node.inputs[index]
        try:
            node.replace_input_with(index, replacement)
            assert structure.validate_source_structure(model).status == "failed"
        finally:
            node.replace_input_with(index, original)
        assert structure.validate_source_structure(model).status == "validated"
        _checked(model)

    choices = nodes["Gather"].inputs[0]
    assert choices is not None
    original_constant = choices.const_value
    try:
        choices.const_value = ir.tensor(np.asarray([0.0, 0.0, np.nan], np.float64))
        assert structure.validate_source_structure(model).status == "failed"
    finally:
        choices.const_value = original_constant
    assert structure.validate_source_structure(model).status == "validated"
    _checked(model)

    gather = nodes["GatherElements"]
    original_axis = gather.attributes["axis"]
    try:
        gather.attributes["axis"] = ir.AttrInt64("axis", 1)
        assert structure.validate_source_structure(model).status == "failed"
    finally:
        gather.attributes["axis"] = original_axis
    assert structure.validate_source_structure(model).status == "validated"
    _checked(model)

    original_output = model.graph.outputs[0]
    try:
        model.graph.outputs[0] = candidate
        assert structure.validate_source_structure(model).status == "failed"
    finally:
        model.graph.outputs[0] = original_output
    assert structure.validate_source_structure(model).status == "validated"
    _checked(model)
