# tests/extra_tests/converter/test_source_structure.py

"""Real SOURCE composition and same-typed structural counterexamples."""

from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace
from typing import Any
import warnings

import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat.jax import Jaxpr, JaxprEqn, Literal, Primitive, Var
from jax2onnx.converter.ir_context import IRContext
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter import source_structure as structure
from jax2onnx.converter.source_representation import normalized_shape_dimensions
from jax2onnx.plugins import _source_structure as rules
from jax2onnx.precision import CheckStatus, get_precision_report


@pytest.mark.parametrize("operation", ["not", "and"])
@pytest.mark.parametrize("supplied_jit", [False, True])
@pytest.mark.parametrize(
    "dtype,shape,opset",
    [
        (np.bool_, (3,), 23),
        (np.bool_, (), 26),
        (np.int32, (), 23),
        (np.int32, (0,), 26),
        (np.uint32, (3,), 23),
        (np.uint32, (0,), 26),
    ],
)
def test_discrete_direct_original_and_supplied_jit_have_complete_structure(
    operation: str, supplied_jit: bool, dtype: Any, shape: tuple[int, ...], opset: int
) -> None:
    def original(*values: Any) -> Any:
        if operation == "not":
            return jnp.bitwise_not(values[0])
        return jnp.bitwise_and(values[0], values[1])

    arity = 1 if operation == "not" else 2
    target = jax.jit(original) if supplied_jit else original
    model = ui._to_onnx(
        target,
        [jax.ShapeDtypeStruct(shape, dtype)] * arity,
        precision_policy=PrecisionPolicy.SOURCE,
        opset=opset,
        return_mode="ir",
    )
    assert isinstance(model, ir.Model)
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "validated"
    assert (
        result.expected_equations
        == result.checked_equations
        == (2 if supplied_jit else 1)
    )
    expected = (
        ("Not" if operation == "not" else "And")
        if dtype is np.bool_
        else ("BitwiseNot" if operation == "not" else "BitwiseAnd")
    )
    assert sum(node.op_type == expected for node in model.graph) == 1
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize("operation", ["not", "and"])
@pytest.mark.parametrize("fault", ["operator", "edge", "attribute"])
def test_discrete_direct_final_ir_mutations_are_rejected(
    operation: str, fault: str
) -> None:
    def original(*values: Any) -> Any:
        if operation == "not":
            return jnp.bitwise_not(values[0]), values[1]
        return jnp.bitwise_and(values[0], values[1]), values[2]

    arity = 2 if operation == "not" else 3
    model = ui._to_onnx(
        original,
        [jax.ShapeDtypeStruct((3,), np.int32)] * arity,
        precision_policy=PrecisionPolicy.SOURCE,
        return_mode="ir",
    )
    assert isinstance(model, ir.Model)
    assert model.meta[structure.RESULT_KEY].status == "validated"
    node = next(
        n
        for n in model.graph
        if n.op_type == ("BitwiseNot" if operation == "not" else "BitwiseAnd")
    )
    if fault == "operator":
        node.op_type = "Neg" if operation == "not" else "BitwiseOr"
    elif fault == "edge":
        assert model.graph.inputs[-1] is not node.inputs[0]
        assert model.graph.outputs[-1] is model.graph.inputs[-1]
        node.replace_input_with(0, model.graph.inputs[-1])
    else:
        node.attributes["unexpected"] = ir.AttrInt64("unexpected", 1)
    assert structure.validate_source_structure(model).status == "failed"


@pytest.mark.parametrize("operation", ["not", "and"])
@pytest.mark.parametrize(
    "fault",
    [
        "primitive",
        "equation",
        "effects",
        "params",
        "params-type",
        "input-arity",
        "output-arity",
        "logical-dtype",
        "physical-dtype",
        "input-dtype",
        "encoding",
        "logical-shape",
        "shape",
        "symbol",
        "symbol-axis",
        "logical-symbol-axis",
        "weakness",
        "weakness-type",
    ],
)
def test_discrete_direct_rule_rejects_unowned_or_incompatible_contracts(
    operation: str, fault: str
) -> None:
    fn = jax.lax.bitwise_not if operation == "not" else jax.lax.bitwise_and
    arity = 1 if operation == "not" else 2
    native = jax.make_jaxpr(fn)(
        *[jax.ShapeDtypeStruct((3,), np.uint32)] * arity
    ).jaxpr.eqns[0]
    eqn = native
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.outvars
    )
    assert rules.freeze_source_semantic_key(eqn, inputs, outputs) == (
        "direct",
        operation,
        "BitwiseNot" if operation == "not" else "BitwiseAnd",
        (),
        False,
    )
    if fault == "primitive":
        eqn = native.replace(primitive=Primitive(operation))
    elif fault == "equation":
        eqn = SimpleNamespace(
            primitive=native.primitive,
            params={},
            effects=(),
            invars=native.invars,
            outvars=native.outvars,
        )
    elif fault == "effects":
        eqn = native.replace(effects=frozenset({object()}))
    elif fault == "params":
        eqn = native.replace(params={"placement": None})
    elif fault == "params-type":

        class Params(dict):
            pass

        eqn = native.replace(params=Params())
    elif fault == "input-arity":
        inputs = inputs[:-1]
    elif fault == "output-arity":
        outputs = outputs * 2
    elif fault == "input-dtype":
        inputs = (
            *inputs[:-1],
            replace(inputs[-1], logical_dtype="int32", dtype=ir.DataType.INT32),
        )
    else:
        changes = {
            "logical-dtype": {"logical_dtype": "float32"},
            "physical-dtype": {"dtype": ir.DataType.INT32},
            "encoding": {"encoding": "complex-real-pair"},
            "logical-shape": {"logical_shape": (1, 3)},
            "shape": {"shape": (1, 3), "logical_shape": (1, 3)},
            "symbol": {"symbol_token": object()},
            "symbol-axis": {"symbol_axis": 0},
            "logical-symbol-axis": {"logical_symbol_axis": 0},
            "weakness": {"weak_type": True},
            "weakness-type": {"weak_type": 0},
        }[fault]
        outputs = (replace(outputs[0], **changes),)
    assert rules.freeze_source_structure_rule(eqn, inputs, outputs) is None
    assert rules.freeze_source_semantic_key(eqn, inputs, outputs) is None


@pytest.mark.parametrize("operation", ["not", "and"])
def test_discrete_direct_rule_retains_native_weak_scalar(operation: str) -> None:
    fn = jax.lax.bitwise_not if operation == "not" else jax.lax.bitwise_and
    with jax.enable_x64(False):
        eqn = jax.make_jaxpr(fn)(*((1,) if operation == "not" else (1, 3))).jaxpr.eqns[
            0
        ]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in eqn.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in eqn.outvars
    )
    assert all(v.weak_type for v in (*inputs, *outputs))
    assert rules.freeze_source_structure_rule(eqn, inputs, outputs) is not None


def _source(
    fn: Any, *, arity: int = 2, shape: Any = (2, 3), **options: Any
) -> ir.Model:
    with jax.enable_x64(True):
        model = ui._to_onnx(
            fn,
            [jax.ShapeDtypeStruct(shape, np.float32)] * arity,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
            **options,
        )
    assert isinstance(model, ir.Model)
    return model


def _composition(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.stop_gradient(jax.lax.transpose(jax.lax.sub(x, y), (1, 0)))


def test_normal_source_composition_is_checked_after_facade_names() -> None:
    model = _source(
        _composition, input_names=["left", "right"], output_names=["answer"]
    )
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "validated"
    assert result.expected_equations == result.checked_equations == 3
    assert result.runtime == result.numerics == "unverified"
    assert [value.name for value in model.graph.inputs] == ["left", "right"]
    assert model.graph.outputs[0].name == "answer"
    report = get_precision_report(model)
    assert report is not None
    assert all(
        check.status is CheckStatus.UNVERIFIED
        for check in report.checks
        if check.category in {"runtime-execution", "numerical-acceptance"}
    )


@pytest.mark.parametrize("mutation", ["rewire", "operator", "permutation", "dtype"])
def test_same_typed_graph_mutations_cannot_reuse_validated_result(
    mutation: str,
) -> None:
    model = _source(_composition)
    assert model.meta[structure.RESULT_KEY].status == "validated"
    if mutation in {"rewire", "operator", "dtype"}:
        node = next(node for node in model.graph if node.op_type == "Sub")
        if mutation == "rewire":
            # Sub is noncommutative; both operands have identical declared types.
            node.replace_input_with(1, model.graph.inputs[0])
        elif mutation == "operator":
            node.op_type = "Add"
        else:
            node.outputs[0].dtype = ir.DataType.DOUBLE
    else:
        node = next(node for node in model.graph if node.op_type == "Transpose")
        node.attributes["perm"] = ir.AttrInt64s("perm", [0, 1])
    assert structure.validate_source_structure(model).status == "failed"


@pytest.mark.parametrize("opset", [26, 27])
@pytest.mark.parametrize("shape", [(), (2, 3), (0, 3), "B, 3"])
@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("uint8", "int8"),
        ("bfloat16", "float16"),
        ("int32", "float32"),
        ("float64", "uint64"),
    ],
)
def test_bitcast_direct_rule_keeps_native_storage_and_shape(
    opset: int, shape: Any, source: str, target: str
) -> None:
    native_shape = jax_export.symbolic_shape(shape) if isinstance(shape, str) else shape
    with jax.enable_x64(True):
        model = ui._to_onnx(
            lambda x: jax.lax.bitcast_convert_type(x, np.dtype(target)),
            [jax.ShapeDtypeStruct(native_shape, np.dtype(source))],
            precision_policy=PrecisionPolicy.SOURCE,
            opset=opset,
            return_mode="ir",
        )
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "validated"
    assert result.expected_equations == result.checked_equations == 1
    node = next(node for node in model.graph if node.op_type == "BitCast")
    assert node.op_type == "BitCast"
    assert node.attributes["to"].as_int() == int(model.graph.outputs[0].dtype.value)
    assert node.inputs[0] is model.graph.inputs[0]
    assert model.graph.opset_imports[""] == opset
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def test_bitcast_frozen_rule_accepts_native_nonleading_zero_tail_symbol() -> None:
    shape = jax_export.symbolic_shape("3, B, 0")
    eqn = jax.make_jaxpr(lambda x: jax.lax.bitcast_convert_type(x, np.uint32))(
        jax.ShapeDtypeStruct(shape, np.float32)
    ).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in eqn.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in eqn.outvars
    )
    assert inputs[0].same_shape(outputs[0])
    assert outputs[0].shape == (3, "B", 0)
    assert outputs[0].symbol_axis == 1
    assert rules.freeze_source_semantic_key(eqn, inputs, outputs) == (
        "direct",
        "bitcast_convert_type",
        "BitCast",
        (("to", 12),),
        False,
    )


@pytest.mark.parametrize("opset", [23, 26, 27])
def test_same_dtype_bitcast_uses_identity_rule_and_semantic_key(opset: int) -> None:
    def native(x: Any) -> Any:
        return jax.lax.bitcast_convert_type(x, np.float32)

    model = _source(native, arity=1, opset=opset)
    assert model.meta[structure.RESULT_KEY].status == "validated"
    assert not any(node.op_type == "BitCast" for node in model.graph)
    graph = jax.make_jaxpr(native)(jax.ShapeDtypeStruct((2, 3), np.float32)).jaxpr
    shapes = structure._ShapeFreezer()
    eqn = graph.eqns[0]
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in eqn.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in eqn.outvars
    )
    assert rules.freeze_source_semantic_key(eqn, inputs, outputs) == (
        "direct",
        "bitcast_convert_type",
        "Identity",
        (),
        True,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "primitive",
        "effects",
        "params",
        "to",
        "weakness",
        "encoding",
        "logical-shape",
        "dtype",
        "width",
        "scope",
        "symbol-axis",
    ],
)
def test_bitcast_rule_does_not_freeze_non_native_representation(fault: str) -> None:
    native = jax.make_jaxpr(lambda x: jax.lax.bitcast_convert_type(x, np.uint32))(
        jax.ShapeDtypeStruct((2,), np.float32)
    ).jaxpr.eqns[0]
    eqn = native.replace(params=dict(native.params))
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.outvars
    )
    assert rules.freeze_source_semantic_key(eqn, inputs, outputs) == (
        "direct",
        "bitcast_convert_type",
        "BitCast",
        (("to", 12),),
        False,
    )
    wrapped = SimpleNamespace(
        primitive=native.primitive, params=dict(native.params), effects=native.effects
    )
    assert rules.freeze_source_structure_rule(wrapped, inputs, outputs) is None
    assert rules.freeze_source_semantic_key(wrapped, inputs, outputs) is None
    if fault == "primitive":
        eqn = eqn.replace(primitive=Primitive("bitcast_convert_type"))
    elif fault == "effects":
        eqn = eqn.replace(effects=frozenset({object()}))
    elif fault == "params":
        eqn = eqn.replace(params={**eqn.params, "sharding": None})
    elif fault == "to":
        eqn = eqn.replace(params={**eqn.params, "new_dtype": np.dtype(np.int32)})
    else:
        changes = {
            "weakness": {"weak_type": True},
            "encoding": {"encoding": "complex-real-pair"},
            "logical-shape": {"logical_shape": (1, 2)},
            "dtype": {"dtype": ir.DataType.INT32},
            "width": {"logical_dtype": "uint16", "dtype": ir.DataType.UINT16},
            "scope": {"symbol_token": object()},
            "symbol-axis": {"symbol_axis": 0, "logical_symbol_axis": 0},
        }[fault]
        outputs = (replace(outputs[0], **changes),)
    assert rules.freeze_source_structure_rule(eqn, inputs, outputs) is None
    assert rules.freeze_source_semantic_key(eqn, inputs, outputs) is None


@pytest.mark.parametrize("fault", ["cast", "to", "rewire", "dtype", "shape"])
def test_bitcast_structure_rejects_changed_operator_attributes_edges_and_metadata(
    fault: str,
) -> None:
    model = _source(
        lambda x, y: (jax.lax.bitcast_convert_type(x, np.uint32), y), opset=26
    )
    assert model.meta[structure.RESULT_KEY].status == "validated"
    node = next(node for node in model.graph if node.op_type == "BitCast")
    if fault == "cast":
        node.op_type = "Cast"
    elif fault == "to":
        node.attributes["to"] = ir.AttrInt64("to", int(ir.DataType.INT32.value))
    elif fault == "rewire":
        node.replace_input_with(0, model.graph.inputs[1])
    elif fault == "dtype":
        node.outputs[0].dtype = ir.DataType.INT32
    else:
        node.outputs[0].shape = ir.Shape((3, 2))
    assert structure.validate_source_structure(model).status == "failed"


def test_bitcast_constant_word_mutation_cannot_reuse_original_storage() -> None:
    words = np.asarray([0x80000000, 0x7FC00001], np.uint32)
    model = _source(
        lambda x: (x, jax.lax.bitcast_convert_type(words, np.float32)),
        arity=1,
        shape=(2,),
        opset=26,
    )
    assert model.meta[structure.RESULT_KEY].status == "validated"
    constant = next(iter(model.graph.initializers.values()))
    changed = words.copy()
    changed[1] ^= np.uint32(1)
    constant.const_value = ir.tensor(changed)
    result = structure.validate_source_structure(model)
    assert result.status == "failed"
    assert "source-constant-storage-differs" in result.reasons


def test_nested_jit_bitcasts_keep_distinct_semantic_keys_and_original_edges() -> None:
    @jax.jit
    def inner(x: Any) -> Any:
        return jax.lax.bitcast_convert_type(x, np.uint32)

    @jax.jit
    def outer(x: Any) -> Any:
        return jax.lax.bitcast_convert_type(inner(x), np.int32)

    model = _source(outer, arity=1, opset=27)
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "validated"
    assert result.expected_equations == result.checked_equations == 4
    first, second = [node for node in model.graph if node.op_type == "BitCast"]
    assert first.attributes["to"].as_int() == int(ir.DataType.UINT32.value)
    assert second.attributes["to"].as_int() == int(ir.DataType.INT32.value)
    assert second.inputs[0] is first.outputs[0]


@pytest.mark.parametrize("mutation", ["signed-zero", "nan-payload"])
def test_exact_source_constant_storage_detects_signed_zero_and_nan_payload_changes(
    mutation: str,
) -> None:
    original = np.array([0x80000000, 0x7FC00001], dtype=np.uint32).view(np.float32)
    model = _source(lambda x: jax.lax.sub(x, original), arity=1, shape=(2,))
    assert model.meta[structure.RESULT_KEY].status == "validated"
    constant = next(iter(model.graph.initializers.values()))
    changed = original.copy().view(np.uint32)
    if mutation == "signed-zero":
        changed[0] = 0x00000000
    else:
        changed[1] = 0x7FC00002
    constant.const_value = ir.tensor(changed.view(np.float32))
    result = structure.validate_source_structure(model)
    assert result.status == "failed"
    assert result.reasons == ("source-constant-storage-differs",)


def test_real_float_casts_remain_explicit_rounding_boundaries() -> None:
    def rounding(x: jax.Array, y: jax.Array) -> jax.Array:
        return jax.lax.sub(x.astype(jnp.float64), y.astype(jnp.float64)).astype(
            jnp.float32
        )

    model = _source(rounding)
    assert model.meta[structure.RESULT_KEY].status == "validated"
    assert sum(node.op_type == "Cast" for node in model.graph) == 3
    cast = next(node for node in model.graph if node.op_type == "Cast")
    cast.attributes["to"] = ir.AttrInt64("to", ir.DataType.FLOAT.value)
    assert structure.validate_source_structure(model).status == "failed"


def test_empty_add_retains_legal_export_with_unverified_structure() -> None:
    model = _source(jax.lax.add, shape=(0, 3))
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "unverified"
    assert result.expected_equations == 1 and result.checked_equations == 0
    assert "equation-0:rule-unavailable" in result.reasons
    assert tuple(model.graph.outputs[0].shape) == (0, 3)
    assert model.graph.outputs[0].dtype == ir.DataType.FLOAT
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def test_fp64_cosine_retains_legal_export_with_unverified_structure() -> None:
    with jax.enable_x64(True):
        model = ui._to_onnx(
            jax.lax.cos,
            [jax.ShapeDtypeStruct((2, 3), np.float64)],
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
        )
    assert isinstance(model, ir.Model)
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "unverified"
    assert result.expected_equations == 1 and result.checked_equations == 0
    assert "equation-0:rule-unavailable" in result.reasons
    assert tuple(model.graph.outputs[0].shape) == (2, 3)
    assert model.graph.outputs[0].dtype == ir.DataType.DOUBLE
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def test_private_witnesses_survive_clone_and_names_but_do_not_serialize() -> None:
    original = _source(_composition)
    model = ir.Model(original.graph.clone(allow_outer_scope_values=True), ir_version=10)
    model.meta.update(original.meta)
    for index, value in enumerate(model.graph.inputs):
        value.name = "renamed_input_" + str(index)
    model.graph.outputs[0].name = "renamed_answer"
    assert structure.validate_source_structure(model).status == "validated"
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    payload = proto.SerializeToString()
    assert all(
        key.encode() not in payload
        for key in (
            structure.RESULT_KEY,
            structure._OWNER,
            structure._ORIGIN,
            structure._VALUES,
            structure._LEDGER,
        )
    )


@pytest.mark.parametrize(
    "fault", ["missing", "wrong-owner", "wrong-producer", "wrong-port", "duplicate"]
)
def test_missing_or_ambiguous_lexical_witness_stays_unverified(fault: str) -> None:
    model = _source(_composition)
    output = model.graph.outputs[0]
    if fault == "missing":
        output.meta.pop(structure._VALUES)
    elif fault == "wrong-owner":
        model.graph.meta[structure._OWNER] = object()
    elif fault == "wrong-producer":
        output.producer().meta[structure._ORIGIN] = object()
    elif fault == "wrong-port":
        original = output.meta[structure._VALUES][0]
        output.meta[structure._VALUES] = (replace(original, port=original.port + 1),)
    else:
        output.meta[structure._VALUES] *= 2
    assert structure.validate_source_structure(model).status == "unverified"


def test_incomplete_verifier_does_not_reject_export_or_validate_partial_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rules, "freeze_source_structure_rule", lambda eqn, inputs, outputs: None
    )
    model = _source(_composition)
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "unverified"
    assert result.expected_equations == 3 and result.checked_equations == 0
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_expected_source_equations_anchor_receipt_completeness(fault: str) -> None:
    model = _source(_composition)
    scope = model.meta[structure._LEDGER].scopes[0]
    if fault == "missing":
        scope.receipts.pop(1)
    else:
        scope.receipts[len(scope.equations)] = scope.receipts[0]
    result = structure.validate_source_structure(model)
    assert result.status == "unverified"
    assert result.expected_equations == 3
    assert result.reasons == ("source-coverage-incomplete",)


def test_optimizer_witness_loss_does_not_disable_optimizer_or_reject_export() -> None:
    model = _source(
        lambda x: jax.lax.transpose(jax.lax.transpose(x, (1, 0)), (1, 0)), arity=1
    )
    assert not any(node.op_type == "Transpose" for node in model.graph)
    assert model.meta[structure.RESULT_KEY].status == "unverified"


def test_capture_time_source_ssa_contradiction_is_failed_without_throwing() -> None:
    jaxpr = jax.make_jaxpr(jax.lax.sub)(
        jax.ShapeDtypeStruct((2,), np.float32), jax.ShapeDtypeStruct((2,), np.float32)
    ).jaxpr
    ctx = IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    for index, var in enumerate(jaxpr.invars):
        ctx.add_input_for_invar(var, index)
    ledger = ctx._source_structure
    assert ledger is not None
    with ledger.scope(ctx, jaxpr, rules.freeze_source_structure_rule):
        ctx.bind_value_for_var(jaxpr.invars[0], ctx.builder.inputs[1])
        ledger.before_equation(ctx, jaxpr.eqns[0], 0)
        ctx.bind_value_for_var(jaxpr.outvars[0], ctx.builder.inputs[0])
        ledger.equation(ctx, jaxpr.eqns[0], 0)
    ctx.builder.outputs.append(ctx.builder.inputs[0])
    model = ctx.builder.to_ir_model(name="contradiction")
    structure.attach_source_structure(model, ledger)
    result = structure.validate_source_structure(model)
    assert result.status == "failed"
    assert "equation-input-capture:source-ssa-binding-differs" in result.reasons


def test_literal_observation_adds_no_nodes_or_initializers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    actual_audit = ui.validate_source_structure

    def audit(model: ir.Model) -> structure.StructureResult:
        before = len(model.graph), len(model.graph.initializers)
        result = actual_audit(model)
        observed.append((before, (len(model.graph), len(model.graph.initializers))))
        return result

    monkeypatch.setattr(ui, "validate_source_structure", audit)
    model = _source(lambda x: jax.lax.sub(x, np.float32(1)), arity=1)
    assert model.meta[structure.RESULT_KEY].status == "validated"
    assert observed and all(before == after for before, after in observed)
    # Scope collection itself is also non-allocating, including root literals.
    jaxpr = jax.make_jaxpr(lambda: np.float32(1))().jaxpr
    ctx = IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    ledger = ctx._source_structure
    assert ledger is not None
    before = len(ctx.builder.graph), len(ctx.builder.graph.initializers)
    with ledger.scope(ctx, jaxpr, rules.freeze_source_structure_rule):
        pass
    assert before == (len(ctx.builder.graph), len(ctx.builder.graph.initializers))
    assert not ledger.gaps and not ledger.failures
    assert ledger.expected_equations == 0
    assert isinstance(ledger.scopes[0].result_values[0], structure.LiteralOperand)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("unary", [False, True])
def test_actual_literal_arithmetic_exports_check_without_rebinding_literals(
    optimized: bool, unary: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not optimized:
        monkeypatch.setattr(
            conversion_api, "optimize_graph", lambda model, **kwargs: None
        )

    def fn(x: jax.Array) -> jax.Array:
        literal = np.float32(1.25)
        if unary:
            literal = jax.lax.neg(literal)
        return jax.lax.sub(x, literal)

    model = _source(fn, arity=1, shape=())
    assert model.meta[structure.RESULT_KEY].status == "validated"
    assert model.meta[structure.RESULT_KEY].runtime == "unverified"
    assert model.meta[structure.RESULT_KEY].numerics == "unverified"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def test_literal_cast_rounding_and_mixed_repeated_root_outputs() -> None:
    def fn(x: jax.Array) -> Any:
        rounded = jax.lax.sub(x.astype(jnp.float64), np.float64(2**-40)).astype(
            jnp.float32
        )
        return rounded, np.float64(0.125), np.float32(-0.0), np.float32(-0.0)

    names = ["rounded", "wide_literal", "negative_zero", "repeated_zero"]
    model = _source(fn, arity=1, shape=(), output_names=names)
    assert model.meta[structure.RESULT_KEY].status == "validated"
    assert [value.name for value in model.graph.outputs] == names
    assert [value.dtype for value in model.graph.outputs] == [
        ir.DataType.FLOAT,
        ir.DataType.DOUBLE,
        ir.DataType.FLOAT,
        ir.DataType.FLOAT,
    ]
    assert all(tuple(value.shape) == () for value in model.graph.outputs)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize("word", [0x80000000, 0x7FC00001])
def test_root_literal_exact_signed_zero_and_nan_payloads(word: int) -> None:
    original = np.array([word], dtype=np.uint32).view(np.float32)[0]
    model = _source(lambda: original, arity=0, output_names=["literal_answer"])
    assert model.meta[structure.RESULT_KEY].status == "validated"
    expected = structure.ConstantContract.from_array(np.asarray(original))
    scope = model.meta[structure._LEDGER].scopes[0]
    assert scope.outputs[0].literal == expected
    output = model.graph.outputs[0]
    if output.producer() is not None:
        output = output.producer().inputs[0]
    changed = (
        np.array([0 if word == 0x80000000 else 0x7FC00002], dtype=np.uint32)
        .view(np.float32)
        .reshape(())
    )
    output.const_value = ir.tensor(changed)
    result = structure.validate_source_structure(model)
    assert result.status == "failed"
    assert result.reasons == ("literal-storage-differs",)


def _scalar_literal_model() -> tuple[ir.Model, ir.Node, ir.Value]:
    model = _source(lambda x: jax.lax.sub(x, np.float32(1.25)), arity=1, shape=())
    assert model.meta[structure.RESULT_KEY].status == "validated"
    consumer = next(node for node in model.graph if node.op_type == "Sub")
    leaf = consumer.inputs[1]
    assert leaf is not None
    return model, consumer, leaf


@pytest.mark.parametrize("fault", ["value", "dtype", "shape", "missing"])
def test_recognized_wrong_literal_operand_fails(fault: str) -> None:
    model, consumer, leaf = _scalar_literal_model()
    if fault == "value":
        leaf.const_value = ir.tensor(np.asarray(2.5, dtype=np.float32))
    elif fault == "dtype":
        leaf.dtype = ir.DataType.DOUBLE
        leaf.const_value = ir.tensor(np.asarray(1.25, dtype=np.float64))
    elif fault == "shape":
        leaf.shape = ir.Shape([1])
        leaf.const_value = ir.tensor(np.asarray([1.25], dtype=np.float32))
    else:
        consumer.replace_input_with(1, None)
    assert structure.validate_source_structure(model).status == "failed"


@pytest.mark.parametrize("overridable", [False, True])
def test_runtime_input_cannot_masquerade_as_literal(overridable: bool) -> None:
    model, consumer, _ = _scalar_literal_model()
    runtime = model.graph.inputs[0]
    runtime.const_value = ir.tensor(np.asarray(1.25, dtype=np.float32))
    if overridable:
        model.graph.register_initializer(runtime)
    consumer.replace_input_with(1, runtime)
    result = structure.validate_source_structure(model)
    assert result.status == "failed"
    assert result.reasons == ("literal-runtime-input-substitute",)


def test_equivalent_literal_cse_keeps_nonliteral_identity_strict() -> None:
    def fn(x: jax.Array) -> Any:
        return jax.lax.sub(x, np.float32(1.25)), jax.lax.add(x, np.float32(1.25))

    model = _source(fn, arity=1, shape=())
    assert model.meta[structure.RESULT_KEY].status == "validated"
    subtract = next(node for node in model.graph if node.op_type == "Sub")
    add = next(node for node in model.graph if node.op_type == "Add")
    add.replace_input_with(1, subtract.inputs[1])
    assert structure.validate_source_structure(model).status == "validated"
    add.replace_input_with(0, subtract.inputs[1])
    assert structure.validate_source_structure(model).status == "failed"


def test_constant_literal_uses_authoritative_tensor_and_covers_its_node() -> None:
    model, consumer, _ = _scalar_literal_model()
    array = np.asarray(1.25, dtype=np.float32)
    leaf = ir.val("literal_constant", ir.DataType.FLOAT, [])
    leaf.const_value = ir.tensor(array.copy())
    constant = ir.Node(
        "",
        "Constant",
        [],
        outputs=[leaf],
        attributes=[ir.AttrTensor("value", ir.tensor(array))],
    )
    model.graph.insert_before(consumer, constant)
    consumer.replace_input_with(1, leaf)
    assert structure.validate_source_structure(model).status == "validated"
    # A stale matching annotation must not hide changed authoritative storage.
    constant.attributes["value"] = ir.AttrTensor(
        "value", ir.tensor(np.asarray(2.5, dtype=np.float32))
    )
    result = structure.validate_source_structure(model)
    assert result.status == "failed"
    assert result.reasons == ("literal-storage-differs",)


@pytest.mark.parametrize("fault", ["unreadable", "computed"])
def test_unreadable_or_unknown_literal_definition_stays_unverified(fault: str) -> None:
    model, consumer, original = _scalar_literal_model()
    if fault == "unreadable":
        original.const_value = None
    else:
        computed = ir.val("computed_literal", ir.DataType.FLOAT, [])
        computed.const_value = ir.tensor(np.asarray(1.25, dtype=np.float32))
        model.graph.insert_before(
            consumer, ir.Node("", "Identity", [original], outputs=[computed])
        )
        consumer.replace_input_with(1, computed)
    assert structure.validate_source_structure(model).status == "unverified"


def test_root_literal_weakness_is_frozen_from_original_native_trace() -> None:
    model = _source(lambda: 1.25, arity=0)
    assert model.meta[structure.RESULT_KEY].status == "validated"
    scope = model.meta[structure._LEDGER].scopes[0]
    report = get_precision_report(model)
    assert report is not None
    assert (
        scope.outputs[0].tensor.weak_type is report.traced_outputs[0].weak_type is True
    )


def test_literal_observation_preserves_serialized_export_report_and_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fn(x: jax.Array) -> Any:
        return jax.lax.sub(x, np.float32(1.25)), np.float64(0.125)

    results = []
    actual_init = IRContext.__init__

    def without_ledger(self: IRContext, *args: Any, **kwargs: Any) -> None:
        actual_init(self, *args, **kwargs)
        self._source_structure = None

    for enabled in (False, True):
        with monkeypatch.context() as change:
            if not enabled:
                change.setattr(IRContext, "__init__", without_ledger)
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                model = _source(fn, arity=1, shape=())
            results.append(
                (
                    ir.to_proto(model).SerializeToString(deterministic=True),
                    get_precision_report(model),
                    tuple((entry.category, str(entry.message)) for entry in observed),
                )
            )
            if enabled:
                assert model.meta[structure.RESULT_KEY].status == "validated"
    assert results[0] == results[1]


def test_symbolic_identity_keeps_frozen_native_scope_authority() -> None:
    shape = jax_export.symbolic_shape("B, 3")
    model = _source(lambda x: x, arity=1, shape=shape)
    assert model.meta[structure.RESULT_KEY].status == "validated"
    scope = model.meta[structure._LEDGER].scopes[0]
    source, result = scope.inputs[0].tensor, scope.outputs[0].tensor
    assert type(source.symbol_token) is object
    assert source.symbol_token is result.symbol_token
    assert source.symbol_axis == source.logical_symbol_axis == result.symbol_axis == 0
    assert source.symbol_token is not shape[0].scope


def _symbolic_direct(x: jax.Array, y: jax.Array) -> Any:
    left = jax.lax.transpose(x, (1, 0, 2))
    right = jax.lax.transpose(y, (1, 0, 2))
    wide = jax.lax.sub(left, right).astype(jnp.float64)
    return wide, jax.lax.transpose(wide, (1, 0, 2)).astype(jnp.float32)


def _symbolic_model() -> ir.Model:
    shape = jax_export.symbolic_shape("B, 384, 257")
    return _source(_symbolic_direct, shape=shape)


@pytest.mark.parametrize("optimized", [False, True])
def test_real_atomic_symbol_survives_nonleading_arithmetic_and_casts(
    optimized: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not optimized:
        monkeypatch.setattr(
            conversion_api, "optimize_graph", lambda model, **kwargs: None
        )
    model = _symbolic_model()
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "validated"
    assert result.runtime == result.numerics == "unverified"
    assert normalized_shape_dimensions(tuple(model.graph.outputs[0].shape)) == (
        384,
        "B",
        257,
    )
    assert normalized_shape_dimensions(tuple(model.graph.outputs[1].shape)) == (
        "B",
        384,
        257,
    )
    scope = model.meta[structure._LEDGER].scopes[0]
    token = scope.inputs[0].tensor.symbol_token
    assert token is not None
    contracts = [
        use.tensor for eqn in scope.equations for use in (*eqn.inputs, *eqn.outputs)
    ]
    assert all(value.symbol_token is token for value in contracts)
    assert any(
        value.shape == (384, "B", 257) and value.symbol_axis == 1 for value in contracts
    )
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def test_real_leading_symbolic_negation_uses_existing_admitted_owner() -> None:
    model = _source(jax.lax.neg, arity=1, shape=jax_export.symbolic_shape("B, 3"))
    assert model.meta[structure.RESULT_KEY].status == "validated"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize(
    "fault", ["axis", "label", "static-tail", "edge", "permutation"]
)
def test_atomic_symbol_does_not_hide_final_graph_mutations(fault: str) -> None:
    model = _symbolic_model()
    assert model.meta[structure.RESULT_KEY].status == "validated"
    if fault == "edge":
        node = next(node for node in model.graph if node.op_type == "Sub")
        node.replace_input_with(1, node.inputs[0])
    elif fault == "permutation":
        node = next(node for node in model.graph if node.op_type == "Transpose")
        node.attributes["perm"] = ir.AttrInt64s("perm", [0, 1, 2])
    else:
        model.graph.outputs[0].shape = ir.Shape(
            {
                "axis": ["B", 384, 257],
                "label": [384, "C", 257],
                "static-tail": [384, "B", 258],
            }[fault]
        )
    assert structure.validate_source_structure(model).status == "failed"


def test_equivalent_ir_symbol_objects_are_only_representation_labels() -> None:
    model = _symbolic_model()
    for value in (*model.graph.inputs, *model.graph.outputs):
        dimensions = normalized_shape_dimensions(tuple(value.shape))
        value.shape = ir.Shape(
            [ir.SymbolicDim(dim) if isinstance(dim, str) else dim for dim in dimensions]
        )
    assert structure.validate_source_structure(model).status == "validated"


def test_equal_printed_native_symbols_from_distinct_scopes_stay_unverified() -> None:
    (local,) = jax_export.symbolic_shape("B")
    (foreign,) = jax_export.symbolic_shape("B")
    assert str(local) == str(foreign) == "B" and local.scope is not foreign.scope
    spec = jax.ShapeDtypeStruct((local, 3), np.float32)
    original = jax.make_jaxpr(jax.lax.sub)(spec, spec).jaxpr
    other = Var(original.invars[1].aval.update(shape=(foreign, 3)))
    equation = original.eqns[0].replace(invars=[original.invars[0], other])
    changed = original.replace(invars=[original.invars[0], other], eqns=[equation])
    ctx = IRContext(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    for index, var in enumerate(changed.invars):
        ctx.add_input_for_invar(var, index)
    ledger = ctx._source_structure
    assert ledger is not None
    with ledger.scope(ctx, changed, rules.freeze_source_structure_rule):
        pass
    output = ir.val("difference", ir.DataType.FLOAT, ["B", 3])
    ctx.builder.op("Sub", list(ctx.builder.inputs), output=output)
    ctx.builder.outputs.append(output)
    model = ctx.builder.to_ir_model(name="distinct_native_scopes")
    structure.attach_source_structure(model, ledger)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    result = structure.validate_source_structure(model)
    assert result.status == "unverified"
    assert result.reasons == ("scope-capture:symbolic-scope-contract-unavailable",)


@pytest.mark.parametrize("form", ["string-only", "derived", "constrained", "two-atoms"])
def test_unreviewed_native_dimension_forms_are_explicit_verifier_gaps(
    form: str,
) -> None:
    (batch,) = jax_export.symbolic_shape("B")
    if form == "string-only":
        shape = ("B", 3)
    elif form == "derived":
        shape = (batch + 1, 3)
    elif form == "constrained":
        (constrained,) = jax_export.symbolic_shape("B", constraints=("B >= 2",))
        shape = (constrained, 3)
    else:
        first, second = jax_export.symbolic_shape("B,C")
        shape = (first, second, 3)
    with pytest.raises(
        structure.StructureUnavailable, match="symbolic-shape-contract-unavailable"
    ):
        structure.TensorContract.from_aval(
            jax.ShapeDtypeStruct(shape, np.float32), structure._ShapeFreezer()
        )


def test_atomic_symbol_tokens_retain_no_native_scope_or_expression() -> None:
    (native,) = jax_export.symbolic_shape("B")
    model = _symbolic_model()

    def inspect(value: Any) -> None:
        assert not isinstance(
            value, (Jaxpr, JaxprEqn, Var, Literal, type(native), type(native.scope))
        )
        if is_dataclass(value) and not isinstance(value, type):
            for member in fields(value):
                inspect(getattr(value, member.name))
        elif isinstance(value, dict):
            for key, child in value.items():
                inspect(key)
                inspect(child)
        elif isinstance(value, (list, tuple, set)):
            for child in value:
                inspect(child)

    inspect(vars(model.meta[structure._LEDGER]))


def test_zero_equation_identity_has_real_boundary_coverage() -> None:
    model = _source(lambda x: x, arity=1, output_names=["identity_answer"])
    assert model.meta[structure.RESULT_KEY].status == "validated"
    model.meta.pop(structure._LEDGER)
    assert structure.validate_source_structure(model).status == "unverified"


@pytest.mark.parametrize(
    ("input_layout", "output_layout"), [(True, False), (False, True), (True, True)]
)
def test_nchw_boundary_bridges_remain_valid_exports_with_unverified_structure(
    input_layout: bool, output_layout: bool
) -> None:
    logical_shape = (1, 2, 3, 4)
    nchw_shape = (1, 4, 2, 3)
    model = _source(
        lambda x: x,
        arity=1,
        shape=logical_shape,
        inputs_as_nchw=[0] if input_layout else None,
        outputs_as_nchw=[0] if output_layout else None,
    )
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "unverified"
    assert result.reasons == ("boundary-layout-contract-unavailable",)
    assert result.runtime == result.numerics == "unverified"
    assert tuple(model.graph.inputs[0].shape) == (
        nchw_shape if input_layout else logical_shape
    )
    assert tuple(model.graph.outputs[0].shape) == (
        nchw_shape if output_layout else logical_shape
    )
    assert (
        model.graph.inputs[0].dtype == model.graph.outputs[0].dtype == ir.DataType.FLOAT
    )
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


def test_same_named_custom_primitive_has_no_canonical_lax_rule() -> None:
    jaxpr = jax.make_jaxpr(jax.lax.sub)(
        jax.ShapeDtypeStruct((2,), np.float32), jax.ShapeDtypeStruct((2,), np.float32)
    ).jaxpr
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes)
        for var in jaxpr.eqns[0].invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes)
        for var in jaxpr.eqns[0].outvars
    )
    assert (
        rules.freeze_source_structure_rule(jaxpr.eqns[0], inputs, outputs) is not None
    )
    fake = jaxpr.eqns[0].replace(primitive=Primitive("sub"))
    assert rules.freeze_source_structure_rule(fake, inputs, outputs) is None
    # Missing primitives are malformed boundary fixtures, not native equations.
    missing: Any = SimpleNamespace(primitive=None)
    assert rules.freeze_source_structure_rule(missing, (), ()) is None


def test_final_registry_retains_no_native_trace_objects() -> None:
    model = _source(_composition)
    ledger = model.meta[structure._LEDGER]

    def inspect(value: Any) -> None:
        assert not isinstance(value, (Jaxpr, JaxprEqn, Var, Literal))
        if is_dataclass(value) and not isinstance(value, type):
            for member in fields(value):
                inspect(getattr(value, member.name))
        elif isinstance(value, dict):
            for key, child in value.items():
                inspect(key)
                inspect(child)
        elif isinstance(value, (list, tuple, set)):
            for child in value:
                inspect(child)

    inspect(vars(ledger))


def test_legacy_and_serialized_results_do_not_gain_structural_claims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    def forbidden(model: ir.Model) -> None:
        raise AssertionError("SOURCE observer ran on LEGACY")

    with monkeypatch.context() as context:
        context.setattr(ui, "validate_source_structure", forbidden)
        legacy = ui.to_onnx(
            jax.lax.sub, [jax.ShapeDtypeStruct((2,), np.float32)] * 2, return_mode="ir"
        )
    assert (
        structure._LEDGER not in legacy.meta and structure.RESULT_KEY not in legacy.meta
    )
    for mode in ("proto", "file"):
        with jax.enable_x64(True):
            exported = ui._to_onnx(
                jax.lax.sub,
                [jax.ShapeDtypeStruct((2,), np.float32)] * 2,
                precision_policy=PrecisionPolicy.SOURCE,
                return_mode=mode,
                **({"output_path": tmp_path / "result.onnx"} if mode == "file" else {}),
            )
        proto = onnx.load_model(exported) if isinstance(exported, str) else exported
        assert not any(
            item.key.startswith("jax2onnx.source.structure")
            for item in proto.metadata_props
        )


@pytest.mark.parametrize("operation", ["or", "xor"])
@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.uint32])
def test_discrete_family_direct_broadcast_is_complete(operation, dtype):
    primitive = jax.lax.bitwise_or if operation == "or" else jax.lax.bitwise_xor

    def original(x, y):
        return primitive(x, y)

    model = ui._to_onnx(
        original,
        [jax.ShapeDtypeStruct((2, 3), dtype), jax.ShapeDtypeStruct((1, 3), dtype)],
        precision_policy=PrecisionPolicy.SOURCE,
        opset=23,
        return_mode="ir",
    )
    result = model.meta[structure.RESULT_KEY]
    assert (
        result.status == "validated"
        and result.expected_equations == result.checked_equations == 1
    )
    expected = (
        ("Or" if operation == "or" else "Xor")
        if dtype is np.bool_
        else ("BitwiseOr" if operation == "or" else "BitwiseXor")
    )
    assert [node.op_type for node in model.graph] == [expected]
    assert tuple(model.graph.outputs[0].shape) == (2, 3)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize("operation", ["or", "xor"])
@pytest.mark.parametrize("fault", ["operator", "edge", "attribute"])
def test_discrete_family_direct_final_mutations_fail(operation, fault):
    primitive = jax.lax.bitwise_or if operation == "or" else jax.lax.bitwise_xor

    def original(x, y, other):
        return primitive(x, y), other

    model = ui._to_onnx(
        original,
        [jax.ShapeDtypeStruct((3,), np.int32)] * 3,
        precision_policy=PrecisionPolicy.SOURCE,
        return_mode="ir",
    )
    assert model.meta[structure.RESULT_KEY].status == "validated"
    node = next(n for n in model.graph if n.op_type in {"BitwiseOr", "BitwiseXor"})
    if fault == "operator":
        node.op_type = "BitwiseAnd"
    elif fault == "edge":
        assert model.graph.inputs[2] is model.graph.outputs[1]
        node.replace_input_with(0, model.graph.inputs[2])
    else:
        node.attributes["unexpected"] = ir.AttrInt64("unexpected", 1)
    assert structure.validate_source_structure(model).status == "failed"


@pytest.mark.parametrize(
    "fault",
    ["params", "weakness", "shape", "encoding", "dtype", "placement", "primitive"],
)
def test_discrete_family_promotion_has_exact_contract(fault):
    native = jax.make_jaxpr(lambda x: jax.lax.convert_element_type(x, np.int32))(
        jax.ShapeDtypeStruct((3,), np.bool_)
    ).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.outvars
    )
    assert rules.freeze_source_semantic_key(native, inputs, outputs) is not None
    eqn = native
    params = dict(native.params)
    if fault == "params":
        params["unreviewed"] = None
    elif fault == "placement":
        params["sharding"] = object()
    elif fault == "primitive":
        eqn = native.replace(primitive=Primitive("convert_element_type"))
    else:
        changes = {
            "weakness": {"weak_type": not outputs[0].weak_type},
            "shape": {"shape": (1,), "logical_shape": (1,)},
            "encoding": {"encoding": "complex-real-pair"},
            "dtype": {"logical_dtype": "float32", "dtype": ir.DataType.FLOAT},
        }[fault]
        outputs = (replace(outputs[0], **changes),)
    if fault != "primitive":
        eqn = eqn.replace(params=params)
    assert rules.freeze_source_structure_rule(eqn, inputs, outputs) is None


@pytest.mark.parametrize(
    "operation", [jax.lax.bitwise_not, jax.lax.bitwise_or, jax.lax.bitwise_xor]
)
def test_discrete_family_native_weak_scalar_is_not_restamped(operation):
    arity = 1 if operation is jax.lax.bitwise_not else 2
    native = jax.make_jaxpr(operation)(*[3] * arity).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.outvars
    )
    assert all(value.weak_type is True for value in (*inputs, *outputs))
    assert rules.freeze_source_structure_rule(native, inputs, outputs) is not None
    forged_inputs = tuple(replace(value, weak_type=False) for value in inputs)
    forged_outputs = tuple(replace(value, weak_type=False) for value in outputs)
    assert (
        rules.freeze_source_structure_rule(native, forged_inputs, forged_outputs)
        is None
    )


def _predicate_source(
    operation: str,
    dtype: Any,
    shapes: tuple[tuple[int, ...], ...],
    tmp_path: Any,
    *,
    supplied_jit: bool = False,
    retain_input: bool = False,
) -> ir.Model:
    fn = {
        "eq": jax.lax.eq,
        "ne": jax.lax.ne,
        "lt": jax.lax.lt,
        "le": jax.lax.le,
        "gt": jax.lax.gt,
        "ge": jax.lax.ge,
        "is_finite": jax.lax.is_finite,
    }[operation]
    arity = 1 if operation == "is_finite" else 2

    def original(*values: Any) -> Any:
        result = fn(*values[:arity])
        return (result, values[-1]) if retain_input else result

    target = jax.jit(original) if supplied_jit else original
    with jax.enable_x64(True):
        model = ui._to_onnx(
            target,
            [jax.ShapeDtypeStruct(shape, dtype) for shape in shapes],
            precision_policy=PrecisionPolicy.SOURCE,
            opset=23,
            return_mode="ir",
        )
    assert isinstance(model, ir.Model)
    # Preserve actual emitted IR/result before asserting a newly admitted rule.
    onnx.save_model(ir.to_proto(model), tmp_path / "predicate-source.onnx")
    (tmp_path / "predicate-source-result.txt").write_text(
        repr(model.meta[structure.RESULT_KEY]) + "\n"
    )
    return model


@pytest.mark.parametrize("operation", ["eq", "ne", "lt", "le", "gt", "ge"])
@pytest.mark.parametrize(
    "dtype,shape",
    [
        (np.float32, (7,)),
        (np.float64, (2, 3)),
        (np.int32, (0,)),
        (np.uint32, ()),
    ],
)
def test_static_predicate_vector_contracts_are_checked(
    operation: str, dtype: Any, shape: tuple[int, ...], tmp_path: Any
) -> None:
    model = _predicate_source(operation, dtype, (shape, shape), tmp_path)
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "validated"
    assert result.expected_equations == result.checked_equations == 1
    expected = {
        "eq": ["Equal"],
        "ne": ["Equal", "Not"],
        "lt": ["Less"],
        "le": ["LessOrEqual"],
        "gt": ["Greater"],
        "ge": ["GreaterOrEqual"],
    }[operation]
    assert [node.op_type for node in model.graph] == expected
    assert model.graph.outputs[0].dtype is ir.DataType.BOOL
    assert normalized_shape_dimensions(tuple(model.graph.outputs[0].shape)) == shape
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize("operation", ["eq", "ne", "lt", "le", "gt", "ge"])
def test_static_predicate_native_broadcast_geometry_is_checked(
    operation: str, tmp_path: Any
) -> None:
    model = _predicate_source(operation, np.float32, ((2, 3), (1, 3)), tmp_path)
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "validated"
    assert result.expected_equations == result.checked_equations == 1
    assert normalized_shape_dimensions(tuple(model.graph.outputs[0].shape)) == (2, 3)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize("operation", ["eq", "ne"])
def test_static_boolean_equality_contract_is_checked(
    operation: str, tmp_path: Any
) -> None:
    model = _predicate_source(operation, np.bool_, ((3,), (3,)), tmp_path)
    assert model.meta[structure.RESULT_KEY].status == "validated"
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize(
    "dtype,shape,supplied_jit",
    [(np.float32, (7,), False), (np.float64, (), True), (np.float32, (0,), False)],
)
def test_static_finite_predicate_composition_is_checked(
    dtype: Any, shape: tuple[int, ...], supplied_jit: bool, tmp_path: Any
) -> None:
    model = _predicate_source(
        "is_finite", dtype, (shape,), tmp_path, supplied_jit=supplied_jit
    )
    result = model.meta[structure.RESULT_KEY]
    assert result.status == "validated"
    assert (
        result.expected_equations
        == result.checked_equations
        == (2 if supplied_jit else 1)
    )
    assert [node.op_type for node in model.graph] == ["IsInf", "IsNaN", "Or", "Not"]
    onnx.checker.check_model(ir.to_proto(model), full_check=True)


@pytest.mark.parametrize("operation", ["ne", "is_finite"])
@pytest.mark.parametrize(
    "fault",
    ["operator", "edge", "attribute", "intermediate-dtype", "intermediate-shape"],
)
def test_composed_predicate_final_ir_counterexamples_are_rejected(
    operation: str, fault: str, tmp_path: Any
) -> None:
    arity = 1 if operation == "is_finite" else 2
    model = _predicate_source(
        operation,
        np.float32,
        ((3,),) * (arity + 1),
        tmp_path,
        retain_input=True,
    )
    assert model.meta[structure.RESULT_KEY].status == "validated"
    node = next(
        node
        for node in model.graph
        if node.op_type == ("Equal" if operation == "ne" else "IsInf")
    )
    if fault == "operator":
        node.op_type = "Greater" if operation == "ne" else "IsNaN"
    elif fault == "edge":
        assert model.graph.inputs[-1] is not node.inputs[0]
        assert model.graph.outputs[-1] is model.graph.inputs[-1]
        node.replace_input_with(0, model.graph.inputs[-1])
    elif fault == "attribute":
        node.attributes["unexpected"] = ir.AttrInt64("unexpected", 1)
    elif fault == "intermediate-dtype":
        node.outputs[0].dtype = ir.DataType.DOUBLE
    else:
        node.outputs[0].shape = ir.Shape((1, 3))
    assert structure.validate_source_structure(model).status == "failed"


@pytest.mark.parametrize("operation", ["eq", "ne", "is_finite"])
@pytest.mark.parametrize(
    "fault",
    [
        "primitive",
        "equation",
        "effects",
        "params",
        "params-type",
        "input-arity",
        "output-arity",
        "logical-dtype",
        "physical-dtype",
        "encoding",
        "logical-shape",
        "shape",
        "symbol",
        "weakness",
        "weakness-type",
    ],
)
def test_static_predicate_rule_rejects_unowned_or_incompatible_contracts(
    operation: str, fault: str
) -> None:
    fn = {"eq": jax.lax.eq, "ne": jax.lax.ne, "is_finite": jax.lax.is_finite}[operation]
    arity = 1 if operation == "is_finite" else 2
    native = jax.make_jaxpr(fn)(
        *[jax.ShapeDtypeStruct((3,), np.float32)] * arity
    ).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.outvars
    )
    assert rules.freeze_source_structure_rule(native, inputs, outputs) is not None
    eqn = native
    if fault == "primitive":
        eqn = native.replace(primitive=Primitive(operation))
    elif fault == "equation":
        eqn = SimpleNamespace(primitive=native.primitive, params={}, effects=())
    elif fault == "effects":
        eqn = native.replace(effects=frozenset({object()}))
    elif fault == "params":
        eqn = native.replace(params={"placement": None})
    elif fault == "params-type":

        class Params(dict):
            pass

        eqn = native.replace(params=Params())
    elif fault == "input-arity":
        inputs = inputs[:-1]
    elif fault == "output-arity":
        outputs = outputs * 2
    else:
        changes = {
            "logical-dtype": {"logical_dtype": "int8"},
            "physical-dtype": {"dtype": ir.DataType.INT32},
            "encoding": {"encoding": "complex-real-pair"},
            "logical-shape": {"logical_shape": (1, 3)},
            "shape": {"shape": (1, 3), "logical_shape": (1, 3)},
            "symbol": {"symbol_token": object()},
            "weakness": {"weak_type": True},
            "weakness-type": {"weak_type": 0},
        }[fault]
        outputs = (replace(outputs[0], **changes),)
    assert rules.freeze_source_structure_rule(eqn, inputs, outputs) is None
    assert rules.freeze_source_semantic_key(eqn, inputs, outputs) is None


@pytest.mark.parametrize("operation", ["eq", "ne", "is_finite"])
def test_static_predicate_weak_scalar_matches_native_conjunction(
    operation: str,
) -> None:
    fn = {"eq": jax.lax.eq, "ne": jax.lax.ne, "is_finite": jax.lax.is_finite}[operation]
    arity = 1 if operation == "is_finite" else 2
    with jax.enable_x64(False):
        eqn = jax.make_jaxpr(fn)(*((1.0,) * arity)).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in eqn.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in eqn.outvars
    )
    assert all(value.weak_type for value in (*inputs, *outputs))
    assert rules.freeze_source_structure_rule(eqn, inputs, outputs) is not None


@pytest.mark.parametrize("operation", ["eq", "ne", "is_finite"])
def test_static_predicate_rejects_nonboolean_native_weak_flag_before_other_fields(
    operation: str,
) -> None:
    from jax2onnx._compat.jax import ShapedArray

    fn = {"eq": jax.lax.eq, "ne": jax.lax.ne, "is_finite": jax.lax.is_finite}[operation]
    arity = 1 if operation == "is_finite" else 2
    native = jax.make_jaxpr(fn)(
        *[jax.ShapeDtypeStruct((3,), np.float32)] * arity
    ).jaxpr.eqns[0]
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(v.aval, shapes) for v in native.outvars
    )
    invalid = object.__new__(ShapedArray)
    object.__setattr__(invalid, "weak_type", 1)
    assert type(invalid) is ShapedArray
    assert type(invalid.weak_type) is int and invalid.weak_type == 1
    var = native.invars[0]
    original = var.aval
    try:
        var.aval = invalid
        assert rules.freeze_source_structure_rule(native, inputs, outputs) is None
        assert rules.freeze_source_semantic_key(native, inputs, outputs) is None
    finally:
        var.aval = original
