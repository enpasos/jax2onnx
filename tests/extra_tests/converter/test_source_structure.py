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
from jax2onnx.plugins import _source_structure as rules
from jax2onnx.precision import CheckStatus, get_precision_report


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
    monkeypatch.setattr(rules, "freeze_source_structure_rule", lambda eqn: None)
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


def test_symbolic_zero_equation_identity_does_not_pass_by_dimension_strings() -> None:
    shape = jax_export.symbolic_shape("B, 3")
    model = _source(lambda x: x, arity=1, shape=shape)
    assert model.meta[structure.RESULT_KEY].status == "unverified"
    assert model.meta[structure.RESULT_KEY].reasons == (
        "scope-capture:symbolic-shape-contract-unavailable",
    )


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
    assert rules.freeze_source_structure_rule(jaxpr.eqns[0]) is not None
    fake = jaxpr.eqns[0].replace(primitive=Primitive("sub"))
    assert rules.freeze_source_structure_rule(fake) is None
    # Missing primitives are malformed boundary fixtures, not native equations.
    missing: Any = SimpleNamespace(primitive=None)
    assert rules.freeze_source_structure_rule(missing) is None


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
