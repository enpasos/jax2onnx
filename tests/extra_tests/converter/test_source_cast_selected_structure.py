# tests/extra_tests/converter/test_source_cast_selected_structure.py

"""Final production graphs must satisfy the prebound selected mathematical rule."""

from dataclasses import FrozenInstanceError, replace
from typing import Any

import jax
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx.converter import _source_cast_authority as authority
from jax2onnx.converter import _source_cast_planning as planning
from jax2onnx.converter import conversion_api
from jax2onnx.converter import source_structure as structure
from jax2onnx.plugins.jax._source_f64_f16_structure import SelectedF64F16Rule
from tests.extra_tests.converter.test_source_cast_selected_emission import (
    assert_selected_model,
    export_selected,
)


def _result(graph: ir.Graph) -> ir.Value:
    result = graph.outputs[0]
    producer = result.producer()
    return producer.inputs[0] if producer.op_type == "Identity" else result


def test_rule_is_prebound_to_original_handle_and_does_not_rewrite_native_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lower = conversion_api._lower_jaxpr_equations
    rules = []
    contexts = []

    def inspect(ctx: Any, body: Any) -> None:
        rule = ctx._source_cast_rule
        assert type(rule) is SelectedF64F16Rule
        assert rule.contract is ctx._source_cast_audit
        assert ctx._source_cast_session._state == "prepared"
        assert ctx._source_structure.scopes == []
        with pytest.raises(FrozenInstanceError):
            rule.contract = replace(rule.contract)
        rules.append(rule)
        contexts.append(ctx)
        lower(ctx, body)
        assert ctx._source_cast_session._state == "emitted"
        ledger = ctx._source_structure
        scope = ledger.scopes[0]
        handle = structure.EquationHandle(scope.invocation, 0)
        assert ledger.handles == []
        assert scope.equations[0].source_id == id(body.eqns[0])
        assert ledger.cast_selections == {handle: rule}
        assert ledger.cast_selections[handle] is scope.equations[0].rule is rule
        assert scope.body.equations[0].semantics[0] == "native-f64-f16-convert"
        assert (
            ctx._source_cast_plan.algorithm is planning.UnresolvedAlgorithm.UNRESOLVED
        )

    monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", inspect)
    with jax.enable_x64(True):
        model = export_selected()
    receipt = assert_selected_model(model)
    assert len(rules) == len(contexts) == 1 and receipt.contract is rules[0].contract
    assert (
        model.meta[planning.PLAN_KEY].algorithm
        is planning.UnresolvedAlgorithm.UNRESOLVED
    )
    assert structure.validate_source_structure(model).status == "validated"
    assert contexts[0]._source_cast_session is contexts[0]._source_cast_rule is None


@pytest.mark.parametrize("fault", ["missing", "copied", "foreign-handle"])
def test_observation_mirror_requires_exact_rule_and_original_handle(fault: str) -> None:
    with jax.enable_x64(True):
        model = export_selected()
    assert_selected_model(model)
    ledger = model.meta[structure._LEDGER]
    assert structure.validate_source_structure(model).status == "validated"
    handle, rule = next(iter(ledger.cast_selections.items()))
    if fault == "missing":
        ledger.cast_selections.clear()
    elif fault == "copied":
        ledger.cast_selections[handle] = replace(rule)
    else:
        ledger.cast_selections = {structure.EquationHandle(object(), 0): rule}
    result = structure.validate_source_structure(model)
    assert result.status == "unverified"
    assert result.reasons == ("selected-cast-handle-coverage-incomplete",)
    # The observational mirror cannot select, consume or rewrite genuine math.
    assert_selected_model(model)


def _mutate(model: ir.Model, session: Any, fault: str) -> None:
    graph = model.graph
    source, result = graph.inputs[0], _result(graph)
    matched = session.rule.match(graph, source, result)
    rounded = next(node for node in graph if node.op_type == "Round")
    if fault == "prefix-source":
        # LOW must use FLOAT->DOUBLE as the math input. DIRECT must use root.
        absolute = next(node for node in graph if node.op_type == "Abs")
        if absolute.inputs[0] is source:
            absolute.replace_input_with(0, rounded.outputs[0])
        else:
            absolute.replace_input_with(0, source)
    elif fault == "round-bypass":
        divided = next(node for node in graph if node.op_type == "Div")
        assert divided.inputs[0] is rounded.outputs[0]
        divided.replace_input_with(0, rounded.inputs[0])
    elif fault == "round-missing":
        graph.remove(rounded)
    elif fault == "producer-order":
        divided = next(node for node in graph if node.op_type == "Div")
        graph.insert_after(divided, rounded)
    elif fault == "literal-bytes":
        matched.constants[0].const_value = ir.tensor(np.asarray(7.0, np.float64))
    elif fault == "literal-type":
        literal = matched.constants[0]
        literal.type = ir.TensorType(ir.DataType.FLOAT)
        literal.const_value = ir.tensor(np.asarray(2.0, np.float32))
    elif fault == "literal-shape":
        matched.constants[0].shape = ir.Shape((1,))
    elif fault == "literal-owner":
        comparison = next(node for node in graph if node.op_type == "GreaterOrEqual")
        foreign = conversion_api.IRContext(opset=23, enable_double_precision=False)
        value = foreign.bind_const_for_var(
            object(), comparison.inputs[1].const_value.numpy()
        )
        comparison.replace_input_with(1, value)
    elif fault == "extra-initializer":
        literal = ir.Value(
            name="foreign", const_value=ir.tensor(np.asarray(7.0, np.float64))
        )
        graph.initializers[literal.name] = literal
    elif fault == "extra-node":
        graph.append(
            ir.Node("", "Identity", [source], outputs=[ir.Value(name="unused")])
        )
    elif fault == "extra-output":
        graph.outputs.append(source)
    elif fault == "facade-alias":
        facade = graph.outputs[0].producer()
        if facade.op_type == "Identity":
            facade.replace_input_with(0, source)
        else:
            alias = ir.Value(name="wrong_alias", type=result.type, shape=result.shape)
            node = ir.Node("", "Identity", [source], outputs=[alias])
            graph.append(node)
            graph.outputs[0] = alias
    elif fault == "cast-default-attribute":
        result.producer().attributes["saturate"] = ir.AttrInt64("saturate", 1)
    elif fault == "opset":
        graph.opset_imports[""] = 24
    elif fault == "source-witness":
        source.meta[structure._VALUES] = ()
    elif fault == "result-witness":
        result.meta[structure._VALUES] = ()
    elif fault == "duplicate-witness":
        rounded.outputs[0].meta[structure._VALUES] = source.meta[structure._VALUES]
    else:
        raise AssertionError(fault)


@pytest.mark.parametrize(
    "fault",
    [
        "prefix-source",
        "round-bypass",
        "round-missing",
        "producer-order",
        "literal-bytes",
        "literal-type",
        "literal-shape",
        "literal-owner",
        "extra-initializer",
        "extra-node",
        "extra-output",
        "facade-alias",
        "cast-default-attribute",
        "opset",
        "source-witness",
        "result-witness",
        "duplicate-witness",
    ],
)
def test_final_graph_corruption_aborts_with_ledger_disabled(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_final_abort(monkeypatch, fault, ledger=False, clone=True)


@pytest.mark.parametrize("ledger", [False, True])
@pytest.mark.parametrize("clone", [False, True])
def test_mandatory_final_gate_cannot_be_disabled_by_ledger_or_clone(
    ledger: bool,
    clone: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_final_abort(monkeypatch, "round-bypass", ledger=ledger, clone=clone)


def _assert_final_abort(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    *,
    ledger: bool,
    clone: bool,
) -> None:
    build = conversion_api._build_and_finalize_ir_model
    create = conversion_api._create_ir_context
    sessions = []
    models = []

    def context(**kwargs: Any) -> Any:
        ctx = create(**kwargs)
        if not ledger:
            ctx._source_structure = None
        return ctx

    def corrupt(ctx: Any, **kwargs: Any) -> ir.Model:
        model = build(ctx, **kwargs)
        session = ctx._source_cast_session
        assert session._state == "emitted" and session._receipt is None
        assert authority.EMISSION_KEY not in model.meta
        sessions.append(session)
        models.append(model)
        _mutate(model, session, fault)
        return model

    monkeypatch.setattr(conversion_api, "_create_ir_context", context)
    monkeypatch.setattr(conversion_api, "_build_and_finalize_ir_model", corrupt)
    with (
        jax.enable_x64(True),
        pytest.raises((structure.StructureMismatch, structure.StructureUnavailable)),
    ):
        export_selected(clone=clone)
    assert len(sessions) == len(models) == 1
    assert authority.EMISSION_KEY not in models[0].meta
    session = sessions[0]
    assert session._state == "closed" and session._receipt is None
    assert session._owner is session._context is session._rule is session._audit is None
    assert session._input_witness is session._output_witness is None
