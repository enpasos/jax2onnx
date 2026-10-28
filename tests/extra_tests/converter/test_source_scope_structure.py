# tests/extra_tests/converter/test_source_scope_structure.py

"""Original Function bodies and every independently expected call attachment."""

from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace
from typing import Any

import jax
from flax import nnx
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import Jaxpr, JaxprEqn, Literal, Var
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.converter import source_structure as structure
from jax2onnx.converter import _source_regions as regions
from jax2onnx.converter import ir_context
from jax2onnx.ir_utils import iter_ir_functions
from jax2onnx.plugins.plugin_system import FunctionPlugin
from jax2onnx.plugins import _source_structure as rules
from jax2onnx.plugins import _source_function_structure as function_rules
from jax2onnx.plugins import plugin_system
from jax2onnx.precision import get_precision_report, precision_report_to_json

_INVOCATIONS: list[None] = []
_FAILURE_INVOCATIONS: list[None] = []
_OBSERVATION_INVOCATIONS: list[tuple[str, tuple[Any, ...]]] = []
_STORAGE = np.asarray([0x80000000, 0x7FC00001], np.uint32).view(np.float32)


@onnx_function
def _counted(value: Any) -> Any:
    _INVOCATIONS.append(None)
    return jax.lax.neg(value)


@onnx_function
def _counted_freezer_failure(value: Any) -> Any:
    _FAILURE_INVOCATIONS.append(None)
    return jax.lax.neg(value)


def _observation_configuration() -> tuple[Any, ...]:
    return (
        bool(jax.config.x64_enabled),
        jax.config.jax_default_matmul_precision,
        str(jax.config.jax_numpy_dtype_promotion),
        str(jax.config.jax_numpy_rank_promotion),
        bool(jax.config.jax_debug_key_reuse),
        active_precision_policy(),
        plugin_system._IN_FUNCTION_BUILD.get(),
    )


@onnx_function
def _observed_difference(left: Any, *, right: Any) -> Any:
    _OBSERVATION_INVOCATIONS.append(("leaf", _observation_configuration()))
    return jax.lax.sub(left, right)


@onnx_function
def _observed_nested(left: Any, right: Any) -> Any:
    _OBSERVATION_INVOCATIONS.append(("outer", _observation_configuration()))
    return jax.lax.sub(
        _observed_difference(left, right=right), _observed_difference(right, right=left)
    )


@jax.jit
def _jitted_difference(left: Any, right: Any) -> Any:
    return jax.lax.sub(left, right)


def _scanned_difference(left: Any, right: Any) -> Any:
    return jax.lax.scan(
        lambda carry, unused: (jax.lax.sub(carry, right), carry),
        left,
        None,
        length=2,
    )


@onnx_function
def _function_with_jit(left: Any, right: Any) -> Any:
    return _jitted_difference(left, right)


@onnx_function
def _function_with_scan(left: Any, right: Any) -> Any:
    return _scanned_difference(left, right)


@onnx_function
def _captured_storage(value: Any) -> Any:
    return jax.lax.sub(value, _STORAGE)


@onnx_function
def _chain_leaf(value: Any) -> Any:
    return jax.lax.neg(value)


@onnx_function
def _chain_middle(value: Any) -> Any:
    return _chain_leaf(value)


@onnx_function
def _chain_outer(value: Any) -> Any:
    return _chain_middle(value)


@onnx_function
def _difference(left: Any, *, right: Any) -> Any:
    return jax.lax.sub(left, right)


@onnx_function
def _product(left: Any, right: Any) -> Any:
    return jax.lax.mul(left, right)


@onnx_function
def _nested(left: Any, right: Any) -> Any:
    return jax.lax.sub(_difference(left, right=right), _difference(right, right=left))


@onnx_function
def _multi_result(left: Any, right: Any) -> Any:
    value = jax.lax.sub(left, right)
    return value, left, value


@onnx_function
def _rounding(value: Any) -> Any:
    return value.astype(np.float32).astype(np.float64)


@onnx_function
def _literal(value: Any) -> Any:
    return jax.lax.sub(value, np.float32(-0.0))


@onnx_function
class _Counter(nnx.Module):
    def __init__(self) -> None:
        self.total = nnx.Variable(np.zeros(2, np.float32))

    def __call__(self, value: Any, *, increment: Any) -> Any:
        self.total[...] = self.total[...] + increment
        return value - self.total[...]


def _source(fn: Any, *, arity: int = 2, dtype: Any = np.float32) -> ir.Model:
    with jax.enable_x64(True):
        model = ui._to_onnx(
            fn,
            [jax.ShapeDtypeStruct((2,), dtype)] * arity,
            precision_policy=PrecisionPolicy.SOURCE,
            return_mode="ir",
        )
    assert isinstance(model, ir.Model)
    return model


def _proof(model: ir.Model) -> structure.StructureResult:
    return structure.validate_source_structure(model)


def _calls(graph: ir.Graph) -> list[ir.Node]:
    return [node for node in graph if node.domain]


def test_nested_reused_function_has_original_logical_and_physical_coverage() -> None:
    model = _source(lambda x, y: _nested(x, y))
    result = _proof(model)
    assert result.status == "validated", result
    assert result.checked_equations == result.expected_equations == 6
    ledger = model.meta[structure._LEDGER]
    definitions = list(iter_ir_functions(model.functions))
    assert len(definitions) == 2
    assert len(ledger.scopes) == 3
    assert len(ledger.function_calls) == 3
    assert len({call.owner for call in ledger.function_calls.values()}) == 2
    assert sum(len(scope.equations) for scope in ledger.scopes) == 5
    assert result.runtime == result.numerics == "unverified"


def test_native_default_mul_has_a_complete_function_body_contract() -> None:
    model = _source(lambda x, y: _product(x, y))
    result = _proof(model)
    assert result.status == "validated", result
    assert result.expected_equations == result.checked_equations == 2


def test_nested_body_hash_visits_each_single_call_child_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _source(lambda x: _chain_outer(x), arity=1)
    assert _proof(model).status == "validated"
    ledger = model.meta[structure._LEDGER]
    root = ledger.scopes[0]
    original_body = ledger.function_calls[
        structure.EquationHandle(root.invocation, 0)
    ].body

    def fresh(value: Any) -> Any:
        if type(value) is tuple:
            return tuple(fresh(item) for item in value)
        if type(value) in (
            regions.FrozenBody,
            regions.FrozenEquation,
            regions.FrozenUse,
            structure.TensorContract,
            structure.ConstantContract,
        ):
            return replace(
                value,
                **{
                    member.name: fresh(getattr(value, member.name))
                    for member in fields(value)
                },
            )
        assert value is None or type(value) in (str, int, bool, ir.DataType)
        return value

    # CPython 3.14 caches tuple hashes. Export has already hashed the original
    # contract, so rebuild every tuple (including semantic keys) and dataclass
    # before measuring. Equality does not hash or prewarm the fresh contract.
    body = fresh(original_body)
    assert body == original_body
    expected = []
    current = body
    while True:
        expected.append(id(current))
        children = [
            child for equation in current.equations for _, child in equation.children
        ]
        if not children:
            break
        assert len(children) == 1
        assert current.equations[0].semantics == ("function",)
        current = children[0]
    assert len(expected) == 3
    actual_hash = regions.FrozenBody.__hash__
    observed = []

    def measured(value: Any) -> int:
        observed.append(id(value))
        return actual_hash(value)

    monkeypatch.setattr(regions.FrozenBody, "__hash__", measured)
    hash(body)
    assert observed == expected


@pytest.mark.parametrize("failure", ["tensor-use", "semantic-discovery"])
def test_root_declaration_survives_nested_semantic_capture_failure(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("deliberately unavailable observation")

    if failure == "tensor-use":
        monkeypatch.setattr(structure, "_use", unavailable)
    else:
        monkeypatch.setattr(function_rules, "original_function_body", unavailable)
    model = _source(lambda x, y: _nested(x, y))
    result = _proof(model)
    assert result.status == "unverified"
    assert result.expected_equations == 6
    assert model.meta[structure._LEDGER].root_body.expected_equations == 6


def test_unavailable_declaration_reports_unknown_total_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("declaration unavailable")

    monkeypatch.setattr(function_rules, "declared_function_body", unavailable)
    model = _source(lambda x, y: jax.lax.sub(x, y))
    result = _proof(model)
    assert result.status == "unverified"
    assert result.expected_equations is None
    assert "expected-region-total-unknown" in result.reasons


@pytest.mark.parametrize("carrier", ["jit", "scan"])
@pytest.mark.parametrize("nested", [False, True])
def test_unhandled_native_body_never_claims_an_exact_outer_prefix_total(
    carrier: str,
    nested: bool,
) -> None:
    if carrier == "jit":
        if nested:
            model = _source(lambda x, y: _function_with_jit(x, y))
        else:
            model = _source(lambda x, y: _jitted_difference(x, y))
    elif nested:
        model = _source(lambda x, y: _function_with_scan(x, y))
    else:
        model = _source(lambda x, y: _scanned_difference(x, y))
    result = _proof(model)
    if carrier == "scan":
        assert result.status == "validated", result
        assert (
            result.expected_equations
            == result.checked_equations
            == (3 if nested else 2)
        )
        assert "expected-region-total-unknown" not in result.reasons
    else:
        assert result.status == "unverified"
        assert result.expected_equations is None
        assert "expected-region-total-unknown" in result.reasons
    assert "scope-capture:source-region-declaration-differs" not in result.reasons


@pytest.mark.parametrize(
    "params",
    [
        {"out_dtype": np.float32},
        {"out_dtype": np.float64},
        {"out_dtype": None, "future_setting": None},
    ],
)
def test_mul_nondefault_and_unknown_params_cannot_borrow_default_proof(
    params: Any,
) -> None:
    specs = (jax.ShapeDtypeStruct((2,), np.float32),) * 2
    native = jax.make_jaxpr(jax.lax.mul)(*specs).jaxpr
    equation = native.eqns[0]
    assert equation.primitive is jax.lax.mul_p
    shapes = structure._ShapeFreezer()
    inputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.invars
    )
    outputs = tuple(
        structure.TensorContract.from_aval(var.aval, shapes) for var in equation.outvars
    )
    altered = equation.replace(params={**equation.params, **params})
    assert rules.freeze_source_structure_rule(altered, inputs, outputs) is None
    assert rules.freeze_source_semantic_key(altered, inputs, outputs) is None


@pytest.mark.parametrize("mutation", ["operand", "capture", "target", "result"])
def test_second_call_same_typed_mutations_are_checked(mutation: str) -> None:
    model = _source(lambda x, y: (_difference(x, right=y), _difference(y, right=x)))
    assert _proof(model).status == "validated"
    first, second = _calls(model.graph)
    if mutation == "operand":
        second.replace_input_with(0, second.inputs[1])
    elif mutation == "capture":
        second.replace_input_with(1, second.inputs[0])
    elif mutation == "target":
        second.op_type = first.op_type + "_different"
    else:
        model.graph.outputs[1] = model.graph.outputs[0]
    assert _proof(model).status == "failed"


@pytest.mark.parametrize("mutation", ["operator", "operand", "constant"])
def test_function_body_mutation_keeps_signature_but_loses_original_proof(
    mutation: str,
) -> None:
    model = _source(lambda x, y: _difference(x, right=y))
    assert _proof(model).status == "validated"
    definition = next(iter_ir_functions(model.functions))
    node = next(node for node in definition.graph if node.op_type == "Sub")
    if mutation == "operator":
        node.op_type = "Add"
    elif mutation == "operand":
        node.replace_input_with(1, node.inputs[0])
    else:
        # An extra same-typed immutable leaf cannot substitute for a formal.
        value = ir.Value(name="foreign", const_value=ir.tensor(np.zeros(2, np.float32)))
        value.type = ir.TensorType(ir.DataType.FLOAT)
        value.shape = ir.Shape((2,))
        definition.graph.initializers["foreign"] = value
        node.replace_input_with(1, value)
    assert _proof(model).status == "failed"


@pytest.mark.parametrize("mutation", ["call", "definition", "body", "owner"])
def test_missing_second_attachment_does_not_shrink_expected_denominator(
    mutation: str,
) -> None:
    model = _source(lambda x, y: (_difference(x, right=y), _difference(y, right=x)))
    ledger = model.meta[structure._LEDGER]
    assert _proof(model).status == "validated"
    total = ledger.expected_equations
    if mutation == "call":
        ledger.function_calls.pop(next(reversed(ledger.function_calls)))
    elif mutation == "definition":
        ledger.scopes.pop()
    elif mutation == "body":
        ledger.function_bodies.clear()
    else:
        next(iter_ir_functions(model.functions)).graph.meta.pop(structure._OWNER)
    result = _proof(model)
    assert result.status == "unverified"
    assert result.expected_equations == total == 4


@pytest.mark.parametrize("receipt", ["none", "contradictory"])
def test_definition_body_receipt_distinguishes_missing_from_known_contradiction(
    receipt: str,
) -> None:
    model = _source(lambda x, y: (_difference(x, right=y), _difference(y, right=x)))
    assert _proof(model).status == "validated"
    ledger = model.meta[structure._LEDGER]
    owner, body = next(iter(ledger.function_bodies.items()))
    if receipt == "none":
        ledger.function_bodies[owner] = None
    else:
        # Same typed interface and equations; a present proof now claims a
        # passthrough result instead of the original subtraction result.
        ledger.function_bodies[owner] = replace(
            body, outputs=(replace(body.outputs[0], kind="input", index=0, port=0),)
        )
    result = _proof(model)
    assert result.expected_equations == 4
    if receipt == "none":
        assert result.status == "unverified"
        assert result.reasons == ("function-definition-source-body-unavailable",)
    else:
        assert result.status == "failed"
        assert result.reasons == ("function-definition-source-body-differs",)


@pytest.mark.parametrize("region", ["root", "body"])
def test_nested_missing_equation_receipt_preserves_declared_tree_and_reason(
    region: str,
) -> None:
    model = _source(lambda x, y: _nested(x, y))
    assert _proof(model).status == "validated"
    ledger = model.meta[structure._LEDGER]
    scope = (
        ledger.scopes[0]
        if region == "root"
        else next(scope for scope in ledger.scopes[1:] if len(scope.equations) == 1)
    )
    scope.receipts.pop(0)
    result = _proof(model)
    assert result.status == "unverified"
    assert result.expected_equations == 6
    assert result.checked_equations < 6
    assert result.reasons == ("source-coverage-incomplete",)


def test_otherwise_complete_scopes_still_enforce_independent_declared_total() -> None:
    model = _source(lambda x, y: _nested(x, y))
    assert _proof(model).status == "validated"
    ledger = model.meta[structure._LEDGER]
    assert ledger.expected_equations == 6
    ledger.expected_equations = 7
    result = _proof(model)
    assert result.status == "unverified"
    assert result.expected_equations == 7
    assert result.checked_equations == 6
    assert result.reasons == ("expected-region-coverage-incomplete",)


def test_same_signature_retrace_changes_are_compared_with_original_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = FunctionPlugin._trace_function_body

    def altered(self: Any, *args: Any, **kwargs: Any) -> Any:
        closed = original(self, *args, **kwargs)
        body = closed.jaxpr
        equation = body.eqns[0]
        assert equation.primitive is jax.lax.sub_p
        changed = body.replace(eqns=[equation.replace(primitive=jax.lax.add_p)])
        return SimpleNamespace(jaxpr=changed, consts=closed.consts)

    monkeypatch.setattr(FunctionPlugin, "_trace_function_body", altered)
    model = _source(lambda x, y: _difference(x, right=y))
    result = _proof(model)
    assert result.status == "failed"
    assert "function-body:original-function-body-differs" in result.reasons


def test_reuse_compares_each_original_body_without_retracing_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = FunctionPlugin._trace_function_body
    builds = []

    def observed(self: Any, *args: Any, **kwargs: Any) -> Any:
        builds.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FunctionPlugin, "_trace_function_body", observed)
    model = _source(lambda x, y: (_difference(x, right=y), _difference(y, right=x)))
    assert _proof(model).status == "validated"
    assert len(builds) == 1
    assert len(model.meta[structure._LEDGER].function_calls) == 2


def test_body_freezing_keeps_the_original_and_lowering_abstract_invocations() -> None:
    _INVOCATIONS.clear()
    model = _source(lambda x: _counted(x), arity=1)
    assert _proof(model).status == "validated"
    assert len(_INVOCATIONS) == 2
    _proof(model)
    assert len(_INVOCATIONS) == 2


def test_binding_semantic_freezer_failure_preserves_two_abstract_invocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures = []

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        failures.append(True)
        raise ValueError("deliberately unavailable semantic freezer")

    _FAILURE_INVOCATIONS.clear()
    monkeypatch.setattr(rules, "freeze_source_semantic_key", unavailable)
    model = _source(lambda x: _counted_freezer_failure(x), arity=1)
    assert failures
    assert len(_FAILURE_INVOCATIONS) == 2
    result = _proof(model)
    assert result.status == "unverified"
    assert result.expected_equations == 2
    assert any("source-semantic-key:ValueError" in reason for reason in result.reasons)
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    _proof(model)
    assert len(_FAILURE_INVOCATIONS) == 2


def test_nested_reused_function_ledger_observation_does_not_change_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fresh_root() -> Any:
        def target(left: Any, right: Any) -> Any:
            return _observed_nested(left, right)

        return target

    configuration = _observation_configuration()
    _OBSERVATION_INVOCATIONS.clear()
    enabled = _source(fresh_root())
    enabled_calls = tuple(_OBSERVATION_INVOCATIONS)
    assert _observation_configuration() == configuration
    assert _proof(enabled).status == "validated"
    assert len(enabled.meta[structure._LEDGER].function_calls) == 3
    assert len(list(iter_ir_functions(enabled.functions))) == 2
    assert {name for name, _ in enabled_calls} == {"outer", "leaf"}
    disabled_constructions = []

    def no_ledger() -> None:
        disabled_constructions.append(True)
        return None

    _OBSERVATION_INVOCATIONS.clear()
    with monkeypatch.context() as patch:
        # Disable only the IRContext ledger factory. SOURCE policy, original
        # callees, trace bodies, configurations and lowering remain unchanged.
        patch.setattr(ir_context, "SourceStructureLedger", no_ledger)
        disabled = _source(fresh_root())
    assert ir_context.SourceStructureLedger is structure.SourceStructureLedger
    assert disabled_constructions
    assert structure._LEDGER not in disabled.meta
    assert _proof(disabled).reasons == ("source-ledger-unavailable",)
    assert tuple(_OBSERVATION_INVOCATIONS) == enabled_calls
    assert _observation_configuration() == configuration
    enabled_proto, disabled_proto = ir.to_proto(enabled), ir.to_proto(disabled)
    onnx.checker.check_model(enabled_proto, full_check=True)
    onnx.checker.check_model(disabled_proto, full_check=True)
    assert enabled_proto.SerializeToString(
        deterministic=True
    ) == disabled_proto.SerializeToString(deterministic=True)
    enabled_report, disabled_report = get_precision_report(
        enabled
    ), get_precision_report(disabled)
    assert enabled_report is not None and disabled_report is not None
    assert enabled_report == disabled_report
    assert precision_report_to_json(enabled_report) == precision_report_to_json(
        disabled_report
    )
    assert tuple(_OBSERVATION_INVOCATIONS) == enabled_calls


def test_identical_call_provenance_loss_cannot_reduce_expected_invocations() -> None:
    model = _source(lambda x, y: (_difference(x, right=y), _difference(x, right=y)))
    ledger = model.meta[structure._LEDGER]
    assert ledger.expected_equations == 4
    assert len(ledger.function_calls) == 2
    result = _proof(model)
    # Existing CSE can discard the second node's value markers. No observed
    # retained-node count is allowed to turn that missing receipt into success.
    if len(_calls(model.graph)) == 1:
        assert result.status == "unverified"
    else:
        assert result.status == "validated"
        assert result.checked_equations == 4


def test_reused_definition_cannot_substitute_a_different_body_contract() -> None:
    model = _source(lambda x, y: (_difference(x, right=y), _difference(y, right=x)))
    assert _proof(model).status == "validated"
    ledger = model.meta[structure._LEDGER]
    handle = next(reversed(ledger.function_calls))
    call = ledger.function_calls[handle]
    changed = replace(
        call.body, outputs=(replace(call.body.outputs[0], kind="input", index=1),)
    )
    ledger.function_calls[handle] = replace(call, body=changed)
    assert _proof(model).status == "failed"


def test_result_repetition_and_passthrough_keep_ordered_ports() -> None:
    model = _source(lambda x, y: _multi_result(x, y))
    assert _proof(model).status == "validated"
    definition = next(iter_ir_functions(model.functions))
    assert len({id(value) for value in definition.graph.outputs}) == 3
    definition.graph.outputs[1] = definition.graph.outputs[0]
    assert _proof(model).status == "failed"


def test_function_rounding_boundaries_remain_explicit() -> None:
    model = _source(lambda x: _rounding(x), arity=1, dtype=np.float64)
    assert _proof(model).status == "validated"
    definition = next(iter_ir_functions(model.functions))
    assert [node.op_type for node in definition.graph] == ["Cast", "Cast"]


def test_function_literal_keeps_signed_zero_storage() -> None:
    model = _source(lambda x: _literal(x), arity=1)
    assert _proof(model).status == "validated"
    definition = next(iter_ir_functions(model.functions))
    constant = next(node for node in definition.graph if node.op_type == "Constant")
    constant.attributes["value"] = ir.AttrTensor(
        "value", ir.tensor(np.asarray(0.0, np.float32))
    )
    assert _proof(model).status == "failed"


@pytest.mark.parametrize("word", [0, 1])
def test_function_closed_constant_reads_actual_attribute_not_cached_payload(
    word: int,
) -> None:
    model = _source(lambda x: _captured_storage(x), arity=1)
    assert _proof(model).status == "validated"
    definition = next(iter_ir_functions(model.functions))
    constant = next(node for node in definition.graph if node.op_type == "Constant")
    changed = _STORAGE.copy().view(np.uint32)
    changed[word] = np.uint32(0) if word == 0 else changed[word] ^ np.uint32(1)
    # output.const_value deliberately stays unchanged; current executable
    # Constant storage, including signed zero/NaN payload, is authoritative.
    constant.attributes["value"] = ir.AttrTensor(
        "value", ir.tensor(changed.view(np.float32))
    )
    result = _proof(model)
    assert result.status == "failed"
    assert result.reasons == ("source-constant-storage-differs",)


def test_function_ledger_retains_no_native_trace_or_arrays() -> None:
    model = _source(lambda x, y: _nested(x, y))
    assert _proof(model).status == "validated"

    def inspect(value: Any) -> None:
        assert not isinstance(
            value, (Jaxpr, JaxprEqn, Var, Literal, np.ndarray, jax.Array)
        )
        if is_dataclass(value) and not isinstance(value, type):
            for member in fields(value):
                inspect(getattr(value, member.name))
        elif isinstance(value, dict):
            for key, item in value.items():
                inspect(key)
                inspect(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                inspect(item)

    inspect(vars(model.meta[structure._LEDGER]))


def test_nnx_state_and_dynamic_keyword_have_distinct_original_ports() -> None:
    graphdef, state = nnx.split(_Counter())
    leaves, tree = jax.tree.flatten(state)
    assert len(leaves) == 1

    def target(value: Any, increment: Any, total: Any) -> Any:
        owner = nnx.merge(graphdef, jax.tree.unflatten(tree, [total]))
        result = owner(value, increment=increment)
        return result, owner.total[...]

    model = _source(target, arity=3)
    result = _proof(model)
    assert result.status == "validated", result
    call = next(iter(model.meta[structure._LEDGER].function_calls.values()))
    assert call.order == (0, 2, 1)
    assert result.expected_equations == result.checked_equations == 3


@pytest.mark.parametrize("control", ["if", "while"])
def test_visible_control_descendants_remain_expected_without_attachment_adapter(
    control: str,
) -> None:
    if control == "if":

        def fn(x, y):
            return jax.lax.cond(
                np.asarray(True), lambda a: a[0] - a[1], lambda a: a[1] - a[0], (x, y)
            )

    else:

        def fn(x, y):
            return jax.lax.while_loop(lambda a: np.asarray(False), lambda a: a - y, x)

    model = _source(fn)
    result = _proof(model)
    assert result.expected_equations > len(
        model.meta[structure._LEDGER].scopes[0].equations
    )
    if control == "while":
        # Keep the original case and both condition invocations. The scalar
        # While adapter now checks all three original roles and actual ports.
        assert result.status == "validated", result
        assert result.checked_equations == result.expected_equations
        assert {key.role for key in model.meta[structure._LEDGER].regions} == {
            "initial-condition",
            "body",
            "next-condition",
        }
    else:
        assert result.status == "validated", result
        assert result.checked_equations == result.expected_equations == 3
        ledger = model.meta[structure._LEDGER]
        assert not ledger.if_calls and len(ledger.literal_if_calls) == 1
        attachment = next(iter(ledger.literal_if_calls.values()))
        assert attachment.rule.index == 1
        assert attachment.discharge.selector == attachment.rule.selector
        assert {key.role for key in ledger.regions} == {"false", "true"}
        assert attachment.false.body.expected_equations == 1
        assert attachment.true.body.expected_equations == 1
        assert attachment.false.invocation is attachment.discharge.inactive_invocation
        assert attachment.false.body == attachment.discharge.inactive_body
        assert attachment.discharge.inactive_role == "false"
        assert attachment.discharge.graph_matched_equations == 1
        assert attachment.discharge.semantically_discharged_equations == 2
        assert attachment.discharge.inactive_nodes == ()
        assert attachment.discharge.inactive_storage == ()
