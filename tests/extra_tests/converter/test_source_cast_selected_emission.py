# tests/extra_tests/converter/test_source_cast_selected_emission.py

"""Genuine selected execution; fixed integer oracles never choose the route."""

from dataclasses import FrozenInstanceError, replace
import copy
import pickle
import sys
from typing import Any

import jax
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx.converter import _source_cast_authority as authority
from jax2onnx.converter import _source_cast_context as acquisition
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import validate_source_jaxpr
from jax2onnx.plugins.jax._source_f64_f16_structure import (
    F64F16RneSpec,
    F64F16DoubleRoundSpec,
    check_f64_f16_rne_graph,
    check_f64_f16_double_round_graph,
)
from tests.extra_tests.converter.test_f64_to_f16_rne_emission import (
    _rne_half_bits,
    _session,
    _ORIGINAL_SIX_WORDS,
)
from tests.extra_tests.converter.test_f64_to_f16_double_round_emission import (
    _double_round_bits,
)


def _direct(value: Any) -> Any:
    return jax.lax.convert_element_type(value, np.float16)


def export_selected(shape: tuple[int, ...] = (6,), *, clone: bool = True) -> ir.Model:
    return conversion_api._convert_to_ir(
        fn=_direct,
        inputs=[jax.ShapeDtypeStruct(shape, np.float64)],
        input_params=None,
        model_name="selected_cast",
        opset=23,
        enable_double_precision=False,
        record_primitive_calls_file=None,
        protective_clone=clone,
        precision_policy=PrecisionPolicy.SOURCE,
        jaxpr_validator=validate_source_jaxpr,
    )


def assert_selected_model(model: ir.Model) -> authority.SelectedCastEmission:
    receipt = model.meta[authority.EMISSION_KEY]
    assert type(receipt) is authority.SelectedCastEmission
    assert receipt.contract is model.meta[authority.AUTHORITY_KEY]
    assert receipt.contract.numerical_activation is False
    assert receipt.structure == "validated" and receipt.runtime == "unverified"
    source, result = model.graph.inputs[0], model.graph.outputs[0]
    producer = result.producer()
    if producer is not None and producer.op_type == "Identity":
        result = producer.inputs[0]
    if receipt.contract.algorithm == "DIRECT_RNE_53_TO_11":
        matched = check_f64_f16_rne_graph(
            F64F16RneSpec(receipt.contract.shape), model.graph, source, result
        )
        compute = 63
    else:
        assert receipt.contract.algorithm == "DOUBLE_ROUND_RNE_53_TO_24_TO_11"
        matched = check_f64_f16_double_round_graph(
            F64F16DoubleRoundSpec(receipt.contract.shape), model.graph, source, result
        )
        compute = 65
    assert dict(model.graph.opset_imports) == {"": 23} and not model.functions
    assert receipt.checked_nodes == len(matched.nodes) == len(tuple(model.graph))
    assert receipt.literal_roles == len(matched.literal_obligations) == 59
    assert receipt.constant_definitions == len(matched.constants)
    assert (
        sum(node.op_type not in {"Constant", "Identity"} for node in matched.nodes)
        == compute
    )
    assert len(matched.nodes) <= 125
    onnx.checker.check_model(ir.to_proto(model), full_check=True)
    return receipt


def _values(shape: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    corpus = (
        *_ORIGINAL_SIX_WORDS,
        0,
        1 << 63,
        0x7FF0000000000000,
        0xFFF0000000000000,
        0x3E60000000000001,
        0xBE60000000000001,
        0x3F10000000000000,
    )
    size = int(np.prod(shape, dtype=np.int64))
    words = tuple(corpus[index % len(corpus)] for index in range(size))
    return np.asarray(words, dtype="<u8").view("<f8").reshape(shape), words


@pytest.mark.parametrize("optimization", ["disabled", "all"])
@pytest.mark.parametrize("shape", [(), (0,), (6,), (17,), (3, 7)])
def test_real_selected_math_matches_fixed_integer_oracle(
    shape: tuple[int, ...],
    optimization: str,
) -> None:
    with jax.enable_x64(True):
        model = export_selected(shape)
    receipt = assert_selected_model(model)
    oracle = (
        _rne_half_bits
        if receipt.contract.algorithm == "DIRECT_RNE_53_TO_11"
        else _double_round_bits
    )
    values, words = _values(shape)
    expected = np.asarray([oracle(word) for word in words], dtype="<u2").reshape(shape)
    session = _session(model, optimization)
    actual = session.run(None, {model.graph.inputs[0].name: values})[0]
    assert actual.dtype == np.dtype("float16") and actual.shape == shape
    np.testing.assert_array_equal(actual.view("<u2"), expected)


def test_clone_and_optional_ledger_do_not_select_or_change_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create = conversion_api._create_ir_context
    lower_native = acquisition.lower_original_cast_jaxpr
    graphs = []
    receipts = []
    for clone in (False, True):
        for ledger in (False, True):
            source_calls, resolver_calls = [], []

            def observer(frame: Any, event: str, arg: Any) -> Any:
                if event == "call" and frame.f_code is _direct.__code__:
                    source_calls.append("original")
                return None

            def context(**kwargs: Any) -> Any:
                result = create(**kwargs)
                if not ledger:
                    result._source_structure = None
                return result

            def native(*args: Any) -> Any:
                result = lower_native(*args)
                resolver_calls.append(result.resolver_requests)
                assert (
                    result.source_compile_requests
                    == result.source_execute_requests
                    == 0
                )
                return result

            previous = sys.gettrace()
            try:
                with jax.enable_x64(True), monkeypatch.context() as patch:
                    patch.setattr(conversion_api, "_create_ir_context", context)
                    patch.setattr(acquisition, "lower_original_cast_jaxpr", native)
                    sys.settrace(observer)
                    model = export_selected(clone=clone)
            finally:
                sys.settrace(previous)
            assert source_calls == ["original"] and resolver_calls == [1]
            receipt = assert_selected_model(model)
            values, words = _values((6,))
            oracle = (
                _rne_half_bits
                if receipt.contract.algorithm == "DIRECT_RNE_53_TO_11"
                else _double_round_bits
            )
            expected = np.asarray([oracle(word) for word in words], dtype="<u2")
            for optimization in ("disabled", "all"):
                output = _session(model, optimization).run(
                    None, {model.graph.inputs[0].name: values}
                )[0]
                assert output.dtype == np.dtype("float16") and output.shape == (6,)
                np.testing.assert_array_equal(output.view("<u2"), expected)
            fields = dict(model.meta[acquisition.CONTEXT_KEY].native_fields)
            assert fields["work"] == (
                "compile=1,serialize-executable=1,execute=0"
                if jax.__version__ == "0.10.2"
                else "compile=0,serialize-executable=0,execute=0"
            )
            graphs.append(ir.to_proto(model).SerializeToString(deterministic=True))
            receipts.append(receipt)
    assert all(value == graphs[0] for value in graphs)
    assert all(value == receipts[0] for value in receipts)


@pytest.mark.parametrize(
    "fault", ["missing", "copied", "cross-context", "audit", "rule"]
)
def test_genuine_prepared_session_cannot_be_replaced_or_retargeted(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lower = conversion_api._lower_jaxpr_equations
    sessions = []

    def inspect(ctx: Any, body: Any) -> None:
        session = ctx._source_cast_session
        assert type(session) is authority._CastExecutionSession
        assert (
            session.rule is ctx._source_cast_rule
            and session.audit is ctx._source_cast_audit
        )
        assert session._receipt is None
        sessions.append(session)
        if fault == "missing":
            ctx._source_cast_session = None
        elif fault == "copied":
            for operation in (copy.copy, copy.deepcopy, pickle.dumps):
                with pytest.raises(TypeError):
                    operation(session)
            ctx._source_cast_session = object()
        elif fault == "cross-context":
            foreign = conversion_api.IRContext(
                opset=23,
                enable_double_precision=False,
                precision_policy=PrecisionPolicy.SOURCE,
            )
            foreign._source_cast_session = session
            foreign._source_cast_selected = True
            foreign._source_cast_rule = session.rule
            with pytest.raises(authority.SourceCastSelectionError):
                authority._take_selected_cast(foreign, body.eqns[0])
            ctx._source_cast_session = None
        elif fault == "audit":
            ctx._source_cast_audit = replace(session.audit, algorithm="forged")
        else:
            ctx._source_cast_rule = replace(session.rule)
        lower(ctx, body)

    monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", inspect)
    with jax.enable_x64(True), pytest.raises(authority.SourceCastSelectionError):
        export_selected()
    assert len(sessions) == 1
    session = sessions[0]
    assert session._state == "closed" and session._owner is session._context is None
    assert (
        session._equation
        is session._value
        is session._input_witness
        is session._output_witness
        is None
    )


def test_emission_exception_aborts_once_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jax2onnx.plugins.jax import _f64_to_f16_rne, _f64_to_f16_double_round

    calls = []
    sessions = []
    lower = conversion_api._lower_jaxpr_equations

    def inspect(ctx: Any, body: Any) -> None:
        sessions.append(ctx._source_cast_session)
        lower(ctx, body)

    def failure(*args: Any) -> Any:
        calls.append("selected")
        raise ArithmeticError("selected-emitter-failure")

    monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", inspect)
    monkeypatch.setattr(_f64_to_f16_rne, "emit_f64_to_f16_rne", failure)
    monkeypatch.setattr(
        _f64_to_f16_double_round, "emit_f64_to_f16_double_round", failure
    )
    with (
        jax.enable_x64(True),
        pytest.raises(ArithmeticError, match="selected-emitter-failure"),
    ):
        export_selected()
    assert calls == ["selected"] and len(sessions) == 1
    assert (
        sessions[0]._state == "closed"
        and sessions[0]._owner is sessions[0]._context is None
    )


def test_final_emission_receipt_is_immutable_and_contains_no_execution_owner() -> None:
    with jax.enable_x64(True):
        model = export_selected()
    receipt = assert_selected_model(model)
    with pytest.raises(FrozenInstanceError):
        receipt.structure = "forged"
    assert not any(
        isinstance(value, authority._CastExecutionSession)
        for value in model.meta.values()
    )
