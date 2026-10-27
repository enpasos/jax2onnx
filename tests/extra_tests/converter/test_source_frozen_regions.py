# tests/extra_tests/converter/test_source_frozen_regions.py

"""Freeze independent real traces without retaining their native authorities.

These controls concern typed SSA and structural availability, not numerical or
runtime acceptance. No lowering receipt supplies the expected source body.
"""

from dataclasses import fields, is_dataclass
import hashlib
from typing import Any

import jax
from jax import export as jax_export
import numpy as np
import onnx_ir as ir
import pytest

from jax2onnx._compat.jax import ClosedJaxpr, Jaxpr, Primitive
from jax2onnx.converter import _source_regions as regions
from jax2onnx.converter.source_structure import (
    ConstantContract,
    StructureUnavailable,
    TensorContract,
)
from jax2onnx.plugins import _source_function_structure as functions
from jax2onnx.plugins import _source_structure as semantics


def _closed(function: Any, *specs: Any) -> Any:
    with jax.enable_x64(True):
        return jax.make_jaxpr(function)(*specs)


def _spec(
    shape: tuple[int, ...] = (2, 3), dtype: Any = np.float32, *, weak: bool = False
) -> Any:
    return jax.ShapeDtypeStruct(shape, dtype, weak_type=weak)


def _body(function: Any, *specs: Any) -> Any:
    return regions.freeze_closed_body(_closed(function, *specs))


def _assert_only_immutable_contracts(value: Any) -> None:
    if value is None or type(value) in (str, bool, int) or type(value) is ir.DataType:
        return
    if type(value) is tuple:
        for item in value:
            _assert_only_immutable_contracts(item)
        return
    assert is_dataclass(value) and type(value) in (
        regions.FrozenBody,
        regions.FrozenEquation,
        regions.FrozenUse,
        TensorContract,
        ConstantContract,
    ), f"Native or opaque owner retained: {type(value)!r}"
    for field in fields(value):
        _assert_only_immutable_contracts(getattr(value, field.name))


def test_alpha_distinct_real_traces_have_equal_hashable_typed_bodies() -> None:
    def original(x: Any, y: Any) -> Any:
        difference = jax.lax.sub(x, y)
        return jax.lax.mul(difference, x)

    def renamed(left: Any, right: Any) -> Any:
        temporary = jax.lax.sub(left, right)
        return jax.lax.mul(temporary, left)

    first, second = _closed(original, _spec(), _spec()), _closed(
        renamed, _spec(), _spec()
    )
    assert first.jaxpr.invars[0] is not second.jaxpr.invars[0]
    before, after = regions.freeze_closed_body(first), regions.freeze_closed_body(
        second
    )
    assert before.gaps == after.gaps == ()
    assert before == after and hash(before) == hash(after)
    assert before.expected_equations == after.expected_equations == 2
    assert before.equations[1].inputs[0].kind == "equation"
    assert before.equations[1].inputs[0].index == 0
    _assert_only_immutable_contracts(before)


@pytest.mark.parametrize(
    "change", ("operand-order", "same-typed-rewire", "output-order", "duplicate-output")
)
def test_same_typed_ssa_and_order_changes_remain_distinct(change: str) -> None:
    if change == "operand-order":
        before = _body(lambda x, y: jax.lax.sub(x, y), _spec(), _spec())
        after = _body(lambda x, y: jax.lax.sub(y, x), _spec(), _spec())
    elif change == "same-typed-rewire":

        def original(x: Any, y: Any) -> Any:
            left, right = jax.lax.sub(x, y), jax.lax.add(x, y)
            return jax.lax.mul(left, right)

        def rewired(x: Any, y: Any) -> Any:
            left, right = jax.lax.sub(x, y), jax.lax.add(x, y)
            assert right.shape == left.shape and right.dtype == left.dtype
            return jax.lax.mul(left, left)

        before, after = _body(original, _spec(), _spec()), _body(
            rewired, _spec(), _spec()
        )
        assert before.expected_equations == after.expected_equations == 3
        assert tuple(equation.semantics for equation in before.equations) == tuple(
            equation.semantics for equation in after.equations
        )
    elif change == "output-order":
        before = _body(lambda x, y: (x, y), _spec(), _spec())
        after = _body(lambda x, y: (y, x), _spec(), _spec())
    else:
        before = _body(lambda x, y: (x, x), _spec(), _spec())
        after = _body(lambda x, y: (x, y), _spec(), _spec())
    assert before.gaps == after.gaps == ()
    assert before.inputs == after.inputs
    assert tuple(value.tensor for value in before.outputs) == tuple(
        value.tensor for value in after.outputs
    )
    assert before != after


@pytest.mark.parametrize("change", ("dtype", "shape", "weak-type"))
def test_input_and_result_tensor_contracts_preserve_real_trace_types(
    change: str,
) -> None:
    if change == "dtype":
        first, second = _spec(dtype=np.float32), _spec(dtype=np.float64)
    elif change == "shape":
        first, second = _spec((2, 3)), _spec((3, 2))
    else:
        first, second = _spec((), np.float64, weak=True), _spec(
            (), np.float64, weak=False
        )
    before, after = _body(jax.lax.neg, first), _body(jax.lax.neg, second)
    assert before.gaps == after.gaps == ()
    assert before.expected_equations == after.expected_equations == 1
    assert before != after and before.inputs != after.inputs
    if change == "weak-type":
        assert before.inputs[0].weak_type is True
        assert after.inputs[0].weak_type is False


@pytest.mark.parametrize(
    "changed_bits", ((0x00000000, 0x7FC00001), (0x80000000, 0x7FC00002))
)
def test_hoisted_signed_zero_and_nan_payload_storage_are_not_normalized(
    changed_bits: tuple[int, int],
) -> None:
    original = np.array((0x80000000, 0x7FC00001), dtype=np.uint32).view(np.float32)
    changed = np.array(changed_bits, dtype=np.uint32).view(np.float32)
    first = _body(lambda x: jax.lax.sub(x, original), _spec((2,)))
    second = _body(lambda x: jax.lax.sub(x, changed), _spec((2,)))
    assert first.gaps == second.gaps == ()
    assert len(first.constants) == len(second.constants) == 1
    assert (
        first.constants[0][1].sha256 == hashlib.sha256(original.tobytes()).hexdigest()
    )
    assert (
        second.constants[0][1].sha256 == hashlib.sha256(changed.tobytes()).hexdigest()
    )
    assert first.constants[0][1].byte_count == second.constants[0][1].byte_count == 8
    assert first != second
    _assert_only_immutable_contracts(first)


def test_frozen_constant_storage_survives_mutation_of_native_captured_array() -> None:
    captured = np.array((1, 2), dtype=np.float32)
    frozen = _body(lambda x: jax.lax.add(x, captured), _spec((2,)))
    assert frozen.gaps == () and len(frozen.constants) == 1
    original_digest = hashlib.sha256(captured.tobytes()).hexdigest()
    original_hash = hash(frozen)
    captured[0] = 123
    assert frozen.constants[0][1].sha256 == original_digest
    assert hash(frozen) == original_hash
    assert frozen != _body(lambda x: jax.lax.add(x, captured.copy()), _spec((2,)))
    _assert_only_immutable_contracts(frozen)


def test_same_typed_endpoints_do_not_erase_intermediate_rounding_casts() -> None:
    def rounded(x: Any) -> Any:
        return jax.lax.convert_element_type(
            jax.lax.convert_element_type(x, np.float32), np.float64
        )

    casted, identity = _body(rounded, _spec(dtype=np.float64)), _body(
        lambda x: x, _spec(dtype=np.float64)
    )
    assert casted.gaps == identity.gaps == ()
    assert casted.inputs == identity.inputs
    assert tuple(value.tensor for value in casted.outputs) == tuple(
        value.tensor for value in identity.outputs
    )
    assert casted.expected_equations == 2 and identity.expected_equations == 0
    assert [equation.outputs[0].logical_dtype for equation in casted.equations] == [
        "float32",
        "float64",
    ]
    assert all(
        equation.semantics[:2] == ("direct", "convert_element_type")
        for equation in casted.equations
    )
    assert casted != identity


def test_same_typed_multiresult_ports_keep_their_original_order_even_when_unknown() -> (
    None
):
    def ordered(x: Any, y: Any) -> Any:
        return jax.lax.sort((x, y), dimension=0, num_keys=1)

    def reversed_outputs(x: Any, y: Any) -> Any:
        first, second = jax.lax.sort((x, y), dimension=0, num_keys=1)
        return second, first

    before = _body(ordered, _spec((3,)), _spec((3,)))
    after = _body(reversed_outputs, _spec((3,)), _spec((3,)))
    assert before.expected_equations == after.expected_equations == 1
    assert before.gaps and after.gaps
    assert [value.port for value in before.outputs] == [0, 1]
    assert [value.port for value in after.outputs] == [1, 0]
    assert tuple(value.tensor for value in before.outputs) == tuple(
        value.tensor for value in after.outputs
    )
    assert before != after
    _assert_only_immutable_contracts(before)


def test_input_permutation_matches_independently_traced_port_order_and_is_immutable() -> (
    None
):
    source = _body(lambda x, y: (jax.lax.sub(x, y), x), _spec(), _spec())
    reordered = _body(lambda y, x: (jax.lax.sub(x, y), x), _spec(), _spec())
    before_hash = hash(source)
    assert source.permuted_inputs((1, 0)) == reordered
    assert source.permuted_inputs((0, 1)) == source
    assert source != reordered and hash(source) == before_hash


@pytest.mark.parametrize(
    "order", ((), (0,), (0, 0), (0, 2), (0, 1, 2), (True, False), (0.0, 1.0), [0, 1])
)
def test_invalid_port_permutations_never_coerce_equal_values(order: Any) -> None:
    source = _body(jax.lax.sub, _spec(), _spec())
    with pytest.raises(StructureUnavailable):
        source.permuted_inputs(order)


def test_counterfeit_known_primitive_name_has_a_gap_not_a_borrowed_semantic_key() -> (
    None
):
    counterfeit = Primitive("add")
    calls = []
    counterfeit.def_impl(lambda x, y: calls.append((x, y)))
    counterfeit.def_abstract_eval(lambda x, y: x)
    native = _closed(lambda x, y: counterfeit.bind(x, y), _spec(), _spec())
    frozen = regions.freeze_closed_body(native)
    assert frozen.expected_equations == len(native.jaxpr.eqns) == 1
    assert frozen.equations[0].semantics is None
    assert frozen.gaps and calls == []
    _assert_only_immutable_contracts(frozen)


def test_effectful_real_trace_preserves_denominator_without_running_callback() -> None:
    calls = []

    def effectful(x: Any) -> Any:
        jax.debug.callback(lambda value: calls.append(value), x, ordered=True)
        return jax.lax.neg(x)

    native = _closed(effectful, _spec())
    assert native.jaxpr.effects and calls == []
    frozen = regions.freeze_closed_body(native)
    assert frozen.expected_equations == len(native.jaxpr.eqns) >= 2
    assert frozen.gaps and "source-effects-contract-unavailable" in frozen.gaps
    assert calls == []
    _assert_only_immutable_contracts(frozen)


def test_symbolic_scope_is_an_explicit_gap_and_never_a_retained_native_authority() -> (
    None
):
    symbolic = jax_export.symbolic_shape("B, 3")
    native = _closed(jax.lax.neg, jax.ShapeDtypeStruct(symbolic, np.float32))
    frozen = regions.freeze_closed_body(native)
    assert frozen.expected_equations == len(native.jaxpr.eqns) == 1
    assert frozen.gaps and any("symbol" in gap for gap in frozen.gaps)
    _assert_only_immutable_contracts(frozen)


def test_constant_observation_error_keeps_available_constant_and_equation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = np.array((1, 2), dtype=np.float32)
    native = _closed(lambda x: jax.lax.add(x, captured), _spec((2,)))
    assert len(native.consts) == 1

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("storage representation unavailable")

    monkeypatch.setattr(regions, "representation_for_aval", unavailable)
    frozen = regions.freeze_closed_body(native)
    assert frozen.expected_equations == len(native.jaxpr.eqns) == 1
    assert len(frozen.constants) == 1 and frozen.constants[0][1] is None
    assert frozen.gaps
    _assert_only_immutable_contracts(frozen)


def test_nested_real_if_keeps_branch_denominators_and_explicit_attachment_gap() -> None:
    def branching(flag: Any, x: Any) -> Any:
        return jax.lax.cond(
            flag,
            lambda value: jax.lax.add(value, value),
            lambda value: jax.lax.sub(value, value),
            x,
        )

    native = _closed(branching, _spec((), np.bool_), _spec())
    conditional = next(
        equation
        for equation in native.jaxpr.eqns
        if equation.primitive is jax.lax.cond_p
    )
    expected = len(native.jaxpr.eqns) + sum(
        len(branch.jaxpr.eqns) for branch in conditional.params["branches"]
    )
    frozen = regions.freeze_closed_body(native)
    equation = next(equation for equation in frozen.equations if equation.children)
    assert tuple(role for role, _ in equation.children) == ("false", "true")
    assert frozen.expected_equations == expected
    assert "if-attachment-contract-unavailable" in frozen.gaps
    _assert_only_immutable_contracts(frozen)


@pytest.mark.parametrize("failure", ("semantic", "body-discovery"))
def test_real_existing_trace_failures_keep_available_equations_and_never_retrace(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _closed(lambda x, y: jax.lax.mul(jax.lax.sub(x, y), x), _spec(), _spec())
    expected = len(native.jaxpr.eqns)
    calls = []

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        calls.append(True)
        raise ValueError("source proof unavailable")

    if failure == "semantic":
        monkeypatch.setattr(semantics, "freeze_source_semantic_key", unavailable)
    else:
        monkeypatch.setattr(functions, "original_function_body", unavailable)
    monkeypatch.setattr(
        jax,
        "make_jaxpr",
        lambda *args, **kwargs: pytest.fail("freezer retraced the source"),
    )
    monkeypatch.setattr(
        jax,
        "eval_shape",
        lambda *args, **kwargs: pytest.fail("freezer evaluated a fallback source"),
    )
    frozen = regions.freeze_closed_body(native)
    assert frozen.expected_equations == expected == 2
    assert frozen.gaps and calls
    assert all(equation.semantics is None for equation in frozen.equations)
    _assert_only_immutable_contracts(frozen)


@pytest.mark.parametrize("kind", ("jit", "scan"))
def test_unhandled_real_native_child_regions_never_report_outer_prefix_as_total(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if kind == "jit":
        compiled = jax.jit(lambda value: jax.lax.mul(value, value), inline=False)
        native = _closed(lambda x: compiled(x), _spec())
    else:

        def scanned(carry: Any, values: Any) -> Any:
            def step(state: Any, value: Any) -> Any:
                return jax.lax.add(state, value), jax.lax.neg(value)

            return jax.lax.scan(step, carry, values)

        native = _closed(scanned, _spec((3,)), _spec((2, 3)))
    outer = tuple(native.jaxpr.eqns)
    children = [
        value
        for equation in outer
        for value in equation.params.values()
        if type(value) in (Jaxpr, ClosedJaxpr)
    ]
    assert children and any(getattr(value, "jaxpr", value).eqns for value in children)
    monkeypatch.setattr(
        jax,
        "make_jaxpr",
        lambda *args, **kwargs: pytest.fail("freezer retraced an unhandled region"),
    )
    monkeypatch.setattr(
        jax,
        "eval_shape",
        lambda *args, **kwargs: pytest.fail("freezer evaluated an unhandled region"),
    )
    declaration = regions.declare_body(native)
    frozen = regions.freeze_closed_body(native)
    assert len(declaration.equations) == len(frozen.equations) == len(outer)
    assert declaration.expected_equations is frozen.expected_equations is None
    assert "expected-region-total-unknown" in declaration.gaps
    assert "expected-region-total-unknown" in frozen.gaps
    assert len(frozen.inputs) == len(native.jaxpr.invars)
    assert len(frozen.outputs) == len(native.jaxpr.outvars)
    _assert_only_immutable_contracts(declaration)
    _assert_only_immutable_contracts(frozen)


@pytest.mark.parametrize("fault", ("native-cycle", "foreign-dict"))
def test_real_equation_metadata_container_corruption_is_bounded_and_passive(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _closed(jax.lax.neg, _spec())
    attempts = []

    class ForeignDictionary(dict):
        def __iter__(self) -> Any:
            attempts.append("iter")
            raise AssertionError("foreign metadata was traversed")

        def __len__(self) -> Any:
            attempts.append("len")
            raise AssertionError("foreign metadata length was requested")

        def items(self) -> Any:
            attempts.append("items")
            raise AssertionError("foreign metadata items were requested")

        def values(self) -> Any:
            attempts.append("values")
            raise AssertionError("foreign metadata values were requested")

        def get(self, *args: Any) -> Any:
            attempts.append("get")
            raise AssertionError("foreign metadata was queried")

    if fault == "native-cycle":
        metadata: Any = []
        metadata.append(metadata)
    else:
        metadata = ForeignDictionary()
        dict.__setitem__(metadata, "native_body", native.jaxpr)
    # Corrupt a real, already traced equation, retaining its actual SSA/tensors.
    # Decline unknown extent without evaluating a provider or treating malformed
    # metadata as evidence that no native child region exists.
    monkeypatch.setitem(
        native.jaxpr.eqns[0].params, "unreviewed_region_metadata", metadata
    )
    declaration = regions.declare_body(native)
    frozen = regions.freeze_closed_body(native)
    assert len(declaration.equations) == len(frozen.equations) == 1
    assert declaration.expected_equations is frozen.expected_equations is None
    assert "expected-region-total-unknown" in frozen.gaps
    assert attempts == []
    _assert_only_immutable_contracts(declaration)
    _assert_only_immutable_contracts(frozen)
