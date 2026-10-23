# tests/extra_tests/converter/test_source_complex_loop_precision.py

"""Finite diagnostic audits of complex storage across existing loop scopes.

Only passthrough, swaps, real/imaginary projection and reconstruction are admitted
for complex values. These tests do not admit complex arithmetic or expand the
product profile. References run before exporter patches are activated.
"""

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, user_interface
from jax2onnx._compat.jax import scan_arity
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.precision import (
    CheckStatus,
    SOURCE_DIAGNOSTIC_PROFILE,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)


@pytest.fixture(autouse=True)
def _restore_jax_settings() -> Iterator[None]:
    previous = (
        bool(jax.config.x64_enabled),
        jax.config.jax_platforms,
        jax.config.jax_default_matmul_precision,
    )
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", None)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous[0])
        jax.config.update("jax_platforms", previous[1])
        jax.config.update("jax_default_matmul_precision", previous[2])


def _role(var: Any) -> tuple[str, tuple[int, ...], bool]:
    aval = var.aval
    dtype = np.dtype(aval.dtype).name
    shape = tuple(aval.shape)
    assert dtype in {
        "complex64",
        "complex128",
        "float32",
        "float64",
        "int32",
        "int64",
        "bool",
    }
    assert all(isinstance(dim, (int, np.integer)) and dim >= 0 for dim in shape)
    return dtype, shape, bool(aval.weak_type)


def _roles(variables: Any) -> list[tuple[str, tuple[int, ...]]]:
    return [_role(var)[:2] for var in variables]


def _body(value: Any) -> Any:
    return getattr(value, "jaxpr", value)


class _ComplexLoopAudit:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, jaxpr: Any, opset: int) -> None:
        assert opset == 23 and not jaxpr.effects
        for var in (*jaxpr.constvars, *jaxpr.invars, *jaxpr.outvars):
            _role(var)
        for eqn in jaxpr.eqns:
            name = eqn.primitive.name
            self.seen.append(name)
            ins, outs = _roles(eqn.invars), _roles(eqn.outvars)
            assert not eqn.effects
            if name in {"add", "lt"}:
                assert len(ins) == 2 and len(outs) == 1
                assert ins[0][0] == ins[1][0] and ins[0][0] in {
                    "int32",
                    "int64",
                }, "This finite audit excludes complex arithmetic"
                shape = tuple(np.broadcast_shapes(*(item[1] for item in ins)))
                assert outs == [("bool" if name == "lt" else ins[0][0], shape)]
            elif name == "not":
                assert len(ins) == 1 and ins == outs and ins[0][0] == "bool"
            elif name in {"real", "imag"}:
                assert len(ins) == len(outs) == 1
                base = {"complex64": "float32", "complex128": "float64"}[ins[0][0]]
                assert outs == [(base, ins[0][1])]
            elif name == "complex":
                assert len(ins) == 2 and len(outs) == 1 and ins[0] == ins[1]
                complex_dtype = {"float32": "complex64", "float64": "complex128"}[
                    ins[0][0]
                ]
                assert outs == [(complex_dtype, ins[0][1])]
            elif name == "convert_element_type":
                assert ins == [("bool", ())] and outs == [("int32", ())]
                assert np.dtype(eqn.params["new_dtype"]) == np.dtype(np.int32)
            elif name == "broadcast_in_dim":
                assert len(ins) == len(outs) == 1
                assert ins[0] == ("int32", ()) and outs[0] == ("int32", (2,))
                assert tuple(eqn.params["broadcast_dimensions"]) == ()
            elif name == "scan":
                step = _body(eqn.params["jaxpr"])
                nc, nk, nx = scan_arity(eqn.params, len(step.invars))
                length = eqn.params["length"]
                assert isinstance(length, int) and length >= 0
                assert not eqn.params["reverse"] and nc + nk + nx == len(ins)
                assert _roles(step.invars) == ins[: nc + nk] + [
                    (dtype, shape[1:]) for dtype, shape in ins[nc + nk :]
                ]
                step_out = _roles(step.outvars)
                assert step_out[:nk] == ins[nc : nc + nk]
                assert outs == step_out[:nk] + [
                    (dtype, (length, *shape)) for dtype, shape in step_out[nk:]
                ]
                self(step, opset)
            elif name == "while":
                condition, step = _body(eqn.params["cond_jaxpr"]), _body(
                    eqn.params["body_jaxpr"]
                )
                nc, nb = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
                carry = ins[nc + nb :]
                assert outs == carry == _roles(step.outvars)
                assert _roles(step.invars) == ins[nc : nc + nb] + carry
                assert _roles(condition.invars) == ins[:nc] + carry
                predicate = _roles(condition.outvars)
                assert len(predicate) == 1 and predicate[0][0] == "bool"
                self(condition, opset)
                self(step, opset)
            elif name == "cond":
                assert ins[0] == ("int32", ()) and len(eqn.params["branches"]) == 2
                for branch in eqn.params["branches"]:
                    step = _body(branch)
                    assert (
                        _roles(step.invars) == ins[1:] and _roles(step.outvars) == outs
                    )
                    self(step, opset)
            elif name in {"jit", "pjit"}:
                step = _body(eqn.params.get("jaxpr", eqn.params.get("call_jaxpr")))
                assert _roles(step.invars) == ins and _roles(step.outvars) == outs
                self(step, opset)
            elif name.startswith("onnx_fn::"):
                # The shared callback audits the actual retraced Function body.
                assert outs
            else:
                raise AssertionError(f"Unaudited complex-loop primitive: {name}")


def _signature(jaxpr: Any) -> Any:
    equations = []
    for eqn in jaxpr.eqns:
        nested = []
        for key in ("jaxpr", "call_jaxpr", "cond_jaxpr", "body_jaxpr"):
            if key in eqn.params:
                nested.append((key, _signature(_body(eqn.params[key]))))
        if "branches" in eqn.params:
            nested.extend(
                ("branch", _signature(_body(branch)))
                for branch in eqn.params["branches"]
            )
        equations.append(
            (
                eqn.primitive.name,
                tuple(map(_role, eqn.invars)),
                tuple(map(_role, eqn.outvars)),
                tuple(nested),
            )
        )
    return (
        tuple(map(_role, jaxpr.invars)),
        tuple(map(_role, jaxpr.outvars)),
        tuple(equations),
    )


def _physical(array: np.ndarray) -> np.ndarray:
    """Independent typed-view oracle; the declaration selects complex storage."""
    array = np.asarray(array)
    if array.dtype.kind != "c":
        return array.copy(order="C")
    base = np.float32 if array.dtype == np.dtype(np.complex64) else np.float64
    return np.ascontiguousarray(array).reshape(-1).view(base).reshape((*array.shape, 2))


def _exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


def _payload(dtype: Any, shape: tuple[int, ...], *, other: bool = False) -> np.ndarray:
    if np.dtype(dtype) == np.dtype(np.complex64):
        words = np.asarray(
            [0x80000000, 0, 0x3F800001, 0x80000000, 0x7FC00021, 0x7FC00042], np.uint32
        )
    else:
        words = np.asarray(
            [
                0x8000000000000000,
                0,
                0x3FF0000000400000,
                0x8000000000000000,
                0x7FF8000000000021,
                0x7FF8000000000042,
            ],
            np.uint64,
        )
    values = words.view(dtype)
    if other:
        values = values[::-1].copy()
    return np.resize(values, int(np.prod(shape)) if shape else 1).reshape(shape)


def _export_exact(
    fn: Callable[..., Any],
    values: list[np.ndarray],
    *,
    compare_trace: bool = True,
    observe_where_signed_zero_gap: bool = False,
) -> tuple[onnx.ModelProto, _ComplexLoopAudit]:
    specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
    original = jax.make_jaxpr(fn)(*specs).jaxpr
    references = [np.asarray(value) for value in jax.tree.leaves(fn(*values))]
    compiled = [np.asarray(value) for value in jax.tree.leaves(jax.jit(fn)(*values))]
    for actual, expected in zip(compiled, references, strict=True):
        _exact(actual, expected)
    audit = _ComplexLoopAudit()
    observed = []

    def validate(jaxpr: Any, opset: int) -> None:
        if not observed:
            if compare_trace:
                assert _signature(jaxpr) == _signature(original)
            assert _roles(jaxpr.invars) == _roles(original.invars)
            assert _roles(jaxpr.outvars) == _roles(original.outvars)
        observed.append(jaxpr)
        audit(jaxpr, opset)

    model = user_interface._to_onnx(
        fn,
        specs,
        opset=23,
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=validate,
    )
    assert observed
    proto = onnx.load_model_from_string(model.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == SOURCE_DIAGNOSTIC_PROFILE
    assert (
        next(check.status for check in report.checks if check.category == "lowering")
        is CheckStatus.UNVERIFIED
    )
    expected_outputs = [_physical(value) for value in references]
    for metadata, expected in zip(proto.graph.output, expected_outputs, strict=True):
        tensor = metadata.type.tensor_type
        assert tensor.elem_type == onnx.helper.np_dtype_to_tensor_dtype(expected.dtype)
        assert all(dim.HasField("dim_value") for dim in tensor.shape.dim)
        assert tuple(dim.dim_value for dim in tensor.shape.dim) == expected.shape
    for mode, level in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = level
        session = ort.InferenceSession(
            proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        feed = {
            item.name: _physical(value)
            for item, value in zip(proto.graph.input, values, strict=True)
        }
        bitwise_failure = False
        for actual, expected in zip(
            session.run(None, feed), expected_outputs, strict=True
        ):
            if observe_where_signed_zero_gap:
                assert actual.dtype == expected.dtype and actual.shape == expected.shape
                # The dedicated witness has only exact zeros and integer
                # counters. Changed values cannot pass as a zero-sign gap.
                assert np.array_equal(actual, expected)
                bitwise_failure |= actual.tobytes(order="C") != expected.tobytes(
                    order="C"
                )
            else:
                _exact(actual, expected)
        if observe_where_signed_zero_gap:
            evidence = validate_source_samples(
                proto,
                samples=[
                    SourceSample(feed, expected_outputs, SampleCriteria("bitwise"))
                ],
                optimization=mode,
            )
            assert all(
                check.status is CheckStatus.VALIDATED for check in evidence.checks
            )
            sample = evidence.samples[0]
            assert sample.criteria.mode == "bitwise"
            if bitwise_failure:
                assert evidence.status is CheckStatus.FAILED
                failures = [
                    check
                    for check in sample.checks
                    if check.status is CheckStatus.FAILED
                ]
                assert failures and all(
                    check.category == "numerical-acceptance"
                    and "values fail bitwise comparison" in check.detail
                    for check in failures
                )
                with pytest.raises(SampleAcceptanceError, match="bitwise"):
                    evidence.require_sample_acceptance()
            else:
                # Retain useful evidence if a runtime fixes its Where kernel.
                evidence.require_sample_acceptance()
            assert get_precision_report(proto) == report
    return proto, audit


def _scan_model(length: int, with_xs: bool) -> Callable[..., Any]:
    def model(a, b, real_pair, flag, captured, *sequences):
        def step(state, item):
            left, right, counter, enabled, ordinary = state
            rebuilt = jax.lax.complex(jax.lax.real(left), jax.lax.imag(left))
            next_left = item if with_xs else right
            return (
                next_left,
                rebuilt,
                counter + np.int32(1),
                jnp.logical_not(enabled),
                ordinary,
            ), (
                left,
                right,
                captured,
                jax.lax.real(left),
                jax.lax.imag(right),
                ordinary,
                counter,
                enabled,
            )

        return jax.lax.scan(
            step,
            (a, b, np.int32(0), flag, real_pair),
            sequences[0] if with_xs else None,
            length=length,
        )

    return model


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(), (3,)], ids=["scalar", "vector"])
@pytest.mark.parametrize("length", [0, 1, 3])
@pytest.mark.parametrize("with_xs", [False, True], ids=["no-xs", "typed-xs"])
def test_complex_scan_preserves_mixed_carry_and_scanned_storage(
    dtype: Any, shape: tuple[int, ...], length: int, with_xs: bool
) -> None:
    a, b = _payload(dtype, shape), _payload(dtype, shape, other=True)
    base = np.float32 if dtype == np.complex64 else np.float64
    values = [
        a,
        b,
        np.asarray([[1, -0.0], [3, 4]], dtype=base),
        np.asarray(True),
        _payload(dtype, shape, other=True),
    ]
    if with_xs:
        values.append(np.stack([a, b, a], axis=0)[:length])
    _, audit = _export_exact(_scan_model(length, with_xs), values)
    assert "scan" in audit.seen and "complex" in audit.seen


def _while_model(a, b, real_pair, enabled, captured, limit):
    def body(state):
        left, right, counter, flag, ordinary = state
        rebuilt = jax.lax.complex(jax.lax.real(right), jax.lax.imag(right))
        return rebuilt, captured, counter + np.int32(1), jnp.logical_not(flag), ordinary

    return jax.lax.while_loop(
        lambda state: state[2] < limit, body, (a, b, np.int32(0), enabled, real_pair)
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(), (3,)], ids=["scalar", "vector"])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_complex_while_preserves_mixed_carries_and_capture(
    dtype: Any, shape: tuple[int, ...], length: int
) -> None:
    base = np.float32 if dtype == np.complex64 else np.float64
    _, audit = _export_exact(
        _while_model,
        [
            _payload(dtype, shape),
            _payload(dtype, shape, other=True),
            np.asarray([[1, -0.0], [3, 4]], dtype=base),
            np.asarray(True),
            _payload(dtype, shape, other=True),
            np.asarray(length, np.int32),
        ],
    )
    assert "while" in audit.seen and "complex" in audit.seen


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", [(), (3,)], ids=["scalar", "vector"])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_native_fori_preserves_complex_swaps(
    dtype: Any, shape: tuple[int, ...], length: int
) -> None:
    def model(a, b):
        return jax.lax.fori_loop(
            0, length, lambda index, state: (state[1], state[0]), (a, b)
        )

    _export_exact(model, [_payload(dtype, shape), _payload(dtype, shape, other=True)])


@onnx_function
def _named_complex_pair(left, right):
    return jax.lax.complex(jax.lax.real(left), jax.lax.imag(left)), right


@onnx_function
def _named_complex_scan(left, right, captured, selector, *, length):
    def step(state, _):
        result = jax.lax.cond(
            selector,
            lambda values: _named_complex_pair(values[1], captured),
            lambda values: _named_complex_pair(values[0], values[1]),
            state,
        )
        return result, result

    return jax.lax.scan(step, (left, right), None, length=length)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("length", [0, 1, 3])
@pytest.mark.parametrize("selector", [False, True])
def test_complex_scan_nested_functions_and_if(
    dtype: Any, length: int, selector: bool
) -> None:
    def model(left, right, captured, choose):
        return _named_complex_scan(left, right, captured, choose, length=length)

    model_proto, audit = _export_exact(
        model,
        [
            _payload(dtype, (3,)),
            _payload(dtype, (3,), other=True),
            _payload(dtype, (3,), other=True),
            np.asarray(selector),
        ],
        compare_trace=False,
    )
    assert {"scan", "cond", "complex"} <= set(audit.seen)
    assert len(model_proto.functions) >= 2


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_complex_nested_scans_keep_empty_channel_axis(dtype: Any, length: int) -> None:
    def model(left, right):
        def step(state, _):
            carry, scanned = jax.lax.scan(
                lambda values, unused: ((values[1], values[0]), values),
                state,
                None,
                length=length,
            )
            return carry, scanned

        return jax.lax.scan(step, (left, right), None, length=3)

    _, audit = _export_exact(
        model, [_payload(dtype, (3,)), _payload(dtype, (3,), other=True)]
    )
    assert audit.seen.count("scan") >= 2


@pytest.mark.parametrize("length", [0, 1, 3])
def test_complex_scan_keeps_distinct_base_types_in_output_order(length: int) -> None:
    def model(narrow, wide, ordinary, flag):
        def step(state, _):
            left, right, counter, enabled = state
            return (
                left,
                right,
                counter + np.int32(1),
                jnp.logical_not(enabled),
            ), (right, ordinary, left, counter, enabled)

        return jax.lax.scan(
            step, (narrow, wide, np.int32(0), flag), None, length=length
        )

    _export_exact(
        model,
        [
            _payload(np.complex64, (3,)),
            _payload(np.complex128, (3,), other=True),
            np.asarray([[1, -0.0], [3, 4]], np.float32),
            np.asarray(True),
        ],
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("with_xs", [False, True], ids=["no-xs", "typed-xs"])
@pytest.mark.parametrize("length,width", [(3, 2), (1, 3)])
def test_scan_length_is_distinct_from_complex_vector_width(
    dtype: Any, with_xs: bool, length: int, width: int
) -> None:
    def model(initial, sequence):
        carry, scanned = jax.lax.scan(
            lambda carry, item: (carry, carry if item is None else item),
            initial,
            sequence if with_xs else None,
            length=length,
        )
        return carry, scanned, jax.lax.real(scanned), jax.lax.imag(scanned)

    value = _payload(dtype, (width,))
    _export_exact(model, [value, np.stack([value] * length)])


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("limits", [(0, 1), (1, 3)])
def test_vmapped_complex_while_masks_both_physical_channels(dtype: Any, limits) -> None:
    # Nonzero finite values isolate the physical channel mask and carry order.
    values = np.asarray(
        [[1 + 2j, 3 + 4j, 5 + 6j], [7 + 8j, 9 + 10j, 11 + 12j]], dtype=dtype
    )
    _export_exact(
        _vmapped_complex_while,
        [
            values,
            values[::-1, ::-1].copy(),
            np.asarray(limits, np.int32),
        ],
    )


def _vmapped_complex_while(left, right, limits):
    def one(a, b, limit):
        return jax.lax.while_loop(
            lambda state: state[2] < limit,
            lambda state: (state[1], state[0], state[2] + np.int32(1)),
            (a, b, np.int32(0)),
        )

    return jax.vmap(one)(left, right, limits)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_vmapped_while_where_signed_zero_has_real_bitwise_sample_evidence(
    dtype: Any,
) -> None:
    base = np.float32 if dtype == np.complex64 else np.float64
    value = np.full((2, 3, 2), -0.0, dtype=base).view(dtype).reshape(2, 3)
    _export_exact(
        _vmapped_complex_while,
        [value, value.copy(), np.asarray([0, 1], np.int32)],
        observe_where_signed_zero_gap=True,
    )


@onnx_function
def _named_complex_while(left, right, captured, limit, selector):
    def run(values):
        a, b = values

        def step(state):
            x, y, counter = state
            pair = _named_complex_pair(y, captured)
            return *pair, counter + np.int32(1)

        return jax.lax.while_loop(
            lambda state: state[2] < limit, step, (a, b, np.int32(0))
        )

    return jax.lax.cond(
        selector, run, lambda values: (*values, np.int32(0)), (left, right)
    )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("length", [0, 1, 3])
@pytest.mark.parametrize("selector", [False, True])
def test_complex_while_in_function_and_if_with_nested_function(
    dtype: Any, length: int, selector: bool
) -> None:
    def model(left, right, captured, limit, choose):
        return _named_complex_while(left, right, captured, limit, choose)

    proto, audit = _export_exact(
        model,
        [
            _payload(dtype, (3,)),
            _payload(dtype, (3,), other=True),
            _payload(dtype, (3,), other=True),
            np.asarray(length, np.int32),
            np.asarray(selector),
        ],
        compare_trace=False,
    )
    assert {"while", "cond", "complex"} <= set(audit.seen)
    assert len(proto.functions) >= 2


def _stored_tensors(message: Any) -> Iterator[onnx.TensorProto]:
    if isinstance(message, onnx.TensorProto):
        yield message
    for descriptor, value in message.ListFields():
        if descriptor.message_type is None:
            continue
        if descriptor.is_repeated:
            for child in value:
                yield from _stored_tensors(child)
        else:
            yield from _stored_tensors(value)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("kind", ["scan-no-xs", "scan-xs", "while", "fori"])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_complex_loop_constant_capture_serializes_exact_packed_payload(
    dtype: Any, kind: str, length: int
) -> None:
    captured = _payload(dtype, (3,), other=True)

    def model(initial):
        if kind.startswith("scan"):

            def step(carry, item):
                return captured, captured

            xs = (
                jnp.asarray(np.arange(length, dtype=np.int32))
                if kind == "scan-xs"
                else None
            )
            return jax.lax.scan(step, initial, xs, length=length)
        if kind == "fori":
            return jax.lax.fori_loop(0, length, lambda index, carry: captured, initial)
        return jax.lax.while_loop(
            lambda state: state[1] < np.int32(length),
            lambda state: (captured, state[1] + np.int32(1)),
            (initial, np.int32(0)),
        )

    proto, _ = _export_exact(model, [_payload(dtype, (3,))])
    tensors = list(_stored_tensors(proto))
    assert all(
        tensor.data_type
        not in (onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128)
        for tensor in tensors
    )
    if length:
        expected = _physical(captured)
        stored = [onnx.numpy_helper.to_array(tensor) for tensor in tensors]
        assert any(
            array.dtype == expected.dtype
            and array.shape == expected.shape
            and array.tobytes(order="C") == expected.tobytes(order="C")
            for array in stored
        )


def test_complex128_x64_off_rejects_before_model_call() -> None:
    jax.config.update("jax_enable_x64", False)
    calls = []

    def model(value):
        calls.append(value)
        return jax.lax.scan(lambda carry, _: (carry, carry), value, None, length=1)

    with pytest.raises(SourcePrecisionError, match="canonicalize") as error:
        user_interface._to_onnx(
            model,
            [jax.ShapeDtypeStruct((3,), np.complex128)],
            opset=23,
            precision_policy=PrecisionPolicy.SOURCE,
            diagnostic_jaxpr_validator=_ComplexLoopAudit(),
        )
    assert error.value.category == "configuration-conflict"
    assert not calls and not jax.config.x64_enabled


def test_complex_arithmetic_is_outside_this_loop_audit() -> None:
    def model(value):
        return jax.lax.scan(
            lambda carry, _: (carry + carry, carry), value, None, length=1
        )

    with pytest.raises(AssertionError, match="excludes complex arithmetic"):
        _export_exact(model, [np.asarray([1 + 2j], np.complex64)])
