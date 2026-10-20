# tests/extra_tests/converter/test_source_loop_admission.py

"""Existing native loop contracts through ordinary private SOURCE admission."""

from copy import copy
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat.jax import ClosedJaxpr, ShapedArray, Var, scan_arity
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from jax2onnx.plugins.jax.numpy.zeros import JnpZerosPlugin
from tests.extra_tests.converter import (
    test_source_complex_loop_precision as complex_cases,
)
from tests.extra_tests.converter import test_source_loop_precision as real_cases
from tests.extra_tests.converter import (
    test_onnx_function_loop_call_params as flag_cases,
)
from tests.extra_tests.converter.source_precision_support import (
    SourceProvenance,
    raw_source_case,
    raw_source_scenarios,
    run_source_case,
)
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _globals,
    _settings,
    _restore_state as _restore_state,
)

_BITWISE = SampleCriteria("bitwise")


def _loop_interfaces(proto: onnx.ModelProto) -> None:
    # Inspect actual top-level and Function-owned Loop bodies.
    for nodes in (proto.graph.node, *(function.node for function in proto.functions)):
        for node in real_cases._nodes(nodes):
            if node.op_type != "Loop":
                continue
            body = next(attr.g for attr in node.attribute if attr.name == "body")
            assert len(body.input) == len(node.input)
            assert len(body.output) == len(node.output) + 1
            available = {value.name for value in body.input}
            available.update(value.name for value in body.initializer)
            for child in body.node:
                assert all(not name or name in available for name in child.input)
                available.update(child.output)


def _product(fn: Any, values: list[np.ndarray], *, opset: int = 23) -> onnx.ModelProto:
    """Keep native eager/JIT witnesses and exact existing finite sample criteria."""
    eager = [np.asarray(v) for v in jax.tree.leaves(fn(*values))]
    compiled = [np.asarray(v) for v in jax.tree.leaves(jax.jit(fn)(*values))]
    for actual, expected in zip(compiled, eager, strict=True):
        complex_cases._exact(actual, expected)
    before = _globals(), _settings()
    proto = ui._to_onnx(
        fn,
        [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values],
        precision_policy=PrecisionPolicy.SOURCE,
        opset=opset,
    )
    assert (_globals(), _settings()) == before
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    proto = onnx.load_model_from_string(proto.SerializeToString())
    onnx.checker.check_model(proto, full_check=True)
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-static-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    _loop_interfaces(proto)
    feeds = {
        port.name: complex_cases._physical(value)
        for port, value in zip(proto.graph.input, values, strict=True)
    }
    expected = [complex_cases._physical(v) for v in eager]
    for optimization in ("disabled", "all"):
        evidence = validate_source_samples(
            proto,
            samples=[SourceSample(feeds, expected, _BITWISE)],
            optimization=optimization,
        )
        evidence.require_sample_acceptance()
    assert get_precision_report(proto) == report
    return proto


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize(
    "fn,values",
    [
        (real_cases._while_literal, [np.asarray(0, np.float32)]),
        (real_cases._while_mixed, [np.asarray(2, np.float32)]),
        (real_cases._fori_bool, [np.asarray(0, np.float32)]),
        (real_cases._fori_mixed, [np.asarray(2, np.float32)]),
        (
            real_cases._fori_dynamic,
            [
                np.asarray(2, np.float32),
                np.asarray(2, np.int32),
                np.asarray(5, np.int32),
            ],
        ),
        (real_cases._scan_i64, [np.asarray([1, 2, 3], np.float32)]),
        (real_cases._scan_captured, [np.asarray([1, 2, 3], np.float32)]),
    ],
    ids=[
        "literal-while-i32",
        "mixed-while",
        "fori-bool",
        "mixed-fori",
        "dynamic-fori-index",
        "scan-i64",
        "scan-captured-f64",
    ],
)
def test_existing_real_loops_use_product_admission(
    fn: Any, values: Any, ambient_x64: bool
) -> None:
    with jax.enable_x64(ambient_x64):
        _product(fn, values)


@pytest.mark.parametrize("length", [0, 1, 3])
@pytest.mark.parametrize("opset", [23, 26])
def test_static_no_xs_scan_preserves_native_empty_and_mixed_outputs(
    length: int, opset: int
) -> None:
    def target(x: Any) -> Any:
        return jax.lax.scan(
            lambda half, _: (-half, (half.astype(jnp.float32), half > np.float16(0))),
            x,
            None,
            length=length,
        )

    _product(target, [np.asarray([-0.0, 2], np.float16)], opset=opset)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("length", [0, 3])
@pytest.mark.parametrize("with_xs", [False, True])
def test_existing_complex_scan_preserves_logical_and_physical_roles(
    dtype: Any, length: int, with_xs: bool
) -> None:
    with jax.enable_x64(True):
        shape = (3,)
        a = complex_cases._payload(dtype, shape)
        b = complex_cases._payload(dtype, shape, other=True)
        base = np.float32 if dtype == np.complex64 else np.float64
        values = [
            a,
            b,
            np.asarray([[1, -0.0], [3, 4]], dtype=base),
            np.asarray(True),
            b,
        ]
        if with_xs:
            values.append(np.stack([a, b, a], axis=0)[:length])
        _product(complex_cases._scan_model(length, with_xs), values)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("length", [0, 3])
def test_existing_complex_while_preserves_captures_and_carries(
    dtype: Any, length: int
) -> None:
    with jax.enable_x64(True):
        base = np.float32 if dtype == np.complex64 else np.float64
        _product(
            complex_cases._while_model,
            [
                complex_cases._payload(dtype, (3,)),
                complex_cases._payload(dtype, (3,), other=True),
                np.asarray([[1, -0.0], [3, 4]], dtype=base),
                np.asarray(True),
                complex_cases._payload(dtype, (3,), other=True),
                np.asarray(length, np.int32),
            ],
        )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("limits", [(0, 1), (1, 3)])
def test_existing_batched_complex_while_masks_both_channels(
    dtype: Any, limits: Any
) -> None:
    with jax.enable_x64(True):
        # Original nonzero finite masking witness; signed-zero failure remains separate.
        values = np.asarray(
            [[1 + 2j, 3 + 4j, 5 + 6j], [7 + 8j, 9 + 10j, 11 + 12j]], dtype=dtype
        )
        _product(
            complex_cases._vmapped_complex_while,
            [values, values[::-1, ::-1].copy(), np.asarray(limits, np.int32)],
        )


@pytest.mark.parametrize("scope", ["shared", "nested", "function"])
def test_existing_zero_trip_and_named_scans_use_product_leaf_checks(scope: str) -> None:
    def target(value: Any) -> Any:
        if scope == "function":
            return real_cases._named_empty_scan(value)
        if scope == "nested":

            def step(state: Any, _: Any) -> Any:
                _, empty = jax.lax.scan(
                    lambda x, unused: (-x, x), state, None, length=0
                )
                return -state, empty

            return jax.lax.scan(step, value, None, length=2)
        carry, empty = jax.lax.scan(
            lambda state, _: (-state, state), value, None, length=0
        )
        return carry, {"empty": empty, "again": (empty,)}

    _product(target, [np.asarray([-0.0, 2], np.float32)])


@pytest.mark.parametrize(
    "origin,ordinal",
    [
        ("plugin::scan", 3),
        ("plugin::scan", 5),
        ("plugin::scan", 6),
        ("plugin::scan", 23),
        ("plugin::while", 20),
    ],
)
def test_original_metadata_contracts_reach_normal_source_unchanged(
    origin: str, ordinal: int
) -> None:
    metadata = raw_source_case(origin, ordinal)
    for scenario in raw_source_scenarios(metadata, SourceProvenance(origin, ordinal)):
        assert scenario.criteria is None
        result = run_source_case(metadata, scenario, optimization=None)
        assert result.model is not None, result.audit.detail
        assert result.audit.status is result.schema is CheckStatus.VALIDATED
        report = get_precision_report(result.model)
        assert report is not None and report.profile == "source-static-v1"
        assert metadata["post_check_onnx_graph"](result.model) is True
        if origin == "plugin::scan" and ordinal == 6:
            assert metadata["testcase"] == "scan_unroll_reuses_loop"
        if origin == "plugin::while":
            assert metadata["opset_version"] == 18


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(), (2,), (0, 2)])
@pytest.mark.parametrize("opset", [23, 26])
def test_native_zero_fill_preserves_root_and_scan_body_storage(
    dtype: Any, shape: Any, opset: int
) -> None:
    def target() -> Any:
        return jnp.zeros(shape, dtype=dtype), jax.lax.scan(
            lambda state, _: (state, jnp.zeros(shape, dtype=dtype)),
            np.int32(0),
            None,
            length=2,
        )

    with jax.enable_x64(True):
        _product(target, [], opset=opset)


@pytest.mark.parametrize(
    "mutation",
    [
        "foreign-primitive",
        "input-arity",
        "output-arity",
        "extra-parameter",
        "placement",
        "missing-shape",
        "missing-dtype",
        "shape-list",
        "shape-mismatch",
        "boolean-dimension",
        "negative-dimension",
        "dtype-string",
        "dtype-mismatch",
        "output-weakness",
    ],
)
def test_malformed_zero_fill_carrier_fails_before_lowering(mutation: str) -> None:
    graph = jax.make_jaxpr(
        lambda: JnpZerosPlugin._PRIM.bind(shape=(2,), dtype=np.dtype(np.float32))
    )().jaxpr
    (equation,) = graph.eqns
    assert equation.primitive is JnpZerosPlugin._PRIM
    primitive, params = equation.primitive, dict(equation.params)
    inputs, outputs = list(equation.invars), list(equation.outvars)
    if mutation == "foreign-primitive":
        primitive = SimpleNamespace(name="jax.numpy.zeros")
    elif mutation == "input-arity":
        inputs.append(Var(ShapedArray((), np.float32)))
    elif mutation == "output-arity":
        outputs.append(Var(ShapedArray((2,), np.float32)))
    elif mutation == "extra-parameter":
        params["unreviewed"] = None
    elif mutation == "placement":
        params["out_sharding"] = None
    elif mutation.startswith("missing-"):
        params.pop(mutation.removeprefix("missing-"))
    elif mutation == "shape-list":
        params["shape"] = [2]
    elif mutation == "shape-mismatch":
        params["shape"] = (3,)
    elif mutation == "boolean-dimension":
        params["shape"] = (True,)
    elif mutation == "negative-dimension":
        params["shape"] = (-1,)
    elif mutation == "dtype-string":
        params["dtype"] = "float32"
    elif mutation == "dtype-mismatch":
        params["dtype"] = np.dtype(np.float64)
    else:
        outputs[0] = Var(ShapedArray((2,), np.float32, weak_type=True))
    corrupted = SimpleNamespace(
        primitive=primitive, invars=inputs, outvars=outputs, params=params, effects=()
    )
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(
            SimpleNamespace(
                invars=inputs,
                constvars=(),
                outvars=outputs,
                eqns=[corrupted],
                effects=(),
            ),
            23,
        )
    assert error.value.category == "lowering-defect"
    assert error.value.primitive == "jax.numpy.zeros"


@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.int32, np.complex64])
def test_zero_fill_admission_keeps_other_dtype_routes_excluded(dtype: Any) -> None:
    with pytest.raises(SourcePrecisionError) as error:
        ui._to_onnx(
            lambda: JnpZerosPlugin._PRIM.bind(shape=(2,), dtype=np.dtype(dtype)),
            [],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert error.value.category == "profile-exclusion"
    assert error.value.primitive == "jax.numpy.zeros"


@pytest.mark.parametrize("opset", [16, 18, 21, 24, 27])
def test_zero_fill_admission_keeps_other_opset_routes_excluded(opset: int) -> None:
    with pytest.raises(SourcePrecisionError) as error:
        ui._to_onnx(
            lambda: jnp.zeros((2,), dtype=np.float32),
            [],
            opset=opset,
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert error.value.category == "profile-exclusion"
    assert error.value.primitive == "jax.numpy.zeros"


@pytest.mark.parametrize("component", ["ones", "full"])
def test_zero_fill_admission_keeps_other_creation_wrappers_excluded(
    component: str,
) -> None:
    def target() -> Any:
        if component == "ones":
            return jnp.ones((2,), dtype=np.float32)
        return jnp.full((2,), np.float32(0))

    with pytest.raises(SourcePrecisionError) as error:
        ui._to_onnx(target, [], precision_policy=PrecisionPolicy.SOURCE)
    assert error.value.category == "profile-exclusion"
    assert error.value.primitive == f"jax.numpy.{component}"


@pytest.mark.parametrize(
    "target,inputs",
    [
        (flag_cases._scan_flag, {"x": flag_cases._X}),
        (
            flag_cases._scan_flag_with_inputs,
            {"x": flag_cases._X, "xs": np.ones((2, 2), np.float32)},
        ),
        (
            flag_cases._scan_flag_in_if,
            {"x": flag_cases._X, "predicate": np.asarray(True)},
        ),
        (
            flag_cases._scan_flag_in_if,
            {"x": flag_cases._X, "predicate": np.asarray(False)},
        ),
        (
            flag_cases._while_flag,
            {"x": flag_cases._X, "limit": np.asarray(2, np.int32)},
        ),
        (
            flag_cases._while_flag,
            {"x": flag_cases._X, "limit": np.asarray(0, np.int32)},
        ),
        (flag_cases._nested_loop_flag, {"x": flag_cases._X}),
    ],
    ids=[
        "scan",
        "scan-xs",
        "scan-if-true",
        "scan-if-false",
        "while",
        "while-zero",
        "nested-loops",
    ],
)
def test_original_named_flags_remain_dynamic_on_the_same_source_artifact(
    target: Any, inputs: Any
) -> None:
    references = {
        flag: [
            np.asarray(v)
            for v in jax.tree.leaves(target(*inputs.values(), deterministic=flag))
        ]
        for flag in (True, False)
    }
    proto = ui._to_onnx(
        target,
        [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in inputs.values()],
        input_names=list(inputs),
        input_params={"deterministic": True},
        precision_policy=PrecisionPolicy.SOURCE,
    )
    onnx.checker.check_model(proto, full_check=True)
    _loop_interfaces(proto)
    assert proto.functions and "deterministic" in {v.name for v in proto.graph.input}
    report = get_precision_report(proto)
    assert report is not None and report.profile == "source-static-v1"
    for optimization in ("disabled", "all"):
        result = validate_source_samples(
            proto,
            samples=[
                SourceSample(
                    {**inputs, "deterministic": np.asarray(flag)},
                    references[flag],
                    _BITWISE,
                )
                for flag in (True, False)
            ],
            optimization=optimization,
        )
        result.require_sample_acceptance()


@pytest.mark.parametrize("kind", ["scan", "while"])
@pytest.mark.parametrize("ambient_x64", [False, True])
def test_native_weak_carry_strengthening_is_preserved(
    kind: str, ambient_x64: bool
) -> None:
    def target(x: Any) -> Any:
        if kind == "scan":
            return jax.lax.scan(lambda n, _: (n + np.int32(1), n), 0, None, length=2), x
        return (
            jax.lax.while_loop(lambda n: n < np.int32(2), lambda n: n + np.int32(1), 0),
            x,
        )

    with jax.enable_x64(ambient_x64):
        _product(target, [np.asarray(1, np.float32)])


def _trace_loop(kind: str) -> Any:
    captured = np.asarray([0.25, 0.5, 0.75], np.float32)
    if kind == "scan":

        def target(x: Any, xs: Any) -> Any:
            return jax.lax.scan(lambda c, y: (c + captured + y, c), x, xs)

        return jax.make_jaxpr(target)(
            jax.ShapeDtypeStruct((3,), np.float32),
            jax.ShapeDtypeStruct((2, 3), np.float32),
        ).jaxpr

    def target(x: Any, limit: Any) -> Any:
        return jax.lax.while_loop(
            lambda s: s[0] < limit,
            lambda s: (s[0] + np.int32(1), s[1] + captured),
            (np.int32(0), x),
        )

    return jax.make_jaxpr(target)(
        jax.ShapeDtypeStruct((3,), np.float32), jax.ShapeDtypeStruct((), np.int32)
    ).jaxpr


@pytest.mark.parametrize("kind", ["scan", "while"])
@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "negative-count",
        "boolean-count",
        "string-count",
        "output-weakness",
        "output-dtype",
        "output-shape",
    ],
)
def test_malformed_loop_contracts_fail_before_lowering(
    kind: str, mutation: str
) -> None:
    graph = _trace_loop(kind)
    equation = next(e for e in graph.eqns if e.primitive.name == kind)
    params = dict(equation.params)
    outputs = list(equation.outvars)
    if mutation == "extra":
        params["unreviewed"] = None
    elif mutation == "missing":
        params.pop("jaxpr" if kind == "scan" else "body_jaxpr")
    elif mutation.endswith("count"):
        value = {"negative-count": -1, "boolean-count": True, "string-count": "1"}[
            mutation
        ]
        if kind == "scan" and "ft_in" in params:
            copied = copy(params["ft_in"])
            copied.elts = (value, *copied.elts[1:])
            params["ft_in"] = copied
        else:
            params["num_carry" if kind == "scan" else "body_nconsts"] = value
    else:
        aval = outputs[0].aval
        outputs[0] = Var(
            jax.core.ShapedArray(
                (1,) if mutation == "output-shape" else aval.shape,
                np.float64 if mutation == "output-dtype" else aval.dtype,
                weak_type=(
                    not aval.weak_type
                    if mutation == "output-weakness"
                    else aval.weak_type
                ),
            )
        )
    corrupted = equation.replace(params=params, outvars=outputs)
    test_graph = SimpleNamespace(
        invars=graph.invars,
        constvars=graph.constvars,
        outvars=outputs,
        effects=graph.effects,
        eqns=[corrupted],
    )
    # Keep any native prefix equations that define literal/initial operands.
    test_graph.eqns = [*graph.eqns[: graph.eqns.index(equation)], corrupted]
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(test_graph, 23)
    assert error.value.category in {"profile-exclusion", "lowering-defect"}
    assert error.value.primitive == kind


@pytest.mark.parametrize("kind", ["scan", "while"])
@pytest.mark.parametrize("length", [0, 2])
def test_unreviewed_body_operations_remain_recursive_product_failures(
    kind: str, length: int
) -> None:
    def target(x: Any) -> Any:
        if kind == "scan":
            return jax.lax.scan(
                lambda c, _: (jax.lax.erf(c), c), x, None, length=length
            )
        return jax.lax.while_loop(
            lambda s: s[0] < length,
            lambda s: (s[0] + np.int32(1), jax.lax.erf(s[1])),
            (np.int32(0), x),
        )

    with pytest.raises(SourcePrecisionError) as error:
        ui._to_onnx(
            target,
            [jax.ShapeDtypeStruct((), np.float32)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert (
        error.value.primitive == "erf" and error.value.category == "profile-exclusion"
    )
    assert ("scan-body" if kind == "scan" else "while-body") in error.value.location


@pytest.mark.parametrize("value", [1e8, -1e8])
def test_existing_fori_keeps_nested_jit_function_and_cond(value: float) -> None:
    def target(x: Any) -> Any:
        flag = x > np.float32(0)
        return jax.lax.fori_loop(
            0, 2, lambda index, state: jax.jit(real_cases._named_cond)(state, flag), x
        )

    with jax.enable_x64(True):
        assert len(_product(target, [np.asarray(value, np.float32)]).functions) >= 2


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("length", [0, 2])
def test_existing_complex_nested_scans_keep_empty_channel_axes(
    dtype: Any, length: int
) -> None:
    def target(left: Any, right: Any) -> Any:
        def step(state: Any, _: Any) -> Any:
            carry, scanned = jax.lax.scan(
                lambda pair, unused: ((pair[1], pair[0]), pair),
                state,
                None,
                length=length,
            )
            return carry, scanned

        return jax.lax.scan(step, (left, right), None, length=3)

    with jax.enable_x64(True):
        _product(
            target,
            [
                complex_cases._payload(dtype, (3,)),
                complex_cases._payload(dtype, (3,), other=True),
            ],
        )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("with_xs", [False, True])
@pytest.mark.parametrize("length,width", [(3, 2), (1, 3)])
def test_scan_length_remains_distinct_from_complex_vector_width(
    dtype: Any, with_xs: bool, length: int, width: int
) -> None:
    def target(initial: Any, sequence: Any) -> Any:
        carry, scanned = jax.lax.scan(
            lambda c, item: (c, c if item is None else item),
            initial,
            sequence if with_xs else None,
            length=length,
        )
        return carry, scanned, jax.lax.real(scanned), jax.lax.imag(scanned)

    with jax.enable_x64(True):
        value = complex_cases._payload(dtype, (width,))
        _product(target, [value, np.stack([value] * length)])


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("kind", ["scan-no-xs", "scan-xs", "while", "fori"])
@pytest.mark.parametrize("length", [0, 3])
def test_complex_loop_captures_keep_exact_stored_payloads(
    dtype: Any, kind: str, length: int
) -> None:
    captured = complex_cases._payload(dtype, (3,), other=True)

    def target(initial: Any) -> Any:
        if kind.startswith("scan"):
            xs = (
                jnp.asarray(np.arange(length, dtype=np.int32))
                if kind == "scan-xs"
                else None
            )
            return jax.lax.scan(
                lambda carry, item: (captured, captured), initial, xs, length=length
            )
        if kind == "fori":
            return jax.lax.fori_loop(0, length, lambda index, carry: captured, initial)
        return jax.lax.while_loop(
            lambda state: state[1] < np.int32(length),
            lambda state: (captured, state[1] + np.int32(1)),
            (initial, np.int32(0)),
        )

    with jax.enable_x64(True):
        proto = _product(target, [complex_cases._payload(dtype, (3,))])
        tensors = list(complex_cases._stored_tensors(proto))
        assert all(
            t.data_type not in (onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128)
            for t in tensors
        )
        if length:
            expected = complex_cases._physical(captured)
            assert any(
                array.dtype == expected.dtype
                and array.shape == expected.shape
                and array.tobytes(order="C") == expected.tobytes(order="C")
                for array in map(onnx.numpy_helper.to_array, tensors)
            )


@pytest.mark.parametrize(
    "mutation", ["xs-extent", "ys-count", "flat-output-partition", "reverse"]
)
def test_scan_sequence_and_output_partitions_are_checked(mutation: str) -> None:
    graph = _trace_loop("scan")
    equation = next(e for e in graph.eqns if e.primitive.name == "scan")
    params, inputs, outputs = (
        dict(equation.params),
        list(equation.invars),
        list(equation.outvars),
    )
    formals = list(graph.invars)
    if mutation == "xs-extent":
        nc, nk, _ = scan_arity(params, len(params["jaxpr"].jaxpr.invars))
        original = inputs[nc + nk]
        changed = Var(jax.core.ShapedArray((3, 3), original.aval.dtype))
        formals[formals.index(original)] = inputs[nc + nk] = changed
    elif mutation == "ys-count":
        outputs.pop()
    elif mutation == "flat-output-partition":
        if "ft_out" in params:
            descriptor = copy(params["ft_out"])
            descriptor.elts = (
                descriptor.elts[0].filter_with_mask([False] * len(descriptor.elts[0])),
                descriptor.elts[1],
            )
            params["ft_out"] = descriptor
        else:
            params["num_carry"] += 1
    else:
        params["reverse"] = True
    corrupted = equation.replace(params=params, invars=inputs, outvars=outputs)
    carrier = SimpleNamespace(
        invars=formals,
        constvars=graph.constvars,
        outvars=outputs,
        effects=(),
        eqns=[*graph.eqns[: graph.eqns.index(equation)], corrupted],
    )
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(carrier, 23)
    assert error.value.primitive == "scan"
    assert error.value.category == (
        "profile-exclusion" if mutation == "reverse" else "lowering-defect"
    )


@pytest.mark.parametrize(
    "dtype,shape", [(np.int32, ()), (np.bool_, (2,)), (np.bool_, (2, 1))]
)
def test_while_checks_predicate_dtype_and_logical_carry_prefix(
    dtype: Any, shape: Any
) -> None:
    graph = _trace_loop("while")
    equation = next(e for e in graph.eqns if e.primitive.name == "while")
    closed = equation.params["cond_jaxpr"]
    wrong = Var(jax.core.ShapedArray(shape, dtype))
    changed = ClosedJaxpr(closed.jaxpr.replace(outvars=[wrong]), closed.consts)
    corrupted = equation.replace(params={**equation.params, "cond_jaxpr": changed})
    carrier = SimpleNamespace(
        invars=graph.invars,
        constvars=graph.constvars,
        outvars=equation.outvars,
        effects=(),
        eqns=[*graph.eqns[: graph.eqns.index(equation)], corrupted],
    )
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(carrier, 23)
    assert (
        error.value.category == "lowering-defect" and error.value.primitive == "while"
    )


def test_cyclic_native_while_body_is_rejected_with_child_path() -> None:
    def target(x: Any) -> Any:
        return jax.lax.while_loop(lambda y: y > 0, lambda y: -y, x)

    graph = jax.make_jaxpr(target)(jax.ShapeDtypeStruct((), np.float32)).jaxpr
    equation = graph.eqns[0]
    closed = copy(equation.params["body_jaxpr"])
    body = closed.jaxpr
    recursive = equation.replace(
        invars=list(body.invars),
        outvars=list(body.outvars),
        params={**equation.params, "body_jaxpr": closed},
    )
    body._eqns = [recursive]
    corrupted = equation.replace(params={**equation.params, "body_jaxpr": closed})
    carrier = SimpleNamespace(
        invars=graph.invars,
        constvars=graph.constvars,
        outvars=equation.outvars,
        effects=(),
        eqns=[corrupted],
    )
    with pytest.raises(SourcePrecisionError, match="cyclic") as error:
        validate_source_jaxpr(carrier, 23)
    assert (
        "while-body" in error.value.location
        and error.value.category == "lowering-defect"
    )


@pytest.mark.parametrize("mutation", ["dtype", "shape", "missing"])
def test_closed_scan_constant_payloads_are_checked_before_lowering(
    mutation: str,
) -> None:
    graph = _trace_loop("scan")
    equation = next(e for e in graph.eqns if e.primitive.name == "scan")
    params = dict(equation.params)
    closed = params["jaxpr"]
    body = closed.jaxpr
    nc, _, _ = scan_arity(params, len(body.invars))
    assert nc >= 1
    captured = body.invars[0]
    rebound = body.replace(
        invars=body.invars[1:], constvars=[captured, *body.constvars]
    )
    payload = np.asarray([0.25, 0.5, 0.75], np.float32)
    if "ft_in" in params:
        descriptor = copy(params["ft_in"])
        descriptor.elts = (
            descriptor.elts[0].filter_with_mask([False, *([True] * (nc - 1))]),
            *descriptor.elts[1:],
        )
        params["ft_in"] = descriptor
    else:
        params["num_consts"] -= 1
    params["jaxpr"] = ClosedJaxpr(rebound, [payload, *closed.consts])
    changed = equation.replace(invars=equation.invars[1:], params=params)
    carrier = SimpleNamespace(
        invars=graph.invars,
        constvars=graph.constvars,
        outvars=changed.outvars,
        effects=(),
        eqns=[*graph.eqns[: graph.eqns.index(equation)], changed],
    )
    # First establish this actual native closed-capture fixture is valid.
    validate_source_jaxpr(carrier, 23)
    if mutation == "dtype":
        bad_payloads = [payload.astype(np.float64), *closed.consts]
    elif mutation == "shape":
        bad_payloads = [payload[:1], *closed.consts]
    else:
        bad_payloads = list(closed.consts)
    damaged = copy(params["jaxpr"])
    damaged._consts = bad_payloads
    params["jaxpr"] = damaged
    carrier.eqns[-1] = changed.replace(params=params)
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(carrier, 23)
    assert error.value.category == "lowering-defect"
    if mutation != "missing":
        assert "constant" in str(error.value) and "scan-body" in error.value.location
    # JAX 0.11 makes a payload-less constant a formal; its resulting role/count
    # mismatch must also fail, rather than pretending the capture disappeared.


@pytest.mark.parametrize("kind", ["scan", "while"])
def test_effectful_loop_bodies_remain_excluded_even_without_iterations(
    kind: str,
) -> None:
    def target(x: Any) -> Any:
        def step(value: Any) -> Any:
            jax.debug.print("value={value}", value=value)
            return -value

        if kind == "scan":
            return jax.lax.scan(lambda c, _: (step(c), c), x, None, length=0)
        return jax.lax.while_loop(lambda c: False, step, x)

    with pytest.raises(SourcePrecisionError, match="effects") as error:
        ui._to_onnx(
            target,
            [jax.ShapeDtypeStruct((), np.float32)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert error.value.category == "profile-exclusion"


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_static_empty_batched_while_preserves_both_complex_channels(dtype: Any) -> None:
    with jax.enable_x64(True):
        values = np.empty((0, 3), dtype=dtype)
        _product(
            complex_cases._vmapped_complex_while,
            [values, values.copy(), np.empty((0,), np.int32)],
        )


def test_logical_key_loop_transport_remains_an_explicit_implementation_gap() -> None:
    key = jax.random.key(0, impl="threefry2x32")

    def target(x: Any) -> Any:
        return jax.lax.scan(lambda state, _: (state, state), (key, x), None, length=1)

    with pytest.raises(SourcePrecisionError, match="typed[- ]key") as error:
        ui._to_onnx(
            target,
            [jax.ShapeDtypeStruct((), np.float32)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert error.value.category == "profile-exclusion"
