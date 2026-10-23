# tests/extra_tests/converter/test_source_loop_precision.py

"""Finite, test-only SOURCE audits for native JAX loop and JIT composition.

This validator checks the typed roles exercised here and recursively inspects
actual bodies. It does not extend the product source profile or admit arbitrary
loop bodies. References are traced and evaluated before activating any patches.
"""

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx import user_interface
from jax2onnx._compat.jax import scan_arity
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import PrecisionPolicy, active_precision_policy
from jax2onnx.precision import (
    CheckStatus,
    SOURCE_DIAGNOSTIC_PROFILE,
    get_precision_report,
)


@pytest.fixture(autouse=True)
def _restore_jax_settings() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    previous_precision = jax.config.jax_default_matmul_precision
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)
        jax.config.update("jax_default_matmul_precision", previous_precision)


_FLOATS = {"float16", "float32", "float64"}
_INTEGERS = {"int32", "int64", "uint32"}


def _spec(var: Any) -> tuple[str, tuple[int, ...], bool]:
    aval = var.aval
    dtype = np.dtype(aval.dtype).name
    shape = tuple(aval.shape)
    assert dtype in _FLOATS | _INTEGERS | {"bool"}, (dtype, shape)
    assert all(isinstance(d, (int, np.integer)) and d >= 0 for d in shape)
    return dtype, shape, bool(aval.weak_type)


def _roles(vars: Any) -> list[tuple[str, tuple[int, ...]]]:
    return [_spec(var)[:2] for var in vars]


def _body(value: Any) -> Any:
    return getattr(value, "jaxpr", value)


class _LoopAudit:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, jaxpr: Any, opset: int) -> None:
        assert opset == 23
        assert not jaxpr.effects
        for var in (*jaxpr.constvars, *jaxpr.invars, *jaxpr.outvars):
            _spec(var)
        for eqn in jaxpr.eqns:
            name = eqn.primitive.name
            self.seen.append(name)
            ins, outs = _roles(eqn.invars), _roles(eqn.outvars)
            assert not eqn.effects
            if name in {"add", "sub", "neg", "mul"}:
                assert len(ins) == (1 if name == "neg" else 2) and len(outs) == 1
                dtype = ins[0][0]
                assert all(role[0] == dtype for role in (*ins, *outs))
                assert dtype in _FLOATS or (
                    dtype in {"int32", "int64"}
                    and name in {"add", "sub"}
                    and all(role[1] == () for role in (*ins, *outs))
                )
                assert (
                    tuple(np.broadcast_shapes(*(role[1] for role in ins))) == outs[0][1]
                )
            elif name in {"lt", "gt"}:
                assert len(ins) == 2 and len(outs) == 1
                assert ins[0][0] == ins[1][0] and ins[0][0] in _FLOATS | _INTEGERS
                assert outs == [
                    ("bool", tuple(np.broadcast_shapes(*(r[1] for r in ins))))
                ]
            elif name == "not":
                assert len(ins) == 1 and ins == outs and ins[0][0] == "bool"
            elif name == "convert_element_type":
                assert len(ins) == len(outs) == 1 and ins[0][1] == outs[0][1]
                assert np.dtype(eqn.params["new_dtype"]).name == outs[0][0]
                pair = (ins[0][0], outs[0][0])
                assert (
                    set(pair) <= _FLOATS
                    or set(pair) <= _INTEGERS
                    or pair == ("bool", "int32")
                    or (pair[0] in _INTEGERS and pair[1] in _FLOATS)
                ), pair
            elif name == "while":
                cond, step = _body(eqn.params["cond_jaxpr"]), _body(
                    eqn.params["body_jaxpr"]
                )
                nc, nb = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
                carry = ins[nc + nb :]
                assert outs == carry == _roles(step.outvars)
                assert _roles(step.invars) == ins[nc : nc + nb] + carry
                assert _roles(cond.invars) == ins[:nc] + carry
                assert _roles(cond.outvars) == [("bool", ())]
                self(cond, opset)
                self(step, opset)
            elif name == "scan":
                step = _body(eqn.params["jaxpr"])
                nc, nk, nx = scan_arity(eqn.params, len(step.invars))
                length = eqn.params["length"]
                assert (
                    isinstance(length, int)
                    and length >= 0
                    and not eqn.params["reverse"]
                )
                assert nc + nk + nx == len(ins)
                assert _roles(step.invars) == ins[: nc + nk] + [
                    (dtype, shape[1:]) for dtype, shape in ins[nc + nk :]
                ]
                step_out = _roles(step.outvars)
                assert step_out[:nk] == ins[nc : nc + nk]
                assert outs == step_out[:nk] + [
                    (dtype, (length, *shape)) for dtype, shape in step_out[nk:]
                ]
                self(step, opset)
            elif name in {"jit", "pjit"}:
                step = _body(eqn.params.get("jaxpr", eqn.params.get("call_jaxpr")))
                assert step is not None
                assert _roles(step.invars) == ins and _roles(step.outvars) == outs
                self(step, opset)
            elif name == "cond":
                branches = eqn.params["branches"]
                assert len(branches) == 2 and ins[0] == ("int32", ())
                for branch in branches:
                    step = _body(branch)
                    assert (
                        _roles(step.invars) == ins[1:] and _roles(step.outvars) == outs
                    )
                    self(step, opset)
            elif name.startswith("onnx_fn::"):
                # The common validator callback audits the actual retraced body.
                assert outs
            else:
                raise AssertionError(f"Unaudited loop-test primitive: {name}")


def _signature(jaxpr: Any) -> Any:
    """Compare native and patched typed equations, including internal weak types."""
    equations = []
    for eqn in jaxpr.eqns:
        nested = []
        for key in ("jaxpr", "call_jaxpr", "cond_jaxpr", "body_jaxpr"):
            if key in eqn.params:
                nested.append((key, _signature(_body(eqn.params[key]))))
        if "branches" in eqn.params:
            nested.extend(
                ("branch", _signature(_body(b))) for b in eqn.params["branches"]
            )
        equations.append(
            (
                eqn.primitive.name,
                tuple(map(_spec, eqn.invars)),
                tuple(map(_spec, eqn.outvars)),
                tuple(nested),
            )
        )
    return (
        tuple(map(_spec, jaxpr.invars)),
        tuple(map(_spec, jaxpr.outvars)),
        tuple(equations),
    )


def _trace(fn: Callable[..., Any], specs: list[jax.ShapeDtypeStruct]) -> Any:
    return conversion_api._trace_to_jaxpr(
        fn=fn,
        inputs=specs,
        input_params=None,
        enable_double_precision=False,
        inputs_as_nchw=None,
        outputs_as_nchw=None,
        input_names=None,
        output_names=None,
        precision_policy=PrecisionPolicy.SOURCE,
    ).jaxpr


def _export(
    fn: Callable[..., Any], values: list[np.ndarray], *, compare_trace: bool = True
) -> tuple[onnx.ModelProto, _LoopAudit]:
    specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
    original = jax.make_jaxpr(fn)(*specs).jaxpr
    expected = [np.asarray(v) for v in jax.tree.leaves(fn(*values))]
    audit = _LoopAudit()
    observed = []

    def check_trace(jaxpr: Any, opset: int) -> None:
        if not observed:
            if compare_trace:
                assert _signature(jaxpr) == _signature(original)
            assert _roles(jaxpr.outvars) == _roles(original.outvars)
        observed.append(jaxpr)
        audit(jaxpr, opset)

    model = user_interface._to_onnx(
        fn,
        specs,
        model_name="source_loop_audit",
        opset=23,
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=check_trace,
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
    assert len(proto.graph.output) == len(expected)
    for output, reference in zip(proto.graph.output, expected):
        tensor = output.type.tensor_type
        assert tensor.elem_type == onnx.helper.np_dtype_to_tensor_dtype(reference.dtype)
        assert all(dim.HasField("dim_value") for dim in tensor.shape.dim)
        assert tuple(dim.dim_value for dim in tensor.shape.dim) == reference.shape
    for optimization in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = optimization
        session = ort.InferenceSession(
            proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        feeds = {item.name: value for item, value in zip(session.get_inputs(), values)}
        actual = session.run(None, feeds)
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected):
            assert result.dtype == reference.dtype and result.shape == reference.shape
            assert result.tobytes() == reference.tobytes()
    assert active_precision_policy() is PrecisionPolicy.LEGACY
    return proto, audit


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize(
    "initial",
    [np.int32(0), np.int64(0), np.uint32(0), True, 0, 0.0],
    ids=["i32", "i64", "u32", "bool", "weak-int", "weak-float"],
)
def test_native_fori_trace_preserves_carry_dtype_and_weak_type(
    ambient_x64: bool, initial: Any
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)

    def target(x: Any) -> Any:
        return jax.lax.fori_loop(2, 4, lambda index, state: state, (initial, x))

    specs = [jax.ShapeDtypeStruct((), np.float32)]
    original = jax.make_jaxpr(target)(*specs).jaxpr
    patched = _trace(target, specs)
    assert _signature(patched) == _signature(original)
    assert any(eqn.primitive.name == "scan" for eqn in patched.eqns)
    _LoopAudit()(patched, 23)


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize(
    "bounds",
    [(2, 5), (np.int32(2), np.int32(5)), (2, np.int32(5)), (np.int64(2), np.int64(5))],
    ids=["python", "i32", "weak-strong", "i64"],
)
def test_native_fori_index_aval_matches_original_bound_semantics(
    ambient_x64: bool, bounds: tuple[Any, Any]
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)

    def target(x: Any) -> Any:
        return jax.lax.fori_loop(
            *bounds, lambda index, state: (index, state[1]), (0, x)
        )

    _export(target, [np.asarray(1, np.float32)])


@pytest.mark.parametrize("ambient_x64", [False, True])
def test_native_fori_specializes_weak_carry_like_original(ambient_x64: bool) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)

    def target(x: Any) -> Any:
        return jax.lax.fori_loop(0, 2, lambda index, n: n + np.int32(1), 0), x

    _export(target, [np.asarray(1, np.float32)])


def _while_literal(x: Any) -> Any:
    return (
        jax.lax.while_loop(
            lambda n: n < np.int32(2), lambda n: n + np.int32(1), np.int32(0)
        ),
        x,
    )


def _while_mixed(x: Any) -> Any:
    with jax.enable_x64(True):

        def step(state: Any) -> Any:
            n, half, wide, unsigned, flag = state
            return (
                n + np.int32(1),
                -half,
                wide + np.float64(0.5),
                unsigned,
                jnp.logical_not(flag),
            )

        return jax.lax.while_loop(
            lambda state: state[0] < np.int32(2),
            step,
            (
                np.int32(0),
                x.astype(jnp.float16),
                x.astype(jnp.float64),
                np.uint32(2**31 + 9),
                x > np.float32(0),
            ),
        )


def _fori_bool(x: Any) -> Any:
    return jax.lax.fori_loop(0, 3, lambda index, flag: jnp.logical_not(flag), True), x


def _fori_mixed(x: Any) -> Any:
    with jax.enable_x64(True):

        def step(index: Any, state: Any) -> Any:
            half, wide, n32, n64, unsigned, flag, weak = state
            return (
                -half,
                wide + np.float64(0.5),
                n32 + np.int32(1),
                n64 + np.int64(1),
                unsigned,
                jnp.logical_not(flag),
                weak + 1,
            )

        return jax.lax.fori_loop(
            1,
            3,
            step,
            (
                x.astype(jnp.float16),
                x.astype(jnp.float64),
                np.int32(0),
                np.int64(2**40),
                np.uint32(2**31 + 9),
                True,
                0,
            ),
        )


@jax.jit
def _fori_dynamic(x: Any, lower: Any, upper: Any) -> Any:
    return jax.lax.fori_loop(
        lower, upper, lambda index, state: (index, -state[1]), (lower, x)
    )


def _scan_i64(xs: Any) -> Any:
    with jax.enable_x64(True):
        return jax.lax.scan(lambda n, y: (n + np.int64(1), y), np.int64(2**40), xs)


_COEFFICIENT = np.asarray(1 + 2**-30, np.float64)


def _scan_captured(xs: Any) -> Any:
    with jax.enable_x64(True):
        return jax.lax.scan(
            lambda carry, y: (carry + _COEFFICIENT, y), np.float64(0), xs
        )


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize(
    "fn,values",
    [
        (_while_literal, [np.asarray(0, np.float32)]),
        (_while_mixed, [np.asarray(2, np.float32)]),
        (_fori_bool, [np.asarray(0, np.float32)]),
        (_fori_mixed, [np.asarray(2, np.float32)]),
        (
            _fori_dynamic,
            [
                np.asarray(2, np.float32),
                np.asarray(2, np.int32),
                np.asarray(5, np.int32),
            ],
        ),
        (_scan_i64, [np.asarray([1, 2, 3], np.float32)]),
        (_scan_captured, [np.asarray([1, 2, 3], np.float32)]),
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
def test_source_loop_types_shapes_and_cpu_values(
    fn: Callable[..., Any], values: list[np.ndarray], ambient_x64: bool
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)
    _, audit = _export(fn, values)
    assert "scan" in audit.seen or "while" in audit.seen
    assert bool(jax.config.x64_enabled) is ambient_x64


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_scan_without_xs_retains_static_empty_and_nonempty_shapes(
    ambient_x64: bool, length: int
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)

    def target(x: Any) -> Any:
        return jax.lax.scan(
            lambda half, _: (-half, (half.astype(jnp.float32), half > np.float16(0))),
            x,
            None,
            length=length,
        )

    _export(target, [np.asarray([-0.0, 2], np.float16)])


@onnx_function
def _named_cancellation(value: Any) -> Any:
    rounded = value.astype(jnp.float32)
    return (rounded + np.float32(1)) - rounded


def _reused_body(index: Any, state: Any) -> Any:
    value, wide = state
    return _named_cancellation(value), wide + _COEFFICIENT


def _reused_fori(value: Any) -> Any:
    return jax.lax.fori_loop(0, 1, _reused_body, (value, np.float64(0)))


def _nodes(nodes: Any) -> Iterator[Any]:
    for node in nodes:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _nodes(attr.g.node)


@pytest.mark.parametrize("source_first", [False, True])
def test_reused_fori_body_keeps_named_functions_and_captures_across_policy_caches(
    source_first: bool,
) -> None:
    jax.config.update("jax_enable_x64", True)
    values = [np.asarray(1e8, np.float32)]
    assert np.asarray(_reused_fori(*values)[0]).tobytes() == np.float32(0).tobytes()
    if source_first:
        _export(_reused_fori, values, compare_trace=False)
    ordinary = jax.make_jaxpr(_reused_fori)(*values).jaxpr
    to_onnx(
        _reused_fori,
        [jax.ShapeDtypeStruct((), np.float32)],
        enable_double_precision=True,
    )
    proto, audit = _export(_reused_fori, values, compare_trace=False)
    assert _signature(jax.make_jaxpr(_reused_fori)(*values).jaxpr) == _signature(
        ordinary
    )
    assert proto.functions
    assert any(name.startswith("onnx_fn::") for name in audit.seen)
    assert "sub" in audit.seen
    calls = {node.op_type for node in _nodes(proto.graph.node)}
    assert any(function.name in calls for function in proto.functions)
    constants = list(proto.graph.initializer)
    for node in _nodes(proto.graph.node):
        constants.extend(
            attr.t for attr in node.attribute if attr.type == onnx.AttributeProto.TENSOR
        )
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                constants.extend(attr.g.initializer)
    assert any(
        actual.dtype == _COEFFICIENT.dtype
        and actual.tobytes() == _COEFFICIENT.tobytes()
        for actual in map(onnx.numpy_helper.to_array, constants)
    )


@onnx_function
def _named_cond(value: Any, flag: Any) -> Any:
    return jax.lax.cond(
        flag, _named_cancellation, lambda x: -_named_cancellation(x), value
    )


@pytest.mark.parametrize("value", [1e8, -1e8])
def test_native_fori_composes_jit_named_function_and_conditional(value: float) -> None:
    jax.config.update("jax_enable_x64", True)

    def target(value: Any) -> Any:
        flag = value > np.float32(0)

        def step(index: Any, state: Any) -> Any:
            return jax.jit(_named_cond)(state, flag)

        return jax.lax.fori_loop(0, 2, step, value)

    proto, audit = _export(target, [np.asarray(value, np.float32)], compare_trace=False)
    assert "jit" in audit.seen and "cond" in audit.seen
    assert len(proto.functions) >= 2


@pytest.mark.parametrize("local_x64", [False, True])
def test_jit_retains_model_local_x64_and_rounding(local_x64: bool) -> None:
    jax.config.update("jax_enable_x64", not local_x64)

    def target(value: Any) -> Any:
        with jax.enable_x64(local_x64):

            def cancellation(x: Any) -> Any:
                wide = x.astype(jnp.float64)
                return ((wide + np.float64(1)) - wide).astype(jnp.float32)

            return jax.jit(cancellation)(value)

    _, audit = _export(target, [np.asarray(1e8, np.float32)])
    assert "jit" in audit.seen
    assert bool(jax.config.x64_enabled) is not local_x64


@pytest.mark.parametrize("failure", ["bounds", "carry-dtype", "carry-tree", "unroll"])
def test_source_native_fori_rejects_the_same_invalid_programs_as_original(
    failure: str,
) -> None:
    jax.config.update("jax_enable_x64", True)

    def target(x: Any) -> Any:
        if failure == "bounds":
            return jax.lax.fori_loop(np.int32(0), np.int64(2), lambda i, n: n, x)
        if failure == "carry-dtype":
            return jax.lax.fori_loop(0, 2, lambda i, n: x, np.int32(0))
        if failure == "carry-tree":
            return jax.lax.fori_loop(0, 2, lambda i, n: (n, n), x)
        return jax.lax.fori_loop(
            np.int32(0), x.astype(jnp.int32), lambda i, n: n, x, unroll=2
        )

    specs = [jax.ShapeDtypeStruct((), np.float32)]
    with pytest.raises((TypeError, ValueError)) as native:
        jax.make_jaxpr(target)(*specs)
    with pytest.raises(type(native.value)) as patched:
        _trace(target, specs)
    assert str(patched.value) == str(native.value)


def test_native_concrete_and_abstract_fori_bounds_have_distinct_index_specializations() -> (
    None
):
    jax.config.update("jax_enable_x64", True)

    def target(value: Any, lower: Any, upper: Any) -> Any:
        return jax.lax.fori_loop(
            lower, upper, lambda i, state: (i, -state[1]), (lower, value)
        )

    values = [
        np.asarray(2, np.float32),
        np.asarray(2, np.int32),
        np.asarray(5, np.int32),
    ]
    # Concrete native bounds select scan with a default-width index. Abstract
    # bounds select while with the actual int32 bound/index aval. Keep this
    # native eager error visible instead of treating it as an export mismatch.
    with pytest.raises(TypeError, match="int32.*int64"):
        target(*values)
    specs = [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in values]
    native = jax.make_jaxpr(target)(*specs).jaxpr
    assert _signature(_trace(target, specs)) == _signature(native)
    assert any(eqn.primitive.name == "while" for eqn in native.eqns)


@pytest.mark.parametrize("unroll", [True, 1])
def test_native_dynamic_fori_unroll_one_semantics_match_original(unroll: Any) -> None:
    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def target(value: Any, upper: Any) -> Any:
        return jax.lax.fori_loop(
            np.int32(0),
            upper,
            lambda i, state: -state,
            value,
            unroll=unroll,
        )

    specs = [jax.ShapeDtypeStruct((), np.float32), jax.ShapeDtypeStruct((), np.int32)]
    try:
        jax.make_jaxpr(target)(*specs)
    except ValueError as native_error:
        # Supported JAX versions can differ in whether True/1 is permitted for
        # a dynamic loop. SOURCE inherits that version's native decision.
        with pytest.raises(ValueError, match="unroll") as patched:
            _trace(target, specs)
        assert str(patched.value) == str(native_error)
    else:
        _export(target, [np.asarray(2, np.float32), np.asarray(3, np.int32)])


@onnx_function
def _named_empty_scan(value: Any) -> Any:
    carry, ys = jax.lax.scan(lambda state, _: (-state, state), value, None, length=0)
    return carry, {"empty": ys, "again": (ys,)}


@pytest.mark.parametrize("scope", ["shared", "nested", "function"])
def test_zero_trip_scan_preserves_repeated_and_nested_empty_outputs(scope: str) -> None:
    def target(value: Any) -> Any:
        if scope == "function":
            return _named_empty_scan(value)
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

    _export(
        target, [np.asarray([-0.0, 2], np.float32)], compare_trace=scope != "function"
    )


@pytest.mark.parametrize("length", [0, 2])
def test_loop_audit_rejects_an_unlisted_primitive_recursively(length: int) -> None:
    def target(x: Any) -> Any:
        return jax.lax.scan(
            lambda carry, _: (jnp.sin(carry), carry), x, None, length=length
        )

    traced = _trace(target, [jax.ShapeDtypeStruct((), np.float32)])
    with pytest.raises(AssertionError, match="Unaudited loop-test primitive: .*sin"):
        _LoopAudit()(traced, 23)
