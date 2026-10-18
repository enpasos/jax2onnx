# tests/extra_tests/converter/test_source_jit_admission.py

"""Typed plain-JIT carrier validation and existing lowering ownership."""

from copy import copy
from types import SimpleNamespace
from typing import Any

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, to_onnx, user_interface as ui
from jax2onnx._compat.jax import ClosedJaxpr, Var, fresh_var_like
from jax2onnx.converter import conversion_api
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.precision import SampleCriteria
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
    _accept,
    _artifact,
    _body,
    _globals,
    _settings,
)
from tests.extra_tests.converter.test_source_where_dispatch import (
    _restore_state as _restore_state,
)

_BITWISE = SampleCriteria("bitwise")
_FINITE = {
    "float32": SampleCriteria("tolerant", rtol=1e-5, atol=1e-5),
    "float64": SampleCriteria("tolerant", rtol=1e-7, atol=1e-7),
}


def _trace(fn: Any = None, specs: Any = None) -> Any:
    if fn is None:
        fn = jax.jit(lambda x: -x)
    return _body(
        jax.make_jaxpr(fn)(*(specs or [jax.ShapeDtypeStruct((2,), np.float32)]))
    )


def _carrier(eqn: Any) -> Any:
    return SimpleNamespace(
        invars=list(eqn.invars),
        constvars=[],
        outvars=list(eqn.outvars),
        effects=(),
        eqns=[eqn],
    )


def _change_body(eqn: Any, **changes: Any) -> Any:
    closed = eqn.params["jaxpr"]
    body = closed.jaxpr.replace(**changes)
    return eqn.replace(params={**eqn.params, "jaxpr": ClosedJaxpr(body, closed.consts)})


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "kind", ["neg", "pointwise", "matmul", "alias", "repeated", "closure"]
)
def test_plain_jit_reuses_typed_body_and_existing_lowerer(
    dtype: Any, kind: str
) -> None:
    with jax.enable_x64(dtype is np.float64):
        captured = np.asarray([[0.25, 0.5], [1.0, 2.0]], dtype)
        before = captured.tobytes()
        if kind == "neg":
            fn = jax.jit(lambda x: -x)
        elif kind == "pointwise":
            fn = jax.jit(lambda x: jnp.exp(x + x) * x)
        elif kind == "matmul":
            fn = jax.jit(lambda x: x @ captured)
        elif kind == "alias":
            fn = jax.jit(lambda x: (x, x, captured, captured))
        elif kind == "repeated":
            child = jax.jit(lambda x: x + captured)

            def fn(x: Any) -> Any:
                return child(x), child(-x)

        else:
            left, right = jax.jit(lambda x: x + captured), jax.jit(
                lambda x: x + captured * 2
            )

            def fn(x: Any) -> Any:
                return left(x), right(x)

        values = np.asarray([[0.25, -0.5], [0.5, 1.0]], dtype)
        feeds = ((values,),)
        _, proto, _, refs = _artifact(
            fn, [jax.ShapeDtypeStruct(values.shape, dtype)], feeds=feeds
        )
        _accept(
            proto,
            feeds,
            refs,
            _FINITE[np.dtype(dtype).name] if kind == "pointwise" else _BITWISE,
        )
        assert captured.tobytes() == before


@onnx_function(name="SourceNamedJit")
def _named(x: Any) -> Any:
    return jax.jit(lambda x: (x + x, x))(x)


@onnx_function(name="SourceNestedNamedJit")
def _nested(x: Any) -> Any:
    return _named(x)


@onnx_function(name="SourceJitCondCarrier")
def _cond(x: Any, gate: Any) -> Any:
    return jax.lax.cond(gate, lambda x: _nested(x), lambda x: _nested(-x), x)


@pytest.mark.parametrize("scope", ["nested", "cond"])
def test_actual_function_and_cond_jit_bodies(scope: str) -> None:
    def target(x: Any, gate: Any) -> Any:
        if scope == "nested":
            return _nested(x)
        return _cond(x, gate)

    values = np.asarray([0.5, 1.0], np.float32)
    feeds = ((values, np.asarray(True)), (values, np.asarray(False)))
    _, proto, _, refs = _artifact(
        target,
        [jax.ShapeDtypeStruct((2,), np.float32), jax.ShapeDtypeStruct((), bool)],
        feeds=feeds,
        named=True,
    )
    assert proto.functions
    _accept(proto, feeds, refs, _BITWISE)


@pytest.mark.parametrize("reverse", [False, True])
def test_precompiled_callable_keeps_native_legacy_source_caches(reverse: bool) -> None:
    compiled = jax.jit(lambda x: jnp.where(x > 0, x, 0.5))
    x = np.asarray([1.0, -2.0], np.float32)
    expected = np.asarray(compiled(x))
    settings = _globals(), _settings()
    sequence = (
        ("native", "source", "legacy", "native")
        if reverse
        else ("native", "legacy", "source", "native")
    )
    for phase in sequence:
        if phase == "native":
            np.testing.assert_array_equal(compiled(x), expected)
        elif phase == "legacy":
            artifact = to_onnx(compiled, [jax.ShapeDtypeStruct((2,), np.float32)])
            onnx.checker.check_model(artifact, full_check=True)
        else:
            _, proto, _, refs = _artifact(
                compiled, [jax.ShapeDtypeStruct((2,), np.float32)], feeds=((x,),)
            )
            _accept(proto, ((x,),), refs, _BITWISE)
        assert (_globals(), _settings()) == settings


@pytest.mark.parametrize("kind", ["add", "mul", "exp", "matmul", "selector"])
@pytest.mark.parametrize("rank", [2, 3])
def test_jit_preserves_native_leading_batch_contract(kind: str, rank: int) -> None:
    (batch,) = export.symbolic_shape("B")
    shape = (batch, 3) if rank == 2 else (batch, 2, 3)
    weight = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], np.float32)
    if kind == "matmul":
        if rank == 2:
            fn = jax.jit(lambda x, c: x @ weight)
        else:
            # Canonical paired MatMul requires equal B, not a symbolic broadcast.
            # Its rank-three path is exercised by an explicit second argument below.
            fn = jax.jit(lambda x, w: x @ w)
            specs = [
                jax.ShapeDtypeStruct(shape, np.float32),
                jax.ShapeDtypeStruct((batch, 3, 2), np.float32),
            ]
    else:
        fn = jax.jit(
            {
                "add": lambda x, c: x + x,
                "mul": lambda x, c: x * x,
                "exp": lambda x, c: jnp.exp(x),
                "selector": lambda x, c: jnp.where(c, x, x + x),
            }[kind]
        )
    if not (kind == "matmul" and rank == 3):
        specs = [
            jax.ShapeDtypeStruct(shape, np.float32),
            jax.ShapeDtypeStruct((), bool),
        ]
    feeds = []
    for size in (2, 5):
        x = np.full((size, 3) if rank == 2 else (size, 2, 3), 0.5, np.float32)
        if kind == "matmul" and rank == 3:
            feeds.append((x, np.broadcast_to(weight, (size, 3, 2)).copy()))
        else:
            feeds.extend(((x, np.asarray(True)), (x, np.asarray(False))))
    _, proto, _, refs = _artifact(fn, specs, feeds=tuple(feeds))
    _accept(
        proto, tuple(feeds), refs, _FINITE["float32"] if kind == "exp" else _BITWISE
    )


@pytest.mark.parametrize(
    "mutation,category",
    [
        ("missing", "lowering-defect"),
        ("unknown", "profile-exclusion"),
        ("lazy", "profile-exclusion"),
        ("open", "lowering-defect"),
        ("sharding_count", "lowering-defect"),
        ("layout_count", "lowering-defect"),
        ("sharding", "profile-exclusion"),
        ("layout", "profile-exclusion"),
        ("donation", "profile-exclusion"),
        ("donation_int", "lowering-defect"),
        ("donation_count", "lowering-defect"),
        ("mesh", "profile-exclusion"),
        ("name", "lowering-defect"),
        ("inline", "lowering-defect"),
        ("keep_unused", "lowering-defect"),
        ("options", "profile-exclusion"),
        ("options_type", "lowering-defect"),
        ("weak", "lowering-defect"),
        ("shape", "lowering-defect"),
        ("dtype", "lowering-defect"),
        ("input_count", "lowering-defect"),
        ("output_count", "lowering-defect"),
    ],
)
def test_jit_native_parameter_and_signature_corruption(
    mutation: str, category: str
) -> None:
    eqn = _trace().eqns[0]
    params = dict(eqn.params)
    if mutation == "missing":
        params.pop("inline")
    elif mutation == "unknown":
        params["unreviewed"] = None
    elif mutation == "lazy":
        params["call_jaxpr_thunk"] = lambda: pytest.fail("must not evaluate thunk")
    elif mutation == "open":
        params["jaxpr"] = SimpleNamespace(jaxpr=params["jaxpr"].jaxpr)
    elif mutation == "sharding_count":
        params["in_shardings"] = ()
    elif mutation == "layout_count":
        params["out_layouts"] = ()
    elif mutation == "sharding":
        params["in_shardings"] = (jax.sharding.SingleDeviceSharding(jax.devices()[0]),)
    elif mutation == "layout":
        params["out_layouts"] = (object(),)
    elif mutation == "donation":
        params["donated_invars"] = (True,)
    elif mutation == "donation_int":
        params["donated_invars"] = (0,)
    elif mutation == "donation_count":
        params["donated_invars"] = ()
    elif mutation == "mesh":
        params["ctx_mesh"] = SimpleNamespace(empty=True)
    elif mutation in {"name", "inline", "keep_unused"}:
        params[mutation] = 0
    elif mutation == "options":
        params["compiler_options_kvs"] = (("xla_cpu_enable_fast_math", False),)
    elif mutation == "options_type":
        params["compiler_options_kvs"] = []
    else:
        body = params["jaxpr"].jaxpr
        if mutation == "input_count":
            eqn = eqn.replace(invars=[])
            params["in_shardings"], params["in_layouts"], params["donated_invars"] = (
                (),
                (),
                (),
            )
        elif mutation == "output_count":
            eqn = eqn.replace(outvars=[])
            params["out_shardings"], params["out_layouts"] = (), ()
        else:
            aval = body.outvars[0].aval
            new = Var(
                aval.update(weak_type=not aval.weak_type)
                if mutation == "weak"
                else (
                    aval.update(shape=(3,))
                    if mutation == "shape"
                    else aval.update(dtype=np.float16)
                )
            )
            eqn = eqn.replace(outvars=[new])
    eqn = eqn.replace(params=params)
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(_carrier(eqn), 23)
    assert error.value.category == category
    assert error.value.primitive == "jit"


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_input",
        "unbound",
        "redefined",
        "unbound_output",
        "graph_effect",
        "equation_effect",
        "cycle",
    ],
)
def test_jit_recursive_binding_and_effect_corruption(mutation: str) -> None:
    eqn = _trace().eqns[0]
    closed, body = eqn.params["jaxpr"], eqn.params["jaxpr"].jaxpr
    if mutation == "duplicate_input":
        eqn = _change_body(eqn, invars=[body.invars[0], body.invars[0]])
        eqn = eqn.replace(
            invars=[*eqn.invars, *eqn.invars],
            params={
                **eqn.params,
                "in_shardings": eqn.params["in_shardings"] * 2,
                "in_layouts": (None, None),
                "donated_invars": (False, False),
            },
        )
        graph = _carrier(eqn)
        # The enclosing binding also rejects duplicate formals before lowering.
    elif mutation == "unbound":
        eqn = _change_body(
            eqn, eqns=[body.eqns[0].replace(invars=[fresh_var_like(body.invars[0])])]
        )
        graph = _carrier(eqn)
    elif mutation == "redefined":
        eqn = _change_body(eqn, eqns=[body.eqns[0], body.eqns[0]])
        graph = _carrier(eqn)
    elif mutation == "unbound_output":
        eqn = _change_body(eqn, outvars=[fresh_var_like(body.outvars[0])])
        graph = _carrier(eqn)
    elif mutation == "graph_effect":
        eqn = _change_body(eqn, effects={object()})
        graph = _carrier(eqn)
    elif mutation == "equation_effect":
        eqn = _change_body(eqn, eqns=[body.eqns[0].replace(effects={object()})])
        graph = _carrier(eqn)
    else:
        copied = copy(closed)
        nested = eqn.replace(
            params={**eqn.params, "jaxpr": copied},
            invars=body.invars,
            outvars=body.outvars,
        )
        # Native structures are mutable; insert one exact typed self-reference.
        copied.jaxpr._eqns = [nested]
        graph = _carrier(eqn.replace(params={**eqn.params, "jaxpr": copied}))
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(graph, 23)
    assert error.value.category in {"profile-exclusion", "lowering-defect"}


@pytest.mark.parametrize("mutation", ["dtype", "shape", "payload_type"])
def test_closed_constant_payloads_fail_before_context_or_nodes(
    mutation: str, monkeypatch: Any
) -> None:
    captured = np.asarray([np.nan, -0.0], np.float64)
    original = captured.tobytes()
    with jax.enable_x64(True):
        fn = jax.jit(lambda x: x + captured)
        trace = conversion_api._trace_to_jaxpr(
            fn=fn,
            inputs=[jax.ShapeDtypeStruct((2,), np.float64)],
            input_params=None,
            enable_double_precision=False,
            inputs_as_nchw=None,
            outputs_as_nchw=None,
            input_names=None,
            output_names=None,
            precision_policy=PrecisionPolicy.SOURCE,
        )
    eqn = trace.jaxpr.eqns[0]
    closed = copy(eqn.params["jaxpr"])
    payload = (
        captured.astype(np.float32)
        if mutation == "dtype"
        else captured[:1] if mutation == "shape" else object()
    )
    closed._consts = [payload]
    graph = copy(trace.jaxpr)
    graph._eqns = [eqn.replace(params={**eqn.params, "jaxpr": closed})]
    monkeypatch.setattr(
        conversion_api,
        "_trace_to_jaxpr",
        lambda **kwargs: SimpleNamespace(**{**vars(trace), "jaxpr": graph}),
    )
    monkeypatch.setattr(
        conversion_api,
        "_create_ir_context",
        lambda **kwargs: pytest.fail("must validate before context/constants/nodes"),
    )
    with jax.enable_x64(True), pytest.raises(SourcePrecisionError) as error:
        ui._to_onnx(
            fn,
            [jax.ShapeDtypeStruct((2,), np.float64)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert error.value.category == "lowering-defect"
    assert captured.tobytes() == original


def test_captured_fp64_payload_is_not_recanonicalized_by_x64_off() -> None:
    captured = np.asarray([np.nan, -0.0, np.inf], np.float64)
    original = captured.tobytes()
    with jax.enable_x64(True):
        graph = _trace(
            jax.jit(lambda x: (captured, x)), [jax.ShapeDtypeStruct((), np.float32)]
        )
    with jax.enable_x64(False):
        validate_source_jaxpr(graph, 23)
    assert (
        graph.eqns[0].params["jaxpr"].consts[0].tobytes()
        == original
        == captured.tobytes()
    )


@pytest.mark.parametrize("kind", ["sin", "scan", "while"])
def test_jit_does_not_admit_unaudited_child_operations(kind: str) -> None:
    child = {
        "sin": lambda x: jnp.sin(x),
        "scan": lambda x: jax.lax.fori_loop(0, 2, lambda i, v: v + v, x),
        "while": lambda x: jax.lax.while_loop(
            lambda pair: pair[0] < 2,
            lambda pair: (pair[0] + 1, pair[1] + pair[1]),
            (0, x),
        )[1],
    }[kind]
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(_trace(jax.jit(child)), 23)
    assert "jit body" in error.value.location
    assert error.value.category == "profile-exclusion"


def test_jit_symbolic_scope_identity_is_not_a_printed_name() -> None:
    (left,), (right,) = export.symbolic_shape("B"), export.symbolic_shape("B")
    graph = _trace(
        jax.jit(lambda x: x + x), [jax.ShapeDtypeStruct((left, 3), np.float32)]
    )
    eqn = graph.eqns[0]
    fake_result = Var(jax.core.ShapedArray((right, 3), np.float32))
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(_carrier(eqn.replace(outvars=[fake_result])), 23)
    assert error.value.category in {"lowering-defect", "profile-exclusion"}


def test_native_inline_and_keep_unused_hints_can_vary() -> None:
    fn = jax.jit(lambda x, y: -x, inline=True, keep_unused=True)
    graph = _trace(fn, [jax.ShapeDtypeStruct((2,), np.float32)] * 2)
    validate_source_jaxpr(graph, 23)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        actual = _trace(fn, [jax.ShapeDtypeStruct((2,), np.float32)] * 2)
    assert typed_trace_manifest(actual) == typed_trace_manifest(graph)


@pytest.mark.parametrize("value", ["AUTO", "auto", 0, 1, None])
def test_inline_rejects_non_native_representations(value: Any) -> None:
    from enum import Enum

    class Inline(Enum):
        AUTO = "auto"
        JAX_LATE = "jax_late"

    native = _trace().eqns[0]
    for candidate in (value, Inline.AUTO, Inline.JAX_LATE):
        equation = native.replace(params={**native.params, "inline": candidate})
        with pytest.raises(SourcePrecisionError) as error:
            validate_source_jaxpr(_carrier(equation), 23)
        assert error.value.category == "lowering-defect"


def test_inline_native_options_retain_the_exact_member() -> None:
    from jax2onnx._compat.jax import native_jit_boolean_inline_hints

    equation = _trace().eqns[0]
    for value in native_jit_boolean_inline_hints():
        updated = equation.replace(params={**equation.params, "inline": value})
        validate_source_jaxpr(_carrier(updated), 23)
        assert updated.params["inline"] is value
    # XLA-specific/early enum variants are not inferred from AUTO/JAX_LATE.
    from jax._src import api

    enum = getattr(api, "Inline", None)
    if enum is not None:
        for value in (enum.JAX_EARLY, enum.XLA_EARLY, enum.XLA_LATE):
            with pytest.raises(SourcePrecisionError):
                validate_source_jaxpr(
                    _carrier(
                        equation.replace(params={**equation.params, "inline": value})
                    ),
                    23,
                )


@pytest.mark.parametrize("mutation", ["count", "duplicate", "const_input_overlap"])
def test_closed_constant_counts_and_bindings_are_exact(mutation: str) -> None:
    first, second = np.asarray([1.0, 2.0], np.float32), np.asarray(
        [3.0, 4.0], np.float32
    )
    graph = _trace(jax.jit(lambda x: (x + first, x + second)))
    eqn = graph.eqns[0]
    closed = copy(eqn.params["jaxpr"])
    body = closed.jaxpr
    if mutation == "count":
        closed._consts = closed.consts[:1]
    elif mutation == "duplicate":
        closed = ClosedJaxpr(
            body.replace(constvars=[body.constvars[0], body.constvars[0]]),
            closed.consts,
        )
    else:
        closed = ClosedJaxpr(body.replace(invars=[body.constvars[0]]), closed.consts)
    corrupted = eqn.replace(params={**eqn.params, "jaxpr": closed})
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(_carrier(corrupted), 23)
    assert error.value.category == "lowering-defect"


def test_success_and_failure_preserve_all_effective_hidden_settings() -> None:
    originals = {
        name: getattr(jnp, name) for name in ("where", "exp", "add", "multiply")
    }
    with (
        jax.enable_x64(True),
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
        jax.default_matmul_precision("highest"),
    ):
        before = _globals(), _settings()
        for fn in (
            jax.jit(lambda x: jnp.sin(x)),
            jax.jit(lambda x: jax.lax.while_loop(lambda x: x > 0, lambda x: -x, x)),
        ):
            with pytest.raises(SourcePrecisionError):
                ui._to_onnx(
                    fn,
                    [jax.ShapeDtypeStruct((), np.float64)],
                    precision_policy=PrecisionPolicy.SOURCE,
                )
            assert (_globals(), _settings()) == before
            assert all(getattr(jnp, name) is value for name, value in originals.items())
        ui._to_onnx(
            jax.jit(lambda x: -x),
            [jax.ShapeDtypeStruct((), np.float64)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
        assert (_globals(), _settings()) == before
        assert all(getattr(jnp, name) is value for name, value in originals.items())


@pytest.mark.parametrize("location", ["body", "branch"])
def test_jit_rejects_high_forms_recursively(location: str) -> None:
    if location == "body":
        eqn = _trace().eqns[0]
        eqn = _change_body(eqn, is_high=True)
    else:

        def fn(x: Any, gate: Any) -> Any:
            return jax.lax.cond(gate, lambda x: -x, lambda x: x + x, x)

        eqn = _trace(
            jax.jit(fn),
            [jax.ShapeDtypeStruct((2,), np.float32), jax.ShapeDtypeStruct((), bool)],
        ).eqns[0]
        body = eqn.params["jaxpr"].jaxpr
        condition = next(e for e in body.eqns if e.primitive.name == "cond")
        branches = list(condition.params["branches"])
        branches[0] = ClosedJaxpr(
            branches[0].jaxpr.replace(is_high=True), branches[0].consts
        )
        updated = condition.replace(
            params={**condition.params, "branches": tuple(branches)}
        )
        eqn = _change_body(
            eqn, eqns=[updated if e is condition else e for e in body.eqns]
        )
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(_carrier(eqn), 23)
    assert error.value.category == "profile-exclusion"


def test_jit_rejects_reference_intermediates_with_tensor_shaped_metadata() -> None:
    from jax._src.state.types import AbstractRef

    equation = _trace().eqns[0]
    body = equation.params["jaxpr"].jaxpr
    reference = Var(AbstractRef(body.invars[0].aval))
    producer = body.eqns[0].replace(outvars=[reference])
    consumer = body.eqns[0].replace(invars=[reference])
    equation = _change_body(equation, eqns=[producer, consumer])
    with pytest.raises(SourcePrecisionError) as error:
        validate_source_jaxpr(_carrier(equation), 23)
    assert error.value.category == "profile-exclusion"
