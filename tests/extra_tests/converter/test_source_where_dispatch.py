# tests/extra_tests/converter/test_source_where_dispatch.py

"""Native Where traces, finite product admission, and strict target evidence."""

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy.where import JnpWherePlugin
from jax2onnx.precision import (
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    validate_source_samples,
)
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

# Frozen before target execution: exact selection is not an accumulation.
_BITWISE = SampleCriteria("bitwise")
_NAN_CLASS = SampleCriteria("exact", equal_nan=True)
_ORIGINAL = jnp.where


@pytest.fixture(autouse=True)
def _restore_state() -> Iterator[None]:
    hidden, effective = _globals(), _settings()
    try:
        with (
            jax.enable_x64(False),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in hidden.items():
            jax.config.update(name, value)
        assert (_globals(), _settings()) == (hidden, effective)
        assert jnp.where is _ORIGINAL
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _spec(value: Any, *, weak: bool = False) -> Any:
    value = np.asarray(value)
    return jax.ShapeDtypeStruct(value.shape, value.dtype, weak_type=weak)


@onnx_function(name="SourceNativeWhere")
def _named(c: Any, x: Any, y: Any) -> Any:
    return jnp.where(c, x, y)


@onnx_function(name="SourceNestedNativeWhere")
def _nested(c: Any, x: Any, y: Any) -> Any:
    return _named(c, x, y)


@onnx_function(name="SourceWhereCondCarrier")
def _cond(c: Any, x: Any, y: Any, gate: Any) -> Any:
    return jax.lax.cond(gate, lambda: _nested(c, x, y), lambda: _nested(c, y, x))


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("promotion", ["standard", "strict"])
@pytest.mark.parametrize("rank", ["allow", "raise"])
@pytest.mark.parametrize(
    "dtype", [np.bool_, np.float16, jnp.bfloat16, np.float32, np.float64]
)
@pytest.mark.parametrize("weak", [False, True])
def test_native_where_complete_trace_and_weakness(
    x64: bool, promotion: str, rank: str, dtype: Any, weak: bool
) -> None:
    with (
        jax.enable_x64(x64),
        jax.numpy_dtype_promotion(promotion),
        jax.numpy_rank_promotion(rank),
    ):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        specs = [
            jax.ShapeDtypeStruct((2,), np.bool_, weak_type=weak),
            *[jax.ShapeDtypeStruct((2,), dtype, weak_type=weak) for _ in range(2)],
        ]

        def fn(c: Any, x: Any, y: Any) -> Any:
            return jnp.where(c, x, y)

        native = jax.make_jaxpr(fn)(*specs)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(fn)(*specs)
        assert typed_trace_manifest(activated) == typed_trace_manifest(native)
        assert _body(native).eqns[0].primitive.name == "jit"
        validate_source_jaxpr(_body(activated), 23)


@pytest.mark.parametrize("dtype", [np.bool_, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("kind", ["scalar", "equal", "broadcast", "empty"])
def test_static_selection_has_exact_runtime_controls(dtype: Any, kind: str) -> None:
    with jax.enable_x64(dtype is np.float64):
        shape = () if kind == "scalar" else (0, 3) if kind == "empty" else (2, 3)
        c = (
            np.asarray(True)
            if kind == "scalar"
            else np.ones((2, 1), bool) if kind == "broadcast" else np.ones(shape, bool)
        )
        x = np.full((1, 3) if kind == "broadcast" else shape, 1, dtype)
        y = (
            np.asarray(0 if dtype is np.bool_ else -2, dtype)
            if kind == "broadcast"
            else np.full(shape, 0 if dtype is np.bool_ else -2, dtype)
        )
        feeds = ((c, x, y), (~c, x, y))
        _, proto, _, refs = _artifact(
            lambda c, x, y: jnp.where(c, x, y), list(map(_spec, feeds[0])), feeds=feeds
        )
        _accept(proto, feeds, refs, _BITWISE)
        ops = [node.op_type for node in proto.graph.node]
        if kind != "empty":
            assert ("Where" in ops) is (dtype is not np.bool_)
            if dtype is np.bool_:
                assert all(op in ops for op in ("Not", "And", "Or"))


@pytest.mark.parametrize("mode", ["ir", "proto", "file"])
@pytest.mark.parametrize("opset", [23, 26])
def test_where_return_modes_and_opsets(mode: str, opset: int, tmp_path: Any) -> None:
    feeds = (
        (
            np.asarray([True, False]),
            np.asarray([1, 2], np.float32),
            np.asarray([-1, -2], np.float32),
        ),
    )
    _, proto, _, refs = _artifact(
        lambda c, x, y: jnp.where(c, x, y),
        list(map(_spec, feeds[0])),
        feeds=feeds,
        mode=mode,
        opset=opset,
        path=tmp_path / "where.onnx" if mode == "file" else None,
    )
    assert (
        next(item.version for item in proto.opset_import if item.domain == "") == opset
    )
    _accept(proto, feeds, refs, _BITWISE)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numeric_condition_zero_nan_and_infinity(dtype: Any) -> None:
    with jax.enable_x64(dtype is np.float64):
        c = np.asarray([0, -0.0, np.nan, np.inf, -np.inf, -1], dtype)
        x = np.asarray([1, 2, 3, 4, 5, 6], dtype)
        y = -x
        feeds = ((c, x, y),)
        _, proto, _, refs = _artifact(
            lambda c, x, y: jnp.where(c, x, y), list(map(_spec, feeds[0])), feeds=feeds
        )
        assert {node.op_type for node in proto.graph.node} >= {"Equal", "Not", "Where"}
        _accept(proto, feeds, refs, _BITWISE)


@pytest.mark.parametrize("scope", ["jit", "nested", "cond"])
def test_where_runtime_boolean_dependencies_in_scopes(scope: str) -> None:
    def target(c: Any, x: Any, y: Any, gate: Any) -> Any:
        if scope == "jit":
            return jax.jit(lambda c, x, y: jnp.where(c, x, y))(c, x, y)
        if scope == "nested":
            return _nested(c, x, y)
        return _cond(c, x, y, gate)

    args = (
        np.asarray([True, False]),
        np.asarray([True, True]),
        np.asarray([False, False]),
        np.asarray(True),
    )
    feeds = (
        args,
        (args[0], args[1], args[2], np.asarray(False)),
        (~args[0], args[1], args[2], args[3]),
    )
    _, proto, _, refs = _artifact(
        target, list(map(_spec, args)), feeds=feeds, named=scope != "jit"
    )
    _accept(proto, feeds, refs, _BITWISE)
    assert len(proto.graph.input) == 4


def test_source_binder_delegates_exactly_once_before_validation() -> None:
    calls = []
    marker = object()

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return marker

    binding = next(
        spec
        for spec in JnpWherePlugin.binding_specs()
        if isinstance(spec, MonkeyPatchSpec)
    )
    storage = "__orig_impl__where"
    previous = getattr(JnpWherePlugin._PRIM, storage, None)
    try:
        patched = binding.make_value(original)
        with precision_policy_scope(PrecisionPolicy.SOURCE):
            assert patched(None, size=2, unknown=marker) is marker
        assert calls == [((None,), {"size": 2, "unknown": marker})]
    finally:
        if previous is None:
            delattr(JnpWherePlugin._PRIM, storage)
        else:
            setattr(JnpWherePlugin._PRIM, storage, previous)


@pytest.mark.parametrize(
    "case", ["one", "none", "missing", "keyword", "unknown", "size", "incompatible"]
)
def test_original_errors_are_preserved(case: str) -> None:
    mask = np.asarray([True, False])
    args, kwargs = {
        "one": ((mask,), {}),
        "none": ((mask, None, 1.0), {}),
        "missing": ((mask, 1.0), {}),
        "keyword": ((), {"condition": mask, "x": 1.0, "y": 2.0}),
        "unknown": ((mask, 1.0, 2.0), {"unknown": True}),
        "size": ((mask, 1.0, 2.0), {"size": 2}),
        "incompatible": ((mask, np.ones(3), np.ones(4)), {}),
    }[case]

    def call() -> Any:
        return jnp.where(*args, **kwargs)

    if case == "one":
        native = jax.make_jaxpr(lambda c: _ORIGINAL(c, size=2))(_spec(mask))
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            actual = jax.make_jaxpr(lambda c: jnp.where(c, size=2))(_spec(mask))
        assert typed_trace_manifest(native) == typed_trace_manifest(actual)
        return
    with pytest.raises(Exception) as native_error:
        call()
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        with pytest.raises(type(native_error.value)) as source_error:
            call()
    assert str(source_error.value) == str(native_error.value)


def test_integer_where_accepts_original_int32_signature() -> None:
    values = np.asarray([2147483646, 2147483647], np.int32)
    feeds = [(np.asarray(mask), values) for mask in ([False, True], [True, False])]
    _, proto, _, references = _artifact(
        lambda c, x: jnp.where(c, x, x),
        [jax.ShapeDtypeStruct((2,), bool), jax.ShapeDtypeStruct((2,), np.int32)],
        feeds=feeds,
    )
    _accept(proto, feeds, references, _BITWISE)


def test_complex_where_remains_excluded() -> None:
    with pytest.raises(SourcePrecisionError) as error:
        ui._to_onnx(
            lambda c, x: jnp.where(c, x, x),
            [
                jax.ShapeDtypeStruct((2,), bool),
                jax.ShapeDtypeStruct((2,), np.complex64),
            ],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert error.value.category == "profile-exclusion"


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_where_negative_zero_retains_strict_failure_gate(
    dtype: Any, record_property: Any
) -> None:
    with jax.enable_x64(dtype is np.float64):
        args = (
            np.asarray([True] * 4),
            np.asarray([-0.0, 0.0, -0.0, -1.0], dtype),
            np.asarray([1.0, -1.0, 2.0, 1.0], dtype),
        )
        _, proto, _, refs = _artifact(
            lambda c, x, y: jnp.where(c, x, y), list(map(_spec, args)), feeds=(args,)
        )
        for optimization in ("disabled", "all"):
            evidence = validate_source_samples(
                proto,
                samples=[SourceSample(args, refs[0][0], _BITWISE)],
                optimization=optimization,
            )
            record_property(f"negative_zero_{optimization}", evidence.status.value)
            record_property(f"artifact_sha256_{optimization}", evidence.artifact_sha256)
            if evidence.status is CheckStatus.VALIDATED:
                evidence.require_sample_acceptance()
            else:
                # Pin the known sign-only failure. Any other numerical failure
                # must fail this regression rather than being blanket expected.
                assert evidence.status is CheckStatus.FAILED
                expected_target = args[1].copy()
                expected_target[expected_target == 0] = dtype(0)
                validate_source_samples(
                    proto,
                    samples=[SourceSample(args, [expected_target], _BITWISE)],
                    optimization=optimization,
                ).require_sample_acceptance()
                with pytest.raises(SampleAcceptanceError):
                    evidence.require_sample_acceptance()


def test_special_selection_and_captured_payloads_stay_separate() -> None:
    captured = np.asarray([np.nan, np.inf, -np.inf, -0.0], np.float64)
    original = captured.tobytes()
    with jax.enable_x64(True):
        args = (
            np.asarray([True, True, True, False]),
            np.asarray([1.0, 2.0, 3.0, 4.0], np.float64),
        )
        _, proto, _, refs = _artifact(
            lambda c, y: jnp.where(c, captured, y),
            list(map(_spec, args)),
            feeds=(args,),
        )
        _accept(proto, (args,), refs, _NAN_CLASS)
        constants = [
            onnx.numpy_helper.to_array(value) for value in proto.graph.initializer
        ]
        assert any(
            array.dtype == captured.dtype
            and array.shape == captured.shape
            and array.tobytes() == original
            for array in constants
        )
        assert captured.tobytes() == original


def test_bfloat16_artifact_has_typed_evidence_without_kernel_claim() -> None:
    values = (
        np.asarray([True, False]),
        np.asarray([1.0, 2.0], jnp.bfloat16),
        np.asarray([3.0, 4.0], jnp.bfloat16),
    )
    _, proto, report, _ = _artifact(
        lambda c, x, y: jnp.where(c, x, y), list(map(_spec, values)), feeds=(values,)
    )
    assert proto.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
    assert {check.category: check.status for check in report.checks}[
        "runtime-execution"
    ] is CheckStatus.UNVERIFIED


@pytest.mark.parametrize("promotion", ["dtype", "rank"])
def test_native_strict_promotion_errors_are_forwarded(promotion: str) -> None:
    specs = [
        jax.ShapeDtypeStruct((2, 3), np.float32),
        jax.ShapeDtypeStruct((3,), np.float64 if promotion == "dtype" else np.float32),
    ]

    def fn(x: Any, y: Any) -> Any:
        return jnp.where(True, x, y)

    with (
        jax.enable_x64(True),
        jax.numpy_dtype_promotion("strict" if promotion == "dtype" else "standard"),
        jax.numpy_rank_promotion("raise" if promotion == "rank" else "allow"),
    ):
        if promotion == "rank":
            # Native Where explicitly broadcasts its arms and accepts this
            # shape pair even under rank-promotion='raise'. Preserve that path.
            native = jax.make_jaxpr(fn)(*specs)
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                activated = jax.make_jaxpr(fn)(*specs)
            assert typed_trace_manifest(activated) == typed_trace_manifest(native)
            validate_source_jaxpr(_body(activated), 23)
            return
        with pytest.raises(Exception) as original:
            jax.make_jaxpr(fn)(*specs)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            with pytest.raises(type(original.value)) as activated:
                jax.make_jaxpr(fn)(*specs)
        assert str(activated.value) == str(original.value)


def _leaf_graph(name: str, inputs: Any, output: Any, params: Any = None) -> Any:
    from types import SimpleNamespace
    from jax2onnx._compat.jax import Var

    invars = [
        Var(jax.core.ShapedArray(shape, dtype, weak_type=weak))
        for dtype, shape, weak in inputs
    ]
    outvar = Var(jax.core.ShapedArray(output[1], output[0], weak_type=output[2]))
    eqn = SimpleNamespace(
        primitive=SimpleNamespace(name=name),
        invars=invars,
        outvars=[outvar],
        params=params or {},
        effects=(),
    )
    return SimpleNamespace(
        invars=invars, constvars=[], outvars=[outvar], eqns=[eqn], effects=()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "arity",
        "selector",
        "arm_dtype",
        "arm_shape",
        "output_dtype",
        "output_shape",
        "weakness",
        "params",
    ],
)
def test_select_native_type_shape_weakness_and_schema_are_strict(mutation: str) -> None:
    graph = _leaf_graph(
        "select_n",
        [(bool, (), False), (np.float32, (2,), False), (np.float32, (2,), False)],
        (np.float32, (2,), False),
    )
    eqn = graph.eqns[0]
    if mutation == "arity":
        eqn.invars.pop()
    elif mutation == "params":
        eqn.params["sharding"] = None
    else:
        from jax2onnx._compat.jax import Var

        index = {"selector": 0, "arm_dtype": 2, "arm_shape": 2}.get(mutation)
        dtype = np.float32 if mutation == "selector" else np.float16
        shape = (3,) if mutation in {"arm_shape", "output_shape"} else (2,)
        aval = jax.core.ShapedArray(
            shape,
            (
                np.float32
                if mutation in {"arm_shape", "output_shape", "weakness"}
                else dtype
            ),
            weak_type=mutation == "weakness",
        )
        if index is not None:
            old, replacement = eqn.invars[index], Var(aval)
            graph.invars[graph.invars.index(old)] = replacement
            eqn.invars[index] = replacement
        else:
            replacement = Var(aval)
            eqn.outvars = [replacement]
            graph.outvars = [replacement]
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(graph, 23)


@pytest.mark.parametrize(
    "name,params",
    [
        (
            "broadcast_in_dim",
            {"shape": (2,), "broadcast_dimensions": (), "sharding": object()},
        ),
        ("broadcast_in_dim", {"shape": (2,), "broadcast_dimensions": (0,)}),
        ("convert_element_type", {"new_dtype": np.dtype(np.bool_), "weak_type": 0}),
        (
            "convert_element_type",
            {"new_dtype": np.dtype(np.bool_), "weak_type": False, "sharding": object()},
        ),
    ],
)
def test_boolean_support_does_not_widen_conversion_or_placement(
    name: str, params: Any
) -> None:
    output = (bool, (2,) if name == "broadcast_in_dim" else (), False)
    graph = _leaf_graph(name, [(bool, (), False)], output, params)
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(graph, 23)


def test_boolean_to_float_accepts_original_explicit_conversion() -> None:
    graph = _leaf_graph(
        "convert_element_type",
        [(bool, (), False)],
        (np.float32, (), False),
        {"new_dtype": np.dtype(np.float32), "weak_type": False},
    )
    validate_source_jaxpr(graph, 23)
    feeds = [(np.asarray(value),) for value in (False, True)]
    _, proto, _, references = _artifact(
        lambda c: jax.lax.convert_element_type(c, np.float32),
        [jax.ShapeDtypeStruct((), bool)],
        feeds=feeds,
    )
    _accept(proto, feeds, references, _BITWISE)


def test_boolean_to_complex_conversion_remains_excluded() -> None:
    graph = _leaf_graph(
        "convert_element_type",
        [(bool, (), False)],
        (np.complex64, (), False),
        {"new_dtype": np.dtype(np.complex64), "weak_type": False},
    )
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(graph, 23)


def test_symbolic_boolean_boundary_accepts_original_same_batch_scope() -> None:
    from jax import export

    (batch,) = export.symbolic_shape("B")
    feeds = [(np.arange(size * 3).reshape(size, 3) % 2 == 0,) for size in (2, 5)]
    _, proto, _, references = _artifact(
        lambda c: c,
        [jax.ShapeDtypeStruct((batch, 3), bool)],
        feeds=feeds,
    )
    _accept(proto, feeds, references, _BITWISE)


def test_symbolic_complex_boundary_remains_excluded() -> None:
    from jax import export

    (batch,) = export.symbolic_shape("B")
    with pytest.raises(SourcePrecisionError) as error:
        ui._to_onnx(
            lambda c: c,
            [jax.ShapeDtypeStruct((batch, 3), np.complex64)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    assert error.value.phase == "input" and error.value.category == "profile-exclusion"
