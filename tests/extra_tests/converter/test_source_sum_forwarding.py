# tests/extra_tests/converter/test_source_sum_forwarding.py

"""Native Sum forwarding contracts; no export, admission or target acceptance."""

from collections.abc import Callable, Iterator
import hashlib
import inspect
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
from onnxruntime.capi import _pybind_state as ort_state
import pytest

from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy.sum import JnpSumPlugin
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)


_ORIGINAL_SUM = jnp.sum
_ORIGINALS = {name: getattr(jnp, name) for name in ("sum", "asarray", "array", "where")}
_SOURCE_PATHS = (Path(inspect.getfile(JnpSumPlugin)), Path(__file__))


def _globals() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


def _settings() -> tuple[Any, ...]:
    return tuple(getattr(jax.config, name) for name in _globals())


@pytest.fixture(autouse=True)
def _restore_sources_settings_callables_policy_and_deny_ort(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    hidden, effective = _globals(), _settings()
    sources = [hashlib.sha256(path.read_bytes()).digest() for path in _SOURCE_PATHS]
    attempts = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        attempts.append((args, kwargs))
        pytest.fail("Native Sum forwarding contracts deny every ORT constructor/run")

    session_type = ort.InferenceSession
    for name in (
        "__init__",
        "run",
        "run_async",
        "run_with_iobinding",
        "run_with_ort_values",
        "run_with_ortvaluevector",
    ):
        if hasattr(session_type, name):
            monkeypatch.setattr(session_type, name, forbidden)
    monkeypatch.setattr(ort_state, "InferenceSession", forbidden)
    try:
        with (
            jax.enable_x64(False),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        try:
            # Assert preservation before cleanup, so fixture repair cannot hide a leak.
            assert (_globals(), _settings()) == (hidden, effective)
            assert all(
                getattr(jnp, name) is original for name, original in _ORIGINALS.items()
            )
            assert active_precision_policy() is PrecisionPolicy.LEGACY
            assert [
                hashlib.sha256(path.read_bytes()).digest() for path in _SOURCE_PATHS
            ] == sources
            assert not attempts
        finally:
            for name, value in hidden.items():
                jax.config.update(name, value)


def _fresh(fn: Callable[..., Any]) -> Callable[..., Any]:
    def call(*args: Any) -> Any:
        return fn(*args)

    return call


def _payload(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value)
    return array.dtype.name, array.shape, array.tobytes()


def _compare(fn: Callable[..., Any], *values: Any) -> None:
    before = (_globals(), _settings())
    native = jax.make_jaxpr(_fresh(fn))(*values)
    eager = fn(*values)
    compiled = jax.jit(_fresh(fn))(*values)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(_fresh(fn))(*values)
        assert typed_trace_manifest(activated) == typed_trace_manifest(native)
        assert [_payload(value) for value in activated.consts] == [
            _payload(value) for value in native.consts
        ]
        for actual, expected in (
            (fn(*values), eager),
            (jax.jit(_fresh(fn))(*values), compiled),
        ):
            assert jax.tree.structure(actual) == jax.tree.structure(expected)
            assert [_payload(value) for value in jax.tree.leaves(actual)] == [
                _payload(value) for value in jax.tree.leaves(expected)
            ]
    assert (_globals(), _settings()) == before


@pytest.mark.parametrize(
    "x64,input_dtype,compute_dtype",
    [
        pytest.param(False, "float16", None, id="fp16_native"),
        pytest.param(False, "bfloat16", None, id="bf16_native"),
        pytest.param(False, "float32", None, id="fp32_native"),
        pytest.param(True, "float64", None, id="fp64_native"),
        pytest.param(False, "float16", "float32", id="fp16_to_fp32"),
        pytest.param(True, "float32", "float64", id="fp32_to_fp64"),
        pytest.param(True, "float64", "float32", id="fp64_to_fp32"),
    ],
)
def test_native_float_computation_types(
    x64: bool, input_dtype: str, compute_dtype: str | None
) -> None:
    with jax.enable_x64(x64):
        value = jnp.asarray(
            [[2048, 1, -2048, 0], [0, -0.0, 3.25, -1.25]], dtype=jnp.dtype(input_dtype)
        )
        dtype = None if compute_dtype is None else jnp.dtype(compute_dtype)
        _compare(lambda x: jnp.sum(x, axis=-1, dtype=dtype, keepdims=True), value)


@pytest.mark.parametrize(
    "x64,input_dtype,promote",
    [
        pytest.param(
            x64,
            dtype,
            promote,
            id=f"{dtype}_x{64 if x64 else 32}_{'promote' if promote else 'preserve'}",
        )
        for x64, dtype in (
            (False, "int8"),
            (False, "uint8"),
            (False, "int32"),
            (False, "uint32"),
            (False, "bool"),
            (True, "int8"),
            (True, "uint8"),
            (True, "int32"),
            (True, "uint32"),
            (True, "int64"),
            (True, "uint64"),
            (True, "bool"),
        )
        for promote in (False, True)
    ],
)
def test_native_integer_and_bool_promotion(
    x64: bool, input_dtype: str, promote: bool
) -> None:
    with jax.enable_x64(x64):
        values = [255, 1, 2, 3] if input_dtype.startswith("uint") else [127, 1, -128, 2]
        if input_dtype == "int32":
            values = [2**24 + 1, 1, -(2**24), 3]
        elif input_dtype == "uint32":
            values = [2**24 + 1, 1, 0, 3]
        elif input_dtype == "int64":
            values = [2**53 + 1, 1, -(2**53), 3]
        elif input_dtype == "uint64":
            values = [2**53 + 1, 1, 0, 3]
        if input_dtype == "bool":
            values = [True, False, True, True]
        value = jnp.asarray(values, dtype=jnp.dtype(input_dtype))
        _compare(lambda x: jnp.sum(x, promote_integers=promote), value)


@pytest.mark.parametrize("axis", [None, 0, -1, (0, 2), (), [0, 2]])
@pytest.mark.parametrize("keepdims", [False, True])
def test_native_axes_and_keepdims(axis: Any, keepdims: bool) -> None:
    value = jnp.asarray(np.arange(24).reshape(2, 3, 4), dtype=jnp.float32)
    _compare(lambda x: jnp.sum(x, axis=axis, keepdims=keepdims), value)


@pytest.mark.parametrize("shape,axis", [((0, 4), None), ((0, 4), 0), ((2, 0, 3), 1)])
@pytest.mark.parametrize("initial", [None, np.float32(0.25)])
def test_native_empty_reductions(
    shape: tuple[int, ...], axis: Any, initial: Any
) -> None:
    value = jnp.asarray(np.empty(shape, dtype=np.float32))
    _compare(lambda x: jnp.sum(x, axis=axis, initial=initial, keepdims=True), value)


def test_native_initial_where_and_captured_constants() -> None:
    with jax.enable_x64(True):
        value = jnp.asarray(
            [[2**24, 1, -(2**24), 0], [0, -0.0, 2, -2]], dtype=jnp.float64
        )
        constant = np.asarray([[0.125, -0.25, 0.5, -1]], dtype=np.float64)
        initial = jnp.asarray(0.25)
        assert initial.weak_type
        mask = jnp.asarray([[True, False, True, True], [False, True, True, False]])
        _compare(
            lambda x, start, selected: jnp.sum(
                x + constant, axis=-1, initial=start, where=selected, keepdims=True
            ),
            value,
            initial,
            mask,
        )


@pytest.mark.parametrize("x64,dtype", [(False, "complex64"), (True, "complex128")])
def test_native_complex_initial_and_mask(x64: bool, dtype: str) -> None:
    with jax.enable_x64(x64):
        value = jnp.asarray(
            [[2048 + 1j, 1 - 1j, -2048 + 2j, 0]], dtype=jnp.dtype(dtype)
        )
        mask = jnp.asarray([[True, False, True, True]])
        initial = np.asarray(0.25 + 0.5j, dtype=np.dtype(dtype))
        _compare(
            lambda x, selected: jnp.sum(x, axis=-1, initial=initial, where=selected),
            value,
            mask,
        )


@pytest.mark.parametrize(
    "mask_kind", ["none", "scalar_true", "scalar_false", "row", "column"]
)
@pytest.mark.parametrize("initial", [None, np.float32(0.25)])
def test_native_broadcast_masks_and_initial(mask_kind: str, initial: Any) -> None:
    value = jnp.asarray(np.arange(8).reshape(2, 4), dtype=jnp.float32)
    mask = None
    if mask_kind == "scalar_true":
        mask = jnp.asarray(True)
    elif mask_kind == "scalar_false":
        mask = jnp.asarray(False)
    elif mask_kind == "row":
        mask = jnp.asarray([True, False, True, False])
    elif mask_kind == "column":
        mask = jnp.asarray([[False], [True]])
    _compare(
        lambda x, selected: jnp.sum(
            x, axis=-1, initial=initial, where=selected, keepdims=True
        ),
        value,
        mask,
    )


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("promote", [False, True])
def test_native_explicit_integer_dtype_overrides_promotion(
    x64: bool, promote: bool
) -> None:
    with jax.enable_x64(x64):
        value = jnp.asarray([127, 1, -128, 2], dtype=jnp.int8)
        _compare(
            lambda x: jnp.sum(x, dtype=np.dtype("int16"), promote_integers=promote),
            value,
        )


@pytest.mark.parametrize("x64", [False, True])
def test_native_weak_scalar_and_omitted_defaults(x64: bool) -> None:
    with jax.enable_x64(x64):
        value = jnp.asarray(1.0)
        assert value.weak_type
        _compare(lambda x: jnp.sum(x), value)
        _compare(lambda x: jnp.sum(x, None, None, None, False, None, None, True), value)


@pytest.mark.parametrize("promotion", ["standard", "strict"])
@pytest.mark.parametrize("rank", ["allow", "raise"])
def test_native_dtype_and_rank_contexts(promotion: str, rank: str) -> None:
    value = jnp.asarray(np.arange(12).reshape(3, 4), dtype=jnp.float32)
    mask = jnp.asarray([True, False, True, False])
    with jax.numpy_dtype_promotion(promotion), jax.numpy_rank_promotion(rank):
        try:
            _ORIGINAL_SUM(value, axis=-1, where=mask)
        except (ValueError, TypeError) as failure:
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                with pytest.raises(type(failure)) as actual:
                    jnp.sum(value, axis=-1, where=mask)
            assert str(actual.value) == str(failure)
        else:
            _compare(
                lambda x, selected: jnp.sum(x, axis=-1, where=selected), value, mask
            )


def test_native_named_jit_body_and_constants() -> None:
    constant = np.asarray([0.125, -0.25, 0.5, -1], dtype=np.float32)

    @jax.jit
    @jax.named_call
    def source_sum_body(x: Any) -> Any:
        return jnp.sum(x + constant, axis=-1, keepdims=True)

    value = jnp.asarray(np.arange(12).reshape(3, 4), dtype=jnp.float32)
    _compare(source_sum_body, value)


def test_native_nondefault_matmul_and_x64_context() -> None:
    with (
        jax.enable_x64(True),
        jax.default_matmul_precision("highest"),
        jax.numpy_dtype_promotion("strict"),
        jax.numpy_rank_promotion("raise"),
    ):
        value = jnp.asarray([[2**24 + 1, 1, -(2**24), 3]], dtype=jnp.float64)
        _compare(lambda x: jnp.sum(x, axis=-1, initial=np.float64(0.125)), value)


@pytest.mark.parametrize("transform", ["grad", "jvp", "vmap", "transpose"])
def test_native_sum_transformations(transform: str) -> None:
    value = jnp.asarray(np.arange(12).reshape(3, 4), dtype=jnp.float32)
    if transform == "grad":
        fn = jax.grad(lambda x: jnp.sum(x * np.float32(0.25)))
    elif transform == "jvp":
        tangent = np.ones(value.shape, dtype=np.float32)

        def fn(x: Any) -> Any:
            return jax.jvp(lambda operand: jnp.sum(operand, axis=-1), (x,), (tangent,))

    elif transform == "vmap":
        fn = jax.vmap(lambda x: jnp.sum(x))
    else:
        value = jnp.asarray(0.25, dtype=jnp.float32)

        def fn(x: Any) -> Any:
            return jax.linear_transpose(
                lambda operand: jnp.sum(operand), jax.ShapeDtypeStruct((4,), np.float32)
            )(x)

    _compare(fn, value)


def test_dispatch_captures_original_arguments_omission_and_failure_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    a, axis, dtype, out, keepdims, initial, where, promote, result = (
        object() for _ in range(9)
    )
    failure = ValueError("native sum failure identity")

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        if kwargs.get("fail"):
            raise failure
        return result

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "SOURCE Sum must use its captured original before array/axis/default processing"
        )

    slot = "__orig_impl__sum"
    monkeypatch.setattr(JnpSumPlugin._PRIM, slot, None, raising=False)
    monkeypatch.setattr(JnpSumPlugin._PRIM, "bind", forbidden)
    spec = next(
        s for s in JnpSumPlugin.binding_specs() if isinstance(s, MonkeyPatchSpec)
    )
    wrapper = spec.make_value(original)
    other_calls = []
    other_result = object()

    def other_original(*args: Any, **kwargs: Any) -> Any:
        other_calls.append((args, kwargs))
        return other_result

    other_wrapper = spec.make_value(other_original)
    monkeypatch.setattr(JnpSumPlugin._PRIM, slot, forbidden)
    variants = [
        ((a,), {}),
        ((), {"a": a}),
        ((a, axis, dtype, out, keepdims, initial, where, promote), {}),
        (
            (a,),
            {
                "axis": None,
                "dtype": None,
                "out": None,
                "keepdims": False,
                "initial": None,
                "where": None,
                "promote_integers": True,
            },
        ),
        ((), {"fail": True}),
    ]
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        for args, kwargs in variants[:-1]:
            assert wrapper(*args, **kwargs) is result
        with pytest.raises(ValueError) as caught:
            wrapper(*variants[-1][0], **variants[-1][1])
        assert other_wrapper(a) is other_result
    assert caught.value is failure
    assert calls == variants
    assert other_calls == [((a,), {})]


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "keyword",
        "out",
        "axis",
        "duplicate_axes",
        "initial_shape",
        "where_shape",
        "dtype",
        "axis_type",
        "extra_positional",
    ],
)
def test_native_api_errors(kind: str) -> None:
    value = jnp.asarray(np.arange(12).reshape(3, 4), dtype=jnp.float32)
    args, kwargs = (value,), {}
    if kind == "missing":
        args = ()
    elif kind == "keyword":
        kwargs = {"unknown": object()}
    elif kind == "out":
        kwargs = {"out": object()}
    elif kind == "axis":
        kwargs = {"axis": 3}
    elif kind == "duplicate_axes":
        kwargs = {"axis": (0, 0)}
    elif kind == "initial_shape":
        kwargs = {"initial": jnp.asarray([1, 2], dtype=jnp.float32)}
    elif kind == "where_shape":
        kwargs = {"where": jnp.asarray([True, False, True])}
    elif kind == "dtype":
        kwargs = {"dtype": "not-a-dtype"}
    elif kind == "axis_type":
        kwargs = {"axis": "not-an-axis"}
    else:
        args = (value, None, None, None, False, None, None, True, None)
    with pytest.raises(Exception) as original:
        _ORIGINAL_SUM(*args, **kwargs)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        with pytest.raises(type(original.value)) as activated:
            jnp.sum(*args, **kwargs)
    assert str(activated.value) == str(original.value)


def test_active_signature_matches_original() -> None:
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        assert inspect.signature(jnp.sum) == inspect.signature(_ORIGINAL_SUM)
