# tests/extra_tests/converter/test_source_model_native_bindings.py

"""Exact native leaf forwarding; full original model gates have a separate owner."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from flax import nnx
import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
from onnxruntime.capi import _pybind_state as ort_state
import pytest

from jax2onnx._compat.jax import (
    _native_context_fields,
    native_context_snapshot,
    native_transformed_execution_context,
    require_native_context_restored,
)
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec, apply_patches
from jax2onnx.plugins.flax.nnx.dot_product_attention import (
    DotProductAttentionPlugin as NnxAttention,
)
from jax2onnx.plugins.flax.nnx.embed import EmbedPlugin
from jax2onnx.plugins.flax.nnx.log_softmax import LogSoftmaxPlugin as NnxLogSoftmax
from jax2onnx.plugins.jax.nn.dot_product_attention import (
    DotProductAttentionPlugin as JaxAttention,
)
from jax2onnx.plugins.jax.numpy.take import JnpTakePlugin
from jax2onnx.plugins.jax.numpy.concatenate import JnpConcatenatePlugin
from jax2onnx.plugins.jax.numpy.split import JnpSplitPlugin
from jax2onnx.plugins.jax.numpy.tile import JnpTilePlugin
from jax2onnx.plugins.jax.numpy.transpose import JnpTransposePlugin
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.source_precision_support import (
    SourceSettings,
    _global_source_configuration,
)

_PLUGINS = (
    EmbedPlugin,
    NnxAttention,
    JnpTransposePlugin,
    JnpTakePlugin,
    JaxAttention,
    JnpTilePlugin,
    JnpSplitPlugin,
    JnpConcatenatePlugin,
    NnxLogSoftmax,
)
_ORIGINALS = (
    nnx.Embed.__call__,
    nnx.dot_product_attention,
    jnp.transpose,
    jnp.take,
    jax.nn.dot_product_attention,
    jnp.tile,
    jnp.split,
    jnp.concatenate,
    nnx.log_softmax,
)
_PATHS = tuple(Path(inspect.getfile(plugin)) for plugin in _PLUGINS)


def _payload(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value)
    return (
        array.dtype.str,
        array.shape,
        bool(getattr(value, "weak_type", False)),
        array.tobytes(),
    )


def _same(actual: Any, expected: Any) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    assert [_payload(x) for x in jax.tree.leaves(actual)] == [
        _payload(x) for x in jax.tree.leaves(expected)
    ]


def _fresh(fn: Callable[..., Any]) -> Callable[..., Any]:
    def call(*args: Any) -> Any:
        return fn(*args)

    return call


def _contexts(closed: Any) -> list[Any]:
    inner = getattr(closed, "jaxpr", None)
    body = inner if inner is not None and inner is not closed else closed
    if not hasattr(body, "eqns"):
        return []
    result = []
    for equation in body.eqns:
        result.append(_native_context_fields(equation.ctx))
        for value in equation.params.values():
            if hasattr(value, "jaxpr") or hasattr(value, "eqns"):
                result.extend(_contexts(value))
            elif isinstance(value, (tuple, list)):
                for item in value:
                    if hasattr(item, "jaxpr") or hasattr(item, "eqns"):
                        result.extend(_contexts(item))
    return result


@contextmanager
def _activated() -> Iterator[None]:
    with (
        precision_policy_scope(PrecisionPolicy.SOURCE),
        apply_patches([spec for plugin in _PLUGINS for spec in plugin.binding_specs()]),
    ):
        yield


@pytest.fixture(autouse=True)
def _restore_exact_owners_sources_settings_and_deny_ort(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    config, settings = _global_source_configuration(), SourceSettings.observe()
    pins = [hashlib.sha256(path.read_bytes()).hexdigest() for path in _PATHS]
    attempts = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        attempts.append((args, kwargs))
        pytest.fail("Scoped native binding contracts forbid ORT construction/execution")

    for name in (
        "__init__",
        "run",
        "run_async",
        "run_with_iobinding",
        "run_with_ort_values",
        "run_with_ortvaluevector",
    ):
        if hasattr(ort.InferenceSession, name):
            monkeypatch.setattr(ort.InferenceSession, name, forbidden)
    monkeypatch.setattr(ort_state, "InferenceSession", forbidden)

    def excluded_carrier(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "SOURCE forwarding must not bind any of the excluded custom carriers"
        )

    for plugin in _PLUGINS:
        monkeypatch.setattr(plugin._PRIM, "bind", excluded_carrier)
    try:
        with native_transformed_execution_context():
            native = native_context_snapshot()
            yield
            require_native_context_restored(native)
    finally:
        # Check before repair so cleanup cannot hide owner/config mutation.
        try:
            assert (_global_source_configuration(), SourceSettings.observe()) == (
                config,
                settings,
            )
            assert (
                nnx.Embed.__call__,
                nnx.dot_product_attention,
                jnp.transpose,
                jnp.take,
                jax.nn.dot_product_attention,
                jnp.tile,
                jnp.split,
                jnp.concatenate,
                nnx.log_softmax,
            ) == _ORIGINALS
            assert active_precision_policy() is PrecisionPolicy.LEGACY
            assert [
                hashlib.sha256(path.read_bytes()).hexdigest() for path in _PATHS
            ] == pins
            assert not attempts
        finally:
            for name, value in config.items():
                jax.config.update(name, value)


def _compare(
    fn: Callable[..., Any],
    *values: Any,
    execute: bool = True,
    corrupt: Callable[[], AbstractContextManager[None]] | None = None,
) -> Any:
    before = (_global_source_configuration(), SourceSettings.observe())
    native = jax.make_jaxpr(_fresh(fn))(*values)
    if execute:
        eager = fn(*values)
        wrapper = jax.jit(_fresh(fn))
        first, warm = wrapper(*values), wrapper(*values)
    with _activated(), nullcontext() if corrupt is None else corrupt():
        activated = jax.make_jaxpr(_fresh(fn))(*values)
        assert typed_trace_manifest(activated) == typed_trace_manifest(native)
        assert _contexts(activated) == _contexts(native)
        assert [_payload(v) for v in activated.consts] == [
            _payload(v) for v in native.consts
        ]
        if execute:
            actual_eager = fn(*values)
            actual_wrapper = jax.jit(_fresh(fn))
            actual_first, actual_warm = actual_wrapper(*values), actual_wrapper(*values)
            for actual, expected in (
                (actual_eager, eager),
                (actual_first, first),
                (actual_warm, warm),
            ):
                _same(actual, expected)
    assert (_global_source_configuration(), SourceSettings.observe()) == before
    return native


@pytest.mark.parametrize(
    "dtype,compute",
    [
        ("float16", None),
        ("float32", None),
        ("float32", "float64"),
        ("float64", "float32"),
    ],
)
def test_embedding_integer_indices_storage_and_compute_dtype(
    dtype: str, compute: str | None
) -> None:
    with jax.enable_x64(True):
        layer = nnx.Embed(
            7,
            4,
            dtype=None if compute is None else jnp.dtype(compute),
            param_dtype=jnp.dtype(dtype),
            rngs=nnx.Rngs(17),
        )
        layer.embedding[...] = jnp.asarray(
            np.linspace(-2, 2, 28).reshape(7, 4), dtype=jnp.dtype(dtype)
        )
        table = _payload(layer.embedding[...])
        indices = jnp.asarray([[0, 6, -1], [2, 0, 3]], dtype=jnp.int32)
        _compare(lambda x: layer(inputs=x, out_sharding=None), indices)
        assert _payload(layer.embedding[...]) == table
        assert indices.dtype == jnp.int32 and not indices.weak_type


@pytest.mark.parametrize("count", [1, 7])
def test_embedding_symbolic_batch_and_original_single_table_case(count: int) -> None:
    layer = nnx.Embed(count, 4, rngs=nnx.Rngs(9))
    b = jax_export.symbolic_shape("B")[0]
    trace = _compare(
        lambda x: layer(x), jax.ShapeDtypeStruct((b, 3), np.int32), execute=False
    )
    assert str(trace.out_avals[0].shape[0]) == "B"
    assert trace.out_avals[0].dtype == np.dtype("float32")


@pytest.mark.parametrize("mode", [None, "fill", "clip", "wrap"])
@pytest.mark.parametrize("dtype", ["int32", "float32", "float64"])
def test_take_original_index_oob_nan_modes_and_fill(
    mode: str | None, dtype: str
) -> None:
    with jax.enable_x64(True):
        x = jnp.asarray([[0, -0.0, 2], [3, 4, 5]], dtype=jnp.dtype(dtype))
        indices = jnp.asarray([-4, -1, 0, 1, 2, 7], dtype=jnp.int32)
        _compare(lambda a, i: jnp.take(a, i, axis=0, mode=mode), x, indices)


@pytest.mark.parametrize("axes", [None, (0, 2, 1), (-1, 0, 1), [2, 1, 0]])
def test_transpose_native_axes_payload_and_symbolic_provenance(axes: Any) -> None:
    with jax.enable_x64(True):
        value = jnp.asarray(np.arange(24, dtype=np.float64).reshape(2, 3, 4))
        _compare(lambda x: jnp.transpose(a=x, axes=axes), value)
        b = jax_export.symbolic_shape("B")[0]
        _compare(
            lambda x: jnp.transpose(a=x, axes=axes),
            jax.ShapeDtypeStruct((b, 3, 4), np.float64),
            execute=False,
        )


@pytest.mark.parametrize("api", ["nnx", "jax"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_attention_native_geometry_mask_bias_precision_and_symbolic_batch(
    api: str, dtype: str
) -> None:
    # FP64 here is exact native-versus-forwarded parity, not CPU numerical acceptance.
    with jax.enable_x64(True), jax.default_matmul_precision("highest"):
        q = jnp.asarray(
            np.linspace(-0.5, 0.5, 2 * 3 * 4 * 8).reshape(2, 3, 4, 8),
            dtype=jnp.dtype(dtype),
        )
        k = jnp.asarray(
            np.linspace(-0.4, 0.6, 2 * 5 * 2 * 8).reshape(2, 5, 2, 8),
            dtype=jnp.dtype(dtype),
        )
        v = jnp.asarray(
            np.linspace(-0.7, 0.3, 2 * 5 * 2 * 8).reshape(2, 5, 2, 8),
            dtype=jnp.dtype(dtype),
        )
        bias = jnp.asarray(
            np.linspace(-0.1, 0.2, 4 * 3 * 5).reshape(1, 4, 3, 5),
            dtype=jnp.dtype(dtype),
        )
        mask = jnp.asarray(np.arange(3 * 5).reshape(1, 1, 3, 5) % 3 != 0)

        def call(query: Any, key: Any, value: Any, b: Any, m: Any) -> Any:
            if api == "nnx":
                # Actual native positional order is bias, mask; LEGACY order is unchanged.
                return nnx.dot_product_attention(
                    query,
                    key,
                    value,
                    b,
                    m,
                    deterministic=True,
                    dropout_rate=0.0,
                    dtype=jnp.dtype(dtype),
                    precision=jax.lax.Precision.HIGHEST,
                )
            return jax.nn.dot_product_attention(
                query,
                key,
                value,
                b,
                m,
                scale=0.375,
                is_causal=False,
                implementation="xla",
            )

        _compare(call, q, k, v, bias, mask)
        b = jax_export.symbolic_shape("B")[0]
        _compare(
            call,
            jax.ShapeDtypeStruct((b, 3, 4, 8), jnp.dtype(dtype)),
            jax.ShapeDtypeStruct((b, 5, 2, 8), jnp.dtype(dtype)),
            jax.ShapeDtypeStruct((b, 5, 2, 8), jnp.dtype(dtype)),
            bias,
            mask,
            execute=False,
        )


@pytest.mark.parametrize("plugin", _PLUGINS)
def test_forwarding_preserves_exact_original_arguments_and_return(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = [s for s in plugin.binding_specs() if isinstance(s, MonkeyPatchSpec)]
    assert len(specs) == 1
    args, token = (object(), object(), object(), object(), object()), object()
    kwargs = {
        "inputs": object(),
        "out_sharding": object(),
        "precision": object(),
        "axes": object(),
        "mode": object(),
        "unique_indices": object(),
    }
    calls = []

    def original(*actual: Any, **named: Any) -> Any:
        calls.append((actual, named))
        return token

    with monkeypatch.context() as cleanup:
        if hasattr(plugin, "_ORIG_CALL"):
            cleanup.setattr(plugin, "_ORIG_CALL", plugin._ORIG_CALL)
        for slot in (
            "__orig_impl__take",
            "__orig_impl___transpose",
            "__orig_impl__tile",
            "__orig_impl__split",
            "__orig_impl__concatenate",
        ):
            cleanup.setattr(
                plugin._PRIM, slot, getattr(plugin._PRIM, slot, None), raising=False
            )
        patched = specs[0].make_value(original)
        with precision_policy_scope(PrecisionPolicy.SOURCE):
            assert patched(*args, **kwargs) is token
    assert len(calls) == 1
    assert all(a is b for a, b in zip(calls[0][0], args, strict=True))
    assert calls[0][1].keys() == kwargs.keys()
    assert all(calls[0][1][k] is v for k, v in kwargs.items())


def test_wrong_transpose_consumer_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _ORIGINALS[2]

    @contextmanager
    def corrupt() -> Iterator[None]:
        # Corruption unwinds while the activated wrapper still owns this slot.
        with monkeypatch.context() as temporary:
            temporary.setattr(
                jnp, "transpose", lambda a, axes=None: original(a, (0, 2, 1))
            )
            yield

    value = jnp.asarray(np.arange(24).reshape(2, 3, 4), dtype=jnp.float32)
    with pytest.raises(AssertionError):
        _compare(lambda x: jnp.transpose(x, (2, 0, 1)), value, corrupt=corrupt)


def test_wrong_forwarding_result_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _ORIGINALS[3]

    @contextmanager
    def corrupt() -> Iterator[None]:
        with monkeypatch.context() as temporary:
            temporary.setattr(
                jnp, "take", lambda a, i, axis=None: original(a, i + 1, axis=axis)
            )
            yield

    value = jnp.asarray(np.arange(12).reshape(3, 4), dtype=jnp.float32)
    indices = jnp.asarray([0, 1], dtype=jnp.int32)
    with pytest.raises(AssertionError):
        _compare(lambda x, i: jnp.take(x, i, axis=0), value, indices, corrupt=corrupt)


@pytest.mark.parametrize("plugin", _PLUGINS)
def test_legacy_keeps_its_original_custom_binding(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = next(s for s in plugin.binding_specs() if isinstance(s, MonkeyPatchSpec))
    token, calls, native_calls = object(), [], []

    def original(*args: Any, **kwargs: Any) -> Any:
        native_calls.append((args, kwargs))
        raise AssertionError("LEGACY carrier case unexpectedly forwarded natively")

    def bind(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return token

    array = jnp.asarray([[1, 2], [3, 4]], dtype=jnp.float32)
    with monkeypatch.context() as cleanup:
        cleanup.setattr(plugin._PRIM, "bind", bind)
        if hasattr(plugin, "_ORIG_CALL"):
            cleanup.setattr(plugin, "_ORIG_CALL", plugin._ORIG_CALL)
        for slot in (
            "__orig_impl__take",
            "__orig_impl___transpose",
            "__orig_impl__tile",
            "__orig_impl__split",
            "__orig_impl__concatenate",
        ):
            cleanup.setattr(
                plugin._PRIM, slot, getattr(plugin._PRIM, slot, None), raising=False
            )
        patched = spec.make_value(original)
        with precision_policy_scope(PrecisionPolicy.LEGACY):
            if plugin is EmbedPlugin:
                receiver = SimpleNamespace(embedding=SimpleNamespace(value=array))
                actual = patched(receiver, jnp.asarray([0], dtype=jnp.int32))
            elif plugin is JnpTransposePlugin:
                actual = patched(array, (1, 0))
            elif plugin is JnpTakePlugin:
                actual = patched(array, jnp.asarray([0], dtype=jnp.int32), axis=0)
            elif plugin is JnpTilePlugin:
                actual = patched(array, (1, 2))
            elif plugin is JnpSplitPlugin:
                actual = patched(array, 2, axis=1)
            elif plugin is JnpConcatenatePlugin:
                actual = patched((array, array), axis=1)
            elif plugin is NnxLogSoftmax:
                actual = patched(array, axis=-1)
            else:
                actual = patched(array, array, array)
        assert actual is token and len(calls) == 1 and not native_calls


@pytest.mark.parametrize("x64", (False, True))
def test_tile_native_counterparts_keep_reps_types_values_and_symbolic_scope(
    x64: bool,
) -> None:
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        value = jnp.asarray(np.arange(12).reshape(3, 4), dtype=dtype)
        _compare(lambda x: jnp.tile(x, (1, 2)), value)
        batch = jax_export.symbolic_shape("B")[0]
        _compare(
            lambda x, token: jnp.tile(token, (x.shape[0], 1, 1)),
            jax.ShapeDtypeStruct((batch, 49, 256), dtype),
            jax.ShapeDtypeStruct((1, 1, 256), dtype),
            execute=False,
        )


def test_wrong_tile_repeats_are_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _ORIGINALS[5]

    @contextmanager
    def corrupt() -> Iterator[None]:
        with monkeypatch.context() as temporary:
            temporary.setattr(jnp, "tile", lambda a, reps: original(a, (1, 1)))
            yield

    value = jnp.asarray(np.arange(6).reshape(2, 3), dtype=jnp.float32)
    with pytest.raises(AssertionError):
        _compare(lambda x: jnp.tile(x, (1, 2)), value, corrupt=corrupt)


@pytest.mark.parametrize("x64", (False, True))
def test_split_native_counterparts_keep_sections_indices_axis_and_result_list(
    x64: bool,
) -> None:
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        value = jnp.asarray(np.arange(24).reshape(3, 8), dtype=dtype)
        _compare(lambda x: jnp.split(x, 2, axis=-1), value)
        _compare(lambda x: jnp.split(x, np.asarray([2, 2, 5]), axis=1), value)
        batch = jax_export.symbolic_shape("B")[0]
        _compare(
            lambda x: jnp.split(x, 2, axis=-1),
            jax.ShapeDtypeStruct((batch, 2, 4), dtype),
            execute=False,
        )


def test_wrong_split_sections_are_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _ORIGINALS[6]

    @contextmanager
    def corrupt() -> Iterator[None]:
        with monkeypatch.context() as temporary:
            temporary.setattr(
                jnp,
                "split",
                lambda a, indices_or_sections, axis=0: original(a, (1,), axis=axis),
            )
            yield

    value = jnp.asarray(np.arange(12).reshape(3, 4), dtype=jnp.float32)
    with pytest.raises(AssertionError):
        _compare(lambda x: jnp.split(x, 2, axis=-1), value, corrupt=corrupt)


@pytest.mark.parametrize("x64", (False, True))
def test_concatenate_native_counterparts_keep_input_container_axis_and_explicit_dtype(
    x64: bool,
) -> None:
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        left = jnp.asarray(np.arange(12).reshape(3, 4), dtype=dtype)
        right = jnp.asarray(np.arange(6).reshape(3, 2), dtype=np.int32)
        _compare(
            lambda x, y: jnp.concatenate([x, y], axis=-1, dtype=dtype), left, right
        )
        _compare(lambda x, y: jnp.concatenate((x, y), axis=None), left, right)
        batch = jax_export.symbolic_shape("B")[0]
        _compare(
            lambda x, token: jnp.concatenate([token, x], axis=1),
            jax.ShapeDtypeStruct((batch, 49, 256), dtype),
            jax.ShapeDtypeStruct((batch, 1, 256), dtype),
            execute=False,
        )


def test_wrong_concatenate_axis_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _ORIGINALS[7]

    @contextmanager
    def corrupt() -> Iterator[None]:
        with monkeypatch.context() as temporary:
            temporary.setattr(
                jnp, "concatenate", lambda arrays, axis=0: original(arrays, axis=0)
            )
            yield

    value = jnp.asarray(np.arange(6).reshape(2, 3), dtype=jnp.float32)
    with pytest.raises(AssertionError):
        _compare(lambda x: jnp.concatenate([x, x], axis=1), value, corrupt=corrupt)


@pytest.mark.parametrize("x64", (False, True))
def test_nnx_log_softmax_native_counterparts_keep_dtype_axis_mask_and_scope(
    x64: bool,
) -> None:
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        value = jnp.asarray(np.arange(20).reshape(2, 10) / 8 - 1, dtype=dtype)
        _compare(lambda x: nnx.log_softmax(x, axis=-1), value)
        _compare(lambda x: nnx.log_softmax(x, axis=(0, 1)), value)
        mask = jnp.asarray(np.arange(20).reshape(2, 10) % 3 != 0, dtype=jnp.bool_)
        _compare(lambda x, m: nnx.log_softmax(x, axis=-1, where=m), value, mask)
        batch = jax_export.symbolic_shape("B")[0]
        _compare(
            lambda x: nnx.log_softmax(x, axis=-1),
            jax.ShapeDtypeStruct((batch, 10), dtype),
            execute=False,
        )
