# tests/extra_tests/converter/test_source_layer_norm_forwarding.py

"""Original LayerNorm calls and native traces; no export or runtime acceptance."""

from collections.abc import Callable, Iterator
import hashlib
import inspect
from pathlib import Path
from typing import Any

import equinox as eqx
from flax import linen as nn
from flax import nnx
import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
from onnxruntime.capi import _pybind_state as ort_state
import pytest

from jax2onnx._compat.jax import TypePromotionError
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.plugins._patching import MonkeyPatchSpec, apply_patches
from jax2onnx.plugins.equinox.eqx.nn.layer_norm import LayerNormPlugin as EqxPlugin
from jax2onnx.plugins.flax.linen.layer_norm import LayerNormPlugin as LinenPlugin
from jax2onnx.plugins.flax.nnx.layer_norm import LayerNormPlugin as NnxPlugin
from jax2onnx.plugins.jax.numpy.mean import JnpMeanPlugin
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)


_ORIGINALS = (nnx.LayerNorm.__call__, nn.LayerNorm.__call__, eqx.nn.LayerNorm.__call__)
_ORIGINAL_MEAN = jnp.mean
_PLUGINS = {"nnx": NnxPlugin, "linen": LinenPlugin, "eqx": EqxPlugin}
_SOURCE_PATHS = tuple(
    Path(inspect.getfile(plugin)) for plugin in (*_PLUGINS.values(), JnpMeanPlugin)
)


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
def _restore_sources_settings_callables_and_deny_ort(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    globals_before, settings_before = _globals(), _settings()
    sources_before = [
        hashlib.sha256(path.read_bytes()).digest() for path in _SOURCE_PATHS
    ]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("LayerNorm native forwarding tests deny all ORT execution")

    # Cover Python constructors, cached class references and execution entrypoints.
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
        for name, value in globals_before.items():
            jax.config.update(name, value)
        assert (_globals(), _settings()) == (globals_before, settings_before)
        assert (
            nnx.LayerNorm.__call__,
            nn.LayerNorm.__call__,
            eqx.nn.LayerNorm.__call__,
        ) == (_ORIGINALS)
        assert jnp.mean is _ORIGINAL_MEAN
        assert active_precision_policy() is PrecisionPolicy.LEGACY
        assert [
            hashlib.sha256(path.read_bytes()).digest() for path in _SOURCE_PATHS
        ] == (sources_before)


def _payload(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value)
    return array.dtype.name, array.shape, array.tobytes()


def _fresh(fn: Callable[..., Any]) -> Callable[..., Any]:
    # A new callable forces each trace/JIT to observe the active patch world.
    # Reusing a cached native trace would conceal the pre-forwarding defect.
    def call(*args: Any) -> Any:
        return fn(*args)

    return call


def _compare(
    fn: Callable[..., Any],
    *values: Any,
    execute: bool = True,
    layer_norm_only: bool = False,
) -> Any:
    original = jax.make_jaxpr(_fresh(fn))(*values)
    if execute:
        eager = fn(*values)
        compiled = jax.jit(_fresh(fn))(*values)
    activation = (
        apply_patches(
            [spec for plugin in _PLUGINS.values() for spec in plugin.binding_specs()]
        )
        if layer_norm_only
        else _activate_plugin_worlds()
    )
    with precision_policy_scope(PrecisionPolicy.SOURCE), activation:
        activated = jax.make_jaxpr(_fresh(fn))(*values)
        assert typed_trace_manifest(activated) == typed_trace_manifest(original)
        assert [_payload(value) for value in activated.consts] == [
            _payload(value) for value in original.consts
        ]
        if execute:
            for actual, expected in (
                (fn(*values), eager),
                (jax.jit(_fresh(fn))(*values), compiled),
            ):
                assert jax.tree.structure(actual) == jax.tree.structure(expected)
                assert [_payload(value) for value in jax.tree.leaves(actual)] == [
                    _payload(value) for value in jax.tree.leaves(expected)
                ]
    return original


def _rows(dtype: Any) -> Any:
    # Freeze zero, constant, near-constant, outlier and source-rounding witnesses.
    return jnp.asarray(
        [
            [0.0] * 8,
            [0.09006345] * 8,
            [1.0, 1.0001, 0.9999, 1.0002, 0.9998, 1.0, 1.0, 1.0],
            [1700.0, -900.0, 0.125, -0.25, 0.5, -1.0, 2.0, 3.0],
            [1.0, 1.0009765625, 1.0078125, -0.0, 0.0, -1.0, 0.125, -0.125],
        ],
        dtype=dtype,
    )


def _flax_model(
    framework: str,
    value: Any,
    *,
    param_dtype: Any = jnp.float32,
    dtype: Any = None,
    fast: bool = True,
    scale: bool = True,
    bias: bool = True,
    reduction_axes: Any = -1,
    feature_axes: Any = -1,
    **options: Any,
) -> tuple[Callable[..., Any], Callable[[], Any]]:
    kwargs = dict(
        epsilon=np.float32(0.0003),
        dtype=dtype,
        param_dtype=param_dtype,
        use_fast_variance=fast,
        use_scale=scale,
        use_bias=bias,
        reduction_axes=reduction_axes,
        feature_axes=feature_axes,
        **options,
    )
    if framework == "nnx":
        model = nnx.LayerNorm(value.shape[-1], rngs=nnx.Rngs(0), **kwargs)
        for name, start, stop in (("scale", 0.5, 1.5), ("bias", -0.25, 0.75)):
            parameter = getattr(model, name)
            if parameter is not None:
                parameter[...] = jnp.asarray(
                    np.linspace(start, stop, parameter.size).reshape(parameter.shape),
                    dtype=param_dtype,
                )

        def state() -> Any:
            return tuple(
                (
                    None
                    if getattr(model, name) is None
                    else _payload(getattr(model, name)[...])
                )
                for name in ("scale", "bias")
            )

        return lambda x, **kw: model(x, **kw), state
    module = nn.LayerNorm(**kwargs)
    variables = module.init(jax.random.key(0), value)
    for name, start, stop in (("scale", 0.5, 1.5), ("bias", -0.25, 0.75)):
        if name in variables.get("params", {}):
            parameter = variables["params"][name]
            variables["params"][name] = jnp.asarray(
                np.linspace(start, stop, parameter.size).reshape(parameter.shape),
                dtype=param_dtype,
            )
    return (
        lambda x, **kw: module.apply(variables, x, **kw),
        lambda: tuple(_payload(v) for v in jax.tree.leaves(variables)),
    )


@pytest.mark.parametrize("framework", ["nnx", "linen"])
@pytest.mark.parametrize(
    "x64,input_dtype,param_dtype,compute_dtype,fast,scale,bias",
    [
        (False, "float32", "float32", None, True, True, True),
        (False, "float32", "float16", None, False, True, True),
        (False, "float16", "float32", None, True, True, True),
        (False, "bfloat16", "float32", "bfloat16", False, True, True),
        (False, "float32", "float16", "float16", True, True, False),
        (False, "float16", "float32", "float32", False, False, True),
        (False, "float32", "float32", None, False, False, False),
        (True, "float64", "float32", None, True, True, True),
        (True, "float32", "float64", None, False, True, True),
        (True, "float64", "float32", "float32", False, True, True),
    ],
)
def test_flax_native_dtype_variance_and_affine_traces(
    framework: str,
    x64: bool,
    input_dtype: str,
    param_dtype: str,
    compute_dtype: str | None,
    fast: bool,
    scale: bool,
    bias: bool,
) -> None:
    with jax.enable_x64(x64):
        value = _rows(jnp.dtype(input_dtype))
        fn, state = _flax_model(
            framework,
            value,
            param_dtype=jnp.dtype(param_dtype),
            dtype=None if compute_dtype is None else jnp.dtype(compute_dtype),
            fast=fast,
            scale=scale,
            bias=bias,
        )
        before = state()
        _compare(fn, value)
        assert state() == before


@pytest.mark.parametrize("framework", ["nnx", "linen"])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("axes", [(1, 2), (0, 2), (1,)])
@pytest.mark.parametrize("masked", [False, True])
def test_flax_native_axes_and_mask(
    framework: str, fast: bool, axes: tuple[int, ...], masked: bool
) -> None:
    value = jnp.asarray(np.linspace(-2, 3, 48).reshape(2, 3, 8), dtype=jnp.float32)
    fn, state = _flax_model(framework, value, fast=fast, reduction_axes=axes)
    mask = jnp.asarray(np.arange(48).reshape(2, 3, 8) % 3 != 0)
    before = state()
    if masked:
        _compare(lambda x, m: fn(x, mask=m), value, mask)
    else:
        _compare(fn, value)
    assert state() == before


@pytest.mark.parametrize("input_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("fast", [False, True])
def test_linen_native_selected_reduction_dtype(input_dtype: str, fast: bool) -> None:
    value = _rows(jnp.dtype(input_dtype))
    fn, _ = _flax_model(
        "linen", value, dtype=value.dtype, fast=fast, force_float32_reductions=False
    )
    _compare(fn, value)


@pytest.mark.parametrize("input_dtype", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize(
    "layer_norm_only", [False, True], ids=["full_world", "wrapper_only"]
)
@pytest.mark.parametrize(
    "scale,bias", [(True, True), (False, False), (True, False), (False, True)]
)
def test_eqx_native_dtype_affine_and_state_traces(
    input_dtype: str, layer_norm_only: bool, scale: bool, bias: bool
) -> None:
    with jax.enable_x64(input_dtype == "float64"):
        value = _rows(jnp.dtype(input_dtype))[3]
        model = eqx.nn.LayerNorm(
            8,
            eps=np.float32(0.0003),
            use_weight=scale,
            use_bias=bias,
            dtype=jnp.float32,
        )
        if scale:
            model = eqx.tree_at(
                lambda m: m.weight,
                model,
                jnp.asarray(np.linspace(0.5, 1.5, 8), dtype=jnp.float32),
            )
        if bias:
            model = eqx.tree_at(
                lambda m: m.bias,
                model,
                jnp.asarray(np.linspace(-0.25, 0.75, 8), dtype=jnp.float32),
            )
        before = tuple(_payload(v) for v in jax.tree.leaves(model))
        key = jax.random.key(3)
        _compare(
            lambda x: model(x, state=None, key=key),
            value,
            layer_norm_only=layer_norm_only,
        )
        _compare(lambda x: model(x), value, layer_norm_only=layer_norm_only)
        assert tuple(_payload(v) for v in jax.tree.leaves(model)) == before


@pytest.mark.parametrize("framework", ["nnx", "linen", "eqx"])
def test_native_weak_scalar_and_ambient_context(framework: str) -> None:
    with jax.enable_x64(True):
        if framework == "eqx":
            model = eqx.nn.LayerNorm((), dtype=jnp.float64)
            with jax.numpy_dtype_promotion("strict"):
                _compare(lambda x: model(x), jnp.asarray(1.0))
        else:
            value = jnp.asarray([[0.5, 1.0, 2.0]], dtype=jnp.float64)
            fn, _ = _flax_model(framework, value, param_dtype=jnp.float64)
            with jax.numpy_dtype_promotion("strict"):
                with pytest.raises(TypePromotionError) as original:
                    fn(value)
                with (
                    precision_policy_scope(PrecisionPolicy.SOURCE),
                    _activate_plugin_worlds(),
                ):
                    with pytest.raises(type(original.value)) as activated:
                        fn(value)
                assert str(activated.value) == str(original.value)


@pytest.mark.parametrize("framework", ["nnx", "linen", "eqx"])
def test_symbolic_batch_native_trace(framework: str) -> None:
    concrete = _rows(jnp.float32)
    if framework == "eqx":
        model = eqx.nn.LayerNorm(8)
        fn = jax.vmap(model)
    else:
        fn, _ = _flax_model(framework, concrete)
    shape = jax_export.symbolic_shape("B,8")
    _compare(fn, jax.ShapeDtypeStruct(shape, jnp.float32), execute=False)


def test_active_layer_norm_signatures_match_actual_originals() -> None:
    signatures = tuple(inspect.signature(fn) for fn in _ORIGINALS)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        assert (
            tuple(
                inspect.signature(fn)
                for fn in (
                    nnx.LayerNorm.__call__,
                    nn.LayerNorm.__call__,
                    eqx.nn.LayerNorm.__call__,
                )
            )
            == signatures
        )
        assert inspect.signature(jnp.mean) == inspect.signature(_ORIGINAL_MEAN)


@pytest.mark.parametrize("framework", ["nnx", "linen", "eqx"])
def test_dispatch_forwards_exact_original_arguments_and_failure_identity(
    framework: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    result, argument, mask, state, key = (object() for _ in range(5))
    failure = ValueError("native LayerNorm failure identity")

    class Untouchable:
        def __getattribute__(self, name: str) -> Any:
            pytest.fail(f"SOURCE read configuration before its original call: {name}")

    token = Untouchable()

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        if kwargs.get("fail"):
            raise failure
        return result

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("SOURCE must use its captured original before custom binding")

    plugin = _PLUGINS[framework]
    monkeypatch.setattr(plugin._PRIM, "bind", forbidden)
    monkeypatch.setattr(plugin, "_ORIGINAL_CALL", None, raising=False)
    spec = next(s for s in plugin.binding_specs() if isinstance(s, MonkeyPatchSpec))
    wrapper = spec.make_value(original)
    monkeypatch.setattr(plugin, "_ORIGINAL_CALL", forbidden)
    variants = [
        ((token, argument), {}),
        ((token,), {"x": argument}),
        ((token, argument), {"mask": mask}),
        ((token, argument), {"mask": None}),
        ((token, argument, state), {"key": key}),
        ((token, argument), {"state": None, "key": key}),
        ((token,), {"fail": True}),
    ]
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        for args, kwargs in variants[:-1]:
            assert wrapper(*args, **kwargs) is result
        with pytest.raises(ValueError) as caught:
            wrapper(*variants[-1][0], **variants[-1][1])
    assert caught.value is failure
    assert calls == variants


@pytest.mark.parametrize("framework", ["nnx", "linen", "eqx"])
@pytest.mark.parametrize(
    "invalid",
    [
        "missing_input",
        "unknown_keyword",
        "extra_positional",
        "shape",
        "mask_positional",
    ],
)
def test_native_api_errors_are_preserved(framework: str, invalid: str) -> None:
    value = _rows(jnp.float32)[0]
    if framework == "eqx":
        model = eqx.nn.LayerNorm(8)

        def fn(*args: Any, **kw: Any) -> Any:
            return model(*args, **kw)

    else:
        fn, _ = _flax_model(framework, value)
    args, kwargs = (value,), {}
    if invalid == "missing_input":
        args = ()
    elif invalid == "unknown_keyword":
        kwargs = {"unknown": object()}
    elif invalid == "extra_positional":
        args = (value, None, None)
    elif invalid == "mask_positional":
        args = (value, None) if framework != "eqx" else (value, None, None)
    elif framework == "eqx":
        args = (jnp.zeros((2, 8)),)
    else:
        args = (jnp.zeros((7,)),)
    with pytest.raises(Exception) as original:
        fn(*args, **kwargs)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        with pytest.raises(type(original.value)) as activated:
            fn(*args, **kwargs)
    assert str(activated.value) == str(original.value)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "x64,input_dtype,compute_dtype,axes,keepdims",
    [
        (False, "float16", None, -1, False),
        (False, "bfloat16", "float32", (0, 1), True),
        (False, "float32", None, None, True),
        (True, "float32", "float64", (0, 2), True),
        (True, "float64", None, -1, False),
        (False, "int32", None, 1, False),
    ],
)
def test_mean_native_dtype_axes_keepdims_and_where(
    x64: bool,
    input_dtype: str,
    compute_dtype: str | None,
    axes: Any,
    keepdims: bool,
    masked: bool,
) -> None:
    with jax.enable_x64(x64):
        value = jnp.asarray(
            np.arange(24).reshape(2, 3, 4), dtype=jnp.dtype(input_dtype)
        )
        dtype = None if compute_dtype is None else jnp.dtype(compute_dtype)
        if masked:
            mask = jnp.asarray(np.arange(24).reshape(2, 3, 4) % 3 != 0)
            _compare(
                lambda x, m: jnp.mean(
                    x, axis=axes, dtype=dtype, keepdims=keepdims, where=m
                ),
                value,
                mask,
            )
        else:
            _compare(lambda x: jnp.mean(x, axes, dtype, None, keepdims), value)


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("promotion", ["standard", "strict"])
def test_mean_native_weak_scalar_and_context(x64: bool, promotion: str) -> None:
    with jax.enable_x64(x64), jax.numpy_dtype_promotion(promotion):
        value = jnp.asarray(1.0)
        assert value.weak_type
        _compare(lambda x: jnp.mean(x, where=None), value)


def test_mean_dispatch_forwards_exact_original_arguments_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    argument, axes, dtype, out, keepdims, mask, result = (object() for _ in range(7))
    failure = ValueError("native mean failure identity")

    def original(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        if kwargs.get("fail"):
            raise failure
        return result

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail(
            "SOURCE mean must use its captured original before legacy processing"
        )

    slot = "__orig_impl__mean"
    monkeypatch.setattr(JnpMeanPlugin._PRIM, slot, None, raising=False)
    monkeypatch.setattr(JnpMeanPlugin._PRIM, "bind", forbidden)
    spec = next(
        s for s in JnpMeanPlugin.binding_specs() if isinstance(s, MonkeyPatchSpec)
    )
    wrapper = spec.make_value(original)
    monkeypatch.setattr(JnpMeanPlugin._PRIM, slot, forbidden)
    variants = [
        ((argument,), {}),
        ((), {"a": argument}),
        ((argument, axes, dtype, out, keepdims), {"where": mask}),
        (
            (argument,),
            {
                "axis": None,
                "dtype": None,
                "out": None,
                "keepdims": False,
                "where": None,
            },
        ),
        ((), {"fail": True}),
    ]
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        for args, kwargs in variants[:-1]:
            assert wrapper(*args, **kwargs) is result
        with pytest.raises(ValueError) as caught:
            wrapper(*variants[-1][0], **variants[-1][1])
    assert caught.value is failure
    assert calls == variants


@pytest.mark.parametrize(
    "invalid", ["missing_input", "axis", "duplicate_axes", "out", "keyword"]
)
def test_mean_native_errors_are_preserved(invalid: str) -> None:
    value = jnp.asarray([[0.5, 1.0, 2.0]], dtype=jnp.float32)
    args, kwargs = (value,), {}
    if invalid == "missing_input":
        args = ()
    elif invalid == "axis":
        kwargs = {"axis": 3}
    elif invalid == "duplicate_axes":
        kwargs = {"axis": (0, 0)}
    elif invalid == "out":
        kwargs = {"out": object()}
    else:
        kwargs = {"unknown": object()}
    with pytest.raises(Exception) as original:
        _ORIGINAL_MEAN(*args, **kwargs)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        with pytest.raises(type(original.value)) as activated:
            jnp.mean(*args, **kwargs)
    assert str(activated.value) == str(original.value)
