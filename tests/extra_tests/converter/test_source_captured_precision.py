# tests/extra_tests/converter/test_source_captured_precision.py

"""Source function reuse preserves promotion metadata in captured JAX values."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import pytest

from jax2onnx import onnx_function, to_onnx
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.plugins import plugin_system as ps
from jax2onnx.plugins._function_keys import (
    snapshot_static_value,
    source_static_value_key,
    static_value_key,
)
from tests.extra_tests.converter.precision_runtime_support import (
    check_cpu_sample,
    require_sample_acceptance,
)
from tests.extra_tests.converter.precision_test_support import export_source_dtypes


@pytest.fixture(autouse=True)
def _restore_jax_config() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    previous_matmul_precision = jax.config.jax_default_matmul_precision
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)
        jax.config.update("jax_default_matmul_precision", previous_matmul_precision)


def _scaled_add(x: jax.Array, bias: jax.Array) -> jax.Array:
    # Both variants have the same result dtype. Weak bias performs the Add in
    # the input dtype before this cast; strong bias first promotes the input.
    return (x + bias).astype(bias.dtype) * np.float32(2**-20)


def _promotion_epsilon(x: jax.Array, bias: jax.Array) -> jax.Array:
    # Under x64=False, weak FP32 and strong FP32 captures select different
    # promotion with an FP16 input. Perform the actual arithmetic in FP32 so
    # this witness does not depend on ORT preserving FP16 arithmetic rounding.
    epsilon = np.float32(jnp.finfo(jnp.result_type(x, bias)).eps)
    return x.astype(jnp.float32) + epsilon


@onnx_function(unique=True)
@jax.tree_util.register_pytree_node_class
class _CapturedBias:
    def __init__(self, bias: jax.Array, *, promotion_probe: bool = False):
        self.bias = bias
        self.promotion_probe = promotion_probe

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.promotion_probe:
            return _promotion_epsilon(x, self.bias)
        return _scaled_add(x, self.bias)

    def tree_flatten(self):
        return (self.bias,), self.promotion_probe

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return cls(children[0], promotion_probe=auxiliary)


@onnx_function(unique=True)
def _static_bias(x: jax.Array, *, config) -> jax.Array:
    if config["promotion_probe"]:
        return _promotion_epsilon(x, config["nested"][0]["bias"])
    return _scaled_add(x, config["nested"][0]["bias"])


def _case(x64_enabled: bool) -> tuple[np.ndarray, tuple[jax.Array, ...]]:
    jax.config.update("jax_enable_x64", x64_enabled)
    bias_value = 1e8 if x64_enabled else 4096.0
    strong_dtype = np.float64 if x64_enabled else np.float32
    input_dtype = np.float32 if x64_enabled else np.float16
    biases = (
        jnp.asarray(bias_value),
        jnp.asarray(bias_value, dtype=strong_dtype),
        jnp.asarray(bias_value),
        jnp.asarray(bias_value, dtype=strong_dtype),
    )
    assert [bias.weak_type for bias in biases] == [True, False, True, False]
    for bias in biases[1:]:
        np.testing.assert_array_equal(bias, biases[0], strict=True)
    return np.asarray(1, dtype=input_dtype), biases


def _model(
    biases: tuple[jax.Array, ...],
    *,
    capture_kind: str,
    reverse: bool,
    promotion_probe: bool = False,
) -> Callable[[jax.Array], tuple[jax.Array, ...]]:
    if reverse:
        biases = (biases[1], biases[0], biases[3], biases[2])
    if capture_kind == "instance":
        blocks = tuple(
            _CapturedBias(bias, promotion_probe=promotion_probe) for bias in biases
        )

        def model(x: jax.Array) -> tuple[jax.Array, ...]:
            return tuple(
                block(x if index < 2 else -x) for index, block in enumerate(blocks)
            )

    else:
        configurations = tuple(
            {"nested": [{"bias": bias}], "promotion_probe": promotion_probe}
            for bias in biases
        )

        def model(x: jax.Array) -> tuple[jax.Array, ...]:
            return tuple(
                _static_bias(x if index < 2 else -x, config=config)
                for index, config in enumerate(configurations)
            )

    return model


def _assert_saved_artifact(
    exported: onnx.ModelProto,
    x: np.ndarray,
    expected: tuple[np.ndarray, ...],
    path: Path,
) -> onnx.ModelProto:
    onnx.save_model(exported, path)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    assert reloaded.graph.input[0].type.tensor_type.elem_type == (
        onnx.helper.np_dtype_to_tensor_dtype(x.dtype)
    )
    assert [value.type.tensor_type.elem_type for value in reloaded.graph.output] == [
        onnx.helper.np_dtype_to_tensor_dtype(value.dtype) for value in expected
    ]
    assert len(reloaded.functions) == 2
    calls = [node for node in reloaded.graph.node if node.domain.startswith("custom.")]
    assert len(calls) == 4
    assert calls[0].domain == calls[2].domain
    assert calls[1].domain == calls[3].domain
    assert calls[0].domain != calls[1].domain
    assert all(
        sum(node.op_type == "Add" for node in function.node) == 1
        for function in reloaded.functions
    )
    add_types = set()
    for function in reloaded.functions:
        value_types = {
            value.name: value.type.tensor_type.elem_type
            for value in function.value_info
        }
        add = next(node for node in function.node if node.op_type == "Add")
        add_type = value_types[add.output[0]]
        assert all(value_types[name] == add_type for name in add.input)
        add_types.add(add_type)
    assert add_types == (
        {onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE}
        if expected[0].dtype == np.float64
        else {onnx.TensorProto.FLOAT}
    )
    feeds = {reloaded.graph.input[0].name: x}
    actual = ReferenceEvaluator(reloaded).run(None, feeds)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference, strict=True)
    for optimize in (False, True):
        require_sample_acceptance(
            check_cpu_sample(reloaded, [x], expected, optimize=optimize)
        )
    return reloaded


@pytest.mark.parametrize("x64_enabled", [False, True])
@pytest.mark.parametrize("capture_kind", ["instance", "static-keyword"])
@pytest.mark.parametrize("reverse", [False, True], ids=["weak-first", "strong-first"])
def test_source_captures_specialize_weak_metadata_and_reuse_equal_states(
    x64_enabled: bool, capture_kind: str, reverse: bool, tmp_path: Path
) -> None:
    x, biases = _case(x64_enabled)
    model = _model(
        biases,
        capture_kind=capture_kind,
        reverse=reverse,
        promotion_probe=not x64_enabled,
    )
    expected = tuple(np.asarray(value) for value in model(jnp.asarray(x)))
    assert expected[0].dtype == expected[1].dtype
    assert expected[0] != expected[1]
    for value, reference in zip(jax.jit(model)(jnp.asarray(x)), expected, strict=True):
        np.testing.assert_array_equal(value, reference, strict=True)
    exported = export_source_dtypes(model, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    _assert_saved_artifact(exported, x, expected, tmp_path / "captured.onnx")
    assert bool(jax.config.x64_enabled) is x64_enabled
    assert active_precision_policy() is PrecisionPolicy.LEGACY


def test_source_static_snapshots_retain_nested_array_promotion_metadata() -> None:
    _, (weak, strong, same_weak, same_strong) = _case(True)

    def nested(bias):
        return {"biases": [(bias,)]}

    weak_snapshot = snapshot_static_value(nested(weak))
    strong_snapshot = snapshot_static_value(nested(strong))
    assert weak_snapshot["biases"][0][0] is weak
    assert strong_snapshot["biases"][0][0] is strong
    assert static_value_key(weak_snapshot) == static_value_key(strong_snapshot)
    assert source_static_value_key(weak_snapshot) != source_static_value_key(
        strong_snapshot
    )
    assert source_static_value_key(weak_snapshot) == source_static_value_key(
        nested(same_weak)
    )
    assert source_static_value_key(strong_snapshot) == source_static_value_key(
        nested(same_strong)
    )


@pytest.mark.parametrize("capture_kind", ["instance", "static-keyword"])
def test_source_legacy_source_captures_leave_ordinary_jax_behavior_intact(
    capture_kind: str, tmp_path: Path
) -> None:
    x, biases = _case(True)
    model = _model(biases, capture_kind=capture_kind, reverse=False)
    array = jnp.asarray(x)
    compiled = jax.jit(model)
    expected = tuple(np.asarray(value) for value in compiled(array))
    specs = [jax.ShapeDtypeStruct(x.shape, x.dtype)]
    first = export_source_dtypes(model, specs)
    _assert_saved_artifact(first, x, expected, tmp_path / "before_legacy.onnx")
    legacy = to_onnx(model, specs, enable_double_precision=False)
    onnx.checker.check_model(legacy, full_check=True)
    second = export_source_dtypes(model, specs)
    _assert_saved_artifact(second, x, expected, tmp_path / "after_legacy.onnx")
    for fn in (model, compiled):
        for value, reference in zip(fn(array), expected, strict=True):
            np.testing.assert_array_equal(value, reference, strict=True)
    assert jax.config.x64_enabled
    assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize("reverse", [False, True], ids=["first", "reversed"])
def test_source_equal_instance_state_reuses_bodies_but_changed_values_do_not(
    reverse: bool, tmp_path: Path
) -> None:
    biases = tuple(jnp.asarray(value, dtype=jnp.float32) for value in (1, 2, 1, 2))
    assert all(not bias.weak_type for bias in biases)
    x = np.asarray(4, dtype=np.float32)
    model = _model(biases, capture_kind="instance", reverse=reverse)
    expected = tuple(np.asarray(value) for value in model(jnp.asarray(x)))
    assert expected[0] != expected[1]
    for value, reference in zip(jax.jit(model)(jnp.asarray(x)), expected, strict=True):
        np.testing.assert_array_equal(value, reference, strict=True)
    exported = export_source_dtypes(model, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    _assert_saved_artifact(exported, x, expected, tmp_path / "state_values.onnx")


def test_source_nonunique_equal_instance_states_keep_separate_exported_bodies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugin = next(
        plugin
        for plugin in ps.ONNX_FUNCTION_PLUGIN_REGISTRY.values()
        if plugin.target is _CapturedBias
    )
    monkeypatch.setattr(plugin, "unique", False)
    biases = tuple(jnp.asarray(1, dtype=jnp.float32) for _ in range(4))
    model = _model(biases, capture_kind="instance", reverse=False)
    x = np.asarray(4, dtype=np.float32)
    expected = tuple(np.asarray(value) for value in model(jnp.asarray(x)))
    exported = export_source_dtypes(model, [jax.ShapeDtypeStruct(x.shape, x.dtype)])
    path = tmp_path / "nonunique.onnx"
    onnx.save_model(exported, path)
    reloaded = onnx.load_model(path)
    onnx.checker.check_model(reloaded, full_check=True)
    calls = [node for node in reloaded.graph.node if node.domain.startswith("custom.")]
    assert len(reloaded.functions) == len(calls) == 4
    assert len({node.domain for node in calls}) == 4
    actual = ReferenceEvaluator(reloaded).run(None, {reloaded.graph.input[0].name: x})
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference, strict=True)
    for optimize in (False, True):
        require_sample_acceptance(
            check_cpu_sample(reloaded, [x], expected, optimize=optimize)
        )


def test_source_reusable_instance_state_keeps_distinct_native_method_bodies() -> None:
    plugin = next(
        plugin
        for plugin in ps.ONNX_FUNCTION_PLUGIN_REGISTRY.values()
        if plugin.target is _CapturedBias
    )
    owner = _CapturedBias(jnp.asarray(2, dtype=jnp.float32))
    spec = jax.ShapeDtypeStruct((), np.float32)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        closed = jax.make_jaxpr(lambda x: owner(x))(spec)
    equation = next(
        equation
        for equation in closed.eqns
        if equation.primitive is plugin._source_primitive
    )
    original_trace = equation.params[ps._SOURCE_TRACE]
    assert original_trace.call.receiver is owner

    def alternate(self: _CapturedBias, x: jax.Array) -> jax.Array:
        return _scaled_add(x, -self.bias)

    alternate_call = ps._SourceFunctionCall.capture(
        alternate.__get__(owner, type(owner)), instance=True, receiver=owner
    )
    alternate_trace = replace(original_trace, call=alternate_call)
    context = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE, deployment_target=None
    )
    captures = ps.FunctionCaptures((), {}, ())
    keys = [
        plugin._function_key(owner, equation, captures, ctx=context, source_trace=trace)
        for trace in (original_trace, alternate_trace, original_trace)
    ]
    assert keys[0] == keys[2]
    assert keys[0] != keys[1]
    x = jnp.asarray(4, dtype=jnp.float32)
    first = np.asarray(original_trace.call.original(x))
    second = np.asarray(alternate_call.original(x))
    assert first.dtype == second.dtype and first.shape == second.shape
    assert first != second
