# tests/extra_tests/converter/test_source_mlp_normalization_activation.py

"""Native BatchNorm, Dense and GELU semantics under SOURCE activation."""

from typing import Any
from types import SimpleNamespace

from flax import linen, nnx
import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx._compat.jax import atomic_symbolic_dimension
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins.flax.nnx.batch_norm import BatchNormPlugin
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy.shape import JnpShapePlugin
from tests.extra_tests.converter.test_source_mlp_state_preparation import (
    PreparedMLP,
    _restore_configuration as _restore_configuration,
    array_snapshot,
    nonzero_mlp_input,
    original_mlp,
)


@pytest.mark.parametrize("kind", ["nnx", "linen"])
@pytest.mark.parametrize(
    "x64,dtype", [(False, np.float32), (True, np.float32), (True, np.float64)]
)
def test_original_running_mlp_source_trace_matches_native(
    kind: str, x64: bool, dtype: Any
) -> None:
    with jax.enable_x64(x64):
        prepared = PreparedMLP.from_original(original_mlp(kind), kind)
        B = jax_export.symbolic_shape("B")[0]
        spec = jax.ShapeDtypeStruct((B, 30), dtype)
        native = prepared.fresh_original()
        source = prepared.fresh_original()
        native_graph = jax.make_jaxpr(lambda x: native(x, deterministic=True))(spec)
        module = prepared.original if kind == "nnx" else prepared.original._model
        snapshot = array_snapshot(module)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            source_graph = jax.make_jaxpr(lambda x: source(x, deterministic=True))(spec)
        validate_source_jaxpr(source_graph, 23)
        assert [eqn.primitive.name for eqn in source_graph.eqns] == [
            eqn.primitive.name for eqn in native_graph.eqns
        ]
        assert all(
            not eqn.primitive.name.startswith(("nnx.", "linen.", "jax.nn.gelu"))
            for eqn in source_graph.eqns
        )
        assert source_graph.outvars[0].aval.dtype == native_graph.outvars[0].aval.dtype
        for actual, expected in zip(
            source_graph.consts, native_graph.consts, strict=True
        ):
            np.testing.assert_array_equal(actual, expected, strict=True)
        assert array_snapshot(module) == snapshot


@pytest.mark.parametrize("approximate", [True, False])
@pytest.mark.parametrize("alias", ["jax", "linen", "linen_activation", "nnx"])
def test_gelu_alias_forwards_the_original_approximation(
    alias: str, approximate: bool
) -> None:
    from flax.linen import activation

    owners = {"jax": jax.nn, "linen": linen, "linen_activation": activation, "nnx": nnx}
    owner = owners[alias]
    x = jnp.asarray([-2.0, -0.5, 0.0, 0.5, 2.0], jnp.float32)
    expected = owner.gelu(x, approximate)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        actual = owner.gelu(x, approximate)
    np.testing.assert_array_equal(actual, expected, strict=True)


def test_nnx_batch_norm_source_keeps_call_override_mask_and_mutation() -> None:
    original_call = nnx.BatchNorm.__call__
    original = nnx.BatchNorm(20, use_running_average=None, rngs=nnx.Rngs(17))
    native = nnx.clone(original)
    source = nnx.clone(original)
    x = jnp.asarray(nonzero_mlp_input(2)[:, :20])
    mask = jnp.asarray([[True], [False]])
    expected = original_call(native, x, use_running_average=False, mask=mask)
    wrapped = BatchNormPlugin._make_patch(original_call)
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        actual = wrapped(source, x, use_running_average=False, mask=mask)
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert array_snapshot(source) == array_snapshot(native)
    assert array_snapshot(source) != array_snapshot(original)


def test_batch_norm_source_preserves_native_python_errors_without_body_call() -> None:
    calls = []

    def original(self: object, x: object, *, mask: object = None) -> object:
        calls.append((self, x, mask))
        return x

    wrapped = BatchNormPlugin._make_patch(original)
    module, value = object(), object()
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        assert wrapped(module, value, mask=value) is value
        with pytest.raises(TypeError, match="unexpected keyword"):
            wrapped(module, value, unknown=True)
    assert calls == [(module, value, value)]


@pytest.mark.parametrize("kind", ["scalar", "tail", "neg"])
def test_native_pointwise_broadcasts_and_negation_keep_actual_scope(kind: str) -> None:
    B = jax_export.symbolic_shape("B")[0]
    left = jax.ShapeDtypeStruct((B, 3), np.float32)
    right = jax.ShapeDtypeStruct(() if kind == "scalar" else (B, 1), np.float32)

    def native(x: jax.Array, y: jax.Array) -> jax.Array:
        return -x if kind == "neg" else x * y

    graph = jax.make_jaxpr(native)(left, right)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(native)(left, right)
    validate_source_jaxpr(graph, 23)
    validate_source_jaxpr(activated, 23)
    assert str(activated) == str(graph)
    assert graph.outvars[0].aval.shape == (B, 3)
    label, scope = atomic_symbolic_dimension(B)
    actual_label, actual_scope = atomic_symbolic_dimension(
        graph.outvars[0].aval.shape[0]
    )
    assert actual_label == label and actual_scope is scope
    # The original negation witness has a real unused second formal.
    assert len(graph.invars) == 2


def test_source_shape_preserves_original_return_dimensions_and_errors_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    dimension = object()
    result = (dimension, 30)
    error = ValueError("native shape error")

    def original(value: Any) -> Any:
        calls.append(value)
        if value is error:
            raise error
        return result

    patch = JnpShapePlugin.binding_specs()[1]
    assert isinstance(patch, MonkeyPatchSpec)
    monkeypatch.setattr(
        JnpShapePlugin._PRIM,
        "__orig_impl__shape",
        getattr(JnpShapePlugin._PRIM, "__orig_impl__shape", jnp.shape),
        raising=False,
    )
    wrapped = patch.make_value(original)
    value = object()
    with precision_policy_scope(PrecisionPolicy.SOURCE):
        assert wrapped(value) is result
        with pytest.raises(ValueError) as caught:
            wrapped(error)
        assert caught.value is error
    assert calls == [value, error]


@pytest.mark.parametrize("primitive", ["rsqrt", "tanh", "integer_pow"])
@pytest.mark.parametrize("corruption", ["parameter", "dtype", "weakness", "scope"])
def test_native_normalization_activation_corruption_fails_before_lowering(
    primitive: str, corruption: str
) -> None:
    B = jax_export.symbolic_shape("B")[0]
    function = {
        "rsqrt": jax.lax.rsqrt,
        "tanh": jax.lax.tanh,
        "integer_pow": lambda x: jax.lax.integer_pow(x, 3),
    }[primitive]
    graph = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((B, 3), np.float32))
    validate_source_jaxpr(graph, 23)
    original = str(graph)
    equation = graph.eqns[0]
    parameters = dict(equation.params)
    output = equation.outvars[0]
    if corruption == "parameter":
        parameters["y" if primitive == "integer_pow" else "accuracy"] = (
            2 if primitive == "integer_pow" else "fast"
        )
    else:
        aval = output.aval
        if corruption == "dtype":
            aval = aval.update(dtype=np.dtype(np.float64))
        elif corruption == "weakness":
            aval = aval.update(weak_type=not aval.weak_type)
        else:
            foreign = jax_export.symbolic_shape("B")[0]
            aval = aval.update(shape=(foreign, 3))
        output = SimpleNamespace(aval=aval)
    corrupted = SimpleNamespace(
        invars=graph.invars,
        constvars=graph.constvars,
        outvars=[output],
        eqns=[
            SimpleNamespace(
                primitive=equation.primitive,
                invars=equation.invars,
                outvars=[output],
                params=parameters,
                effects=equation.effects,
            )
        ],
        effects=graph.effects,
    )
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(corrupted, 23)
    assert str(graph) == original
