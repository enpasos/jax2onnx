# tests/extra_tests/converter/test_source_function_ownership.py

"""Completed SOURCE exports release owners while live equations retain them."""

from dataclasses import dataclass
import gc
from typing import Any
import weakref

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.plugins import plugin_system as ps


@dataclass
class _StaticOffset:
    offset: Any


@onnx_function
@dataclass
class _OwnedWeightedCall:
    weight: Any

    def __call__(self, x: Any, *, config: _StaticOffset) -> Any:
        return x * self.weight + config.offset


@onnx_function
class _OwnedNNXCall(nnx.Module):
    def __init__(self, weight: Any) -> None:
        self.weight = nnx.Param(weight)

    def __call__(self, x: Any, *, config: _StaticOffset) -> Any:
        return x * self.weight.value + config.offset


def _owned_call(
    kind: str,
) -> tuple[Any, _StaticOffset, tuple[weakref.ReferenceType[Any], ...]]:
    weight = jnp.asarray([2, 3], dtype=jnp.float32)
    config = _StaticOffset(jnp.asarray([1, -1], dtype=jnp.float32))
    owner = _OwnedNNXCall(weight) if kind == "nnx" else _OwnedWeightedCall(weight)
    return (
        owner,
        config,
        tuple(weakref.ref(value) for value in (owner, weight, config, config.offset)),
    )


def _export_owned_call(
    kind: str, nested_jit: bool
) -> tuple[onnx.ModelProto, tuple[weakref.ReferenceType[Any], ...]]:
    owner, config, references = _owned_call(kind)

    def call(x: Any) -> Any:
        return owner(x, config=config)

    artifact = ui._to_onnx(
        jax.jit(call) if nested_jit else call,
        [jax.ShapeDtypeStruct((2,), np.float32)],
        precision_policy=PrecisionPolicy.SOURCE,
    )
    assert isinstance(artifact, onnx.ModelProto) and artifact.functions
    return artifact, references


def _assert_released(references: tuple[weakref.ReferenceType[Any], ...]) -> None:
    # No JAX cache clearing: an artifact may outlive all of its Python owners.
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("kind", ("plain", "nnx"))
@pytest.mark.parametrize("nested_jit", (False, True))
def test_completed_source_exports_release_models_weights_and_static_payloads(
    kind: str, nested_jit: bool
) -> None:
    artifacts = []
    for _ in range(3):
        artifact, references = _export_owned_call(kind, nested_jit)
        artifacts.append(artifact)
        _assert_released(references)
    assert len(artifacts) == 3
    assert all(artifact.SerializeToString() for artifact in artifacts)


def _failed_owned_export(kind: str) -> tuple[weakref.ReferenceType[Any], ...]:
    owner, config, references = _owned_call(kind)

    def call(x: Any) -> Any:
        return owner(x, config=config)

    # Keep neither exception objects nor traceback frames beyond this helper.
    with pytest.raises(RuntimeError, match="ownership failure after staging"):
        ui._to_onnx(
            call,
            [jax.ShapeDtypeStruct((2,), np.float32)],
            precision_policy=PrecisionPolicy.SOURCE,
        )
    return references


@pytest.mark.parametrize("kind", ("plain", "nnx"))
def test_failed_source_export_releases_staged_owner_and_weights(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_after_staging(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("ownership failure after staging")

    monkeypatch.setattr(ps.FunctionPlugin, "_lower_in_context", fail_after_staging)
    _assert_released(_failed_owned_export(kind))


def _owned_equation() -> tuple[Any, tuple[weakref.ReferenceType[Any], ...]]:
    owner, config, references = _owned_call("plain")

    def call(x: Any) -> Any:
        return owner(x, config=config)

    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        closed = jax.make_jaxpr(call)(jax.ShapeDtypeStruct((2,), np.float32))
    (equation,) = closed.eqns
    return equation, references


def _exercise_retained_equation(
    equation: Any, references: tuple[weakref.ReferenceType[Any], ...]
) -> None:
    metadata = equation.params[ps._SOURCE_TRACE]
    assert metadata.call.receiver is references[0]()
    assert metadata.call.original.__self__ is references[0]()
    assert equation.params["config"].value is references[2]()
    assert not equation.effects
    x = np.asarray([-2, 3], dtype=np.float32)
    expected = np.asarray([-3, 8], dtype=np.float32)
    (actual,) = equation.primitive.impl(x, **equation.params)
    np.testing.assert_array_equal(actual, expected, strict=True)
    nested = jax.vmap(
        jax.vmap(lambda row: equation.primitive.bind(row, **equation.params))
    )
    (batched,) = nested(np.broadcast_to(x, (2, 3, 2)))
    np.testing.assert_array_equal(
        batched, np.broadcast_to(expected, (2, 3, 2)), strict=True
    )


def test_only_live_equation_retains_exact_owner_for_delayed_impl_and_native_vmap() -> (
    None
):
    equation, references = _owned_equation()
    gc.collect()
    assert all(reference() is not None for reference in references)
    _exercise_retained_equation(equation, references)
    del equation
    _assert_released(references)


def test_source_staging_preserves_native_parameter_hashability_requirement() -> None:
    equation, _ = _owned_equation()
    params = {**equation.params, "invalid": []}

    def call(x: Any) -> Any:
        return equation.primitive.bind(x, **params)

    with pytest.raises(TypeError, match="parameter invalid was not hashable"):
        jax.make_jaxpr(call)(jax.ShapeDtypeStruct((2,), np.float32))
