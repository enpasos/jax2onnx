# tests/extra_tests/framework/test_eqx_metadata_factories.py

"""Finite requested-dtype constructor contracts; existing exports own numerics."""

from __future__ import annotations

import ast
import functools
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest

from jax2onnx.plugins.equinox.eqx.nn import conv, recurrent, spectral_norm, weight_norm
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    with_prng_key,
    with_requested_dtype,
)


_RECIPES = (
    ("conv", conv, "_EQX_CONV_EXAMPLE"),
    ("conv-batched", conv, "_EQX_CONV_BATCHED_EXAMPLE"),
    ("gru", recurrent, "_EQX_GRU_CELL"),
    ("lstm", recurrent, "_EQX_LSTM_CELL"),
    ("weight-linear", weight_norm, "_EQX_WEIGHT_NORM_LINEAR"),
    ("weight-conv", weight_norm, "_EQX_WEIGHT_NORM_CONV2D"),
    ("spectral", spectral_norm, "_EQX_SPECTRAL_NORM"),
)
_NATIVE = {
    "Linear": eqx.nn.Linear,
    "Conv2d": eqx.nn.Conv2d,
    "GRUCell": eqx.nn.GRUCell,
    "LSTMCell": eqx.nn.LSTMCell,
    "WeightNorm": eqx.nn.WeightNorm,
    "SpectralNorm": eqx.nn.SpectralNorm,
    "State": eqx.nn.State,
}


def test_eqx_metadata_setup_defers_constructors_keys_and_vmap_without_registry_reload() -> (
    None
):
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "metadata collection must defer native construction/key/vmap"
        )

    for module in (conv, recurrent, weight_norm, spectral_norm):
        source = ast.parse(Path(module.__file__).read_text())
        selected = []
        for node in source.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_make_eqx_"):
                selected.append(node)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id.startswith("_EQX_")
            ):
                selected.append(node)
        assert selected
        block = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *selected,
            ],
            type_ignores=[],
        )
        namespace = {
            "eqx": SimpleNamespace(
                nn=SimpleNamespace(**{name: forbidden for name in _NATIVE})
            ),
            "jax": SimpleNamespace(
                vmap=forbidden, random=SimpleNamespace(PRNGKey=forbidden)
            ),
            "construct_and_call": construct_and_call,
            "with_requested_dtype": with_requested_dtype,
            "with_prng_key": with_prng_key,
        }
        exec(
            compile(ast.fix_missing_locations(block), str(module.__file__), "exec"),
            namespace,
        )
        names = [name for _, owner, name in _RECIPES if owner is module]
        assert {
            node.target.id for node in selected if isinstance(node, ast.AnnAssign)
        } == set(names)
        assert all(namespace[name].__jax2onnx_factory__ is True for name in names)
        assert all(getattr(module, name).__jax2onnx_factory__ is True for name in names)


def _native_reference(kind: str, dtype: Any) -> list[tuple[str, Any]]:
    if kind in {"conv", "conv-batched"}:
        return [
            (
                "Conv2d",
                eqx.nn.Conv2d(
                    in_channels=3,
                    out_channels=8,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    dtype=dtype,
                    key=jax.random.PRNGKey(0),
                ),
            )
        ]
    if kind == "gru":
        return [
            (
                "GRUCell",
                eqx.nn.GRUCell(
                    input_size=3, hidden_size=4, dtype=dtype, key=jax.random.PRNGKey(0)
                ),
            )
        ]
    if kind == "lstm":
        return [
            (
                "LSTMCell",
                eqx.nn.LSTMCell(
                    input_size=3, hidden_size=4, dtype=dtype, key=jax.random.PRNGKey(1)
                ),
            )
        ]
    if kind == "weight-linear":
        layer = eqx.nn.Linear(8, 4, dtype=dtype, key=jax.random.PRNGKey(0))
        return [("Linear", layer), ("WeightNorm", eqx.nn.WeightNorm(layer))]
    if kind == "weight-conv":
        layer = eqx.nn.Conv2d(
            3, 5, kernel_size=3, dtype=dtype, key=jax.random.PRNGKey(1)
        )
        return [("Conv2d", layer), ("WeightNorm", eqx.nn.WeightNorm(layer))]
    assert kind == "spectral"
    layer = eqx.nn.Linear(8, 4, dtype=dtype, key=jax.random.PRNGKey(0))
    module = eqx.nn.SpectralNorm(
        layer=layer, weight_name="weight", inference=True, key=jax.random.PRNGKey(1)
    )
    return [
        ("Linear", layer),
        ("SpectralNorm", module),
        ("State", eqx.nn.State(module)),
    ]


def _parameter_arrays(name: str, value: Any, paired: Any = None) -> list[Any]:
    assert type(value) is _NATIVE[name]
    if name in {"Linear", "Conv2d"}:
        arrays = [value.weight, value.bias]
    elif name == "GRUCell":
        arrays = [value.weight_ih, value.weight_hh, value.bias, value.bias_n]
    elif name == "LSTMCell":
        arrays = [value.weight_ih, value.weight_hh, value.bias]
    elif name == "WeightNorm":
        nested = "Linear" if type(value.layer) is eqx.nn.Linear else "Conv2d"
        arrays = [*_parameter_arrays(nested, value.layer), value.g]
        norm = value._norm
        assert type(norm) is functools.partial and norm.args == ()
        assert set(norm.keywords) == {"axis", "norm"} and norm.keywords["axis"] == 0
        inner = norm.keywords["norm"]
        assert type(inner) is functools.partial and inner.func is jax.numpy.linalg.norm
        assert inner.args == () and inner.keywords == {"keepdims": True}
    elif name == "SpectralNorm":
        assert value.inference is True and value.weight_name == "weight"
        assert value.num_power_iterations == 1 and value.eps == 1e-12
        arrays = [*_parameter_arrays("Linear", value.layer), *value.uv_index.init]
    else:
        assert name == "State" and type(paired) is eqx.nn.SpectralNorm
        arrays = list(value.get(paired.uv_index))
    assert arrays and all(eqx.is_array(array) for array in arrays)
    observed = []
    for leaf in jax.tree.leaves(value):
        if eqx.is_array(leaf):
            observed.append(leaf)
        elif name == "WeightNorm" and leaf is value._norm:
            assert type(leaf) is functools.partial
        elif name == "SpectralNorm" and leaf is value.inference:
            assert type(leaf) is bool
        else:
            raise AssertionError("unreviewed native parameter/state leaf")
    assert len(observed) == len(arrays)
    assert all(left is right for left, right in zip(observed, arrays, strict=True))
    return arrays


def _check_wrapper(
    kind: str,
    target: Any,
    records: list[Any],
    vmapped: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = {name: value for name, value, _, _ in records}
    if kind in {"conv-batched", "weight-linear"}:
        expected = models["Conv2d"] if kind == "conv-batched" else models["WeightNorm"]
        assert len(vmapped) == 1 and vmapped[0] is expected and callable(target)
        return
    if kind in {"conv", "weight-conv"}:
        assert target is models["Conv2d" if kind == "conv" else "WeightNorm"]
        assert vmapped == []
        return
    assert vmapped == []
    owner = models[
        {"gru": "GRUCell", "lstm": "LSTMCell", "spectral": "SpectralNorm"}[kind]
    ]
    h, c, x, result, next_state = (object() for _ in range(5))
    calls = []

    def observe(instance: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((instance, args, kwargs))
        return (result, next_state) if kind == "spectral" else result

    with monkeypatch.context() as change:
        change.setattr(type(owner), "__call__", observe)
        actual = (
            target(h, x)
            if kind == "gru"
            else target(h, c, x) if kind == "lstm" else target(x)
        )
    assert (
        actual is result
        and len(calls) == 1
        and calls[0][0] is owner
        and calls[0][2] == {}
    )
    expected_args = (
        (x, h)
        if kind == "gru"
        else (x, (h, c)) if kind == "lstm" else (x, models["State"])
    )
    assert calls[0][1] == expected_args


@pytest.mark.parametrize(
    "kind,module,recipe_name", _RECIPES, ids=[item[0] for item in _RECIPES]
)
@pytest.mark.parametrize(
    "caller,dtype", [(False, np.float32), (True, np.float32), (True, np.float64)]
)
def test_eqx_actual_recipes_allocate_requested_dtype_and_preserve_fresh_native_state(
    kind: str,
    module: Any,
    recipe_name: str,
    caller: bool,
    dtype: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = []
    vmapped = []

    def constructor(name: str) -> Any:
        def build(*args: Any, **kwargs: Any) -> Any:
            value = _NATIVE[name](*args, **kwargs)
            records.append((name, value, args, kwargs))
            return value

        return build

    def vmap(value: Any) -> Any:
        vmapped.append(value)
        return jax.vmap(value)

    proxy = SimpleNamespace(
        nn=SimpleNamespace(**{name: constructor(name) for name in _NATIVE})
    )
    before = bool(jax.config.x64_enabled), jax.config.jax_default_matmul_precision
    previous = None
    try:
        with jax.enable_x64(caller), monkeypatch.context() as change:
            change.setattr(module, "eqx", proxy)
            change.setattr(module, "jax", SimpleNamespace(vmap=vmap))
            recipe = getattr(module, recipe_name)
            for _ in range(2):
                records.clear()
                vmapped.clear()
                target = recipe.with_dtype(dtype).instantiate()
                assert callable(target)
                baseline = _native_reference(kind, dtype)
                assert [name for name, _, _, _ in records] == [
                    name for name, _ in baseline
                ]
                paired = dict((name, value) for name, value, _, _ in records).get(
                    "SpectralNorm"
                )
                baseline_pair = dict(baseline).get("SpectralNorm")
                if kind == "spectral":
                    assert records[-1][2][0] is paired
                    assert records[-1][1] is not paired
                    if previous is not None:
                        with pytest.raises(KeyError):
                            records[-1][1].get(previous[-2][1].uv_index)
                for index, ((name, actual, _, kwargs), (_, expected)) in enumerate(
                    zip(records, baseline, strict=True)
                ):
                    actual_arrays = _parameter_arrays(name, actual, paired)
                    expected_arrays = _parameter_arrays(name, expected, baseline_pair)
                    assert len(actual_arrays) == len(expected_arrays)
                    for left, right in zip(actual_arrays, expected_arrays, strict=True):
                        left, right = np.asarray(left), np.asarray(right)
                        assert left.dtype == right.dtype == np.dtype(dtype)
                        assert (
                            left.shape == right.shape
                            and left.tobytes() == right.tobytes()
                        )
                    if "key" in kwargs:
                        seed = (
                            1
                            if kind in {"lstm", "weight-conv"} or name == "SpectralNorm"
                            else 0
                        )
                        np.testing.assert_array_equal(
                            kwargs["key"], jax.random.PRNGKey(seed)
                        )
                    if previous is not None:
                        assert actual is not previous[index][1]
                _check_wrapper(kind, target, records, vmapped, monkeypatch)
                previous = list(records)
    finally:
        assert (
            bool(jax.config.x64_enabled),
            jax.config.jax_default_matmul_precision,
        ) == before
