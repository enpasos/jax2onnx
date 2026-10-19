# tests/extra_tests/converter/test_source_layer_norm_exports.py

"""Unchanged native LayerNorm modules in static SOURCE artifacts and consumers."""

from typing import Any
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx import onnx_function
from tests.extra_tests.converter.test_source_layer_norm_forwarding import (
    _flax_model,
    _rows,
)
from tests.extra_tests.converter.test_source_square_variance import (
    _BOUNDS,
    _exercise,
    _nodes,
    _payload,
    _serialized_residual,
    source_settings,  # noqa: F401
)

_FLAX_CASES = [
    (kind, fast, dtype, affine, axes, shape)
    for kind in ("nnx", "linen")
    for fast in (False, True)
    for dtype, affine, axes, shape in [
        ("float32", True, (-1,), (5, 8)),
        ("float64", True, (-1,), (5, 8)),
        ("float16", True, (-1,), (5, 8)),
        ("float32", False, (-1,), (5, 8)),
        ("float64", True, (1, 2), (2, 3, 8)),
    ]
]
_EQX_CASES = [
    (dtype, affine, (8,))
    for dtype in ("float16", "float32", "float64")
    for affine in (False, True)
] + [(dtype, True, (3, 8)) for dtype in ("float32", "float64")]


@onnx_function
class _NamedResidual:
    def __init__(self, layer: Any) -> None:
        self.layer = layer

    def __call__(self, x: Any) -> Any:
        return x + self.layer(x)


def _assert_unfused(model: Any, dtype: str) -> None:
    nodes = list(_nodes(model))
    assert not any(
        node.op_type in {"LayerNormalization", "RMSNormalization"} for node in nodes
    )
    assert any(node.op_type == "ReduceSum" for node in nodes)
    assert any(node.op_type == "Mul" for node in nodes)
    assert any(node.op_type == "Sqrt" for node in nodes)
    assert (
        model.graph.output[0].type.tensor_type.elem_type
        == {"float16": 10, "float32": 1, "float64": 11}[dtype]
    )


@pytest.mark.parametrize("framework,fast,dtype,affine,axes,shape", _FLAX_CASES)
def test_original_flax_layer_norm_complete_static_export(
    framework: str,
    fast: bool,
    dtype: str,
    affine: bool,
    axes: Any,
    shape: Any,
    tmp_path: Path,
    record_property: Any,
) -> None:
    value = (
        _rows(jnp.dtype(dtype))
        if shape == (5, 8)
        else jnp.asarray(np.linspace(-2, 3, np.prod(shape)).reshape(shape), dtype=dtype)
    )
    fn, state = _flax_model(
        framework,
        value,
        param_dtype=jnp.float32 if dtype == "float16" else jnp.dtype(dtype),
        dtype=jnp.float16 if dtype == "float16" else None,
        fast=fast,
        scale=affine,
        bias=affine,
        reduction_axes=axes,
    )
    original = state()
    model = _exercise(
        fn, [(value,)], tmp_path, _BOUNDS["layer_norm"][dtype], record_property
    )
    _assert_unfused(model, dtype)
    assert state() == original


@pytest.mark.parametrize("dtype,affine,shape", _EQX_CASES)
def test_original_equinox_layer_norm_complete_static_export(
    dtype: str, affine: bool, shape: Any, tmp_path: Path, record_property: Any
) -> None:
    parameter_dtype = jnp.float64 if dtype == "float64" else jnp.float32
    model = eqx.nn.LayerNorm(
        shape,
        eps=np.float32(0.0003),
        use_weight=affine,
        use_bias=affine,
        dtype=parameter_dtype,
    )
    if affine:
        model = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            model,
            (
                jnp.asarray(
                    np.linspace(0.5, 1.5, np.prod(shape)).reshape(shape),
                    dtype=parameter_dtype,
                ),
                jnp.asarray(
                    np.linspace(-0.25, 0.75, np.prod(shape)).reshape(shape),
                    dtype=parameter_dtype,
                ),
            ),
        )
    original = tuple(_payload(v) for v in jax.tree.leaves(model))
    values = (
        list(_rows(jnp.dtype(dtype)))
        if shape == (8,)
        else [
            jnp.asarray(np.linspace(-2, 3, np.prod(shape)).reshape(shape), dtype=dtype)
        ]
    )
    exported = _exercise(
        model,
        [(v,) for v in values],
        tmp_path,
        _BOUNDS["layer_norm"][dtype],
        record_property,
    )
    _assert_unfused(exported, dtype)
    assert tuple(_payload(v) for v in jax.tree.leaves(model)) == original


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("boundary", ["inline", "function", "cond"])
def test_original_layer_norm_residual_function_and_both_conditional_branches(
    dtype: str, boundary: str, tmp_path: Path, record_property: Any
) -> None:
    value = _rows(jnp.dtype(dtype))
    layer, state = _flax_model("nnx", value, param_dtype=jnp.dtype(dtype), fast=False)

    def residual(x: Any) -> Any:
        return x + layer(x)

    named = _NamedResidual(layer)

    def call_function(x: Any) -> Any:
        return named(x)

    def call_conditional(x: Any, selector: Any) -> Any:
        return jax.lax.cond(
            selector > np.float32(0), lambda y: named(y), lambda y: -named(-y), x
        )

    fn: Any
    if boundary == "inline":
        fn = residual
    elif boundary == "function":
        fn = call_function
    else:
        fn = call_conditional
    feeds = (
        [(value,)]
        if boundary != "cond"
        else [(value, jnp.asarray(v, dtype=jnp.float32)) for v in (1, -1)]
    )
    original = state()
    exported = _exercise(
        fn,
        feeds,
        tmp_path,
        _BOUNDS["layer_norm"][dtype],
        record_property,
        function_body=((named, (value,), state) if boundary != "inline" else None),
        boundary=boundary,
        serialized_check=lambda model, observations: _serialized_residual(
            model, boundary, dtype, observations
        ),
    )
    _assert_unfused(exported, dtype)
    assert state() == original
    if boundary in {"function", "cond"}:
        assert exported.functions
        assert any(
            node.op_type == "ReduceSum"
            for function in exported.functions
            for node in function.node
        )
    if boundary == "cond":
        branch = next(node for node in exported.graph.node if node.op_type == "If")
        graphs = [
            attribute.g
            for attribute in branch.attribute
            if attribute.name in {"then_branch", "else_branch"}
        ]
        assert len(graphs) == 2
        functions = {
            (function.domain, function.name) for function in exported.functions
        }
        assert all(
            any((node.domain, node.op_type) in functions for node in graph.node)
            for graph in graphs
        )
