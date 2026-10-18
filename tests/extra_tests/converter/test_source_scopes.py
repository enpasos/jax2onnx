# tests/extra_tests/converter/test_source_scopes.py

"""Bounded source-dtype evidence across real function and If child contexts.

These exact scalar arithmetic witnesses do not establish general numerical
equivalence, matrix-kernel behavior, or IEEE rounding behavior.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest

from tests.extra_tests.converter.precision_test_support import (
    PrototypeLimit,
    export_source_dtypes,
)
from jax2onnx import onnx_function

_CONSTANT32 = np.asarray([1.25, -2.5], dtype=np.float32)
_CONSTANT64 = np.asarray([1 + 2**-30, -2.5], dtype=np.float64)
_OUTPUT_TYPES = [
    onnx.TensorProto.DOUBLE,
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.DOUBLE,
]


def _make_region(
    constant32: np.ndarray, constant64: np.ndarray
) -> Callable[[jax.Array], tuple[Any, ...]]:
    def region(x: jax.Array) -> tuple[Any, ...]:
        rounded = x.astype(jnp.float32)
        return (
            x + constant64,
            rounded,
            (rounded + np.float32(1)) - rounded,
            -rounded,
            constant32,
            constant64,
        )

    return region


_mixed_region = _make_region(_CONSTANT32, _CONSTANT64)


@onnx_function
def _function_region(x: jax.Array) -> tuple[Any, ...]:
    return _mixed_region(x)


def _call_function_region(x: jax.Array) -> tuple[Any, ...]:
    return _function_region(x)


@onnx_function(unique=True)
def _static_tree_region(
    x: jax.Array, *, negate: bool, mode: str
) -> tuple[jax.Array, list[Any]]:
    source = -x if negate else x
    if mode == "offset":
        wide = source + np.float64(1)
    elif mode == "double":
        wide = source + source
    else:
        raise ValueError(f"Unknown static mode: {mode}")
    rounded = source.astype(jnp.float32)
    return wide, [rounded, {"cancelled": (rounded + np.float32(1)) - rounded}]


def _call_static_tree_variants(x: jax.Array) -> tuple[Any, ...]:
    variants = (
        _static_tree_region(x, negate=False, mode="offset"),
        _static_tree_region(x, negate=False, mode="offset"),
        _static_tree_region(x, negate=True, mode="offset"),
        _static_tree_region(x, negate=False, mode="double"),
    )
    # Consume the tuple/list/dict returned by each Function while tracing.
    # A flattened primitive result cannot satisfy this caller's indexing.
    return tuple((wide, narrow[0], narrow[1]["cancelled"]) for wide, narrow in variants)


@onnx_function
def _alias_tree_region(x: jax.Array) -> dict[str, Any]:
    return {"same": x, "again": (x, [x, _CONSTANT32])}


def _call_alias_tree_region(x: jax.Array) -> tuple[Any, ...]:
    result = _alias_tree_region(x)
    return (
        result["same"],
        result["again"][0],
        result["again"][1][0],
        result["again"][1][1],
    )


def _conditional_region(x: jax.Array, selector: jax.Array) -> tuple[Any, ...]:
    return jax.lax.cond(
        selector > np.float32(0),
        _mixed_region,
        lambda value: _mixed_region(-value),
        x,
    )


def _nested_jit(x: jax.Array) -> jax.Array:
    return jax.jit(lambda value: -value)(x)


def _nested_scan(x: jax.Array) -> jax.Array:
    return jax.lax.scan(lambda carry, _: (-carry, carry), x, None, length=2)[0]


@onnx_function
def _function_with_jit(x: jax.Array) -> jax.Array:
    return _nested_jit(x)


@onnx_function
def _function_with_scan(x: jax.Array) -> jax.Array:
    return _nested_scan(x)


@onnx_function
def _function_with_sin(x: jax.Array) -> jax.Array:
    return jax.lax.sin(x)


@pytest.fixture(autouse=True)
def _restore_jax_config() -> Iterator[None]:
    previous = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)
        jax.config.update("jax_platforms", previous_platforms)


def _nodes_with_subgraphs(nodes: Any) -> Iterator[onnx.NodeProto]:
    for node in nodes:
        yield node
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _nodes_with_subgraphs(attribute.g.node)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph in attribute.graphs:
                    yield from _nodes_with_subgraphs(graph.node)


def _constant_payloads(model: onnx.ModelProto) -> list[np.ndarray]:
    payloads = [onnx.numpy_helper.to_array(t) for t in model.graph.initializer]
    roots = [model.graph.node, *(function.node for function in model.functions)]
    for nodes in roots:
        for node in _nodes_with_subgraphs(nodes):
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    payloads.append(onnx.numpy_helper.to_array(attribute.t))
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    payloads.extend(
                        onnx.numpy_helper.to_array(t) for t in attribute.g.initializer
                    )
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    payloads.extend(
                        onnx.numpy_helper.to_array(t)
                        for graph in attribute.graphs
                        for t in graph.initializer
                    )
    return payloads


def _assert_constant_payloads(model: onnx.ModelProto) -> None:
    payloads = _constant_payloads(model)
    for expected in (_CONSTANT32, _CONSTANT64):
        assert any(
            actual.dtype == expected.dtype and np.array_equal(actual, expected)
            for actual in payloads
        ), f"Missing exact {expected.dtype} closed-over constant payload"


def _session(
    model_path: Path, optimization: ort.GraphOptimizationLevel
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = optimization
    return ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )


@pytest.mark.parametrize(
    ("scope", "fn"),
    [
        ("inline", _mixed_region),
        ("function", _call_function_region),
        ("cond", _conditional_region),
    ],
    ids=["inline", "onnx_function", "cond"],
)
@pytest.mark.parametrize(
    "optimization",
    [
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ],
    ids=["ort_optimization_off", "ort_optimization_on"],
)
def test_mixed_source_region_survives_scope_and_serialization(
    scope: str,
    fn: Callable[..., tuple[Any, ...]],
    optimization: ort.GraphOptimizationLevel,
    tmp_path: Path,
) -> None:
    x = np.asarray(1e8, dtype=np.float64)
    specs = [jax.ShapeDtypeStruct(x.shape, x.dtype)]
    if scope == "cond":
        specs.append(jax.ShapeDtypeStruct((), np.float32))
    model = export_source_dtypes(fn, specs)
    onnx.checker.check_model(model, full_check=True)
    assert [value.type.tensor_type.elem_type for value in model.graph.output] == (
        _OUTPUT_TYPES
    )
    _assert_constant_payloads(model)

    if scope == "function":
        assert model.functions
        assert any(
            node.domain == function.domain and node.op_type == function.name
            for node in model.graph.node
            for function in model.functions
        )
    elif scope == "cond":
        (if_node,) = [node for node in model.graph.node if node.op_type == "If"]
        branches = [
            attribute.g
            for attribute in if_node.attribute
            if attribute.type == onnx.AttributeProto.GRAPH
        ]
        assert len(branches) == 2
        for branch in branches:
            assert not branch.input
            assert [
                value.type.tensor_type.elem_type for value in branch.output
            ] == _OUTPUT_TYPES[:4]

    model_path = tmp_path / f"{scope}.onnx"
    onnx.save_model(model, model_path)
    reloaded = onnx.load_model(model_path)
    onnx.checker.check_model(reloaded, full_check=True)
    _assert_constant_payloads(reloaded)
    session = _session(model_path, optimization)
    # A single exported If must select each body at runtime. The false body
    # negates x, so its returned rounded value proves which body executed.
    selectors = (np.float32(1), np.float32(-1)) if scope == "cond" else (None,)
    for selector in selectors:
        inputs = [x]
        if selector is not None:
            inputs.append(np.asarray(selector))
        expected = fn(*(jnp.asarray(value) for value in inputs))
        actual = session.run(
            None,
            {
                spec.name: value
                for spec, value in zip(session.get_inputs(), inputs, strict=True)
            },
        )
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected, strict=True):
            reference_array = np.asarray(reference)
            assert result.dtype == reference_array.dtype
            np.testing.assert_array_equal(result, reference_array)
        np.testing.assert_array_equal(actual[2], np.asarray(0, dtype=np.float32))
        assert actual[2].dtype == np.float32
        expected_sign = -1 if selector is not None and selector < 0 else 1
        np.testing.assert_array_equal(
            actual[1], np.asarray(expected_sign * 1e8, dtype=np.float32)
        )
        assert jax.config.x64_enabled


@pytest.mark.parametrize(
    "optimization",
    [
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ],
    ids=["ort_optimization_off", "ort_optimization_on"],
)
def test_static_function_variants_keep_their_mixed_result_trees(
    optimization: ort.GraphOptimizationLevel, tmp_path: Path
) -> None:
    x = np.asarray(1e8, dtype=np.float64)
    model = export_source_dtypes(
        _call_static_tree_variants, [jax.ShapeDtypeStruct(x.shape, x.dtype)]
    )
    onnx.checker.check_model(model, full_check=True)
    # One repeated call reuses its definition; each bool/string configuration
    # selects an independent definition with just the actual tensor input.
    assert len(model.functions) == 3
    assert len({(function.domain, function.name) for function in model.functions}) == 3
    assert len(model.graph.input) == 1
    assert all(len(function.input) == 1 for function in model.functions)
    assert [value.type.tensor_type.elem_type for value in model.graph.output] == (
        [onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT] * 4
    )

    model_path = tmp_path / "static_tree_variants.onnx"
    onnx.save_model(model, model_path)
    onnx.checker.check_model(onnx.load_model(model_path), full_check=True)
    session = _session(model_path, optimization)
    actual = session.run(None, {session.get_inputs()[0].name: x})
    expected = jax.tree.leaves(_call_static_tree_variants(jnp.asarray(x)))
    for result, reference in zip(actual, expected, strict=True):
        reference_array = np.asarray(reference)
        assert result.dtype == reference_array.dtype
        np.testing.assert_array_equal(result, reference_array)
    for offset in range(0, len(actual), 3):
        np.testing.assert_array_equal(actual[offset + 2], np.float32(0))
    np.testing.assert_array_equal(actual[0], x + np.float64(1))
    np.testing.assert_array_equal(actual[6], -x + np.float64(1))
    np.testing.assert_array_equal(actual[9], x + x)


@pytest.mark.parametrize(
    "optimization",
    [
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ],
    ids=["ort_optimization_off", "ort_optimization_on"],
)
def test_structured_function_aliases_and_constants_survive_roundtrip(
    optimization: ort.GraphOptimizationLevel, tmp_path: Path
) -> None:
    x = np.asarray([1e8, 1 + 2**-30], dtype=np.float64)
    model = export_source_dtypes(
        _call_alias_tree_region, [jax.ShapeDtypeStruct(x.shape, x.dtype)]
    )
    onnx.checker.check_model(model, full_check=True)
    (function,) = model.functions
    assert len(set(function.output)) == len(function.output) == 4
    assert set(function.output).isdisjoint(function.input)
    assert [value.type.tensor_type.elem_type for value in model.graph.output] == [
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.FLOAT,
    ]

    model_path = tmp_path / "alias_tree.onnx"
    onnx.save_model(model, model_path)
    reloaded = onnx.load_model(model_path)
    onnx.checker.check_model(reloaded, full_check=True)
    assert any(
        payload.dtype == _CONSTANT32.dtype and np.array_equal(payload, _CONSTANT32)
        for payload in _constant_payloads(reloaded)
    )
    session = _session(model_path, optimization)
    actual = session.run(None, {session.get_inputs()[0].name: x})
    expected = _call_alias_tree_region(jnp.asarray(x))
    for result, reference in zip(actual, expected, strict=True):
        reference_array = np.asarray(reference)
        assert result.dtype == reference_array.dtype
        np.testing.assert_array_equal(result, reference_array)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: _function_with_jit(x),
        lambda x: _function_with_scan(x),
        lambda x: _function_with_sin(x),
    ],
    ids=["nested_jit", "nested_scan", "unlisted_arithmetic"],
)
def test_function_bodies_keep_the_recursive_subset_limit(
    fn: Callable[..., Any],
) -> None:
    with pytest.raises(PrototypeLimit, match="outside the PoC subset"):
        export_source_dtypes(fn, [jax.ShapeDtypeStruct((), np.float64)])


@pytest.mark.parametrize(
    "unsupported",
    [_nested_jit, _nested_scan, jax.lax.sin],
    ids=["nested_jit", "nested_scan", "unlisted_arithmetic"],
)
def test_cond_bodies_keep_the_recursive_subset_limit(
    unsupported: Callable[[jax.Array], jax.Array],
) -> None:
    def fn(x: jax.Array, selector: jax.Array) -> jax.Array:
        return jax.lax.cond(
            selector > np.float32(0), lambda value: -value, unsupported, x
        )

    with pytest.raises(PrototypeLimit, match="outside the PoC subset"):
        export_source_dtypes(
            fn,
            [
                jax.ShapeDtypeStruct((), np.float64),
                jax.ShapeDtypeStruct((), np.float32),
            ],
        )
