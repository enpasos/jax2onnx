# tests/extra_tests/converter/test_onnx_function_call_params.py

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import onnxruntime as ort
import pytest

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    onnx_function,
    with_rng_seed,
)
from jax2onnx.user_interface import to_onnx


@onnx_function
class FunctionDropout(nnx.Module):
    def __init__(self, rate: float, *, rngs: nnx.Rngs):
        super().__init__()
        self.dropout = nnx.Dropout(rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        return self.dropout(x, deterministic=deterministic)


def test_onnx_function_consumes_call_time_param():
    fn = construct_and_call(
        FunctionDropout,
        rate=0.25,
        rngs=with_rng_seed(0),
    ).instantiate()

    model = to_onnx(
        fn=fn,
        inputs=[jax.ShapeDtypeStruct((2, 3), jnp.float32)],
        input_params={"deterministic": True},
        model_name="function_dropout",
        opset=21,
        enable_double_precision=False,
    )

    check = EG(
        ["Dropout"],
        search_functions=True,
        no_unused_inputs=True,
        no_unused_function_inputs=True,
    )
    assert check(model)


@onnx_function
class _SimpleBlock(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x)


class _SimpleStack(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.block1 = _SimpleBlock(dim, rngs=rngs)
        self.block2 = _SimpleBlock(dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.block2(self.block1(x))


def test_function_node_names_are_human_friendly():
    model = _SimpleStack(8, rngs=nnx.Rngs(0))
    onnx_model = to_onnx(
        model,
        inputs=[jax.ShapeDtypeStruct((2, 8), jnp.float32)],
        return_mode="proto",
        model_name="simple_stack",
    )

    fn_nodes = [
        node
        for node in onnx_model.graph.node
        if node.domain.startswith("custom") and node.op_type == "_SimpleBlock"
    ]
    assert [node.op_type for node in fn_nodes] == ["_SimpleBlock", "_SimpleBlock"]
    assert [node.name for node in fn_nodes] == ["_SimpleBlock_1", "_SimpleBlock_2"]

    functions = list(getattr(onnx_model, "functions", []))
    assert len(functions) == 2
    assert {fn.name for fn in functions} == {"_SimpleBlock"}
    domains = {fn.domain for fn in functions}
    assert len(domains) == 2
    assert domains == {
        "custom._SimpleBlock.1",
        "custom._SimpleBlock.2",
    }
    assert any(domain.startswith("custom.") for domain in domains)


@onnx_function
def _ordered_tensor_captures(x, *, right, left, unused):
    return x * left + right


def _reuse_ordered_tensor_captures(x, right, left, unused):
    first = _ordered_tensor_captures(x, right=right, left=left, unused=unused)
    return _ordered_tensor_captures(first, right=left, left=right, unused=unused)


def test_tensor_keyword_capture_order_and_unused_formal_survive_function_reuse():
    model = to_onnx(
        _reuse_ordered_tensor_captures,
        inputs=[jax.ShapeDtypeStruct((3,), jnp.float32)] * 4,
        input_names=["x", "right", "left", "unused"],
        output_names=["result"],
        model_name="ordered_tensor_captures",
        return_mode="proto",
    )

    functions = [
        function
        for function in model.functions
        if function.name == "_ordered_tensor_captures"
    ]
    assert len(functions) == 1
    (function,) = functions
    calls = [
        node
        for node in model.graph.node
        if (node.domain, node.op_type) == (function.domain, function.name)
    ]
    assert len(calls) == 2
    assert list(calls[0].input) == ["x", "right", "left", "unused"]
    assert list(calls[1].input) == [calls[0].output[0], "left", "right", "unused"]
    assert len(function.input) == 4

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    x = np.asarray([1, 2, 3], dtype=np.float32)
    right = np.asarray([4, 5, 6], dtype=np.float32)
    left = np.asarray([7, 8, 9], dtype=np.float32)
    expected = (x * left + right) * right + left
    feeds = {"x": x, "right": right, "left": left, "unused": x}
    (actual,) = session.run(None, feeds)
    np.testing.assert_allclose(actual, expected)

    feeds["unused"] = np.full((3,), 100.0, dtype=np.float32)
    (with_other_unused_capture,) = session.run(None, feeds)
    np.testing.assert_allclose(with_other_unused_capture, expected)


_CAPTURED_BIAS = np.asarray([0.5, 1.5, 2.5], dtype=np.float32)


@onnx_function
def _static_captured_affine(x, *, scale):
    return x * scale + _CAPTURED_BIAS


def _reuse_static_captured_affine(x):
    first = _static_captured_affine(x, scale=2.0)
    return _static_captured_affine(first, scale=2.0), _static_captured_affine(
        x, scale=3.0
    )


def test_static_keyword_signature_reuses_body_and_keeps_captured_constants():
    model = to_onnx(
        _reuse_static_captured_affine,
        inputs=[jax.ShapeDtypeStruct((3,), jnp.float32)],
        input_names=["x"],
        return_mode="proto",
    )
    functions = [
        function
        for function in model.functions
        if function.name == "_static_captured_affine"
    ]
    assert len(functions) == 2
    assert all(len(function.input) == 1 for function in functions)
    assert all(
        any(node.op_type == "Constant" for node in function.node)
        for function in functions
    )
    calls = [
        node for node in model.graph.node if node.op_type == "_static_captured_affine"
    ]
    assert len(calls) == 3
    assert calls[0].domain == calls[1].domain
    assert calls[2].domain != calls[0].domain

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    x = np.asarray([1, 2, 3], dtype=np.float32)
    repeated, changed = session.run(None, {"x": x})
    np.testing.assert_allclose(repeated, (x * 2 + _CAPTURED_BIAS) * 2 + _CAPTURED_BIAS)
    np.testing.assert_allclose(changed, x * 3 + _CAPTURED_BIAS)


@onnx_function
def _constant_keyword_scale(x, *, scale):
    return x * scale


def _reuse_scalar_constant_keyword(x):
    first = _constant_keyword_scale(x, scale=jnp.asarray(2.0, dtype=jnp.float32))
    return (
        _constant_keyword_scale(first, scale=jnp.asarray(2.0, dtype=jnp.float32)),
        _constant_keyword_scale(x, scale=jnp.asarray(3.0, dtype=jnp.float32)),
    )


def _reuse_array_constant_keyword(x):
    first = _constant_keyword_scale(
        x, scale=jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float32)
    )
    return (
        _constant_keyword_scale(
            first, scale=jnp.asarray([2.0, 3.0, 4.0], dtype=jnp.float32)
        ),
        _constant_keyword_scale(
            x, scale=jnp.asarray([3.0, 4.0, 5.0], dtype=jnp.float32)
        ),
    )


@pytest.mark.parametrize(
    "fn,scale,changed_scale",
    [
        (_reuse_scalar_constant_keyword, 2.0, 3.0),
        (
            _reuse_array_constant_keyword,
            np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
            np.asarray([3.0, 4.0, 5.0], dtype=np.float32),
        ),
    ],
)
def test_traced_constant_keywords_reuse_functions_without_retaining_tracers(
    fn, scale, changed_scale
):
    model = to_onnx(
        fn,
        inputs=[jax.ShapeDtypeStruct((3,), jnp.float32)],
        input_names=["x"],
        return_mode="proto",
    )
    calls = [
        node for node in model.graph.node if node.op_type == "_constant_keyword_scale"
    ]
    assert len(calls) == 3
    assert calls[0].domain == calls[1].domain
    assert [value.name for value in model.graph.input] == ["x"]

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    x = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    repeated, changed = session.run(None, {"x": x})
    np.testing.assert_allclose(repeated, x * scale * scale)
    np.testing.assert_allclose(changed, x * changed_scale)
