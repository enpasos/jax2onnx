# tests/extra_tests/test_bfloat16_opset_validity.py

"""bfloat16 exports stay inside the ONNX schemas of the target opset.

ONNX admitted bfloat16 operator by operator: ``Conv`` and pooling from opset
22, ``TopK`` from 24, ``Einsum`` from 28. Below that opset the export computes
the operator in float32 between bfloat16 values. See discussion #290.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
from onnx.reference import ReferenceEvaluator

from jax2onnx import StateInput, onnx_function, to_onnx

BF16 = np.dtype(jnp.bfloat16)
BFLOAT16 = onnx.TensorProto.BFLOAT16
FLOAT = onnx.TensorProto.FLOAT


def _array(shape: tuple[int, ...], seed: int, scale: float = 2.0) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=shape) * scale
    return values.astype(np.float32).astype(BF16)


_KERNEL = jnp.asarray(_array((4, 3, 3, 3), seed=1, scale=1.0))
_MATRIX = jnp.asarray(_array((5, 6), seed=2, scale=1.0))


def _conv(x: jax.Array, kernel: jax.Array = _KERNEL) -> jax.Array:
    return jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME")


def _export(fn: Callable[..., Any], *xs: np.ndarray, opset: int) -> onnx.ModelProto:
    model = to_onnx(fn, [jnp.asarray(x) for x in xs], opset=opset)
    onnx.checker.check_model(model, full_check=True)
    return model


def _elem_types(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph
    return {
        info.name: info.type.tensor_type.elem_type
        for info in (*graph.input, *graph.value_info, *graph.output)
    }


def _node(model: onnx.ModelProto, op_type: str) -> onnx.NodeProto:
    (node,) = [node for node in model.graph.node if node.op_type == op_type]
    return node


def _run(
    model: onnx.ModelProto, feeds: list[np.ndarray], expected: list[np.ndarray]
) -> list[np.ndarray]:
    """Run on ONNX Runtime; its NumPy interface takes bfloat16 only as raw bits."""
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    binding = session.io_binding()
    for meta, array in zip(session.get_inputs(), feeds, strict=True):
        if array.dtype == BF16:
            bits = np.ascontiguousarray(array).view(np.uint16)
            binding.bind_ortvalue_input(
                meta.name,
                ort.OrtValue.ortvalue_from_numpy_with_onnx_type(bits, BFLOAT16),
            )
        else:
            binding.bind_cpu_input(meta.name, array)
    results: list[np.ndarray] = []
    for meta, reference in zip(session.get_outputs(), expected, strict=True):
        if reference.dtype == BF16:
            bits = np.empty(reference.shape, dtype=np.uint16)
            binding.bind_ortvalue_output(
                meta.name,
                ort.OrtValue.ortvalue_from_numpy_with_onnx_type(bits, BFLOAT16),
            )
            results.append(bits.view(BF16))
        else:
            buffer = np.empty(reference.shape, dtype=reference.dtype)
            binding.bind_ortvalue_output(
                meta.name, ort.OrtValue.ortvalue_from_numpy(buffer)
            )
            results.append(buffer)
    session.run_with_iobinding(binding)
    return results


def _outputs(result: Any) -> list[np.ndarray]:
    leaves = result if isinstance(result, (tuple, list)) else [result]
    return [np.asarray(leaf) for leaf in leaves]


def _steps_apart(got: np.ndarray, expected: np.ndarray) -> int:
    """Largest distance in representable bfloat16 values."""

    def ordinal(array: np.ndarray) -> np.ndarray:
        bits = array.view(np.uint16).astype(np.int64)
        return np.where(bits & 0x8000, -(bits & 0x7FFF), bits)

    return int(np.max(np.abs(ordinal(got) - ordinal(expected))))


# One JAX primitive lowered to one ONNX operator that does not admit bfloat16 at
# this opset. JAX on CPU evaluates the primitive in float32 as well and rounds
# its result once, so the export reproduces JAX bit for bit.
_SINGLE_PRIMITIVE_CASES: dict[
    str, tuple[Callable[..., Any], tuple[int, ...], int, str]
] = {
    "conv_opset21": (_conv, (2, 3, 8, 8), 21, "Conv"),
    "matmul_opset23": (lambda x: x @ _MATRIX, (2, 4, 5), 23, "Einsum"),
    "einsum_opset23": (
        lambda x: jnp.einsum("bij,jk->bik", x, _MATRIX),
        (2, 4, 5),
        23,
        "Einsum",
    ),
    "top_k_opset23": (lambda x: jax.lax.top_k(x, 3), (4, 9), 23, "TopK"),
    "sin_opset21": (lambda x: jnp.sin(x), (4, 64), 21, "Sin"),
}


@pytest.mark.parametrize("name", _SINGLE_PRIMITIVE_CASES)
def test_operator_below_its_bfloat16_opset_runs_in_float32_and_matches_jax(
    name: str,
) -> None:
    fn, shape, opset, op_type = _SINGLE_PRIMITIVE_CASES[name]
    x = _array(shape, seed=0)
    model = _export(fn, x, opset=opset)

    types = _elem_types(model)
    operator = _node(model, op_type)
    floating_inputs = [
        types[value]
        for value in operator.input
        if types.get(value) != onnx.TensorProto.INT64
    ]
    assert floating_inputs and set(floating_inputs) == {FLOAT}
    assert types[operator.output[0]] == FLOAT
    assert types[model.graph.input[0].name] == BFLOAT16
    assert types[model.graph.output[0].name] == BFLOAT16

    expected = _outputs(fn(jnp.asarray(x)))
    got = _run(model, [x], expected)
    for actual, reference in zip(got, expected, strict=True):
        assert actual.dtype == reference.dtype
        np.testing.assert_array_equal(
            actual.view(np.uint16) if actual.dtype == BF16 else actual,
            reference.view(np.uint16) if reference.dtype == BF16 else reference,
        )


@pytest.mark.parametrize(
    ("fn", "op_type"),
    [
        (lambda x: jax.nn.softplus(x), "Softplus"),
        (lambda x: jax.nn.selu(x), "Selu"),
    ],
    ids=["softplus", "selu"],
)
def test_operator_that_fuses_several_primitives_rounds_once(
    fn: Callable[[jax.Array], jax.Array], op_type: str
) -> None:
    # JAX rounds to bfloat16 after every primitive of the activation; the single
    # ONNX operator computes in float32 and rounds once at its boundary.
    x = _array((4, 64), seed=0)
    model = _export(fn, x, opset=21)
    types = _elem_types(model)
    operator = _node(model, op_type)
    assert types[operator.input[0]] == FLOAT
    assert types[model.graph.output[0].name] == BFLOAT16

    expected = _outputs(fn(jnp.asarray(x)))
    (got,) = _run(model, [x], expected)
    assert _steps_apart(got, expected[0]) <= 4


@pytest.mark.parametrize("opset", [22, 23])
def test_operator_stays_in_bfloat16_once_the_opset_admits_it(opset: int) -> None:
    model = _export(_conv, _array((2, 3, 8, 8), seed=0), opset=opset)

    assert [node.op_type for node in model.graph.node] == ["Conv"]
    assert set(_elem_types(model).values()) == {BFLOAT16}
    assert {tensor.data_type for tensor in model.graph.initializer} == {BFLOAT16}


def test_constants_keep_their_bfloat16_payload() -> None:
    model = _export(_conv, _array((2, 3, 8, 8), seed=0), opset=21)

    (kernel,) = model.graph.initializer
    assert kernel.data_type == BFLOAT16
    assert onnx.numpy_helper.to_array(kernel).tobytes() == np.asarray(_KERNEL).tobytes()


def test_control_flow_bodies_are_valid_below_opset_22() -> None:
    def branches(x: jax.Array) -> jax.Array:
        return jax.lax.cond(x.sum() > 0, lambda v: _conv(v), lambda v: -_conv(v), x)

    def steps(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        def body(carry: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
            carry = jnp.tanh(_conv(carry)[:, :3])
            return carry, carry.sum()

        return jax.lax.scan(body, x, None, length=3)

    x = _array((1, 3, 8, 8), seed=0)
    for fn, op_type, attribute in (
        (branches, "If", "then_branch"),
        (steps, "Loop", "body"),
    ):
        model = _export(fn, x, opset=21)
        (body,) = [
            attr.g for attr in _node(model, op_type).attribute if attr.name == attribute
        ]
        body_ops = [node.op_type for node in body.node]
        assert "Conv" in body_ops and "Cast" in body_ops
        assert model.graph.output[0].type.tensor_type.elem_type == BFLOAT16


def test_bfloat16_through_control_flow_requires_opset_16() -> None:
    def branches(x: jax.Array) -> jax.Array:
        return jax.lax.cond(x.sum() > 0, lambda v: v, lambda v: -v, x)

    with pytest.raises(NotImplementedError, match=r"If .*opset 15.*opset >= 16"):
        to_onnx(branches, [jnp.asarray(_array((2, 3), seed=0))], opset=15)


@onnx_function
def _conv_block(x: jax.Array) -> jax.Array:
    return jnp.tanh(_conv(x))


def test_onnx_function_body_is_valid_below_opset_22() -> None:
    model = _export(
        lambda x: _conv_block(x) + 1, _array((1, 3, 8, 8), seed=0), opset=21
    )

    (function,) = model.functions
    body_ops = [node.op_type for node in function.node]
    assert body_ops.count("Conv") == 1 and "Cast" in body_ops
    assert model.graph.output[0].type.tensor_type.elem_type == BFLOAT16


class _ConvNet(eqx.Module):
    kernel: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.conv_general_dilated(x, self.kernel, (1, 1), "SAME")


def test_state_input_keeps_its_bfloat16_type_default_and_metadata() -> None:
    net = _ConvNet(_KERNEL)
    x = _array((2, 3, 8, 8), seed=0)
    model = to_onnx(
        net,
        [jnp.asarray(x)],
        opset=21,
        state=[StateInput(net.kernel, "kernel", {"pkg.demo.trainable": "true"})],
    )
    onnx.checker.check_model(model, full_check=True)

    assert [
        (value.name, value.type.tensor_type.elem_type) for value in model.graph.input
    ] == [("in_0", BFLOAT16), ("kernel", BFLOAT16)]
    (default,) = model.graph.initializer
    assert (default.name, default.data_type) == ("kernel", BFLOAT16)
    assert (
        onnx.numpy_helper.to_array(default).tobytes() == np.asarray(_KERNEL).tobytes()
    )
    assert {prop.key: prop.value for prop in default.metadata_props} == {
        "pkg.demo.trainable": "true"
    }

    expected = _outputs(net(jnp.asarray(x)))
    (got,) = _run(model, [x], expected)
    np.testing.assert_array_equal(got.view(np.uint16), expected[0].view(np.uint16))


def test_generator_typed_by_an_attribute_is_rejected_below_its_opset() -> None:
    def noisy(x: jax.Array) -> jax.Array:
        return x + jax.random.normal(jax.random.PRNGKey(0), x.shape, dtype=x.dtype)

    x = jnp.asarray(_array((2, 3), seed=0))
    with pytest.raises(
        NotImplementedError, match=r"RandomNormal .*opset 21.*opset >= 22"
    ):
        to_onnx(noisy, [x], opset=21)

    onnx.checker.check_model(to_onnx(noisy, [x], opset=22), full_check=True)


_NEGATIVE_INFINITY = np.array(-np.inf, dtype=BF16)
_POSITIVE_INFINITY = np.array(np.inf, dtype=BF16)

_WINDOW_CASES: dict[str, Callable[[jax.Array], jax.Array]] = {
    # lax.max with its identity reaches the converter as reduce_window_max.
    "max": lambda x: jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (2, 2), (2, 2), "VALID"
    ),
    "max_reducer": lambda x: jax.lax.reduce_window(
        x, _NEGATIVE_INFINITY, lambda a, b: jnp.maximum(a, b), (2, 2), (1, 1), "VALID"
    ),
    "min_reducer": lambda x: jax.lax.reduce_window(
        x, _POSITIVE_INFINITY, lambda a, b: jnp.minimum(a, b), (2, 2), (2, 2), "VALID"
    ),
}


@pytest.mark.parametrize("opset", [21, 23])
@pytest.mark.parametrize("name", _WINDOW_CASES)
def test_reduce_window_max_and_min_export_in_bfloat16(name: str, opset: int) -> None:
    fn = _WINDOW_CASES[name]
    x = _array((8, 8), seed=0)
    model = _export(fn, x, opset=opset)

    types = _elem_types(model)
    pool = _node(model, "MaxPool")
    # MaxPool admits bfloat16 from opset 22 and runs in float32 below it.
    assert types[pool.input[0]] == (BFLOAT16 if opset >= 22 else FLOAT)
    assert types[model.graph.output[0].name] == BFLOAT16

    (expected,) = _outputs(fn(jnp.asarray(x)))
    (got,) = ReferenceEvaluator(model).run(None, {model.graph.input[0].name: x})
    assert got.dtype == BF16
    np.testing.assert_array_equal(got.view(np.uint16), expected.view(np.uint16))
