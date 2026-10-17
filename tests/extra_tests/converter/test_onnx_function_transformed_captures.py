# tests/extra_tests/converter/test_onnx_function_transformed_captures.py

"""Dynamic function keywords participate in JAX transformations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function, to_onnx


@onnx_function
def _transformed_affine(
    x: jax.Array, *, scale: jax.Array, bias: jax.Array
) -> jax.Array:
    return x * scale + bias


def _call_derived_affine(x: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    return _transformed_affine(x, scale=scale + 1.0, bias=bias * 2.0)


def _jit_derived_affine(x, scale, bias):
    return jax.jit(_call_derived_affine)(x, scale, bias)


def _vmap_derived_affine(x, scale, bias):
    return jax.vmap(_call_derived_affine, in_axes=(0, 0, None))(x, scale, bias)


def _vmap_keyword_only(x, scale, bias):
    return jax.vmap(_call_derived_affine, in_axes=(None, 0, None))(x, scale, bias)


def _vmap_nonleading_axis(x, scale, bias):
    return jax.vmap(_call_derived_affine, in_axes=(1, 1, None))(x, scale, bias)


def _jit_vmap_derived_affine(x, scale, bias):
    return jax.jit(_vmap_derived_affine)(x, scale, bias)


def _checked_session(model: onnx.ModelProto) -> ort.InferenceSession:
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


@pytest.mark.parametrize(
    "target,x,scale,expected",
    [
        (
            _jit_derived_affine,
            [[1, 2], [3, 4], [5, 6]],
            [[2, 3], [4, 5], [6, 7]],
            [[3.5, 7], [15.5, 23], [35.5, 47]],
        ),
        (
            _vmap_derived_affine,
            [[1, 2], [3, 4], [5, 6]],
            [[2, 3], [4, 5], [6, 7]],
            [[3.5, 7], [15.5, 23], [35.5, 47]],
        ),
        (
            _vmap_keyword_only,
            [1, 2],
            [[2, 3], [4, 5], [6, 7]],
            [[3.5, 7], [5.5, 11], [7.5, 15]],
        ),
        (
            _vmap_nonleading_axis,
            [[1, 3, 5], [2, 4, 6]],
            [[2, 4, 6], [3, 5, 7]],
            [[3.5, 7], [15.5, 23], [35.5, 47]],
        ),
        (
            _jit_vmap_derived_affine,
            [[1, 2], [3, 4], [5, 6]],
            [[2, 3], [4, 5], [6, 7]],
            [[3.5, 7], [15.5, 23], [35.5, 47]],
        ),
    ],
    ids=["jit", "vmap", "vmap-keyword-only", "vmap-nonleading-axis", "jit-vmap"],
)
def test_transforms_preserve_dynamic_keyword_values(target, x, scale, expected) -> None:
    x = np.asarray(x, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    bias = np.asarray([0.25, -0.5], dtype=np.float32)
    model = to_onnx(
        target,
        inputs=[
            jax.ShapeDtypeStruct(value.shape, value.dtype) for value in (x, scale, bias)
        ],
        input_names=["x", "scale", "bias"],
    )
    session = _checked_session(model)
    (actual,) = session.run(None, {"x": x, "scale": scale, "bias": bias})
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float32))

    # Reusing the exported graph with new keyword values must change its result.
    (changed,) = session.run(None, {"x": x, "scale": scale, "bias": bias + 1.0})
    np.testing.assert_allclose(changed, np.asarray(expected, dtype=np.float32) + 2.0)


def _call_literal_keywords(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return (
        _transformed_affine(
            x,
            scale=jnp.asarray(2.0, dtype=jnp.float32),
            bias=jnp.asarray(0.5, dtype=jnp.float32),
        ),
        _transformed_affine(
            x,
            scale=jnp.asarray(3.0, dtype=jnp.float32),
            bias=jnp.asarray(-0.5, dtype=jnp.float32),
        ),
    )


def _jit_literal_keywords(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jax.jit(_call_literal_keywords)(x)


def test_jit_keeps_distinct_scalar_literal_keyword_values() -> None:
    model = to_onnx(
        _jit_literal_keywords,
        inputs=[jax.ShapeDtypeStruct((3,), jnp.float32)],
        input_names=["x"],
    )
    assert [value.name for value in model.graph.input] == ["x"]
    session = _checked_session(model)
    for values in ([1, 2, 3], [-4, 0, 7]):
        x = np.asarray(values, dtype=np.float32)
        first, second = session.run(None, {"x": x})
        np.testing.assert_allclose(first, x * 2.0 + 0.5)
        np.testing.assert_allclose(second, x * 3.0 - 0.5)


@onnx_function
def _select_literal_flag(
    x: jax.Array, *, deterministic: bool | jax.Array = True
) -> jax.Array:
    return jnp.where(deterministic, x, x * 2.0)


@onnx_function
def _override_external_flag(
    x: jax.Array, *, deterministic: bool | jax.Array = True
) -> jax.Array:
    return _select_literal_flag(x, deterministic=jnp.asarray(False))


def _call_overridden_flag(
    x: jax.Array, *, deterministic: bool | jax.Array = True
) -> jax.Array:
    return _override_external_flag(x, deterministic=deterministic)


def test_explicit_literal_capture_overrides_same_named_external_flag() -> None:
    model = to_onnx(
        _call_overridden_flag,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_names=["x"],
        input_params={"deterministic": True},
    )
    session = _checked_session(model)
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    for external_flag in (True, False):
        (actual,) = session.run(
            None,
            {"x": x, "deterministic": np.asarray(external_flag, dtype=np.bool_)},
        )
        np.testing.assert_allclose(actual, x * 2.0)


def _literal_flag_conditional(x, predicate, flag):
    return jax.lax.cond(
        predicate,
        lambda value: _select_literal_flag(value, deterministic=flag),
        lambda value: value - 1.0,
        x,
    )


@onnx_function
def _python_false_in_function_if(x, predicate, *, deterministic=True):
    return _literal_flag_conditional(x, predicate, False)


@onnx_function
def _python_true_in_function_if(x, predicate, *, deterministic=True):
    return _literal_flag_conditional(x, predicate, True)


@onnx_function
def _array_false_in_function_if(x, predicate, *, deterministic=True):
    return _literal_flag_conditional(x, predicate, jnp.asarray(False))


@onnx_function
def _array_true_in_function_if(x, predicate, *, deterministic=True):
    return _literal_flag_conditional(x, predicate, jnp.asarray(True))


def _call_python_false_in_function_if(x, predicate, *, deterministic=True):
    return _python_false_in_function_if(x, predicate, deterministic=deterministic)


def _call_python_true_in_function_if(x, predicate, *, deterministic=True):
    return _python_true_in_function_if(x, predicate, deterministic=deterministic)


def _call_array_false_in_function_if(x, predicate, *, deterministic=True):
    return _array_false_in_function_if(x, predicate, deterministic=deterministic)


def _call_array_true_in_function_if(x, predicate, *, deterministic=True):
    return _array_true_in_function_if(x, predicate, deterministic=deterministic)


@pytest.mark.parametrize(
    "target,literal",
    [
        (_call_python_false_in_function_if, False),
        (_call_python_true_in_function_if, True),
        (_call_array_false_in_function_if, False),
        (_call_array_true_in_function_if, True),
    ],
    ids=["python-false", "python-true", "array-false", "array-true"],
)
def test_function_if_keeps_explicit_literal_keyword(target, literal) -> None:
    model = to_onnx(
        target,
        inputs=[
            jax.ShapeDtypeStruct((2,), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        ],
        input_names=["x", "predicate"],
        input_params={"deterministic": True},
    )
    assert any(
        node.op_type == "If" for function in model.functions for node in function.node
    ), "The explicit literal must cross an If inside an ONNX function."
    session = _checked_session(model)
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    for external_flag in (True, False):
        for predicate in (True, False):
            (actual,) = session.run(
                None,
                {
                    "x": x,
                    "predicate": np.asarray(predicate, dtype=np.bool_),
                    "deterministic": np.asarray(external_flag, dtype=np.bool_),
                },
            )
            expected = (x if literal else x * 2.0) if predicate else x - 1.0
            np.testing.assert_allclose(actual, expected)


def _root_if_promoted_flag(x, predicate, *, deterministic=True):
    return _literal_flag_conditional(x, predicate, deterministic)


def test_root_if_promotes_declared_runtime_flag() -> None:
    model = to_onnx(
        _root_if_promoted_flag,
        inputs=[
            jax.ShapeDtypeStruct((2,), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        ],
        input_names=["x", "predicate"],
        input_params={"deterministic": True},
    )
    assert any(node.op_type == "If" for node in model.graph.node)
    session = _checked_session(model)
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    for external_flag in (True, False):
        for predicate in (True, False):
            (actual,) = session.run(
                None,
                {
                    "x": x,
                    "predicate": np.asarray(predicate, dtype=np.bool_),
                    "deterministic": np.asarray(external_flag, dtype=np.bool_),
                },
            )
            expected = (x if external_flag else x * 2.0) if predicate else x - 1.0
            np.testing.assert_allclose(actual, expected)


@onnx_function
def _middle_with_explicit_false(x, *, deterministic=True):
    return _select_literal_flag(x)


@onnx_function
def _outer_with_defaulted_inner_flag(x, *, deterministic=True):
    return _middle_with_explicit_false(x, deterministic=False)


@onnx_function
def _outer_with_direct_defaulted_flag(x, *, deterministic=True):
    return _select_literal_flag(x)


def _call_outer_with_defaulted_inner_flag(x, *, deterministic=True):
    return _outer_with_defaulted_inner_flag(x, deterministic=deterministic)


def _call_outer_with_direct_defaulted_flag(x, *, deterministic=True):
    return _outer_with_direct_defaulted_flag(x, deterministic=deterministic)


@pytest.mark.parametrize(
    "target",
    [_call_outer_with_direct_defaulted_flag, _call_outer_with_defaulted_inner_flag],
    ids=["direct-default", "through-explicit-false"],
)
def test_nested_function_default_stays_local_without_dynamic_formal(target) -> None:
    model = to_onnx(
        target,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_names=["x"],
        input_params={"deterministic": True},
    )
    session = _checked_session(model)
    x = np.asarray([2.0, 3.0], dtype=np.float32)
    for external_flag in (True, False):
        (actual,) = session.run(
            None,
            {"x": x, "deterministic": np.asarray(external_flag, dtype=np.bool_)},
        )
        # The middle call explicitly receives False; the inner call omits the
        # keyword and must retain its own True default instead of a global flag.
        np.testing.assert_allclose(actual, x)
