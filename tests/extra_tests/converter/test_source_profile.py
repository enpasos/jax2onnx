# tests/extra_tests/converter/test_source_profile.py

"""Finite profile admission is typed, recursive, and independent of execution."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)


@pytest.fixture(autouse=True)
def _restore_jax_config() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    previous_precision = jax.config.jax_default_matmul_precision
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", None)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)
        jax.config.update("jax_default_matmul_precision", previous_precision)


def _trace(fn: Any, *specs: jax.ShapeDtypeStruct) -> Any:
    return conversion_api._trace_to_jaxpr(
        fn=fn,
        inputs=specs,
        input_params=None,
        enable_double_precision=False,
        inputs_as_nchw=None,
        outputs_as_nchw=None,
        input_names=None,
        output_names=None,
        precision_policy=PrecisionPolicy.SOURCE,
    ).jaxpr


def _spec(shape: tuple[int, ...] = (), dtype: Any = np.float32) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, dtype)


def _graph(
    name: str,
    inputs: list[jax.ShapeDtypeStruct],
    outputs: list[jax.ShapeDtypeStruct],
    params: dict[str, Any] | None = None,
) -> Any:
    invars = [SimpleNamespace(aval=spec) for spec in inputs]
    outvars = [SimpleNamespace(aval=spec) for spec in outputs]
    eqn = SimpleNamespace(
        primitive=SimpleNamespace(name=name),
        invars=invars,
        outvars=outvars,
        params=dict(params or {}),
        effects=(),
    )
    return SimpleNamespace(
        invars=invars, constvars=(), outvars=outvars, eqns=[eqn], effects=()
    )


def _dot(x: jax.Array, w: jax.Array, **kwargs: Any) -> jax.Array:
    batch = (0,) if x.ndim == 3 else ()
    return jax.lax.dot_general(
        x,
        w,
        (((x.ndim - 1,), (w.ndim - 2,)), (batch, batch)),
        **kwargs,
    )


@onnx_function
def _mixed_results(x: jax.Array) -> tuple[jax.Array, ...]:
    return (
        x.astype(jnp.float16),
        x > np.float32(0),
        jax.lax.bitcast_convert_type(x, jnp.uint32),
    )


@onnx_function
def _unsupported_function(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


@onnx_function
def _function_with_cond(x: jax.Array) -> tuple[jax.Array, ...]:
    return jax.lax.cond(
        x > np.float32(0),
        lambda value: _mixed_results(value),
        lambda value: _mixed_results(-value),
        x,
    )


@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.float32, np.float64])
def test_profile_admits_floating_arithmetic_and_explicit_conversions(
    dtype: Any,
) -> None:
    def model(x: jax.Array) -> tuple[jax.Array, ...]:
        positive = x * x
        wide = x.astype(jnp.float64)
        return (
            x + x,
            x - x,
            positive,
            x / x,
            -x,
            jax.lax.sqrt(positive),
            jnp.sqrt(positive),
            jax.lax.broadcast_in_dim(x, (2, 3), (0,)),
            jax.lax.reduce_sum(wide, (0,)),
            wide.astype(dtype),
            x > x,
        )

    validate_source_jaxpr(_trace(model, _spec((2,), dtype)), 23)


def test_profile_preserves_special_constant_payload_and_weak_metadata() -> None:
    captured = np.asarray([np.nan, np.inf, -np.inf, -0.0], dtype=np.float32)
    original = captured.tobytes()
    spec = jax.ShapeDtypeStruct((), np.float64, weak_type=True)
    trace = _trace(lambda x: (captured, x + captured), spec)
    before = [
        (var.aval.dtype, var.aval.shape, var.aval.weak_type) for var in trace.invars
    ]
    validate_source_jaxpr(trace, 23)
    assert captured.tobytes() == original
    assert [
        (var.aval.dtype, var.aval.shape, var.aval.weak_type) for var in trace.invars
    ] == before


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("precision", [None, jax.lax.Precision.DEFAULT])
def test_profile_admits_only_bounded_preferred_matrix_case(
    rank: int, precision: Any
) -> None:
    batch = (2,) if rank == 3 else ()
    trace = _trace(
        lambda x, w: _dot(
            x, w, preferred_element_type=jnp.float32, precision=precision
        ),
        _spec(batch + (3, 4), np.float16),
        _spec(batch + (4, 2), np.float16),
    )
    validate_source_jaxpr(trace, 23)


@pytest.mark.parametrize(
    "precision",
    [
        jax.lax.Precision.HIGH,
        jax.lax.Precision.HIGHEST,
        jax.lax.DotAlgorithmPreset.F16_F16_F32,
    ],
)
def test_profile_rejects_matrix_precision_requirements_with_context(
    precision: Any,
) -> None:
    trace = _trace(
        lambda x, w: _dot(
            x, w, preferred_element_type=jnp.float32, precision=precision
        ),
        _spec((3, 4), np.float16),
        _spec((4, 2), np.float16),
    )
    with pytest.raises(SourcePrecisionError) as captured:
        validate_source_jaxpr(trace, 23)
    error = captured.value
    assert error.category == "profile-exclusion"
    assert error.phase == "profile-validation"
    assert error.location == "equation 0"
    assert error.primitive == "dot_general"
    assert error.requested_dtypes == ("float16", "float16", "float32")
    assert error.opset == 23
    assert "precision" in error.precision_params
    assert "default precision" in error.requirement
    assert error.action
    nested = error.with_context("function named_model")
    assert nested.location == "function named_model/equation 0"
    assert nested.precision_params == error.precision_params


@pytest.mark.parametrize(
    ("shape", "dtype", "preferred"),
    [
        ((0, 2), np.float16, np.float32),
        ((2, 2), np.float16, None),
        ((2, 2), jnp.bfloat16, np.float32),
    ],
)
def test_profile_rejects_unaudited_matrix_types_and_empty_shapes(
    shape: tuple[int, ...], dtype: Any, preferred: Any
) -> None:
    trace = _trace(
        lambda x, w: _dot(x, w, preferred_element_type=preferred),
        _spec(shape, dtype),
        _spec((shape[-1], 2), dtype),
    )
    with pytest.raises(SourcePrecisionError, match="matrix .* outside the profile"):
        validate_source_jaxpr(trace, 23)


def test_profile_admits_ordinary_fp32_with_same_dtype_preference() -> None:
    trace = _trace(
        lambda x, w: _dot(x, w, preferred_element_type=np.float32),
        _spec((2, 2)),
        _spec((2, 2)),
    )
    validate_source_jaxpr(trace, 23)


def test_profile_rejects_valid_but_unaudited_matrix_axes() -> None:
    trace = _trace(
        lambda x, w: jax.lax.dot_general(
            x, w, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
        ),
        _spec((2, 3), np.float16),
        _spec((2, 4), np.float16),
    )
    with pytest.raises(SourcePrecisionError, match="matrix dimensions are outside"):
        validate_source_jaxpr(trace, 23)


def test_profile_bitcast_representation_failure_is_actionable() -> None:
    trace = _trace(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32), _spec((2,)))
    validate_source_jaxpr(trace, 26)
    with pytest.raises(SourcePrecisionError) as captured:
        validate_source_jaxpr(trace, 23)
    error = captured.value
    assert error.category == "representation-limit"
    assert error.phase == "representation"
    assert error.primitive == "bitcast_convert_type"
    assert error.requested_dtypes == ("float32", "uint32")
    assert error.opset == 23
    assert "opset=26" in error.action
    assert "never upgrades" in error.action


@pytest.mark.parametrize(
    ("source", "target"),
    [(np.float32, np.int32), (np.uint32, np.float32), (np.float64, np.uint32)],
)
def test_profile_excludes_other_bitcast_pairs(source: Any, target: Any) -> None:
    trace = _trace(
        lambda x: jax.lax.bitcast_convert_type(x, target), _spec((2,), source)
    )
    with pytest.raises(SourcePrecisionError) as captured:
        validate_source_jaxpr(trace, 26)
    assert captured.value.category == "profile-exclusion"


@pytest.mark.parametrize("function_contains_cond", [False, True])
def test_profile_admits_nested_function_and_cond_with_mixed_result_roles(
    function_contains_cond: bool,
) -> None:
    def model(x: jax.Array) -> tuple[jax.Array, ...]:
        if function_contains_cond:
            return _function_with_cond(x)
        return jax.lax.cond(
            x > np.float32(0),
            lambda value: _mixed_results(value),
            lambda value: _mixed_results(-value),
            x,
        )

    exported = conversion_api._convert_to_ir(
        fn=model,
        inputs=[_spec()],
        input_params=None,
        model_name="profile_nested_scopes",
        opset=26,
        enable_double_precision=False,
        record_primitive_calls_file=None,
        precision_policy=PrecisionPolicy.SOURCE,
        jaxpr_validator=validate_source_jaxpr,
    )
    assert [value.dtype for value in exported.graph.outputs] == [
        ir.DataType.FLOAT16,
        ir.DataType.BOOL,
        ir.DataType.UINT32,
    ]
    assert exported.functions
    onnx.checker.check_model(ir.to_proto(exported), full_check=True)


def test_profile_reports_nested_cond_location_and_validates_actual_function_body() -> (
    None
):
    trace = _trace(
        lambda x: jax.lax.cond(
            x > np.float32(0), lambda y: -y, lambda y: jnp.sin(y), x
        ),
        _spec(),
    )
    with pytest.raises(SourcePrecisionError) as captured:
        validate_source_jaxpr(trace, 23)
    assert "/branch " in captured.value.location
    assert captured.value.primitive in {"sin", "jax.numpy.sin"}
    with pytest.raises(SourcePrecisionError) as function_error:
        conversion_api._convert_to_ir(
            fn=lambda x: _unsupported_function(x),
            inputs=[_spec()],
            input_params=None,
            model_name="unsupported_body",
            opset=23,
            enable_double_precision=False,
            record_primitive_calls_file=None,
            precision_policy=PrecisionPolicy.SOURCE,
            jaxpr_validator=validate_source_jaxpr,
        )
    assert function_error.value.primitive in {"sin", "jax.numpy.sin"}


@pytest.mark.parametrize(
    "model",
    [
        lambda x: jnp.sin(x),
        lambda x: jax.jit(lambda value: jnp.sin(value))(x),
        lambda x: jax.lax.fori_loop(0, 2, lambda index, value: value + value, x),
        lambda x: x.astype(jnp.int32) + np.int32(1),
        lambda x: (jax.debug.print("value={x}", x=x), x)[1],
    ],
)
def test_profile_rejects_current_functionality_outside_the_finite_milestone(
    model: Any,
) -> None:
    with pytest.raises(SourcePrecisionError) as captured:
        validate_source_jaxpr(_trace(model, _spec()), 23)
    assert captured.value.category == "profile-exclusion"


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (_graph("add", [_spec()], [_spec()]), "operand/result count"),
        (_graph("add", [_spec(), _spec()], [_spec((), np.float64)]), "result dtype"),
        (_graph("gt", [_spec(), _spec()], [_spec()]), "result dtype"),
        (_graph("neg", [_spec((2,))], [_spec((3,))]), "result shape"),
        (
            _graph(
                "broadcast_in_dim",
                [_spec((2, 2))],
                [_spec((2, 2))],
                {"shape": (2, 2), "broadcast_dimensions": (0, 0)},
            ),
            "broadcast dimensions",
        ),
        (
            _graph("reduce_sum", [_spec((2, 3))], [_spec((3,))], {"axes": (0, 0)}),
            "reduction axes",
        ),
        (
            _graph(
                "convert_element_type",
                [_spec()],
                [_spec((), np.float16)],
                {"new_dtype": np.dtype(np.float64), "weak_type": False},
            ),
            "cast attribute",
        ),
        (
            _graph(
                "bitcast_convert_type",
                [_spec((2,))],
                [_spec((3,), np.uint32)],
                {"new_dtype": np.uint32},
            ),
            "result shape",
        ),
        (
            _graph("reduce_sum", [_spec((2,))], [_spec()]),
            "missing primitive parameters",
        ),
    ],
)
def test_profile_distinguishes_inconsistent_traced_metadata(
    graph: Any, message: str
) -> None:
    with pytest.raises(SourcePrecisionError, match=message) as captured:
        validate_source_jaxpr(graph, 26)
    assert captured.value.category == "lowering-defect"


def test_profile_accepts_integer_data_and_requires_boolean_cond_origin() -> None:
    graph = _graph(
        "convert_element_type",
        [_spec((), np.bool_)],
        [_spec((), np.int32)],
        {"new_dtype": np.dtype(np.int32), "weak_type": False},
    )
    # Ordinary integer data may be returned. Explicit placement remains a
    # negative control, and cond still requires its own Boolean-origin proof.
    validate_source_jaxpr(graph, 23)
    graph.eqns[0].params["sharding"] = object()
    with pytest.raises(SourcePrecisionError, match="placement"):
        validate_source_jaxpr(graph, 23)
    malformed = _graph("cond", [_spec(), _spec()], [_spec()], {"branches": ()})
    with pytest.raises(
        SourcePrecisionError, match="lacks an audited Boolean conversion"
    ):
        validate_source_jaxpr(malformed, 23)
    trace = _trace(lambda x: jax.lax.cond(x > 0, lambda y: -y, lambda y: y, x), _spec())
    validate_source_jaxpr(trace, 23)


def test_profile_rejects_symbolic_shapes_and_unsupported_opsets() -> None:
    graph = _graph("neg", [jax.ShapeDtypeStruct(("B",), np.float32)], [_spec((2,))])
    with pytest.raises(SourcePrecisionError, match="rank-two") as captured:
        validate_source_jaxpr(graph, 23)
    assert captured.value.location == "input 0"
    assert captured.value.requested_dtypes == ("float32",)
    with pytest.raises(SourcePrecisionError, match="opset 25 is outside"):
        validate_source_jaxpr(_trace(lambda x: -x, _spec()), 25)


def test_profile_rejects_unaudited_arithmetic_attributes() -> None:
    graph = _graph("sqrt", [_spec()], [_spec()], {"accuracy": "highest"})
    with pytest.raises(SourcePrecisionError, match="accuracy is outside") as captured:
        validate_source_jaxpr(graph, 23)
    assert captured.value.category == "profile-exclusion"
    assert captured.value.precision_params == {"accuracy": "'highest'"}
