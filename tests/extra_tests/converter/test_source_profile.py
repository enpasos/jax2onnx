# tests/extra_tests/converter/test_source_profile.py

"""Finite profile admission is typed, recursive, and independent of execution."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function
from jax2onnx._compat.jax import ClosedJaxpr
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


@pytest.mark.parametrize("boundary", ["root", "jit"])
@pytest.mark.parametrize("role", ["formal", "constant"])
def test_opset16_rejects_unused_native_scope_types(boundary: str, role: str) -> None:
    """Unused captured/formal BF16 still belongs to the actual native scope."""
    unused = _spec((2,), jnp.bfloat16)
    kept = _spec((2,), np.float32)

    def target(ignored: Any, value: Any) -> Any:
        return value

    trace = jax.make_jaxpr(
        jax.jit(target, keep_unused=True) if boundary == "jit" else target
    )(unused, kept)
    body = trace.jaxpr
    if role == "constant":
        # Rebind an actual unused native formal as an actual captured constant;
        # no output/equation is changed and the FP32 identity stays independent.
        body = body.replace(invars=body.invars[1:], constvars=body.invars[:1])
        trace = ClosedJaxpr(body, [np.zeros((2,), dtype=jnp.bfloat16)])
    validate_source_jaxpr(trace.jaxpr, 23)
    validate_source_jaxpr(trace.jaxpr, 26)
    with pytest.raises(SourcePrecisionError) as captured:
        validate_source_jaxpr(trace.jaxpr, 16)
    error = captured.value
    assert error.category == "profile-exclusion" and error.opset == 16
    assert error.requested_dtypes == ("bfloat16",)
    assert "scope dtype" in str(error)
    if boundary == "jit":
        assert "/jit body" in error.location
        assert "input 0" in error.location
    else:
        assert error.location == ("input 0" if role == "formal" else "constant 0")


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
def _unadmitted_erf_function(x: jax.Array) -> jax.Array:
    return jax.lax.erf(x)


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
    shape: tuple[int, ...], dtype: Any, preferred: Any, tmp_path: Path
) -> None:
    def native(x: Any, w: Any) -> Any:
        return _dot(x, w, preferred_element_type=preferred)

    trace = _trace(
        native,
        _spec(shape, dtype),
        _spec((shape[-1], 2), dtype),
    )
    if shape == (0, 2) and dtype is np.float16 and preferred is np.float32:
        from tests.extra_tests.converter.test_source_generalized_contraction import (
            observe_contraction,
        )

        validate_source_jaxpr(trace, 23)
        values = np.ones(shape, dtype), np.ones((shape[-1], 2), dtype)
        observe_contraction(
            native, values, tmp_path, expected=(np.empty((0, 2), np.float32),)
        )
    else:
        with pytest.raises(SourcePrecisionError, match="matrix .* outside the profile"):
            validate_source_jaxpr(trace, 23)


def test_profile_admits_ordinary_fp32_with_same_dtype_preference() -> None:
    trace = _trace(
        lambda x, w: _dot(x, w, preferred_element_type=np.float32),
        _spec((2, 2)),
        _spec((2, 2)),
    )
    validate_source_jaxpr(trace, 23)


def test_profile_rejects_valid_but_unaudited_matrix_axes(tmp_path: Path) -> None:
    from tests.extra_tests.converter.test_source_generalized_contraction import (
        observe_contraction,
    )

    def native(x: Any, w: Any) -> Any:
        return jax.lax.dot_general(
            x, w, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )

    trace = _trace(native, _spec((2, 3), np.float16), _spec((2, 4), np.float16))
    validate_source_jaxpr(trace, 23)
    values = np.ones((2, 3), np.float16), np.ones((2, 4), np.float16)
    observe_contraction(
        native, values, tmp_path, expected=(np.full((3, 4), 2, np.float32),)
    )


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
    validate_source_jaxpr(trace, 23)
    admitted = conversion_api._convert_to_ir(
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
    assert admitted.graph.outputs[0].dtype == ir.DataType.FLOAT
    assert any(
        node.op_type == "Sin"
        for function in admitted.functions.values()
        for node in function.graph
    )
    unadmitted = _trace(
        lambda x: jax.lax.cond(
            x > np.float32(0), lambda y: -y, lambda y: jax.lax.erf(y), x
        ),
        _spec(),
    )
    with pytest.raises(SourcePrecisionError) as captured:
        validate_source_jaxpr(unadmitted, 23)
    assert "/branch " in captured.value.location
    assert captured.value.primitive == "erf"
    with pytest.raises(SourcePrecisionError) as function_error:
        conversion_api._convert_to_ir(
            fn=lambda x: _unadmitted_erf_function(x),
            inputs=[_spec()],
            input_params=None,
            model_name="unadmitted_erf_body",
            opset=23,
            enable_double_precision=False,
            record_primitive_calls_file=None,
            precision_policy=PrecisionPolicy.SOURCE,
            jaxpr_validator=validate_source_jaxpr,
        )
    assert function_error.value.primitive == "erf"


@pytest.mark.parametrize(
    "model",
    [
        lambda x: jnp.sin(x),
        lambda x: jax.jit(lambda value: jnp.sin(value))(x),
        lambda x: jax.lax.fori_loop(0, 2, lambda index, value: value + value, x),
        lambda x: (jax.debug.print("value={x}", x=x), x)[1],
    ],
)
def test_profile_rejects_current_functionality_outside_the_finite_milestone(
    model: Any,
) -> None:
    trace = _trace(model, _spec())
    # Preserve the original four parameter IDs/callables. Native Sin/JIT and
    # the static loop carrier are admitted; effects retain their negative contract.
    if len(trace.eqns) == 1 and trace.eqns[0].primitive.name in {
        "sin",
        "jax.numpy.sin",
        "jit",
        "pjit",
        "scan",
    }:
        validate_source_jaxpr(trace, 23)
    else:
        with pytest.raises(SourcePrecisionError) as captured:
            validate_source_jaxpr(trace, 23)
        assert captured.value.category == "profile-exclusion"


@pytest.mark.parametrize("model", [lambda x: x.astype(jnp.int32) + np.int32(1)])
def test_profile_admits_original_integer_cast_add_with_native_overflow(
    model: Any,
) -> None:
    from jax2onnx.precision import SampleCriteria
    from tests.extra_tests.converter.test_source_pointwise_mul_exp import (
        _accept,
        _artifact,
    )

    # Retain the original scalar F32 callable and prove its actually consumed
    # conversion, native I32 addition and exact discrete result contract.
    original = _trace(model, _spec())
    validate_source_jaxpr(original, 23)
    cast = next(
        eqn for eqn in original.eqns if eqn.primitive.name == "convert_element_type"
    )
    addition = next(eqn for eqn in original.eqns if eqn.primitive.name == "add")
    assert np.dtype(cast.params["new_dtype"]) == np.dtype(np.int32)
    assert addition.invars[0] is cast.outvars[0]
    assert all(
        np.dtype(var.aval.dtype) == np.dtype(np.int32)
        for var in (*addition.invars, *addition.outvars, *original.outvars)
    )
    assert original.outvars[0].aval.weak_type is False

    def observed(x: Any) -> Any:
        result = model(x)
        return x, result, jax.lax.eq(result, np.int32(-(2**31))), x.astype(jnp.int32)

    # F64 represents both I32 endpoints exactly. Keep the original operation,
    # then expose its result, threshold consumer and unchanged source value.
    values = (np.asarray([1.75, -1.75, 2**31 - 1, -(2**31)], np.float64),)
    specs = [jax.ShapeDtypeStruct((4,), np.float64)]
    _, artifact, _, references = _artifact(observed, specs, feeds=(values,))
    np.testing.assert_array_equal(
        references[0][0][1],
        np.asarray([2, 0, -(2**31), -(2**31) + 1], np.int32),
        strict=True,
    )
    _accept(artifact, (values,), references, SampleCriteria("bitwise"))


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
    graph = _graph(
        "neg", [jax.ShapeDtypeStruct(("B", 2, 3, 4, 5), np.float32)], [_spec((2,))]
    )
    with pytest.raises(
        SourcePrecisionError, match="rank-one through rank-four"
    ) as captured:
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


@pytest.mark.parametrize("primitive", ["and", "or", "xor", "not"])
def test_opset16_distinguishes_boolean_and_integer_bitwise(primitive: str) -> None:
    arity = 1 if primitive == "not" else 2
    boolean = _graph(
        primitive, [_spec((2, 3), np.bool_)] * arity, [_spec((2, 3), np.bool_)]
    )
    integer = _graph(
        primitive, [_spec((2, 3), np.int32)] * arity, [_spec((2, 3), np.int32)]
    )
    for opset in (16, 23, 26):
        validate_source_jaxpr(boolean, opset)
    for opset in (23, 26):
        validate_source_jaxpr(integer, opset)
    with pytest.raises(
        SourcePrecisionError, match="integer bitwise.*opset-16"
    ) as caught:
        validate_source_jaxpr(integer, 16)
    assert caught.value.primitive == primitive
    assert caught.value.opset == 16
    assert caught.value.category == "profile-exclusion"


def test_opset16_keeps_bitcast_26_only() -> None:
    trace = _trace(lambda x: jax.lax.bitcast_convert_type(x, jnp.uint32), _spec((2,)))
    validate_source_jaxpr(trace, 26)
    for opset in (16, 23):
        with pytest.raises(
            SourcePrecisionError, match="BitCast requires opset 26"
        ) as caught:
            validate_source_jaxpr(trace, opset)
        assert caught.value.primitive == "bitcast_convert_type"
        assert caught.value.opset == opset
        assert caught.value.category == "representation-limit"


def test_shared_shape_widening_keeps_symbolic_rank_one_matrix_rejection() -> None:
    (batch,) = export.symbolic_shape("B")
    graph = _graph(
        "dot_general",
        [
            jax.ShapeDtypeStruct((batch,), np.float32),
            jax.ShapeDtypeStruct((batch,), np.float32),
        ],
        [jax.ShapeDtypeStruct((), np.float32)],
        {"dimension_numbers": (((0,), (0,)), ((), ()))},
    )
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 23)
    assert caught.value.primitive == "dot_general"
    assert caught.value.category == "profile-exclusion"


def test_profile_admits_native_canonical_symbolic_rank_four_tensor_matrix() -> None:
    (batch,) = export.symbolic_shape("B")
    specs = (
        jax.ShapeDtypeStruct((batch, 2, 3, 4), np.float32),
        jax.ShapeDtypeStruct((4, 5), np.float32),
    )
    dimensions = (((3,), (0,)), ((), ()))

    def model(x: jax.Array, weight: jax.Array) -> jax.Array:
        return jax.lax.dot_general(x, weight, dimensions)

    native = jax.make_jaxpr(model)(*specs).jaxpr
    source = _trace(model, *specs)

    def metadata(graph: Any) -> tuple[Any, ...]:
        variables = (*graph.invars, *graph.outvars)
        return (
            tuple(
                (
                    tuple(var.aval.shape),
                    np.dtype(var.aval.dtype),
                    bool(var.aval.weak_type),
                )
                for var in variables
            ),
            tuple((eqn.primitive.name, dict(eqn.params)) for eqn in graph.eqns),
        )

    native_metadata = metadata(native)
    source_metadata = metadata(source)
    assert source_metadata == native_metadata
    for graph in (native, source):
        assert not graph.effects and not graph.constvars
        assert len(graph.invars) == 2 and len(graph.outvars) == 1
        assert tuple(graph.invars[0].aval.shape) == (batch, 2, 3, 4)
        assert tuple(graph.invars[1].aval.shape) == (4, 5)
        assert tuple(graph.outvars[0].aval.shape) == (batch, 2, 3, 5)
        assert all(
            np.dtype(var.aval.dtype) == np.dtype(np.float32) and not var.aval.weak_type
            for var in (*graph.invars, *graph.outvars)
        )
        assert len(graph.eqns) == 1
        equation = graph.eqns[0]
        assert equation.primitive.name == "dot_general"
        assert equation.params["dimension_numbers"] == dimensions
        assert equation.params.get("precision") is None
        assert equation.params.get("preferred_element_type") is None
        assert equation.params.get("out_sharding") is None
        validate_source_jaxpr(graph, 23)
    assert metadata(native) == native_metadata
    assert metadata(source) == source_metadata
