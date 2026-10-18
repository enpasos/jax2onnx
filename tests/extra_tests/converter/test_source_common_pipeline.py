# tests/extra_tests/converter/test_source_common_pipeline.py

"""The bounded source adapter shares the ordinary IR conversion pipeline."""

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
)
from tests.extra_tests.converter import precision_test_support
from tests.extra_tests.converter.precision_test_support import (
    PrototypeLimit,
    _validate_trace,
    export_source_dtypes,
)


@pytest.fixture(autouse=True)
def _restore_jax_config() -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_platforms = jax.config.jax_platforms
    previous_precision = jax.config.jax_default_matmul_precision
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_platforms", previous_platforms)
        jax.config.update("jax_default_matmul_precision", previous_precision)


def _source_convert(fn: Any, **options: Any) -> ir.Model:
    kwargs: dict[str, Any] = {
        "fn": fn,
        "inputs": [jax.ShapeDtypeStruct((), np.float32)],
        "input_params": None,
        "model_name": "source_common_pipeline",
        "opset": 23,
        "enable_double_precision": False,
        "record_primitive_calls_file": None,
        "strict_optimizer_failures": True,
        "precision_policy": PrecisionPolicy.SOURCE,
        "jaxpr_validator": _validate_trace,
    }
    kwargs.update(options)
    return conversion_api._convert_to_ir(**kwargs)


@onnx_function
def _rounded_cancellation(x: jax.Array) -> jax.Array:
    rounded = x.astype(jnp.float32)
    return (rounded + np.float32(1)) - rounded


@onnx_function
def _unsupported_function(x: jax.Array) -> jax.Array:
    return jnp.sin(x)


@pytest.mark.parametrize("dtype", [np.float16, jnp.bfloat16, np.float32, np.float64])
def test_source_adapter_retains_captured_payload_through_the_common_path(
    dtype: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    jax.config.update("jax_enable_x64", True)
    captured = np.asarray([1 + 2**-30, -0.0, -2.5], dtype=dtype)
    calls = []
    convert = conversion_api._convert_to_ir

    def observed(**kwargs: Any) -> ir.Model:
        calls.append(kwargs["precision_policy"])
        return convert(**kwargs)

    monkeypatch.setattr(conversion_api, "_convert_to_ir", observed)
    model = export_source_dtypes(
        lambda x: (captured, x + captured),
        [jax.ShapeDtypeStruct(captured.shape, captured.dtype)],
    )
    assert calls == [PrecisionPolicy.SOURCE]
    payloads = [onnx.numpy_helper.to_array(value) for value in model.graph.initializer]
    assert any(
        value.dtype == captured.dtype and value.tobytes() == captured.tobytes()
        for value in payloads
    )
    expected = int(ir.DataType.from_numpy(captured.dtype))
    assert model.graph.input[0].type.tensor_type.elem_type == expected
    assert [value.type.tensor_type.elem_type for value in model.graph.output] == [
        expected,
        expected,
    ]


def test_common_source_path_rejects_wrong_lowered_result_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lower = conversion_api._lower_jaxpr_equations

    def corrupt_result(ctx: Any, jaxpr: Any) -> None:
        lower(ctx, jaxpr)
        value = ctx.require_value_for_var(jaxpr.outvars[0])
        value.type = ir.TensorType(ir.DataType.DOUBLE)

    monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", corrupt_result)
    with pytest.raises(RuntimeError, match="Source result type FLOAT became DOUBLE"):
        _source_convert(lambda x: -x)


@pytest.mark.parametrize("boundary", ["input", "output"])
def test_common_source_path_rechecks_finalized_boundary_types(
    boundary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalize = conversion_api._build_and_finalize_ir_model

    def corrupt_boundary(*args: Any, **kwargs: Any) -> ir.Model:
        model = finalize(*args, **kwargs)
        values = model.graph.inputs if boundary == "input" else model.graph.outputs
        values[0].type = ir.TensorType(ir.DataType.DOUBLE)
        return model

    monkeypatch.setattr(
        conversion_api, "_build_and_finalize_ir_model", corrupt_boundary
    )
    with pytest.raises(
        RuntimeError, match=f"Final {boundary} .* lost source type FLOAT"
    ):
        _source_convert(lambda x: -x)


def test_source_adapter_rechecks_boundary_types_after_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def corrupt_alias(model: ir.Model, **kwargs: Any) -> None:
        # Identity input/output aliases stay valid ONNX even with the wrong type.
        model.graph.inputs[0].type = ir.TensorType(ir.DataType.DOUBLE)

    monkeypatch.setattr(precision_test_support, "postprocess_ir_model", corrupt_alias)
    with pytest.raises(RuntimeError, match="Final input .* lost source type FLOAT"):
        export_source_dtypes(lambda x: x, [jax.ShapeDtypeStruct((), np.float32)])


@pytest.mark.parametrize("ambient_x64", [False, True])
@pytest.mark.parametrize("failure", [None, "trace", "lowering", "optimizer"])
def test_common_source_path_preserves_ambient_config_on_success_and_failure(
    ambient_x64: bool, failure: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    jax.config.update("jax_enable_x64", ambient_x64)
    observed = []

    def model(x: jax.Array) -> jax.Array:
        observed.append(bool(jax.config.x64_enabled))
        if failure == "trace":
            raise RuntimeError("trace failure")
        return -x

    def fail(*args: Any, **kwargs: Any) -> None:
        assert bool(jax.config.x64_enabled) is ambient_x64
        raise RuntimeError(f"{failure} failure")

    if failure == "lowering":
        monkeypatch.setattr(conversion_api, "_lower_jaxpr_equations", fail)
    elif failure == "optimizer":
        monkeypatch.setattr(conversion_api, "optimize_graph", fail)
    if failure is None:
        result = _source_convert(model)
        assert result.graph.outputs[0].dtype == ir.DataType.FLOAT
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failure"):
            _source_convert(model)
    assert observed == [ambient_x64]
    assert bool(jax.config.x64_enabled) is ambient_x64
    assert active_precision_policy() is PrecisionPolicy.LEGACY


def test_common_source_path_forwards_subset_validation_into_functions() -> None:
    seen = []

    def validate(jaxpr: Any, opset: int) -> None:
        seen.append(tuple(eqn.primitive.name for eqn in jaxpr.eqns))
        _validate_trace(jaxpr, opset)

    with pytest.raises(PrototypeLimit, match="sin.*outside the PoC subset"):
        _source_convert(lambda x: _unsupported_function(x), jaxpr_validator=validate)
    assert any(any(name.startswith("onnx_fn::") for name in names) for names in seen)
    assert any(any(name.endswith("sin") for name in names) for names in seen)


@pytest.mark.parametrize("strict_optimizer_failures", [None, False, True])
def test_common_source_path_always_propagates_optimizer_failures(
    strict_optimizer_failures: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAX2ONNX_STRICT_OPTIMIZER_FAILURES", "0")

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("source optimizer failure")

    monkeypatch.setattr(conversion_api, "optimize_graph", fail)
    with pytest.raises(RuntimeError, match="source optimizer failure"):
        _source_convert(
            lambda x: -x, strict_optimizer_failures=strict_optimizer_failures
        )


def test_common_source_path_isolates_function_cache_from_legacy_conversion() -> None:
    jax.config.update("jax_enable_x64", True)

    def model(x: jax.Array) -> jax.Array:
        return _rounded_cancellation(x)

    before = _source_convert(model)
    legacy = conversion_api.to_onnx(
        fn=model,
        inputs=[jax.ShapeDtypeStruct((), np.float32)],
        input_params=None,
        model_name="legacy_between_source_calls",
        opset=23,
        enable_double_precision=True,
        record_primitive_calls_file=None,
    )
    after = _source_convert(model)
    assert legacy.graph.outputs[0].dtype == ir.DataType.DOUBLE
    for exported in (before, after):
        assert exported.graph.outputs[0].dtype == ir.DataType.FLOAT
        assert len(exported.functions) == 1
        assert (
            next(iter(exported.functions.values())).graph.outputs[0].dtype
            == ir.DataType.FLOAT
        )
        onnx.checker.check_model(ir.to_proto(exported), full_check=True)
    assert jax.config.x64_enabled
    assert active_precision_policy() is PrecisionPolicy.LEGACY


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"enable_double_precision": True}, "legacy double promotion"),
        ({"jaxpr_validator": None}, "requires a subset validator"),
        ({"input_params": {"scale": 2}}, "does not support input_params"),
        ({"normalization_mode": "prefer_native"}, "does not support normalization"),
        ({"inputs": [()]}, "requires typed inputs"),
        (
            {"inputs": [jax.ShapeDtypeStruct(("B",), np.float32)]},
            "rank-two",
        ),
    ],
)
def test_common_source_path_rejects_unaudited_options_before_tracing(
    options: dict[str, Any], message: str
) -> None:
    called = []

    def model(x: jax.Array) -> jax.Array:
        called.append(True)
        return x

    with pytest.raises(ValueError, match=message):
        _source_convert(model, **options)
    assert not called


def test_common_source_path_rejects_disabled_x64_before_tracing() -> None:
    jax.config.update("jax_enable_x64", False)
    called = []

    def model(x: jax.Array) -> jax.Array:
        called.append(True)
        return x

    with pytest.raises(ValueError, match="float64, but JAX x64 is disabled"):
        _source_convert(model, inputs=[jax.ShapeDtypeStruct((), np.float64)])
    assert not called
    assert not jax.config.x64_enabled


@pytest.mark.parametrize("option", ["inputs_as_nchw", "outputs_as_nchw"])
@pytest.mark.parametrize("empty", [[], ()])
def test_common_source_path_empty_layout_selection_is_a_noop(
    option: str, empty: Any
) -> None:
    model = _source_convert(lambda x: -x, **{option: empty})
    assert [node.op_type for node in model.graph] == ["Neg"]
