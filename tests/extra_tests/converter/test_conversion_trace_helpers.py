# tests/extra_tests/converter/test_conversion_trace_helpers.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jax2onnx.converter.conversion_api import _trace_to_jaxpr
from jax2onnx.converter.precision_policy import PrecisionPolicy


def _scale(x, *, scale: float):
    return x * scale


def test_trace_to_jaxpr_captures_params_and_layout_indices() -> None:
    result = _trace_to_jaxpr(
        fn=_scale,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_params={"scale": 2.0},
        enable_double_precision=False,
        inputs_as_nchw=[0],
        outputs_as_nchw=[0],
        input_names=["x"],
        output_names=["y"],
    )

    assert result.frozen_params == {"scale": 2.0}
    assert result.inputs_as_nchw == (0,)
    assert result.outputs_as_nchw == (0,)
    assert len(result.jaxpr.invars) == 1
    assert len(result.jaxpr.outvars) == 1


def test_trace_to_jaxpr_validates_input_name_arity() -> None:
    with pytest.raises(ValueError, match="input_names length"):
        _trace_to_jaxpr(
            fn=_scale,
            inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
            input_params={"scale": 2.0},
            enable_double_precision=False,
            inputs_as_nchw=None,
            outputs_as_nchw=None,
            input_names=["x", "extra"],
            output_names=["y"],
        )


def test_trace_to_jaxpr_validates_layout_indices() -> None:
    with pytest.raises(ValueError, match="inputs_as_nchw index 1 is out of range"):
        _trace_to_jaxpr(
            fn=_scale,
            inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
            input_params={"scale": 2.0},
            enable_double_precision=False,
            inputs_as_nchw=[1],
            outputs_as_nchw=None,
            input_names=["x"],
            output_names=["y"],
        )


def test_source_trace_binds_named_operands_and_keeps_layout_positional() -> None:
    result = _trace_to_jaxpr(
        fn=lambda x, *, flag: jax.lax.select(flag, x, -x),
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_params={"flag": False},
        enable_double_precision=False,
        inputs_as_nchw=[0],
        outputs_as_nchw=None,
        input_names=["features"],
        output_names=None,
        precision_policy=PrecisionPolicy.SOURCE,
    )
    assert result.frozen_params == {}
    assert len(result.sds_list) == len(result.jaxpr.invars) == 2
    assert result.inputs_as_nchw == (0,)
    assert [binding.keyword_name for binding in result.source_inputs] == [None, "flag"]
    assert all(
        binding.var is var
        for binding, var in zip(result.source_inputs, result.jaxpr.invars, strict=True)
    )


@pytest.mark.parametrize("option", ["input_names", "inputs_as_nchw"])
def test_source_trace_named_ports_do_not_expand_positional_options(option: str) -> None:
    options = {"input_names": None, "inputs_as_nchw": None}
    options[option] = ["features", "flag"] if option == "input_names" else [1]
    with pytest.raises(ValueError, match="input_names length|out of range"):
        _trace_to_jaxpr(
            fn=lambda x, *, flag: jax.lax.select(flag, x, -x),
            inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
            input_params={"flag": True},
            enable_double_precision=False,
            outputs_as_nchw=None,
            output_names=None,
            precision_policy=PrecisionPolicy.SOURCE,
            **options,
        )
