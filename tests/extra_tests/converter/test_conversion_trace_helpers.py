# tests/extra_tests/converter/test_conversion_trace_helpers.py

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx.converter.conversion_api import _trace_to_jaxpr


def _scale(x, *, scale: float):
    return x * scale


def test_trace_to_jaxpr_captures_params_and_layout_indices() -> None:
    result = _trace_to_jaxpr(
        fn=_scale,
        inputs=[jax.ShapeDtypeStruct((2,), jnp.float32)],
        input_params={"scale": 2.0},
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
            inputs_as_nchw=[1],
            outputs_as_nchw=None,
            input_names=["x"],
            output_names=["y"],
        )


def test_trace_shape_default_stays_fp32_beside_explicit_fp64_inside_x64() -> None:
    with jax.enable_x64():
        result = _trace_to_jaxpr(
            fn=lambda fp32, fp64: (fp32, fp64),
            inputs=[(2,), jax.ShapeDtypeStruct((2,), np.float64)],
            input_params=None,
            inputs_as_nchw=None,
            outputs_as_nchw=None,
            input_names=["fp32", "fp64"],
            output_names=["out32", "out64"],
        )

    assert [spec.dtype for spec in result.sds_list] == [np.float32, np.float64]
    assert [tuple(spec.shape) for spec in result.sds_list] == [(2,), (2,)]
    assert [var.aval.dtype for var in result.jaxpr.invars] == [np.float32, np.float64]
    assert [var.aval.dtype for var in result.jaxpr.outvars] == [np.float32, np.float64]
