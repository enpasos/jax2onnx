# jax2onnx/plugins/jax/lax/clamp.py
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import jax.numpy as jnp
from jax import core, lax
from jax.extend.core import Var
from onnx import helper

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


def _np_dtype(x) -> np.dtype:
    return x if isinstance(x, np.dtype) else np.dtype(x)


def _cast_to(
    s: "Jaxpr2OnnxConverter",
    name: str,
    cur_dt: np.dtype,
    tgt_dt: np.dtype,
    *,
    ctx: str,
    shape_hint: tuple[Any, ...],
) -> str:
    """Insert Cast(name -> tgt_dt) if needed, keep original shape."""
    if cur_dt == tgt_dt:
        return name
    out = s.builder.get_unique_name(f"{ctx}_cast")
    s.add_node(
        helper.make_node(
            "Cast",
            inputs=[name],
            outputs=[out],
            to=int(s.builder._numpy_dtype_to_onnx(tgt_dt)),
            name=s.builder.get_unique_name(f"{ctx}_Cast"),
        )
    )
    s.add_shape_info(out, shape_hint, tgt_dt)
    return out


@register_primitive(
    jaxpr_primitive=lax.clamp_p.name,  # jnp.clip lowers to lax.clamp
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.clamp.html",
    onnx=[
        {"component": "Max", "doc": "https://onnx.ai/onnx/operators/onnx__Max.html"},
        {"component": "Min", "doc": "https://onnx.ai/onnx/operators/onnx__Min.html"},
    ],
    since="v0.7.5",
    context="primitives.lax",
    component="clamp",
    testcases=[
        # Basic integer bounds on int32 vector
        {
            "testcase": "clamp_i32_scalar_bounds",
            # JAX requires same dtype for all 3 args → cast Python ints to x.dtype.
            "callable": lambda x: lax.clamp(
                jnp.asarray(0, dtype=x.dtype), x, jnp.asarray(4, dtype=x.dtype)
            ),
            "input_values": [np.array([-3, 1, 9, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
        },
        # Float32 with scalar float bounds; keep bounds in x.dtype
        {
            "testcase": "clamp_scalar_float_bounds_match_x",
            "callable": lambda x: lax.clamp(
                jnp.asarray(-1.5, dtype=x.dtype), x, jnp.asarray(2.5, dtype=x.dtype)
            ),
            "input_values": [np.array([-2.0, 0.5, 3.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
        },
        # Vector bounds same shape as x (no broadcasting)
        {
            "testcase": "clamp_vector_bounds_match",
            "callable": lambda x, lo, hi: lax.clamp(lo, x, hi),
            "input_values": [
                np.array([-5, -1, 0, 1, 5], dtype=np.float64),
                np.array([-1, -1, -1, -1, -1], dtype=np.float64),
                np.array([1, 1, 1, 1, 1], dtype=np.float64),
            ],
            "expected_output_shapes": [(5,)],
            "expected_output_dtypes": [np.float64],
            "run_only_f64_variant": True,
        },
        # Bool-friendly case (x float32, bounds given as python ints -> cast to float32)
        {
            "testcase": "clamp_pyint_bounds_promote_to_x_dtype",
            "callable": lambda x: lax.clamp(
                jnp.asarray(0, dtype=x.dtype), x, jnp.asarray(1, dtype=x.dtype)
            ),
            "input_values": [np.array([-2.0, 0.25, 3.0], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
        },
    ],
)
class ClampPlugin(PrimitiveLeafPlugin):
    """Lower `lax.clamp(min, x, max)` via ONNX Max/Min, with bounds cast to x.dtype."""

    @staticmethod
    def abstract_eval(
        min_av: core.ShapedArray,
        x_av: core.ShapedArray,
        max_av: core.ShapedArray,
        **params,
    ) -> core.ShapedArray:
        # JAX: result has x's dtype/shape; min/max broadcast to x
        return core.ShapedArray(x_av.shape, x_av.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        v_min, v_x, v_max = node_inputs
        v_out = node_outputs[0]

        x_name = s.get_name(v_x)
        min_name = s.get_name(v_min)
        max_name = s.get_name(v_max)
        out_name = s.get_name(v_out)

        x_dt = _np_dtype(v_x.aval.dtype)
        mn_dt = _np_dtype(v_min.aval.dtype)
        mx_dt = _np_dtype(v_max.aval.dtype)

        # Cast bounds to x dtype (prevents ORT type mismatches and “f64 drift” in double mode)
        min_name = _cast_to(
            s, min_name, mn_dt, x_dt, ctx="ClampMin", shape_hint=v_min.aval.shape
        )
        max_name = _cast_to(
            s, max_name, mx_dt, x_dt, ctx="ClampMax", shape_hint=v_max.aval.shape
        )

        # y = max(x, min)
        y_name = s.builder.get_unique_name("clamp_lowered_max")
        s.add_node(
            helper.make_node(
                "Max",
                inputs=[x_name, min_name],
                outputs=[y_name],
                name=s.builder.get_unique_name("Max"),
            )
        )
        s.add_shape_info(y_name, v_x.aval.shape, x_dt)  # result shape equals x

        # z = min(y, max)
        z_name = out_name
        s.add_node(
            helper.make_node(
                "Min",
                inputs=[y_name, max_name],
                outputs=[z_name],
                name=s.builder.get_unique_name("Min"),
            )
        )
        s.add_shape_info(z_name, v_x.aval.shape, x_dt)
