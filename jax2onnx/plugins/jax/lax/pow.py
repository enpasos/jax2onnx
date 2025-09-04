# file: jax2onnx/plugins/jax/lax/pow.py

from typing import TYPE_CHECKING
import numpy as np

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.pow_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.pow.html",
    onnx=[{"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"}],
    since="v0.8.2",
    context="primitives.lax",
    component="pow",
    testcases=[
        {
            "testcase": "pow_lax",
            "callable": lambda x1, x2: jax.lax.pow(x1, x2),
            "input_shapes": [(3,), (3,)],
        }
    ],
)
class PowPlugin(PrimitiveLeafPlugin):
    """Elementwise power; cast exponent to base dtype to satisfy ONNX."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        base_v, exp_v = node_inputs
        out_v = node_outputs[0]

        base_name = s.get_name(base_v)
        exp_name = s.get_name(exp_v)
        out_name = s.get_name(out_v)

        # Prefer builder-known numpy dtypes (order-safe); fall back to aval dtype.
        def _np_dt(name, var):
            try:
                _, dt = s.builder.get_shape_dtype(name)
                # builder may return either np.dtype or ONNX enum; normalize to np.dtype
                if isinstance(dt, np.dtype):
                    return dt
                # If it's an ONNX enum, map via private helper
                return s.builder._onnx_dtype_to_numpy(dt)  # present in builder
            except Exception:
                return np.dtype(var.aval.dtype)

        base_dt = _np_dt(base_name, base_v)
        exp_dt = _np_dt(exp_name, exp_v)

        # ONNX Pow requires matching input dtypes – cast exponent when needed
        if base_dt != exp_dt:
            cast_out = s.get_unique_name("pow_exp_cast_out")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[exp_name],
                    outputs=[cast_out],
                    name=s.get_unique_name("pow_exp_cast"),
                    to=int(s.builder._numpy_dtype_to_onnx(base_dt)),
                )
            )
            # Register value_info for the cast output (required in subgraphs like Scan bodies)
            s.add_shape_info(cast_out, exp_v.aval.shape, base_dt)
            exp_name = cast_out

        s.add_node(
            helper.make_node(
                "Pow",
                inputs=[base_name, exp_name],
                outputs=[out_name],
                name=s.get_unique_name("pow"),
            )
        )

        # Output takes the resolved base dtype and jaxpr's inferred shape
        s.add_shape_info(out_name, out_v.aval.shape, base_dt)
