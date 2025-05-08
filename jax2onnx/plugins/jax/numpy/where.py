from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import jax
import jax.numpy as jnp
from jax import core, lax
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.converter.onnx_builder import OnnxBuilder
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.where")

# Define the primitive for jnp.where
jnp.where_p = Primitive("where")
jnp.where_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jnp.where_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html",
    onnx=[
        {"component": "Where", "doc": "https://onnx.ai/onnx/operators/onnx__Where.html"}
    ],
    since="0,5,2",
    context="primitives.jnp",
    component="where",
    testcases=[
        {
            "testcase": "where_simple",
            "callable": lambda c, x, y: jnp.where(c, x, y),
            "input_shapes": [(3,), (3,), (3,)],
            "expected_output_shapes": [(3,)],
        },
        {
            "testcase": "where_broadcast",
            "callable": lambda c, x, y: jnp.where(c[:, None], x, y),
            "input_shapes": [(4,), (4, 5), (4, 5)],
            "expected_output_shapes": [(4, 5)],
        },
    ],
)
class WherePlugin(PrimitiveLeafPlugin):
    """Lower `jnp.where` to ONNX Where operator."""

    @staticmethod
    def abstract_eval(
        cond_av: core.AbstractValue,
        x_av: core.AbstractValue,
        y_av: core.AbstractValue,
        **kwargs,
    ) -> core.AbstractValue:
        # Accept both jnp.bool_ and np.bool_ for compatibility, but allow float/bool for tracing
        import numpy as np

        bool_types = (jnp.bool_, np.bool_, bool)
        # Allow float condition for tracing (JAX will trace with float32 for symbolic shapes)
        if not isinstance(cond_av, core.ShapedArray):
            raise TypeError(
                f"jnp.where condition must be a ShapedArray, got {type(cond_av)}"
            )
        if cond_av.dtype not in bool_types:
            # Accept float32 for tracing, but warn
            if cond_av.dtype == np.float32:
                # Allow, but do not raise
                pass
            else:
                raise TypeError(
                    f"jnp.where condition must be boolean, got {cond_av.dtype}"
                )

        # Compute broadcasted shape
        out_shape = lax.broadcast_shapes(cond_av.shape, x_av.shape, y_av.shape)

        # Validate dtypes
        if x_av.dtype != y_av.dtype:
            raise TypeError(f"Dtype mismatch in where: {x_av.dtype} vs {y_av.dtype}")

        return core.ShapedArray(out_shape, x_av.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[core.Var],
        node_outputs: Sequence[core.Var],
        params: dict[str, Any],
    ) -> None:
        # Map inputs
        cond_v, x_v, y_v = node_inputs
        cond_name = s.get_name(cond_v)
        x_name = s.get_name(x_v)
        y_name = s.get_name(y_v)
        out_v = node_outputs[0]
        out_name = s.get_name(out_v)

        # --- PATCH: Ensure condition is cast to BOOL for ONNX ---
        import numpy as np
        from onnx import TensorProto

        cond_dtype = getattr(cond_v.aval, "dtype", None)
        if cond_dtype is not None and cond_dtype != np.bool_:
            cond_cast_name = s.builder.get_unique_name("where_cond_cast")
            s.builder.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[cond_name],
                    outputs=[cond_cast_name],
                    to=TensorProto.BOOL,
                    name=s.builder.get_unique_name("cast_where_cond"),
                )
            )
            s.add_shape_info(cond_cast_name, cond_v.aval.shape, np.bool_)
            cond_name = cond_cast_name

        # Create ONNX Where node
        node = helper.make_node(
            "Where",
            inputs=[cond_name, x_name, y_name],
            outputs=[out_name],
            name=s.builder.get_unique_name("WhereOp"),
        )
        s.add_node(node)
        s.add_shape_info(out_name, out_v.aval.shape, out_v.aval.dtype)

    @staticmethod
    def patch_info():
        # Monkey-patch jnp.where and lax.select to emit our primitive in the jaxpr
        def patched_where(cond, x=None, y=None):
            if x is None or y is None:
                raise NotImplementedError(
                    "Only `jnp.where(cond, x, y)` is supported for ONNX conversion."
                )
            return jnp.where_p.bind(cond, x, y)

        return {
            "patch_targets": [jnp, lax],
            "target_attribute": "where",
            "patch_function": lambda orig: patched_where,
        }


# Bind abstract evaluation
jnp.where_p.def_abstract_eval(WherePlugin.abstract_eval)
