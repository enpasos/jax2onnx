# file: jax2onnx/plugins/jax/numpy/where.py


from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import jax.numpy as jnp
from jax import core, lax
from jax.extend.core import Primitive, Var
import numpy as np
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.where")

# Define the primitive for jnp.where
jnp.where_p = Primitive("jnp.where")
jnp.where_p.multiple_results = False


# Example definition (ensure it's globally accessible for the test generator):
def create_problematic_where_sequence(cond_input, data_input):
    scalar_true_val = jnp.array(1.0, dtype=data_input.dtype)
    scalar_false_val = jnp.array(0.0, dtype=data_input.dtype)
    where_output = jnp.where(cond_input, scalar_true_val, scalar_false_val)
    processed_data = data_input * where_output
    return processed_data


@register_primitive(
    jaxpr_primitive=jnp.where_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.where.html",
    onnx=[
        {"component": "Where", "doc": "https://onnx.ai/onnx/operators/onnx__Where.html"}
    ],
    since="v0.5.2",
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
        {
            "testcase": "where_multidim_condition_scalar_branches_broadcast",
            "callable": lambda c, t, f: jnp.where(c, t, f),
            "input_shapes": [(201, 1, 1), (), ()],
        },
        {
            "testcase": "where_multidim_condition_scalar_branches_broadcast",
            "callable": lambda c, t, f: jnp.where(c, t, f),
            "input_shapes": [(201, 1, 1), (), ()],
        },
        {
            "testcase": "where_A",
            "callable": create_problematic_where_sequence,
            "input_values": [
                np.random.choice([True, False], size=(201, 1, 1)),
                np.random.rand(201, 1, 201).astype(np.float32),
            ],
            "expected_output_shapes": [(201, 1, 201)],
        },
        {
            "testcase": "where_B",
            "callable": create_problematic_where_sequence,
            "input_values": [
                np.random.choice([True, False], size=(201, 1, 1)),
                np.random.rand(201, 1, 201).astype(np.int32),
            ],
            "expected_output_shapes": [(201, 1, 201)],
            "expected_output_dtypes": [np.int32],
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
        import numpy as np  # local import

        bool_types = (jnp.bool_, np.bool_, bool)
        if not isinstance(cond_av, core.ShapedArray):  # Should be ShapedArray
            raise TypeError(
                f"jnp.where condition must be a ShapedArray, got {type(cond_av)}"
            )

        if cond_av.dtype not in bool_types:
            # accept float32 AND float64 when tracing – they will be lowered to Bool later
            if cond_av.dtype in (np.float32, np.float64):
                pass
            else:
                raise TypeError(
                    f"jnp.where condition must be boolean, got {cond_av.dtype}"
                )

        # Standard JAX abstract evaluation for where (can be simplified)
        # The output shape is determined by broadcasting rules between x_av and y_av.
        # The output dtype is the result of type promotion of x_av.dtype and y_av.dtype.
        if not isinstance(x_av, core.ShapedArray) or not isinstance(
            y_av, core.ShapedArray
        ):
            raise TypeError(
                "x and y in jnp.where must be ShapedArrays for abstract_eval."
            )

        # Simplified output shape/dtype logic based on JAX's lax.select_n_p
        try:
            # This is how JAX's own lax.select_n (used by jnp.where) performs abstract eval
            out_aval = lax.select_n_p.abstract_eval(cond_av, x_av, y_av, **kwargs)
        except Exception as e:
            # Fallback or re-raise with more context if needed
            logger.error(f"Error during lax.select_n_p.abstract_eval for where: {e}")
            # As a basic fallback, try to use broadcasted shape and promoted type:
            # This might not perfectly match JAX in all edge cases.
            try:
                # This is a simplified shape inference. JAX's own is more robust.
                promoted_dtype = jnp.promote_types(x_av.dtype, y_av.dtype)

                # First, broadcast the shapes of the true/false branches
                shape_xy = np.broadcast_shapes(x_av.shape, y_av.shape)

                # Then, broadcast the result with the condition's shape
                output_shape = np.broadcast_shapes(cond_av.shape, shape_xy)

                out_aval = core.ShapedArray(output_shape, promoted_dtype)
                logger.warning(
                    f"Used fallback shape/dtype inference for where (corrected): {out_aval}"
                )

            except Exception as fallback_e:
                logger.error(
                    f"Fallback shape/dtype inference for where also failed: {fallback_e}"
                )
                raise TypeError(
                    f"Cannot determine output aval for jnp.where with inputs: {cond_av}, {x_av}, {y_av}"
                ) from fallback_e

        return out_aval

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
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
