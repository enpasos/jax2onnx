from __future__ import annotations

import logging
from functools import reduce
from typing import TYPE_CHECKING, Any, Sequence

import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive, Var
import numpy as np
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.select")

# Define the primitive for jnp.select
jnp_select_p = Primitive("jnp_select")
jnp_select_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jnp_select_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.select.html",
    onnx=[
        {"component": "Where", "doc": "https://onnx.ai/onnx/operators/onnx__Where.html"}
    ],
    since="v0.7.1",
    context="primitives.jnp",
    component="select",
    testcases=[
        {
            "testcase": "select_simple",
            "callable": lambda: jnp.select(
                [jnp.array([True, False]), jnp.array([False, True])],
                [jnp.array([1, 2]), jnp.array([3, 4])],
                default=jnp.array([0, 0]),
            ),
        },
        {
            "testcase": "select_broadcast",
            "callable": lambda: jnp.select(
                [jnp.array([True, False]), jnp.array([False, True])],
                [jnp.array([1, 2]), jnp.array([3, 4])],
                default=0,  # broadcast default
            ),
        },
        {
            "testcase": "select_gpt2_attention_mask",
            "callable": lambda scores, mask: jnp.select([mask], [scores], default=-1e9),
            "input_shapes": [("B", 12, "T", "T"), ("B", 1, "T", "T")],
            "input_dtypes": [jnp.float32, jnp.bool_],
        },
    ],
)
class SelectPlugin(PrimitiveLeafPlugin):
    """Lower `jnp.select` to a series of nested ONNX Where operators."""

    @staticmethod
    def abstract_eval(*avals, num_conds, num_choices):
        # Partition the abstract values based on the counts
        cond_avals = avals[:num_conds]
        choice_avals = avals[num_conds : num_conds + num_choices]
        default_aval = avals[-1]

        # Determine the output shape by broadcasting all input shapes
        all_shapes = (
            [a.shape for a in cond_avals]
            + [a.shape for a in choice_avals]
            + [default_aval.shape]
        )
        output_shape = jnp.broadcast_shapes(*all_shapes)

        # Determine the output dtype by promoting all choices and the default
        all_dtypes = [a.dtype for a in choice_avals] + [default_aval.dtype]

        # jnp.promote_types only takes two arguments, so we must reduce.
        output_dtype = reduce(jnp.promote_types, all_dtypes)

        return core.ShapedArray(output_shape, output_dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Var],
        node_outputs: Sequence[Var],
        params: dict[str, Any],
    ) -> None:
        num_conds = params["num_conds"]
        num_choices = params["num_choices"]

        # Partition the input variables based on the counts
        cond_vars = node_inputs[:num_conds]
        choice_vars = node_inputs[num_conds : num_conds + num_choices]
        default_var = node_inputs[-1]

        # The last choice is the default value.
        current_else = s.get_name(default_var)

        # Pre-compute full broadcast shape and dtype for all branches
        all_shapes = (
            [v.aval.shape for v in cond_vars]
            + [v.aval.shape for v in choice_vars]
            + [default_var.aval.shape]
        )
        result_shape = jnp.broadcast_shapes(*all_shapes)
        all_dtypes = [v.aval.dtype for v in choice_vars] + [default_var.aval.dtype]
        result_dtype = reduce(jnp.promote_types, all_dtypes)
        # Build nested Where's from the inside out.
        # Inner nodes get unique names; the outermost writes directly to the JAXPR output var.
        for i in range(len(cond_vars) - 1, -1, -1):
            cond_name = s.get_name(cond_vars[i])
            then_name = s.get_name(choice_vars[i])
            # Outer-most iteration writes to the true output var:
            if i == 0:
                output_name = s.get_name(node_outputs[0])
            else:
                output_name = s.builder.get_unique_name(f"select_where_{i}")

            # Ensure the condition is boolean for ONNX's Where
            cond_aval = cond_vars[i].aval
            if cond_aval.dtype != np.bool_:
                cast_name = s.builder.get_unique_name(f"cond_cast_{i}")
                s.builder.add_node(
                    helper.make_node(
                        "Cast", [cond_name], [cast_name], to=TensorProto.BOOL
                    )
                )
                s.add_shape_info(cast_name, cond_aval.shape, np.bool_)
                cond_name = cast_name

            node = helper.make_node(
                "Where",
                inputs=[cond_name, then_name, current_else],
                outputs=[output_name],
            )
            s.add_node(node)
            # register shape & dtype for this intermediate
            s.add_shape_info(output_name, result_shape, result_dtype)
            current_else = output_name
        # done — no s.rename() needed anymore

    @staticmethod
    def patch_info():
        # Only bind our jnp_select_p primitive when under a JAX trace;
        # otherwise fall back to the original numeric implementation.
        def make_patched(orig_select):
            def patched_select(condlist, choicelist, default=0):
                # if any cond is a JAX Tracer, we must emit our ONNX primitive
                from jax import core as _core

                if any(isinstance(c, _core.Tracer) for c in condlist):
                    # ensure default has an aval
                    default_arr = jnp.asarray(default)
                    return jnp_select_p.bind(
                        *condlist,
                        *choicelist,
                        default_arr,
                        num_conds=len(condlist),
                        num_choices=len(choicelist),
                    )
                else:
                    # just do normal jnp.select for eager/numpy evaluation
                    return orig_select(condlist, choicelist, default)

            return patched_select

        return {
            "patch_targets": [jnp],
            "target_attribute": "select",
            "patch_function": make_patched,
        }


# Bind abstract evaluation
jnp_select_p.def_abstract_eval(SelectPlugin.abstract_eval)
