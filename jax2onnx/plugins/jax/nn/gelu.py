# file: jax2onnx/plugins/jax/nn/gelu.py

from typing import TYPE_CHECKING

import jax
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper
from onnx import TensorProto
import numpy as _np

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.gelu_p = Primitive("jax.nn.gelu")
jax.nn.gelu_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.gelu_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.gelu.html",
    onnx=[
        {
            "component": "Gelu",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gelu.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="gelu",
    testcases=[
        {
            "testcase": "jaxnn_gelu",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_gelu_1",
            "callable": lambda x: jax.nn.gelu(x, approximate=False),
            "input_shapes": [(2, 5)],
        },
        {
            "testcase": "jaxnn_gelu_approx",
            "callable": lambda x: jax.nn.gelu(x, approximate=True),
            "input_shapes": [(3, 3)],
        },
    ],
)
class JaxGeluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.gelu calls to the ONNX Gelu operator.
    Supports both exact and approximate (tanh-based) variants.
    """

    @staticmethod
    def abstract_eval(x, approximate=True):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_var = node_inputs[0]
        output_var = node_outputs[0]

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)

        approximate = params.get("approximate", True)
        # ONNX expects 'tanh' for approximate=True, 'none' otherwise
        approximation = "tanh" if approximate else "none"

        # figure out dtype & shape
        dtype = input_var.aval.dtype
        shape = input_var.aval.shape

        if dtype == _np.float32:
            # native Gelu for float32
            gelu_node = helper.make_node(
                "Gelu",
                inputs=[input_name],
                outputs=[output_name],
                name=s.get_unique_name("gelu"),
                approximate=approximation,
            )
            s.add_node(gelu_node)
            s.add_shape_info(output_name, shape, dtype)
        else:
            # fallback for float64: Cast→Gelu(f32)→Cast back
            cast_in = s.get_unique_name("gelu_cast_in")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[cast_in],
                    name=s.get_unique_name("cast_in"),
                    to=TensorProto.FLOAT,
                )
            )
            s.add_shape_info(cast_in, shape, _np.float32)

            gelu_f32 = s.get_unique_name("gelu_f32")
            s.add_node(
                helper.make_node(
                    "Gelu",
                    inputs=[cast_in],
                    outputs=[gelu_f32],
                    name=s.get_unique_name("gelu"),
                    approximate=approximation,
                )
            )
            s.add_shape_info(gelu_f32, shape, _np.float32)

            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[gelu_f32],
                    outputs=[output_name],
                    name=s.get_unique_name("cast_out"),
                    to=TensorProto.DOUBLE,
                )
            )
            s.add_shape_info(output_name, shape, _np.float64)

    @staticmethod
    def get_monkey_patch():
        def patched_gelu(x, approximate=True):
            return jax.nn.gelu_p.bind(x, approximate=approximate)

        return patched_gelu

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxGeluPlugin.get_monkey_patch(),
            "target_attribute": "gelu",
        }


def gelu_batching_rule(batched_args, batch_dims, *, approximate):
    """
    Batching rule for jax.nn.gelu.
    Since GELU is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.gelu_p.bind(x, approximate=approximate)
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.gelu_p.def_abstract_eval(JaxGeluPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.gelu_p] = gelu_batching_rule
