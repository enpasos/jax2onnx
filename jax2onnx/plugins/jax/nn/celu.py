# file: jax2onnx/plugins/jax/nn/celu.py

from typing import TYPE_CHECKING

import jax
import numpy as np
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper
from onnx import TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define our own primitive
jax.nn.celu_p = Primitive("jax.nn.celu")
jax.nn.celu_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jax.nn.celu_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.celu.html",
    onnx=[
        {
            "component": "Celu",
            "doc": "https://onnx.ai/onnx/operators/onnx__Celu.html",
        }
    ],
    since="v0.7.1",
    context="primitives.nn",
    component="celu",
    testcases=[
        {
            "testcase": "jaxnn_celu",
            "callable": lambda x: jax.nn.celu(x, alpha=0.1),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "jaxnn_celu_1",
            "callable": lambda x: jax.nn.celu(x, alpha=0.2),
            "input_shapes": [(2, 5)],
        },
    ],
)
class JaxCeluPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.nn.celu calls to the ONNX Celu operator.
    """

    @staticmethod
    def abstract_eval(x, alpha=1.0):
        return x.update(shape=x.shape, dtype=x.dtype, weak_type=False)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        # unpack inputs/outputs and get dtype/shape
        (input_var,) = node_inputs
        (output_var,) = node_outputs
        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)
        alpha = params.get("alpha", 1.0)

        dt = input_var.aval.dtype
        shape = tuple(input_var.aval.shape)

        # ONNX Celu doesn't support double, so cast→Celu(float32)→cast back
        if np.issubdtype(dt, np.floating) and dt == np.dtype(np.float64):
            # 1) cast input down to float32
            cast_in = s.get_unique_name("celu_cast_in")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[cast_in],
                    to=TensorProto.FLOAT,
                )
            )
            s.add_shape_info(cast_in, shape, TensorProto.FLOAT)

            # 2) float32 Celu
            celu_f32 = s.get_unique_name("celu_f32")
            s.add_node(
                helper.make_node(
                    "Celu",
                    inputs=[cast_in],
                    outputs=[celu_f32],
                    name=s.get_unique_name("celu"),
                    alpha=alpha,
                )
            )
            s.add_shape_info(celu_f32, shape, TensorProto.FLOAT)

            # 3) cast result back to float64
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[celu_f32],
                    outputs=[output_name],
                    to=TensorProto.DOUBLE,
                )
            )
        else:
            # float32 (or any non‐double) Celu
            s.add_node(
                helper.make_node(
                    "Celu",
                    inputs=[input_name],
                    outputs=[output_name],
                    name=s.get_unique_name("celu"),
                    alpha=alpha,
                )
            )

    @staticmethod
    def get_monkey_patch():
        def patched_celu(x, alpha=1.0):
            return jax.nn.celu_p.bind(x, alpha=alpha)

        return patched_celu

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jax.nn],
            "patch_function": lambda _: JaxCeluPlugin.get_monkey_patch(),
            "target_attribute": "celu",
        }


def celu_batching_rule(batched_args, batch_dims, *, alpha):
    """
    Batching rule for jax.nn.celu.
    Since celu is elementwise, we simply apply the primitive to the batched input.
    """
    (x,) = batched_args
    (bdim,) = batch_dims

    y = jax.nn.celu_p.bind(x, alpha=alpha)
    return y, bdim


# === Registration ===

# Register the abstract evaluation function
jax.nn.celu_p.def_abstract_eval(JaxCeluPlugin.abstract_eval)

# Register the batching rule
batching.primitive_batchers[jax.nn.celu_p] = celu_batching_rule
