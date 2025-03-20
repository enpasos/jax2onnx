import contextlib
from typing import TYPE_CHECKING

from flax import nnx
from jax.extend.core import Primitive
from onnx import helper
from jax import core
import numpy as np
from jax2onnx.plugin_system import register_primitive, PrimitivePlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the new primitive for dropout.
nnx.dropout_p = Primitive("nnx.dropout")
nnx.dropout_p.multiple_results = False  # Correctly set at initialization


@register_primitive(
    jaxpr_primitive=nnx.dropout_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/stochastic.html#flax.nnx.Dropout",
    onnx=[
        {
            "component": "Dropout",
            "doc": "https://onnx.ai/onnx/operators/onnx__Dropout.html",
        }
    ],
    since="v0.1.0",
    context="primitives.nnx",
    testcases=[
        {
            "testcase": "dropout_inference",
            "callable": nnx.Dropout(rate=0.5, deterministic=True, rngs=nnx.Rngs(0)),
            "input_shapes": [(5, 10)],
        },
        # comparison with training mode is not possible due to the random nature of dropout
        # {
        #     "testcase": "dropout_training",
        #     "callable": nnx.Dropout(
        #         rate=0.5, deterministic=False, rngs=nnx.Rngs(0)
        #     ),
        #     "input_shapes": [(5, 10)],
        # },
    ],
)
class DropoutPlugin(PrimitivePlugin):
    """
    Plugin for converting flax.nnx.Dropout to ONNX.
    """

    @staticmethod
    def abstract_eval(x, rate, deterministic):
        """Abstract evaluation function for dropout."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of dropout to ONNX format."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])

        # Retrieve dropout parameters
        rate = params.get("rate", 0.0)
        deterministic = params.get("deterministic", True)

        # ONNX Dropout expects:
        # - ratio as a **second input** instead of an attribute in newer versions.
        # - training_mode (bool) in newer versions.  We handle this implicitly.
        dropout_inputs = [input_name]

        if not deterministic:
            ratio_name = s.get_constant_name(np.array(rate, dtype=np.float32))
            dropout_inputs.append(
                ratio_name
            )  # Add ratio as an input instead of an attribute.

        dropout_node = helper.make_node(
            "Dropout",
            inputs=dropout_inputs,
            outputs=[output_name],
            name=s.get_unique_name("dropout"),
        )

        s.add_node(dropout_node)

    @staticmethod
    def _dropout(x, rate, deterministic):
        """Defines the primitive binding for dropout."""
        return nnx.dropout_p.bind(x, rate=rate, deterministic=deterministic)

    @staticmethod
    def get_monkey_patch():
        """Returns a patched version of Dropout's call method."""

        def patched_dropout_call(self, x, deterministic=None):
            det = deterministic if deterministic is not None else self.deterministic
            return DropoutPlugin._dropout(x, self.rate, det)

        return patched_dropout_call

    @staticmethod
    def patch_info():
        """Provides patching information for dropout."""
        return {
            "patch_targets": [nnx.Dropout],
            "patch_function": lambda _: DropoutPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# Register abstract evaluation function
nnx.dropout_p.def_abstract_eval(DropoutPlugin.abstract_eval)
