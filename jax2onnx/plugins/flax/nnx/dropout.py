# file: jax2onnx/plugins/flax/nnx/dropout.py


from typing import TYPE_CHECKING
import jax
import numpy as np
import onnx
from flax import nnx
from jax.core import ShapedArray
from jax.extend.core import Literal, Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
import logging

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the new primitive for dropout.
nnx.dropout_p = Primitive("nnx.dropout")
nnx.dropout_p.multiple_results = False  # Single output


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
    component="dropout",
    testcases=[
        {
            "testcase": "dropout_init_params",
            "callable": nnx.Dropout(rate=0.5, deterministic=True, rngs=nnx.Rngs(5)),
            "input_shapes": [("B", 10)],
        },
        {
            "testcase": "dropout_call_params",
            "callable": nnx.Dropout(rate=0.5, deterministic=False, rngs=nnx.Rngs(5)),
            "input_shapes": [("B", 10)],
            "input_params": {
                "deterministic": True,
            },
        },
    ],
)
class DropoutPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting flax.nnx.Dropout to ONNX.
    Supports static and dynamic (call-time) 'deterministic'.
    """

    @staticmethod
    def abstract_eval(x, deterministic, *, rate):
        """Abstract evaluation function for dropout."""
        return ShapedArray(x.shape, x.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        x_name = s.get_name(node_inputs[0])
        det_input = node_inputs[1]
        output_name = s.get_name(node_outputs[0])

        logging.debug("[DEBUG] Dropout to_onnx called")
        logging.debug(f"[DEBUG] Input tensor name: {x_name}")
        logging.debug(f"[DEBUG] Deterministic input: {det_input}")
        logging.debug(f"[DEBUG] Output name: {output_name}")

        det_name = "deterministic"
        # if det_input is a Variable then change_var_name

        if isinstance(det_input, jax._src.core.Var):
            s.change_var_name(det_input, det_name)

        # Static parameter: rate
        rate = params.get("rate", 0.0)
        logging.debug(f"[DEBUG] Dropout rate: {rate}")
        ratio_tensor = np.array(rate, dtype=np.float32)
        ratio_name = s.builder.get_constant_name(ratio_tensor)

        # Handle deterministic: static or dynamic
        if isinstance(det_input, Literal):
            training_mode = not bool(det_input.val)
            logging.debug(
                f"[DEBUG] Static deterministic value: {det_input.val} → training_mode: {training_mode}"
            )
            training_tensor = np.array(training_mode, dtype=bool)
            training_mode_name = s.builder.get_constant_name(training_tensor)

            # Add value_info for the training mode tensor
            s.builder.add_value_info(
                training_mode_name, shape=(), dtype=onnx.TensorProto.BOOL
            )
            logging.debug(
                f"[DEBUG] Added value_info for training_mode: {training_mode_name}"
            )
        else:
            logging.debug("[DEBUG] Dynamic deterministic input detected")
            det_aval = det_input.aval
            det_shape = det_aval.shape
            det_dtype_enum = onnx.TensorProto.BOOL
            # Register the input with the correct name and type
            s.builder.register_value_info_metadata(
                det_name, shape=det_shape, dtype=det_dtype_enum
            )
            s.builder.add_value_info(det_name, shape=det_shape, dtype=det_dtype_enum)
            flipped_name = s.get_unique_name("training_mode")
            not_node = helper.make_node(
                "Not",
                inputs=[det_name],
                outputs=[flipped_name],
                name=s.get_unique_name("not_deterministic"),
            )
            s.add_node(not_node)
            # Add value_info for the flipped training_mode variable
            s.builder.add_value_info(
                flipped_name, shape=det_shape, dtype=det_dtype_enum
            )
            logging.debug(
                f"[DEBUG] Added NOT node to invert deterministic: input={det_name}, output={flipped_name}"
            )
            logging.debug(f"[DEBUG] Added value_info for training_mode: {flipped_name}")
            training_mode_name = flipped_name

        # ONNX Dropout node
        dropout_inputs = [x_name, ratio_name, training_mode_name]
        dropout_node = helper.make_node(
            "Dropout",
            inputs=dropout_inputs,
            outputs=[output_name],
            name=s.get_unique_name("Dropout"),
        )
        s.add_node(dropout_node)
        logging.debug(
            f"[DEBUG] Added Dropout node with inputs: {dropout_inputs}, output: {output_name}"
        )

    @staticmethod
    def _dropout(x, deterministic, rate):
        """Defines the primitive binding for dropout."""
        return nnx.dropout_p.bind(x, deterministic, rate=rate)

    @staticmethod
    def get_monkey_patch():
        """Returns a patched version of Dropout's __call__ method."""

        def patched_dropout_call(self, x, deterministic=None):
            det = deterministic if deterministic is not None else self.deterministic
            return DropoutPlugin._dropout(x, det, self.rate)

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
