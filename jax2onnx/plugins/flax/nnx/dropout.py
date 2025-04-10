# file: jax2onnx/plugins/flax/nnx/dropout.py


from typing import TYPE_CHECKING

from flax import nnx
from jax.extend.core import Primitive, Literal
from jax.core import ShapedArray
from onnx import helper
import numpy as np
import onnx
from jax2onnx.plugin_system import register_primitive, PrimitiveLeafPlugin

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

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
            "input_shapes": [(5, 10)],
        },
        {
            "testcase": "dropout_call_params",
            "callable": nnx.Dropout(rate=0.5, deterministic=False, rngs=nnx.Rngs(5)),
            "input_shapes": [(5, 10)],
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

        # Static parameter: rate
        rate = params.get("rate", 0.0)
        ratio_tensor = np.array(rate, dtype=np.float32)
        ratio_name = s.builder.get_constant_name(ratio_tensor)  # Use builder helper

        # Handle deterministic: static or dynamic
        if isinstance(det_input, Literal):
            training_mode = not bool(det_input.val)
            training_tensor = np.array(training_mode, dtype=bool)
            training_mode_name = s.builder.get_constant_name(
                training_tensor
            )  # Use builder helper
        else:
            # Dynamic: flip the value (ONNX: training_mode = not deterministic)
            det_name = s.get_name(det_input)
            # --- Get aval info for the dynamic deterministic input ---
            det_aval = det_input.aval
            det_shape = det_aval.shape  # Should be scalar ()
            det_dtype_enum = onnx.TensorProto.BOOL  # Expecting bool

            flipped_name = s.get_unique_name("training_mode")  # More descriptive name
            not_node = helper.make_node(
                "Not",
                inputs=[det_name],
                outputs=[flipped_name],
                name=s.get_unique_name("not_deterministic"),  # Node name
            )
            s.add_node(not_node)
            # --- ADD ValueInfo for the 'Not' node output ---
            s.builder.register_value_info_metadata(
                flipped_name, shape=det_shape, dtype=det_dtype_enum
            )
            s.builder.add_value_info(
                flipped_name, shape=det_shape, dtype=det_dtype_enum
            )
            # --- END Add ValueInfo ---
            training_mode_name = flipped_name

        # ONNX Dropout node
        # Inputs: data, ratio (optional float scalar), training_mode (optional bool scalar)
        # Dropout op version >= 12: ratio and training_mode are optional inputs
        dropout_inputs = [x_name]
        # Only add ratio if rate > 0 (it's optional)
        if rate > 0.0:
            dropout_inputs.append(ratio_name)
            # Only add training_mode if ratio is also present (as per ONNX spec order)
            dropout_inputs.append(training_mode_name)
        # If rate is 0, dropout is identity, no need for ratio or training_mode

        dropout_node = helper.make_node(
            "Dropout",
            inputs=dropout_inputs,
            outputs=[output_name],  # Only specify first output (data)
            name=s.get_unique_name("dropout"),
        )
        s.add_node(dropout_node)

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
