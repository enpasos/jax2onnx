import contextlib
from typing import TYPE_CHECKING

from flax import nnx
from jax.extend.core import Primitive
from onnx import helper
from jax import core

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the new primitive for layer norm.
nnx.layer_norm_p = Primitive("nnx.layer_norm")


def get_primitive():
    return nnx.layer_norm_p


def _get_monkey_patch():
    def layer_norm(x, scale, bias, epsilon, axis):
        def layer_norm_abstract_eval(x, scale, bias, epsilon, axis):
            return core.ShapedArray(x.shape, x.dtype)

        nnx.layer_norm_p.multiple_results = False
        nnx.layer_norm_p.def_abstract_eval(layer_norm_abstract_eval)
        return nnx.layer_norm_p.bind(
            x,
            scale,
            bias,
            epsilon=epsilon,
            axis=axis,
        )

    def patched_layer_norm_call(self, x):
        # Default to axis=-1 if no reduction_axes are provided.
        norm_axis = -1
        if hasattr(self, "reduction_axes"):
            # If reduction_axes is iterable (list/tuple), take the minimum; otherwise, use it directly.
            if isinstance(self.reduction_axes, (list, tuple)):
                norm_axis = min(self.reduction_axes)
            else:
                norm_axis = self.reduction_axes
        return layer_norm(
            x,
            self.scale.value if self.scale is not None else None,
            self.bias.value if self.bias is not None else None,
            epsilon=self.epsilon,
            axis=norm_axis,
        )

    return patched_layer_norm_call


@contextlib.contextmanager
def temporary_patch():
    original_call = nnx.LayerNorm.__call__
    nnx.LayerNorm.__call__ = _get_monkey_patch()
    try:
        yield
    finally:
        nnx.LayerNorm.__call__ = original_call


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_layer_norm(node_inputs, node_outputs, params):
        # Expect node_inputs: [x, scale, bias]
        input_name = s.get_name(node_inputs[0])
        scale_name = s.get_name(node_inputs[1])
        bias_name = s.get_name(node_inputs[2])
        output_name = s.get_name(node_outputs[0])

        epsilon = params.get("epsilon")
        axis = params.get("axis", -1)  # Default normalization axis: last dimension

        # ONNX LayerNormalization expects three inputs: input, scale, bias.
        ln_node = helper.make_node(
            "LayerNormalization",
            inputs=[input_name, scale_name, bias_name],
            outputs=[output_name],
            name=s.get_unique_name("layer_norm"),
            axis=axis,
            epsilon=epsilon,
        )
        s.add_node(ln_node)
        # LayerNormalization does not change the shape.
        # s.add_shape_info(output_name, node_inputs[0].aval.shape)

    return handle_layer_norm


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "nnx.layer_norm",
        "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm",
        "onnx": [
            {
                "component": "LayerNormalization",
                "doc": "https://onnx.ai/onnx/operators/onnx__LayerNormalization.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.nnx",
        "testcases": [
            {
                "testcase": "layer_norm",
                "callable": nnx.LayerNorm(
                    num_features=32, epsilon=1e-5, rngs=nnx.Rngs(0)
                ),
                "input_shapes": [(10, 20, 32)],
            },
            {
                "testcase": "layer_norm_multiaxis",
                "callable": nnx.LayerNorm(
                    3 * 3 * 64,
                    reduction_axes=(1, 2, 3),
                    feature_axes=(1, 2, 3),
                    rngs=nnx.Rngs(0),
                ),
                "input_shapes": [(1, 3, 3, 64)],
            },
        ],
    }
