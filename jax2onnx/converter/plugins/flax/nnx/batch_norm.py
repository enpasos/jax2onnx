# file: jax2onnx/converter/plugins/flax/nnx/batch_norm.py
import contextlib
from typing import TYPE_CHECKING

import jax
from flax import nnx
from jax.core import Primitive
from onnx import helper
from jax import numpy as jnp  # Import jnp
from jax import core
import numpy as np

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

batch_norm_p = Primitive("batch_norm")


def get_primitive():
    return batch_norm_p


def _get_monkey_patch():
    def batch_norm(x, scale, bias, mean, var, epsilon, use_running_average, momentum):
        def batch_norm_abstract_eval(x, scale, bias, mean, var, *args, **kwargs):
            return core.ShapedArray(x.shape, x.dtype)

        batch_norm_p.multiple_results = False
        batch_norm_p.def_abstract_eval(batch_norm_abstract_eval)
        return batch_norm_p.bind(
            x,
            scale,
            bias,
            mean,
            var,
            epsilon=epsilon,
            use_running_average=use_running_average,
            momentum=momentum,
        )

    def patched_batch_norm_call(self, x):
        return batch_norm(
            x,
            self.scale.value,
            self.bias.value,
            self.mean.value,
            self.var.value,  # using self.var.value as per your API
            epsilon=self.epsilon,
            use_running_average=self.use_running_average,
            momentum=self.momentum,
        )

    return patched_batch_norm_call


@contextlib.contextmanager
def temporary_patch():
    original_call = nnx.BatchNorm.__call__
    nnx.BatchNorm.__call__ = _get_monkey_patch()
    try:
        yield
    finally:
        nnx.BatchNorm.__call__ = original_call


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_batch_norm(node_inputs, node_outputs, params):
        # Expect node_inputs: [x, scale, bias, mean, var]
        input_name = s.get_name(node_inputs[0])
        scale_name = s.get_name(node_inputs[1])
        bias_name = s.get_name(node_inputs[2])
        mean_name = s.get_name(node_inputs[3])
        variance_name = s.get_name(node_inputs[4])
        final_output_name = s.get_name(node_outputs[0])
        epsilon = params.get("epsilon")
        use_running_average = params.get("use_running_average")
        momentum = params.get("momentum")

        # Assume the JAX BatchNorm input is in NHWC.
        jax_shape = node_inputs[0].aval.shape  # e.g. (11, 2, 2, 64)

        # === Pre-Transpose: NHWC -> NCHW ===
        pre_transpose_name = s.get_unique_name("bn_pre_transpose")
        pre_transpose_node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[pre_transpose_name],
            name=s.get_unique_name("bn_transpose_pre"),
            perm=[0, 3, 1, 2],  # NHWC -> NCHW
        )
        s.add_node(pre_transpose_node)
        # Compute pre-transposed shape: (B, C, H, W)
        pre_transposed_shape = (jax_shape[0], jax_shape[3], jax_shape[1], jax_shape[2])
        s.add_shape_info(pre_transpose_name, pre_transposed_shape)

        # === Create BatchNormalization Node in ONNX ===
        bn_output_name = s.get_unique_name("bn_output")
        batch_norm_node = helper.make_node(
            "BatchNormalization",
            inputs=[
                pre_transpose_name,
                scale_name,
                bias_name,
                mean_name,
                variance_name,
            ],
            outputs=[bn_output_name],
            name=s.get_unique_name("batch_norm"),
            epsilon=epsilon,
        )
        s.add_node(batch_norm_node)
        # The output of the ONNX BatchNormalization node remains in NCHW.
        s.add_shape_info(bn_output_name, pre_transposed_shape)

        # === Post-Transpose: NCHW -> NHWC ===
        post_transpose_node = helper.make_node(
            "Transpose",
            inputs=[bn_output_name],
            outputs=[final_output_name],
            name=s.get_unique_name("bn_transpose_post"),
            perm=[0, 2, 3, 1],  # NCHW -> NHWC
        )
        s.add_node(post_transpose_node)
        # Compute final shape by reversing the pre-transposition.
        final_shape = (
            pre_transposed_shape[0],
            pre_transposed_shape[2],
            pre_transposed_shape[3],
            pre_transposed_shape[1],
        )
        s.add_shape_info(final_output_name, final_shape)

    return handle_batch_norm


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "batch_norm",
        "jax_doc": "https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm",
        "onnx": [
            {
                "component": "BatchNormalization",
                "doc": "https://onnx.ai/onnx/operators/onnx__BatchNormalization.html",
            },
        ],
        "since": "v0.1.0",
        "testcases": [
            {
                "testcase": "batch_norm",
                "callable": nnx.BatchNorm(
                    num_features=64, epsilon=1e-5, momentum=0.9, rngs=nnx.Rngs(0)
                ),
                "input_shapes": [(11, 2, 2, 64)],
            }
        ],
    }
