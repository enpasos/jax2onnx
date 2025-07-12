from typing import TYPE_CHECKING, Callable, Any

import equinox as eqx
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the LayerNorm primitive
eqx.nn.layer_norm_p = Primitive("eqx.nn.layer_norm")
eqx.nn.layer_norm_p.multiple_results = False  # Correctly set at initialization


@register_primitive(
    jaxpr_primitive=eqx.nn.layer_norm_p.name,
    jax_doc="https://docs.kidger.site/equinox/api/nn/normalisation/#equinox.nn.LayerNorm",
    onnx=[
        {
            "component": "LayerNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__LayerNormalization.html",
        }
    ],
    since="v0.7.2",
    context="primitives.eqx",
    component="layer_norm",
    testcases=[
        # TODO: enable testcases
        # {
        #     "testcase": "layer_norm",
        #     "callable": eqx.nn.LayerNorm(32, eps=1e-5),
        #     "input_shapes": [(32)],
        # },
        # {
        #     "testcase": "layer_norm_multiaxis",
        #     "callable": eqx.nn.LayerNorm((20, 32)),
        #     "input_shapes": [(20, 32)],
        # },
        # {
        #     "testcase": "batched_layer_norm",
        #     "callable": jax.vmap(eqx.nn.LayerNorm(32, eps=1e-5)),
        #     "input_shapes": [("B", 32)],
        # },
        # {
        #     "testcase": "layer_norm_no_bias_no_scale",
        #     "callable": eqx.nn.LayerNorm(32, use_bias=False, use_weight=False),
        #     "input_shapes": [(32)],
        # },
    ],
)
class LayerNormPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting equinox.nn.LayerNorm to ONNX.
    """

    _ORIGINAL_LAYERNORM_CALL: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(x, scale, bias, *, epsilon):
        """Abstract evaluation function for LayerNorm."""
        # LayerNorm's output shape is always the same as the input's shape.
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: list,
        node_outputs: list,
        params: dict,
    ):
        """Handles conversion of LayerNorm to ONNX format."""
        x_var, scale_var, bias_var = node_inputs
        y_var = node_outputs[0]

        input_name = s.get_name(x_var)
        scale_name = s.get_name(scale_var)
        bias_name = s.get_name(bias_var)
        output_name = s.get_name(y_var)

        epsilon = params["epsilon"]

        # Equinox LayerNorm normalizes the whole tensor. `jax.vmap` is used
        # for batching, which adds leading dimensions. We detect these batch
        # dimensions by comparing the rank of the input tensor with the rank
        # of the parameter tensors (scale/bias). The difference is the
        # starting axis for normalization in ONNX.
        x_rank = len(x_var.aval.shape)
        scale_rank = len(scale_var.aval.shape)
        onnx_axis = x_rank - scale_rank

        ln_node = helper.make_node(
            "LayerNormalization",
            inputs=[input_name, scale_name, bias_name],
            outputs=[output_name],
            name=s.get_unique_name("layer_norm"),
            axis=onnx_axis,
            epsilon=epsilon,
        )
        s.add_node(ln_node)
        s.add_shape_info(output_name, x_var.aval.shape, x_var.aval.dtype)

    @staticmethod
    def _layer_norm(x, scale, bias, *, epsilon):
        """Defines the primitive binding for LayerNorm."""
        return eqx.nn.layer_norm_p.bind(x, scale, bias, epsilon=epsilon)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable) -> Callable:
        """Returns a patched version of LayerNorm's call method."""
        LayerNormPlugin._ORIGINAL_LAYERNORM_CALL = orig_fn

        def patched_call(self, x, state=None, *, key=None):
            # Maintain original implementation's strict shape check
            if x.shape != self.shape:
                raise ValueError(
                    "`LayerNorm(shape)(x)` must satisfy the invariant `shape == x.shape`"
                    f"Received `shape={self.shape} and `x.shape={x.shape}`. You might need "
                    "to replace `layer_norm(x)` with `jax.vmap(layer_norm)(x)`."
                )

            # The ONNX LayerNormalization op requires scale and bias inputs.
            # If they are disabled in the Equinox layer, we create default
            # constant tensors of ones (for scale) and zeros (for bias).
            # The parameters must have the same dtype as the input.
            dtype = x.dtype
            scale = (
                self.weight if self.use_weight else jnp.ones(self.shape, dtype=dtype)
            )
            bias = self.bias if self.use_bias else jnp.zeros(self.shape, dtype=dtype)

            # Bind to the primitive.
            out = LayerNormPlugin._layer_norm(x, scale, bias, epsilon=self.eps)

            # The original __call__ can return a tuple with state.
            if state is None:
                return out
            else:
                return out, state

        return patched_call

    @staticmethod
    def patch_info():
        """Provides patching information for LayerNorm."""
        return {
            "patch_targets": [eqx.nn.LayerNorm],
            "patch_function": LayerNormPlugin.get_monkey_patch,
            "target_attribute": "__call__",
        }


# Register abstract evaluation function
eqx.nn.layer_norm_p.def_abstract_eval(LayerNormPlugin.abstract_eval)


def _eqx_layernorm_batching_rule(batched_args, batch_dims, *, epsilon):
    """Batching rule for `eqx.nn.layer_norm_p`."""
    x, scale, bias = batched_args
    x_bdim, scale_bdim, bias_bdim = batch_dims

    # Batching is only supported for the data input `x`, not parameters.
    if scale_bdim is not None or bias_bdim is not None:
        raise NotImplementedError(
            "Batching over equinox.nn.LayerNorm parameters is not supported."
        )

    # The primitive is applied to the batched `x`. The `to_onnx` logic
    # will correctly infer the normalization axis based on the rank
    # difference between the batched `x` and the un-batched parameters.
    out = eqx.nn.layer_norm_p.bind(x, scale, bias, epsilon=epsilon)

    # The output has a batch dimension at the same axis as the input.
    return out, x_bdim


# Register the batching rule for our primitive
batching.primitive_batchers[eqx.nn.layer_norm_p] = _eqx_layernorm_batching_rule
