from __future__ import annotations

from typing import TYPE_CHECKING, List

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive, Var
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# -----------------------------------------------------------------------------
# Define the JAX primitive that will be emitted during tracing
# -----------------------------------------------------------------------------

nnx.rms_norm_p = Primitive("nnx.rms_norm")
nnx.rms_norm_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=nnx.rms_norm_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.RMSNorm",
    onnx=[
        {
            "component": "RMSNormalization",
            "doc": "https://onnx.ai/onnx/operators/onnx__RMSNormalization.html",
        },
    ],
    since="v0.3.0",
    context="primitives.nnx",
    component="rms_norm",
    testcases=[
        {
            "testcase": "rms_norm_basic",
            "callable": nnx.RMSNorm(num_features=6, rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 6)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "rms_norm_use_scale_false",
            "callable": nnx.RMSNorm(num_features=6, use_scale=False, rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 6)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "rms_norm_4d_dynamic",
            "callable": nnx.RMSNorm(num_features=3, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "rms_norm_4d_dynamic_no_scale",
            "callable": nnx.RMSNorm(num_features=3, use_scale=False, rngs=nnx.Rngs(0)),
            "input_shapes": [("B", 4, 4, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class RMSNormPlugin(PrimitiveLeafPlugin):
    """Convert *flax.nnx.RMSNorm* to ONNX.

    * **If** `builder.opset_version >= 23` &rarr; emit a single
      `RMSNormalization` node (native ONNX op).
    * **Else** fall back to the explicit graph that reproduces the same maths.
    """

    _ORIG_CALL = None

    # ------------------------------------------------------------------
    # JAX abstract evaluation
    # ------------------------------------------------------------------
    @staticmethod
    def abstract_eval(x, scale, *_, **kwargs):
        """Shape inference via :pyfunc:`jax.eval_shape`."""
        return core.ShapedArray(x.shape, x.dtype)

    # ------------------------------------------------------------------
    # ONNX lowering
    # ------------------------------------------------------------------

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: List[Var],
        node_outputs: List[Var],
        params,
    ) -> None:
        x_var, scale_var = node_inputs
        y_var = node_outputs[0]

        input_name = s.get_name(x_var)
        scale_name = s.get_name(scale_var)
        output_name = s.get_name(y_var)
        epsilon = float(params.get("epsilon", 1e-5))

        input_shape = tuple(x_var.aval.shape)
        input_dtype = x_var.aval.dtype
        axis = len(input_shape) - 1

        opset = getattr(s.builder, "opset_version", 0)
        if opset >= 23:
            s.add_node(
                helper.make_node(
                    "RMSNormalization",
                    [input_name, scale_name],
                    [output_name],
                    axis=axis,
                    epsilon=epsilon,
                    name=s.get_unique_name("rms_norm"),
                )
            )
            s.builder.add_value_info(output_name, tuple(input_shape), input_dtype)
            return

        # Fallback for older opsets
        pow2 = s.get_unique_name("pow2")
        two_const = s.get_constant_name(np.array(2.0, dtype=np.float32))
        s.add_node(helper.make_node("Pow", [input_name, two_const], [pow2], name=pow2))
        s.builder.add_value_info(pow2, tuple(input_shape), input_dtype)

        axes_tensor = s.get_constant_name(np.array([axis], dtype=np.int64))
        mean = s.get_unique_name("mean")
        s.add_node(
            helper.make_node(
                "ReduceMean",
                [pow2, axes_tensor],
                [mean],
                keepdims=1,
                name=mean,
            )
        )
        mean_shape = list(input_shape)
        mean_shape[-1] = 1
        s.builder.add_value_info(mean, tuple(mean_shape), input_dtype)

        add_eps = s.get_unique_name("add_eps")
        eps_const = s.get_constant_name(np.array(epsilon, dtype=np.float32))
        s.add_node(helper.make_node("Add", [mean, eps_const], [add_eps], name=add_eps))
        s.builder.add_value_info(add_eps, tuple(mean_shape), input_dtype)

        sqrt = s.get_unique_name("sqrt")
        s.add_node(helper.make_node("Sqrt", [add_eps], [sqrt], name=sqrt))
        s.builder.add_value_info(sqrt, tuple(mean_shape), input_dtype)

        div = s.get_unique_name("div")
        s.add_node(helper.make_node("Div", [input_name, sqrt], [div], name=div))
        s.builder.add_value_info(div, tuple(input_shape), input_dtype)

        s.add_node(
            helper.make_node(
                "Mul", [div, scale_name], [output_name], name=s.get_unique_name("mul")
            )
        )
        s.builder.add_value_info(output_name, tuple(input_shape), input_dtype)

    # ------------------------------------------------------------------
    # Runtime binding and monkey patching
    # ------------------------------------------------------------------
    @staticmethod
    def _rms_norm(x, scale, epsilon):
        return nnx.rms_norm_p.bind(x, scale, epsilon=epsilon)

    @staticmethod
    def rms_norm(x, scale, epsilon):
        return RMSNormPlugin._rms_norm(x, scale, epsilon)

    @staticmethod
    def get_monkey_patch():
        def patched_rms_norm_call(self, x):
            param_dtype = self.param_dtype if self.param_dtype is not None else x.dtype

            if self.use_scale:
                scale_value = self.scale.value
            else:
                scale_value = jnp.ones((self.num_features,), dtype=param_dtype)

            return RMSNormPlugin._rms_norm(x, scale_value, self.epsilon)

        return patched_rms_norm_call

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [nnx.RMSNorm],
            "patch_function": lambda orig_fn: RMSNormPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
            "store_original": lambda orig_fn: setattr(
                RMSNormPlugin, "_ORIG_CALL", orig_fn
            ),
        }


# -----------------------------------------------------------------------------
# Register abstract-eval fn
# -----------------------------------------------------------------------------
nnx.rms_norm_p.def_abstract_eval(RMSNormPlugin.abstract_eval)
