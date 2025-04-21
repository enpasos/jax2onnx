"""
RMS Norm Plugin for JAX → ONNX conversion

This plugin converts **`flax.nnx.RMSNorm`** layers into the native ONNX
**`RMSNormalization`** operator whenever the exported model uses opset ≥ 23.
For lower opsets it transparently falls back to the manual graph that was
previously implemented (Pow → ReduceMean → Add → Sqrt → Div → Mul).

The public behaviour of the primitive (signature, monkey‑patch, tests) is
unchanged – only the ONNX emission differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive
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
            "testcase": "rms_norm",
            "callable": nnx.RMSNorm(6, rngs=nnx.Rngs(0)),
            "input_shapes": [(11, 2, 2, 6)],
        },
        {
            "testcase": "rms_norm_2",
            "callable": nnx.RMSNorm(num_features=20, rngs=nnx.Rngs(0)),
            "input_shapes": [(2, 20)],
        },
    ],
)
class RMSNormPlugin(PrimitiveLeafPlugin):
    """Convert *flax.nnx.RMSNorm* to ONNX.

    * **If** `builder.opset_version >= 23` &rarr; emit a single
      `RMSNormalization` node (native ONNX op).
    * **Else** fall back to the explicit graph that reproduces the same maths.
    """

    # ------------------------------------------------------------------
    # JAX abstract evaluation – shape/dtype passthrough
    # ------------------------------------------------------------------

    @staticmethod
    def abstract_eval(x, scale, *_, **__):  # noqa: D401 – simple passthrough
        return core.ShapedArray(x.shape, x.dtype)

    # ------------------------------------------------------------------
    # ONNX lowering
    # ------------------------------------------------------------------

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: List[str],
        node_outputs: List[str],
        params,
    ) -> None:
        # ------------------------------------------------------------------
        # Resolve names / shapes / dtypes
        # ------------------------------------------------------------------
        input_name = s.get_name(node_inputs[0])
        scale_name = s.get_name(node_inputs[1])
        output_name = s.get_name(node_outputs[0])
        epsilon = float(params.get("epsilon", 1e-5))

        input_shape = s.shape_env[input_name]
        input_dtype = s.builder.dtype_env[input_name]
        axis = len(input_shape) - 1  # normalise over the last dimension

        # ------------------------------------------------------------------
        # Decide whether we can use the native op
        # ------------------------------------------------------------------
        opset = getattr(s.builder, "opset_version", 0)
        if opset >= 23:
            # ---------------- native RMSNormalization -----------------
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

        # ---------------- fallback: manual construction ------------------
        # 1. x²
        pow2 = s.get_unique_name("pow2")
        two_const = s.get_constant_name(np.array(2.0, dtype=np.float32))
        s.add_node(helper.make_node("Pow", [input_name, two_const], [pow2], name=pow2))
        s.builder.add_value_info(pow2, tuple(input_shape), input_dtype)

        # 2. mean(x²) over last axis (axes as tensor, ONNX ≥ 13)
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

        # 3. add epsilon
        add_eps = s.get_unique_name("add_eps")
        eps_const = s.get_constant_name(np.array(epsilon, dtype=np.float32))
        s.add_node(helper.make_node("Add", [mean, eps_const], [add_eps], name=add_eps))
        s.builder.add_value_info(add_eps, tuple(mean_shape), input_dtype)

        # 4. sqrt
        sqrt = s.get_unique_name("sqrt")
        s.add_node(helper.make_node("Sqrt", [add_eps], [sqrt], name=sqrt))
        s.builder.add_value_info(sqrt, tuple(mean_shape), input_dtype)

        # 5. x / sqrt
        div = s.get_unique_name("div")
        s.add_node(helper.make_node("Div", [input_name, sqrt], [div], name=div))
        s.builder.add_value_info(div, tuple(input_shape), input_dtype)

        # 6. * scale
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
    def _rms_norm(x, scale, epsilon):  # type: ignore[override]
        return nnx.rms_norm_p.bind(x, scale, epsilon=epsilon)

    @staticmethod
    def rms_norm(x, scale, epsilon):  # noqa: D401 – public helper
        return RMSNormPlugin._rms_norm(x, scale, epsilon)

    @staticmethod
    def get_monkey_patch():
        def patched_rms_norm_call(self, x):  # noqa: D401 – inline patch fn
            return RMSNormPlugin._rms_norm(x, self.scale.value, self.epsilon)

        return patched_rms_norm_call

    @staticmethod
    def patch_info():  # noqa: D401 – required by PrimitiveLeafPlugin
        return {
            "patch_targets": [nnx.RMSNorm],
            "patch_function": lambda _: RMSNormPlugin.get_monkey_patch(),
            "target_attribute": "__call__",
        }


# -----------------------------------------------------------------------------
# Register abstract‑eval fn so that JAX knows the primitive’s output shape/dtype
# -----------------------------------------------------------------------------

nnx.rms_norm_p.def_abstract_eval(RMSNormPlugin.abstract_eval)
