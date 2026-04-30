# jax2onnx/plugins/flax/nnx/relu.py

from __future__ import annotations
from typing import Any, Final, cast

import jax
from jax.extend.core import Primitive as JaxPrimitive
from flax import nnx
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.jax.nn._builder_utils import (
    lower_unary_elementwise,
)


# --- Define a JAX Primitive for nnx.relu and keep a reference on flax.nnx ---
# We do this so tracing (make_jaxpr) “sees” a primitive instead of a Python function.


def _init_relu_prim() -> JaxPrimitive:
    relu_prim = getattr(nnx, "relu_p", None)
    if relu_prim is None:
        relu_prim = JaxPrimitive("nnx.relu")
        relu_prim.multiple_results = False
        nnx.relu_p = relu_prim  # attach for visibility / reuse
    return cast(JaxPrimitive, relu_prim)


nnx_relu_p: Final[JaxPrimitive] = _init_relu_prim()


def _relu_abstract_eval(x_aval: jax.core.ShapedArray) -> jax.core.ShapedArray:
    # ReLU preserves shape & dtype
    return jax.core.ShapedArray(x_aval.shape, x_aval.dtype)


# Idempotent abstract eval registration
try:
    nnx_relu_p.def_abstract_eval(_relu_abstract_eval)
except Exception:
    pass


@register_primitive(
    jaxpr_primitive=nnx_relu_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.relu.html",
    onnx=[
        {"component": "Relu", "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html"}
    ],
    since="0.2.0",
    context="primitives.nnx",
    component="relu",
    testcases=[
        {
            "testcase": "relu_1d",
            "callable": lambda x: nnx.relu(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": expect_graph(
                ["Relu:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "relu_4d",
            "callable": lambda x: nnx.relu(x),
            "input_shapes": [("B", 28, 28, 32)],
            "post_check_onnx_graph": expect_graph(
                ["Relu:Bx28x28x32"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReluPlugin(PrimitiveLeafPlugin):
    """
    plugins IR converter for flax.nnx.relu → ONNX Relu.
    """

    # ---------------- lowering (IR) ----------------
    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Relu",
            input_hint="relu_in",
            output_hint="relu_out",
        )

    # ---------------- monkey patch binding ----------------
    @staticmethod
    def patch_info() -> dict[str, Any]:
        """
        Provide a small patch so `flax.nnx.relu` binds our primitive during tracing.
        The converter machinery will enter/exit this patch via plugin_binding().
        """

        def _patched_relu(x: ArrayLike) -> ArrayLike:
            return cast(ArrayLike, nnx_relu_p.bind(x))

        return {
            "patch_targets": [nnx],
            "target_attribute": "relu",
            "patch_function": lambda _: _patched_relu,
        }
