from __future__ import annotations
from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
import jax
from jax.extend.core import Primitive as JaxPrimitive
from jax.core import ShapedArray
from flax import nnx

import onnx_ir as ir
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    _ensure_value_info as _add_value_info,
)

if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


# --- Define a JAX Primitive for nnx.relu and keep a reference on flax.nnx ---
# We do this so tracing (make_jaxpr) “sees” a primitive instead of a Python function.
nnx_relu_p = getattr(nnx, "relu_p", None)
if nnx_relu_p is None:
    nnx_relu_p = JaxPrimitive("nnx.relu")
    nnx_relu_p.multiple_results = False
    nnx.relu_p = nnx_relu_p  # attach for visibility / reuse

def _relu_abstract_eval(x_aval: ShapedArray) -> ShapedArray:
    # ReLU preserves shape & dtype
    return ShapedArray(x_aval.shape, x_aval.dtype)

# Idempotent abstract eval registration
try:
    nnx_relu_p.def_abstract_eval(_relu_abstract_eval)  # type: ignore[arg-type]
except Exception:
    pass


@register_primitive(
    jaxpr_primitive=nnx_relu_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.relu.html",
    onnx=[{"component": "Relu", "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html"}],
    since="v0.2.0",
    context="primitives2.nnx",
    component="relu",
    testcases=[
        {
            "testcase": "relu_1d",
            "callable": lambda x: nnx.relu(x),
            "input_shapes": [(3,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "relu_4d",
            "callable": lambda x: nnx.relu(x),
            "input_shapes": [("B", 28, 28, 32)],
            "use_onnx_ir": True,
        },
    ],
)
class ReluPlugin(PrimitiveLeafPlugin):
    """
    plugins2 IR converter for flax.nnx.relu → ONNX Relu.
    """

    # ---------------- lowering (IR) ----------------
    def lower(self, ctx: "IRBuildContext", eqn):
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]

        # Materialize IR values
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        # Emit ONNX Relu
        ctx.add_node(
            ir.Node(
                op_type="Relu",
                domain="",
                inputs=[x_val],
                outputs=[y_val],
                name=ctx.fresh_name("Relu"),
            )
        )

        # Stamp output type/shape (preserve symbolic labels from input)
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        final_dims: List[Union[int, str]] = []
        for i, d in enumerate(x_shape):
            if isinstance(d, (int, np.integer)):
                final_dims.append(int(d))
            else:
                final_dims.append(_dim_label_from_value_or_aval(x_val, x_shape, i))

        _stamp_type_and_shape(y_val, tuple(final_dims))
        _add_value_info(ctx, y_val)

    # ---------------- monkey patch binding ----------------
    @staticmethod
    def patch_info():
        """
        Provide a small patch so `flax.nnx.relu` binds our primitive during tracing.
        The converter2 machinery will enter/exit this patch via plugin_binding().
        """
        def _patched_relu(x):
            return nnx_relu_p.bind(x)

        return {
            "patch_targets": [nnx],
            "target_attribute": "relu",
            "patch_function": lambda _: _patched_relu,
        }
