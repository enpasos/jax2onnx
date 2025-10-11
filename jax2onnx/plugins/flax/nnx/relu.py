# jax2onnx/plugins/flax/nnx/relu.py

from __future__ import annotations
from typing import TYPE_CHECKING, List, Union, Any, Final

import numpy as np
from jax.extend.core import Primitive as JaxPrimitive
from jax.core import ShapedArray
from flax import nnx

import onnx_ir as ir
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
)

if TYPE_CHECKING:
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


# --- Define a JAX Primitive for nnx.relu and keep a reference on flax.nnx ---
# We do this so tracing (make_jaxpr) “sees” a primitive instead of a Python function.


def _init_relu_prim() -> Any:
    relu_prim = getattr(nnx, "relu_p", None)
    if relu_prim is None:
        relu_prim = JaxPrimitive("nnx.relu")
        relu_prim.multiple_results = False
        nnx.relu_p = relu_prim  # attach for visibility / reuse
    return relu_prim


nnx_relu_p: Final[Any] = _init_relu_prim()


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
    onnx=[
        {"component": "Relu", "doc": "https://onnx.ai/onnx/operators/onnx__Relu.html"}
    ],
    since="v0.2.0",
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
    def lower(self, ctx: "IRBuildContext", eqn):
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]

        # Materialize IR values
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))

        # Emit ONNX Relu
        out_name = getattr(y_val, "name", None) or ctx.fresh_name("Relu")
        builder: Any = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for Relu lowering")
        result = builder.Relu(x_val, _outputs=[out_name])

        # Stamp output type/shape (preserve symbolic labels from input)
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        final_dims: List[Union[int, str]] = []
        for i, d in enumerate(x_shape):
            if isinstance(d, (int, np.integer)):
                final_dims.append(int(d))
            else:
                final_dims.append(_dim_label_from_value_or_aval(x_val, x_shape, i))

        dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if dtype is not None:
            result.type = ir.TensorType(dtype)

        _stamp_type_and_shape(result, tuple(final_dims))
        _ensure_value_metadata(ctx, result)
        bind_value = getattr(ctx, "bind_value_for_var", None)
        if callable(bind_value):
            bind_value(y_var, result)
        else:
            raise AttributeError("IR build context missing bind_value_for_var")

    # ---------------- monkey patch binding ----------------
    @staticmethod
    def patch_info():
        """
        Provide a small patch so `flax.nnx.relu` binds our primitive during tracing.
        The converter machinery will enter/exit this patch via plugin_binding().
        """

        def _patched_relu(x):
            return nnx_relu_p.bind(x)

        return {
            "patch_targets": [nnx],
            "target_attribute": "relu",
            "patch_function": lambda _: _patched_relu,
        }
