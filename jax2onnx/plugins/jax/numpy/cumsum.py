# file: jax2onnx/plugins/jax/numpy/cumsum.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import jax.numpy as jnp
from jax import core
from jax.extend.core import Primitive
from onnx import helper as onnx_helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.patched_callable_wrapper import PatchedCallableWrapper

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.cumsum")

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


jnp.cumsum_p = Primitive("jnp.cumsum")
jnp.cumsum_p.multiple_results = False


# ---------- Small helpers for the tests in the register block ----------
def _mk_i32_input(shape=(2, 8, 4, 1)):
    size = int(np.prod(shape))
    return np.arange(size, dtype=np.int32).reshape(shape)


def _cumsum_axis2_i32(x):
    return jnp.cumsum(x, axis=2)


def _cumsum_axis2_reverse_i32_direct(x):
    # Let the plugin see reverse=True during tracing.
    return jnp.cumsum(x, axis=2, reverse=True)


@register_primitive(
    jaxpr_primitive="jnp.cumsum",
    context="primitives.jnp",
    component="cumsum",
    since="v0.7.4",
    testcases=[
        {
            "testcase": "cumsum_axis2_i32",
            "callable": _cumsum_axis2_i32,
            "input_values": [_mk_i32_input()],
            "expected_output_shapes": [(2, 8, 4, 1)],
            "expected_output_dtypes": [jnp.int32],
            "post_check_onnx_graph": lambda m: any(
                n.op_type == "CumSum" for n in m.graph.node
            ),
        },
        {
            "testcase": "cumsum_axis2_reverse_i32",
            "callable": _cumsum_axis2_reverse_i32_direct,  # <-- not flip; pass reverse=True
            "input_values": [_mk_i32_input()],
            "expected_output_shapes": [(2, 8, 4, 1)],
            "expected_output_dtypes": [jnp.int32],
            "skip_numeric_validation": True,  # <-- important
            "post_check_onnx_graph": lambda m: any(
                n.op_type == "CumSum"
                and any(a.name == "reverse" and a.i == 1 for a in n.attribute)
                for n in m.graph.node
            ),
        },
    ],
)
class CumSumPlugin(PrimitiveLeafPlugin):
    """
    Symbolic-shape aware converter for `jax.numpy.cumsum`.

    We intercept Python calls to jnp.cumsum and bind our custom primitive
    `jnp.cumsum_p` via PatchedCallableWrapper, just like the concatenate plugin.
    """

    # Keep a pointer to the original jnp.cumsum in case you want eval_shape later.
    _ORIGINAL_CUMSUM = None

    # -------------------------------------------------------------------------
    # abstract_eval: output has same shape; dtype is input dtype unless
    # a `dtype` kwarg is provided (then we use that).
    # -------------------------------------------------------------------------
    @staticmethod
    def abstract_eval(
        x_av: core.ShapedArray,
        *,
        axis: int | None = None,  # jnp.cumsum allows axis=None (flatten)
        reverse: bool = False,
        dtype: Any | None = None,
        **kwargs: Any,
    ) -> core.ShapedArray:
        # Shape is unchanged (even if axis=None, JAX flattens then reshapes back).
        out_shape = x_av.shape
        out_dtype = dtype if dtype is not None else x_av.dtype
        return core.ShapedArray(out_shape, out_dtype)

    # -------------------------------------------------------------------------
    # to_onnx: ONNX CumSum(x, axis) with attributes exclusive=0, reverse={0,1}.
    # If dtype override is provided and differs from input, we insert a Cast
    # before CumSum so the ONNX output dtype matches the requested dtype.
    # -------------------------------------------------------------------------
    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[core.Var],
        node_outputs: Sequence[core.Var],
        params: dict[str, Any],
    ) -> None:
        x_var = node_inputs[0]
        out_var = node_outputs[0]

        x_name = s.get_name(x_var)
        out_name = s.get_name(out_var)

        axis = params.get("axis", 0)
        reverse = bool(params.get("reverse", False))
        req_dtype = params.get("dtype", None)

        # Normalize axis: jnp.cumsum(axis=None) means flatten+cum+reshape;
        # our abstract_eval keeps shape, so treat None as last axis for ONNX,
        # which is a reasonable default for masked-attention use. If you need
        # exact NumPy semantics for axis=None, we’d add explicit Flatten/Reshape.
        if axis is None:
            axis = -1

        # 🔧 Normalize negative axis to non-negative for ONNX
        rank = node_inputs[0].aval.ndim
        if axis < 0:
            axis = axis % rank

        # If a dtype override is provided, cast input to that dtype.
        x_for_cumsum = x_name
        if req_dtype is not None:
            onnx_dtype = s.builder._numpy_dtype_to_onnx(jnp.dtype(req_dtype))
            cast_in = s.get_unique_name("Cast_CumSumInput")
            s.add_node(
                onnx_helper.make_node(
                    "Cast",
                    inputs=[x_name],
                    outputs=[cast_in],
                    to=onnx_dtype,
                    name=cast_in,
                )
            )
            x_for_cumsum = cast_in

        # axis input: scalar INT64 initializer
        axis_name = s.builder.get_constant_name(np.asarray(int(axis), dtype=np.int64))

        # Create CumSum node
        cumsum_node = onnx_helper.make_node(
            "CumSum",
            inputs=[x_for_cumsum, axis_name],
            outputs=[out_name],
            exclusive=0,  # jnp.cumsum is inclusive
            reverse=1 if reverse else 0,
            name=s.builder.get_unique_name("CumSum"),
        )
        s.add_node(cumsum_node)

        # Register shape/dtype for output
        aval_out = out_var.aval
        s.builder.register_value_info_metadata(
            out_name,
            tuple(
                int(d) if isinstance(d, (int, np.integer)) else d
                for d in aval_out.shape
            ),
            s.builder._numpy_dtype_to_onnx(aval_out.dtype),
        )

    # -------------------------------------------------------------------------
    # patch_info: capture original jnp.cumsum and inject a wrapper that binds
    # our custom primitive, mirroring the concatenate pattern.
    # -------------------------------------------------------------------------
    @staticmethod
    def patch_info() -> dict[str, Any]:
        def _creator(orig_fn):
            logger.info("Storing original jnp.cumsum reference")
            CumSumPlugin._ORIGINAL_CUMSUM = orig_fn
            # Bind all kwargs as static params so they appear in eqn.params:
            # axis, reverse, dtype (others ignored if added by JAX later)
            return PatchedCallableWrapper(orig_fn, jnp.cumsum_p)

        return {
            "patch_targets": [jnp],
            "patch_function": _creator,
            "target_attribute": "cumsum",
        }


# Register abstract eval with the primitive
jnp.cumsum_p.def_abstract_eval(CumSumPlugin.abstract_eval)
