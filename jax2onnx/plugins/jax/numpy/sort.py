# file: jax2onnx/plugins/jax/numpy/sort.py
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Any, Dict, List

import numpy as np
import jax
from jax import numpy as jnp, core
from jax.extend.core import Primitive
from jax._src.export.shape_poly import _DimExpr as DimExpr  # noqa: F401

from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# ---------------------------------------------------------------------- #
# 1.  A dedicated primitive for jnp.sort                                 #
# ---------------------------------------------------------------------- #
sort_p = Primitive("jnp.sort")
sort_p.multiple_results = False  # only the sorted values


# ---------------------------------------------------------------------- #
# 2.  Plugin registration                                                #
# ---------------------------------------------------------------------- #
@register_primitive(
    primitive_obj=sort_p,
    binding_factory=lambda: jnp.sort,  # the function we monkey-patch
    jaxpr_primitive=sort_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.sort.html",
    onnx=[
        {
            "component": "TopK",
            "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html",
        }
    ],
    since="v0.5.2",
    context="primitives.jnp",
    component="sort",
    testcases=[
        {
            "testcase": "sort_1d",
            "callable": lambda x: jnp.sort(x),
            "input_shapes": [(7,)],
        },
        {
            "testcase": "sort_2d_axis0",
            "callable": lambda x: jnp.sort(x, axis=0),
            "input_shapes": [("B", 4)],
        },
    ],
)
class SortPlugin(PrimitiveLeafPlugin):
    """
    Lower `jnp.sort` (ascending, values-only) to ONNX `TopK`
    with `largest=0, sorted=1`.
    """

    _ORIG_CALL: Callable[..., Any] | None = None
    primitive = sort_p

    # ----------------------------------------------------------
    # abstract_eval – delegate to the original jnp.sort
    # ----------------------------------------------------------
    @staticmethod
    def abstract_eval(x: core.ShapedArray, *, axis: int | None = -1, **_):
        if SortPlugin._ORIG_CALL is None:  # pragma: no cover
            raise RuntimeError("Original jnp.sort not captured")
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        out = jax.eval_shape(lambda a: SortPlugin._ORIG_CALL(a, axis=axis), spec)
        return core.ShapedArray(out.shape, out.dtype)

    # ----------------------------------------------------------
    # ONNX lowering
    # ----------------------------------------------------------
    def to_onnx(  # noqa: D401
        self,
        conv: "Jaxpr2OnnxConverter",
        invars: List,
        outvars: List,
        params: Dict[str, Any],
    ):
        x_var = invars[0]
        x_name = conv.get_name(x_var)

        axis = params.get("axis", -1)
        if axis < 0:
            axis += len(x_var.aval.shape)

        # -------------------------------------------------- #
        # shape(x)  -> gather(axis)  -> k_scalar            #
        # -------------------------------------------------- #
        shape_name = conv.get_unique_name("shape_of")
        conv.add_node(
            helper.make_node(
                "Shape",
                inputs=[x_name],
                outputs=[shape_name],
                name=conv.get_unique_name("Shape"),
            )
        )
        conv.add_shape_info(shape_name, (len(x_var.aval.shape),), np.int64)

        axis_const = conv.get_constant_name(np.array(axis, dtype=np.int64))  # 0-D
        k_scalar = conv.get_unique_name("dim_size")
        conv.add_node(
            helper.make_node(
                "Gather",
                inputs=[shape_name, axis_const],
                outputs=[k_scalar],
                axis=0,
                name=conv.get_unique_name("Gather_dim"),
            )
        )
        conv.add_shape_info(k_scalar, (), np.int64)

        # -------------------------------------------------- #
        # Unsqueeze k to 1-D so TopK accepts it              #
        # -------------------------------------------------- #
        axes_const = conv.get_constant_name(np.array([0], dtype=np.int64))  # 1-D
        k_name = conv.get_unique_name("k_unsqueezed")
        conv.add_node(
            helper.make_node(
                "Unsqueeze",
                inputs=[k_scalar, axes_const],
                outputs=[k_name],
                name=conv.get_unique_name("Unsqueeze_k"),
            )
        )
        conv.add_shape_info(k_name, (1,), np.int64)

        # -------------------------------------------------- #
        # TopK                                              #
        # -------------------------------------------------- #
        values_out = conv.get_name(outvars[0])
        indices_tmp = conv.get_unique_name("topk_indices")  # ignored
        conv.add_node(
            helper.make_node(
                "TopK",
                inputs=[x_name, k_name],
                outputs=[values_out, indices_tmp],
                axis=axis,
                largest=0,
                sorted=1,
                name=conv.get_unique_name("TopK_sort"),
            )
        )

        # Meta-data for the sorted values
        out_shape = tuple(conv._dim_to_symbol_safe(d) for d in x_var.aval.shape)
        out_dtype_enum = conv._ensure_onnx_dtype(x_var.aval.dtype)
        conv.add_shape_info(values_out, out_shape, out_dtype_enum)
        conv.add_shape_info(indices_tmp, out_shape, np.int64)

    # ----------------------------------------------------------
    # monkey-patch helpers
    # ----------------------------------------------------------
    @staticmethod
    def _sort_binding(a, *, axis=-1):
        return sort_p.bind(a, axis=axis)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        SortPlugin._ORIG_CALL = orig_fn

        def patched_sort(a, axis=-1):
            return SortPlugin._sort_binding(a, axis=axis)

        return patched_sort

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jnp],
            "target_attribute": "sort",
            "patch_function": SortPlugin.get_monkey_patch,
        }


# Register abstract-eval with the primitive
sort_p.def_abstract_eval(SortPlugin.abstract_eval)
