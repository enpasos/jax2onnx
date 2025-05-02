# file: jax2onnx/plugins/jax/core/dim_as_value.py

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence
import numpy as np
from onnx import helper, TensorProto
from jax import core
from jax.extend.core import Primitive

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# ----------------------------------------------------------------------
# 1.  Primitive alias  (there is no public symbol, grab it by name)
# ----------------------------------------------------------------------
dim_as_value_p: Primitive = Primitive("dim_as_value")  # just the name is enough


# ----------------------------------------------------------------------
@register_primitive(
    jaxpr_primitive=dim_as_value_p.name,
    jax_doc="returns the (possibly-symbolic) size of a dimension",
    onnx=[{"component": "(Shape → Gather → Squeeze → Cast)", "doc": ""}],
    since="v0.5.0",
    context="primitives.core",
    component="dim_as_value",
    testcases=[
        {
            "testcase": "dim_as_value",
            "callable": lambda x: x.shape[0],
            "input_shapes": [("B", 8)],
        },
    ],
)
class DimAsValuePlugin(PrimitiveLeafPlugin):

    # ------------------------------------------------------------------
    # abstract_eval
    # ------------------------------------------------------------------
    @staticmethod
    def abstract_eval(*__, **___):
        # always a scalar int32
        return core.ShapedArray((), np.dtype("int32"))

    # ------------------------------------------------------------------
    # lowering
    # ------------------------------------------------------------------
    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],  # NOTE: there are none
        node_outputs: Sequence[Any],
        params: dict[str, Any],  # params["dim"]  is the _DimExpr
    ):
        import numpy as np
        from onnx import helper, TensorProto

        out_var = node_outputs[0]
        out_name = s.get_name(out_var)
        dim_expr = params["dim"]

        # --- Static-dimension: emit a Cast from the constant to i32 ---
        if isinstance(dim_expr, int):
            # 1) make a (int64) initializer for the value
            const_array = np.array(dim_expr, dtype=np.int64)
            const_name = s.get_constant_name(const_array)

            # 2) Cast it down to INT32 for JAX's i32
            cast_name = s.get_unique_name("dim_as_value_static_cast")
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[const_name],
                    outputs=[cast_name],
                    to=int(TensorProto.INT32),
                    name=s.get_unique_name("dim_as_value_static_cast"),
                )
            )
            s.add_shape_info(cast_name, (), np.int32)

            # 3) (Optional) fresh name for the final output
            s.add_node(
                helper.make_node(
                    "Identity",
                    inputs=[cast_name],
                    outputs=[out_name],
                    name=s.get_unique_name("dim_as_value_static_id"),
                )
            )
            s.add_shape_info(out_name, (), np.int32)
            return

        # --- Dynamic-dimension: look up origin axis and extract at runtime ---
        if dim_expr not in s.symbolic_dim_to_origin:
            raise ValueError(
                f"Symbolic dimension '{dim_expr}' has no registered input origin."
            )
        source_name, axis = s.symbolic_dim_to_origin[dim_expr]

        # determine the rank robustly
        if source_name in s.builder.value_info_metadata:
            rank = len(s.builder.value_info_metadata[source_name]["shape"])
        elif source_name in s.symbolic_shapes:
            rank = len(s.symbolic_shapes[source_name])
        else:
            raise RuntimeError(
                f"No shape information available for tensor '{source_name}'."
            )

        # 1) Shape → vector of dims
        shape_out = s.get_unique_name("shape_of_tensor")
        s.add_node(helper.make_node("Shape", inputs=[source_name], outputs=[shape_out]))
        s.add_shape_info(shape_out, (rank,), np.int64)

        # 2) Gather out the desired axis
        axis_const = s.get_constant_name(np.array([axis], dtype=np.int64))
        gather_out = s.get_unique_name("gather_dim")
        s.add_node(
            helper.make_node(
                "Gather",
                inputs=[shape_out, axis_const],
                outputs=[gather_out],
                axis=0,
                name=s.get_unique_name("gather_dim"),
            )
        )
        s.add_shape_info(gather_out, (1,), np.int64)

        # 3) Reshape the 1-vector to scalar (avoids Squeeze/axes issues)
        reshape_out = s.get_unique_name("reshape_to_scalar")
        shape_scalar = s.get_constant_name(np.array([], dtype=np.int64))
        s.add_node(
            helper.make_node(
                "Reshape",
                inputs=[gather_out, shape_scalar],
                outputs=[reshape_out],
                name=s.get_unique_name("reshape_to_scalar"),
            )
        )
        s.add_shape_info(reshape_out, (), np.int64)

        # 4) Cast to INT32 to match JAX's i32 for dim_as_value
        s.add_node(
            helper.make_node(
                "Cast",
                inputs=[reshape_out],
                outputs=[out_name],
                to=int(TensorProto.INT32),
                name=s.get_unique_name("cast_to_i32"),
            )
        )
        s.add_shape_info(out_name, (), np.int32)
