from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from jax import lax, core
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Primitive alias
iota_p = lax.iota_p

# Map numpy dtypes to ONNX dtypes
NUMPY_TO_ONNX_DTYPE = {
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.float64): TensorProto.DOUBLE,
    np.dtype(np.int8): TensorProto.INT8,
    np.dtype(np.int16): TensorProto.INT16,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.int64): TensorProto.INT64,
    np.dtype(np.uint8): TensorProto.UINT8,
    np.dtype(np.uint16): TensorProto.UINT16,
    np.dtype(np.uint32): TensorProto.UINT32,
    np.dtype(np.uint64): TensorProto.UINT64,
    np.dtype(np.bool_): TensorProto.BOOL,
}


@register_primitive(
    jaxpr_primitive=iota_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.iota.html",
    onnx=[
        {"component": "Range", "doc": "https://onnx.ai/onnx/operators/onnx__Range.html"}
    ],
    since="v0.5.0",
    context="primitives.lax",
    component="iota",
    testcases=[
        {
            "testcase": "iota_int32",
            "callable": lambda: lax.iota(np.int32, 5),
            "input_shapes": [],
        },
        {
            "testcase": "iota_float32",
            "callable": lambda: lax.iota(np.float32, 10),
            "input_shapes": [],
        },
        {
            "testcase": "broadcasted_iota",
            "callable": lambda: lax.broadcasted_iota(np.int32, (3, 4), 1),
            "input_shapes": [],
        },
    ],
)
class IotaPlugin(PrimitiveLeafPlugin):
    @staticmethod
    def abstract_eval(*dynamic_shape, dtype, shape, dimension, **__):
        return core.ShapedArray(shape, dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        out_name = s.get_name(node_outputs[0])
        dtype = params["dtype"]
        shape = params["shape"]
        dimension = params["dimension"]

        # Process dynamic dimensions if present
        for i, dim in enumerate(node_inputs):
            shape_dim_name = s.get_name(dim)
            if not isinstance(shape, list):
                shape = list(shape)
            shape[i] = shape_dim_name

        # Create 1D range for the specified dimension
        dim_size = shape[dimension] if isinstance(shape[dimension], int) else None

        # Range operator constants
        start_name = s.get_constant_name(np.array(0, dtype=np.int64))
        delta_name = s.get_constant_name(np.array(1, dtype=np.int64))

        if dim_size is not None:
            limit_name = s.get_constant_name(np.array(dim_size, dtype=np.int64))
        else:
            limit_name = shape[dimension]  # Dynamic dimension name

        # Create Range node
        range_output = s.get_unique_name("range_output")
        s.add_node(
            helper.make_node(
                "Range",
                inputs=[start_name, limit_name, delta_name],
                outputs=[range_output],
                name=s.get_unique_name("range"),
            )
        )

        if dim_size is not None:
            s.add_shape_info(range_output, (dim_size,), np.dtype(np.int64))

        # For 1D case with static shape
        if len(shape) == 1 and isinstance(shape[0], int):
            self._add_cast_if_needed(s, range_output, out_name, dtype)
            return

        # For multidimensional case
        # Create unsqueeze axes for dimensions other than the target dimension
        unsqueeze_axes = [i for i in range(len(shape)) if i != dimension]
        unsqueeze_axes_name = s.get_constant_name(
            np.array(unsqueeze_axes, dtype=np.int64)
        )

        # Unsqueeze to add singleton dimensions
        unsqueezed_output = s.get_unique_name("unsqueezed_output")
        s.add_node(
            helper.make_node(
                "Unsqueeze",
                inputs=[range_output, unsqueeze_axes_name],
                outputs=[unsqueezed_output],
                name=s.get_unique_name("unsqueeze"),
            )
        )

        # Calculate intermediate shape after unsqueezing for shape info
        if all(isinstance(dim, int) for dim in shape):
            intermediate_shape = [1] * len(shape)
            intermediate_shape[dimension] = shape[dimension]
            s.add_shape_info(
                unsqueezed_output, tuple(intermediate_shape), np.dtype(np.int64)
            )

            # Create shape constant for Expand
            shape_name = s.get_constant_name(np.array(shape, dtype=np.int64))
            expanded_output = s.get_unique_name("expanded_output")

            # Expand to final shape
            s.add_node(
                helper.make_node(
                    "Expand",
                    inputs=[unsqueezed_output, shape_name],
                    outputs=[expanded_output],
                    name=s.get_unique_name("expand"),
                )
            )

            s.add_shape_info(expanded_output, tuple(shape), np.dtype(np.int64))
            self._add_cast_if_needed(s, expanded_output, out_name, dtype)
        else:
            raise NotImplementedError(
                "Dynamic shape handling for iota not fully implemented"
            )

    def _add_cast_if_needed(self, s, input_name, output_name, dtype):
        """Helper to add a Cast node if needed or an Identity node otherwise."""
        onnx_dtype = NUMPY_TO_ONNX_DTYPE.get(np.dtype(dtype), TensorProto.FLOAT)
        if np.dtype(dtype) != np.int64:
            s.add_node(
                helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[output_name],
                    name=s.get_unique_name("cast"),
                    to=int(onnx_dtype),
                )
            )
        else:
            s.add_node(
                helper.make_node(
                    "Identity",
                    inputs=[input_name],
                    outputs=[output_name],
                    name=s.get_unique_name("identity"),
                )
            )


# Register abstract eval
iota_p.def_abstract_eval(IotaPlugin.abstract_eval)
