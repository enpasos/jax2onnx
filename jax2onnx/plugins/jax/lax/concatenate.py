# file: jax2onnx/plugins/jax/lax/concatenate.py


from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from onnx import helper, TensorProto

from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.concatenate_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {
            "component": "Cast",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html",
        },
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="concatenate",
    testcases=[
        {
            "testcase": "concatenate",
            "callable": lambda a, b: jax.lax.concatenate(
                (a, b), dimension=0
            ),  # Corrected callable
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "concatenate_axis1",
            "callable": lambda a, b: jax.lax.concatenate(
                (a, b), dimension=1
            ),  # Corrected callable
            "input_shapes": [("B", 3), ("B", 4)],
        },
        {
            "testcase": "concatenate_axis0",
            "callable": lambda a, b: jax.lax.concatenate(
                (a, b), dimension=0
            ),  # Corrected callable
            "input_shapes": [(7, 3), (4, 3)],
        },
        {  # 3D inputs
            "testcase": "concatenate_3d",
            "callable": lambda a, b: jax.lax.concatenate(
                (a, b), dimension=1
            ),  # Corrected callable
            "input_shapes": [(2, 3, 4), (2, 5, 4)],
        },
        {
            # Regression: zero-arg; internal int32s -> concatenate -> cast to f32.
            # Ensures the Concat value_info dtype is recorded as int32 and ORT loads.
            "testcase": "concatenate_internal_int32_then_cast_to_f32_zeroarg",
            "callable": (
                lambda: jax.lax.concatenate(
                    (jnp.array([1], dtype=jnp.int32), jnp.array([2], dtype=jnp.int32)),
                    dimension=0,
                ).astype(jnp.float32)
            ),
            "expected_output_shapes": [(2,)],
            # In f64 variant the harness tends to expect DOUBLE for floating outputs.
            # We instead assert explicitly that the graph output type is FLOAT (f32).
            "post_check_onnx_graph": (
                lambda m: len(m.graph.output) == 1
                and m.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
            ),
            "run_only_f64_variant": True,
        },
    ],
)
class ConcatenatePlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.concatenate to ONNX.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX concatenate primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_name(node_outputs[0])
        dimension = params["dimension"]

        # Shapes & dtypes (prefer aval metadata during tracing)
        input_shapes = []
        input_dtypes = []
        for inp in node_inputs:
            if hasattr(inp, "aval"):
                input_shapes.append(inp.aval.shape)
                input_dtypes.append(np.dtype(inp.aval.dtype))
            elif hasattr(inp, "shape") and hasattr(inp, "dtype"):
                input_shapes.append(tuple(inp.shape))
                input_dtypes.append(np.dtype(inp.dtype))
            else:
                raise ValueError(f"Input to concatenate lacks shape/dtype: {inp!r}")

        # Decide target dtype: promote across inputs (usually identical in JAX)
        tgt_dtype = input_dtypes[0]
        for dt in input_dtypes[1:]:
            tgt_dtype = np.promote_types(tgt_dtype, dt)

        output_shape = list(input_shapes[0])  # Start with the shape of the first input
        # Normalize the axis
        rank = len(output_shape)
        dimension = dimension % rank if dimension < 0 else dimension

        for shape in input_shapes[1:]:
            if len(shape) != rank:
                raise ValueError("Inputs must have the same rank.")
            for i in range(rank):
                if i != dimension:
                    if output_shape[i] != shape[i] and not (
                        isinstance(output_shape[i], str) or isinstance(shape[i], str)
                    ):
                        raise ValueError(
                            f"Shapes are incompatible along dimension {i} for Concatenate: {output_shape} vs {shape}"
                        )
                    else:
                        if isinstance(shape[i], str):
                            output_shape[i] = shape[i]

        # Accumulate on the concat axis
        for shape in input_shapes[1:]:
            output_shape[dimension] = (
                output_shape[dimension] + shape[dimension]
                if isinstance(output_shape[dimension], int)
                and isinstance(shape[dimension], int)
                else f"{output_shape[dimension]}+{shape[dimension]}"  # Keep strings
            )

        # Ensure ONNX Concat sees uniform input dtype; insert Casts if needed.
        norm_inputs = []
        for name, dt, shp in zip(input_names, input_dtypes, input_shapes):
            if np.dtype(dt) != np.dtype(tgt_dtype):
                cast_out = s.builder.get_unique_name("Concat_cast")
                s.add_node(
                    helper.make_node(
                        "Cast",
                        [name],
                        [cast_out],
                        to=int(s.builder._numpy_dtype_to_onnx(tgt_dtype)),
                        name=s.get_unique_name("Cast"),
                    )
                )
                s.add_shape_info(cast_out, tuple(shp), np.dtype(tgt_dtype))
                norm_inputs.append(cast_out)
            else:
                norm_inputs.append(name)

        node = helper.make_node(
            "Concat",
            inputs=norm_inputs,
            outputs=[output_name],
            name=s.get_unique_name("concat"),
            axis=dimension,
        )
        s.add_node(node)
        # Record the correct output dtype (fixes #74 regression with zero-arg case)
        s.add_shape_info(output_name, tuple(output_shape), np.dtype(tgt_dtype))
