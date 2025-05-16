from typing import TYPE_CHECKING  # added Optional

import jax
import jax.core as core  # import core for ShapedArray, AbstractValue
from onnx import helper

from jax2onnx.converter.dynamic_utils import encode_dims
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax import lax

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


reshape_p = lax.reshape_p


@register_primitive(
    jaxpr_primitive=jax.lax.reshape_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reshape.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        }
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="reshape",
    testcases=[
        {
            "testcase": "reshape",
            "callable": lambda x: jax.lax.reshape(x, (9,)),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "reshape_valid_squeeze_middle_dim_from_problematic_source",
            # Input shape is {201,1,201} from the error.
            # A valid reshape could be to {201,201} (squeezing the middle dim).
            "callable": lambda x: jax.lax.reshape(
                x,
                new_sizes=(x.shape[0], x.shape[2]),  # Target (201, 201)
                dimensions=(0, 1, 2),
            ),  # Old dims to map new_sizes
            "input_shapes": [(201, 1, 201)],
        },
        {
            "testcase": "reshape_valid_flatten_trailing",
            # Reshape (N, M, K) to (N, M*K)
            "callable": lambda x: jax.lax.reshape(
                x, new_sizes=(x.shape[0], x.shape[1] * x.shape[2]), dimensions=(0, 1, 2)
            ),
            "input_shapes": [(201, 1, 5)],
        },
        {
            "testcase": "reshape_with_target_shape_from_symbolic_dim_computation",
            # This tests if jax2onnx correctly handles new_sizes derived from symbolic computations,
            # relevant to the user's directive on `dim_as_value.to_onnx`.
            "callable": lambda x: jax.lax.reshape(
                x,
                new_sizes=(
                    jax.lax.dynamic_dimension_size(x, 0),  # N
                    jax.lax.dynamic_dimension_size(x, 1)
                    * jax.lax.dynamic_dimension_size(x, 2),  # M*K
                ),
                dimensions=(0, 1, 2),
            ),
            "input_shapes": [("N", "M", "K")],
        },
    ],
)
class ReshapePlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.reshape to ONNX Reshape."""

    # -----------------------------------------------------------------
    # abstract-eval
    # -----------------------------------------------------------------
    @staticmethod
    def abstract_eval(
        operand: core.ShapedArray,
        new_sizes: core.ShapedArray,
        dimensions: core.ShapedArray,
        *,
        num_leaves: int = 0,
        **kwargs,
    ) -> core.AbstractValue:  # return a single AbstractValue
        # Accept and ignore extra keyword args (e.g., 'sharding', 'body_jaxpr')
        if hasattr(new_sizes, "shape"):
            shape = new_sizes.shape
            dtype = new_sizes.dtype
        else:  # new_sizes is a Python tuple / list
            shape = tuple(new_sizes)
            dtype = operand.dtype  # keep operand dtype

        # reshape outputs a SINGLE array → return the ShapedArray directly
        return core.ShapedArray(shape, dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX reshape primitive."""
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_var_name(node_outputs[0])
        new_shape = params["new_sizes"]
        input_shape = node_inputs[0].aval.shape
        dtype = node_inputs[0].aval.dtype

        def _process_newshape(newshape):
            if isinstance(newshape, (int, str)):
                newshape = [newshape]
            else:
                newshape = list(newshape)
            neg_one_count = sum(1 for dim in newshape if dim == -1)
            if neg_one_count > 1:
                raise ValueError("Only one dimension can be -1 (inferred).")
            return newshape

        def _concretize_shape(shape, concrete_value=2):
            return tuple(
                concrete_value if isinstance(dim, str) else dim for dim in shape
            )

        processed_newshape = _process_newshape(new_shape)
        _concretize_shape(processed_newshape)

        if len(new_shape) == 2 and new_shape[0] == 1 and input_shape == (new_shape[1],):
            s.var_to_name[node_outputs[0]] = input_name
            return

        # ✅ FIX HERE: Use get_constant_name to reliably register metadata
        shape_name = s.get_constant_name(encode_dims(new_shape))

        node = helper.make_node(
            "Reshape",
            inputs=[input_name, shape_name],
            outputs=[output_name],
            name=s.get_unique_name("reshape"),
        )
        s.add_node(node)

        s.builder.register_value_info_metadata(
            output_name, shape=tuple(processed_newshape), dtype=dtype
        )


# Bind abstract evaluation
reshape_p.def_abstract_eval(ReshapePlugin.abstract_eval)
