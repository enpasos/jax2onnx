from functools import reduce
import operator
from typing import TYPE_CHECKING, Any, Callable, List, Union
import jax
import numpy as np
from onnx import helper
from jax import core, lax
from jax._src.export.shape_poly import _DimExpr as DimExpr

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the primitive for lax.reshape
reshape_p = lax.reshape_p


@register_primitive(
    jaxpr_primitive=lax.reshape_p.name,
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
            "callable": lambda x: jax.lax.reshape(
                x,
                new_sizes=(x.shape[0], x.shape[2]),
                dimensions=(0, 1, 2),
            ),
            "input_shapes": [(201, 1, 201)],
        },
        {
            "testcase": "reshape_valid_flatten_trailing",
            "callable": lambda x: jax.lax.reshape(x, new_sizes=(x.shape[0], -1)),
            "input_shapes": [(201, 1, 5)],
        },
        {
            "testcase": "reshape_with_target_shape_from_symbolic_dim_computation",
            "callable": lambda x: jax.lax.reshape(x, new_sizes=(x.shape[0], -1)),
            "input_shapes": [("N", "M", "K")],
        },
        {
            "testcase": "reshape_with_inferred_dimension_from_input_dynamic",
            "callable": lambda x: jax.lax.reshape(x, new_sizes=(x.shape[0], -1)),
            "input_shapes": [("B", 10, 10)],
        },
        {
            "testcase": "reshape_with_inferred_dimension_from_input",
            "callable": lambda x: jax.lax.reshape(x, new_sizes=(x.shape[0], -1)),
            "input_shapes": [(3, 10, 10)],
        },
        {
            "testcase": "reshape_merge_symbolic_with_static_and_check_name",
            "callable": lambda x: jax.lax.reshape(x, new_sizes=(-1, x.shape[2])),
            "input_shapes": [("B", 4, 16)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: (
                m.graph.output[0].type.tensor_type.shape.dim[0].dim_param != "B"
                and m.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 16
            ),
        },
    ],
)
class ReshapePlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.reshape to ONNX Reshape."""

    _ORIG_CALL: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(operand_aval: core.ShapedArray, *, new_sizes, **kwargs):
        """
        Manually compute the output shape for reshape, handling -1 for inferred dimensions.
        """
        if not operand_aval.shape:
            input_nelem = 1
        else:
            input_nelem = reduce(operator.mul, operand_aval.shape)

        neg_one_idx = -1
        known_dims_prod = 1
        for i, d in enumerate(new_sizes):
            if d == -1:
                if neg_one_idx != -1:
                    raise ValueError(
                        "Only one '-1' is allowed in new_sizes for reshape."
                    )
                neg_one_idx = i
            else:
                known_dims_prod *= d

        output_shape_list = list(new_sizes)
        if neg_one_idx != -1:
            inferred_dim = input_nelem // known_dims_prod
            output_shape_list[neg_one_idx] = inferred_dim

        return core.ShapedArray(tuple(output_shape_list), operand_aval.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        data_input = node_inputs[0]
        data_input_name = s.get_name(data_input)
        output_var = node_outputs[0]
        output_name = s.get_var_name(output_var)

        new_sizes = params["new_sizes"]
        dynamic_dims_iter = iter(node_inputs[1:])

        shape_components = []
        data_shape_name = None

        for dim in new_sizes:
            if isinstance(dim, (int, np.integer)):
                const_name = s.get_constant_name(np.array([dim], dtype=np.int64))
                shape_components.append(const_name)
            elif isinstance(dim, DimExpr):
                if data_shape_name is None:
                    data_shape_name = s.get_unique_name(f"{data_input_name}_shape")
                    s.add_node(
                        helper.make_node("Shape", [data_input_name], [data_shape_name])
                    )
                    s.add_shape_info(
                        data_shape_name, (len(data_input.aval.shape),), np.int64
                    )
                try:
                    axis_index = data_input.aval.shape.index(dim)
                except ValueError:
                    raise ValueError(
                        f"Could not find symbolic dimension {dim} in input shape {data_input.aval.shape}"
                    )
                axis_const = s.get_constant_name(np.array(axis_index, dtype=np.int64))
                gathered_dim_scalar = s.get_unique_name(
                    f"{data_input_name}_dim{axis_index}"
                )
                s.add_node(
                    helper.make_node(
                        "Gather",
                        [data_shape_name, axis_const],
                        [gathered_dim_scalar],
                        axis=0,
                    )
                )
                s.add_shape_info(gathered_dim_scalar, (), np.int64)
                unsqueezed_dim = s.get_unique_name(f"{gathered_dim_scalar}_unsqueezed")
                unsqueeze_axes = s.get_constant_name(np.array([0], dtype=np.int64))
                s.add_node(
                    helper.make_node(
                        "Unsqueeze",
                        [gathered_dim_scalar, unsqueeze_axes],
                        [unsqueezed_dim],
                    )
                )
                s.add_shape_info(unsqueezed_dim, (1,), np.int64)
                shape_components.append(unsqueezed_dim)
            elif hasattr(dim, "dtype") and np.issubdtype(
                getattr(dim, "dtype", None), np.integer
            ):
                dynamic_dim_var = next(dynamic_dims_iter)
                dynamic_dim_name = s.get_name(dynamic_dim_var)
                unsqueezed_dim_name = s.get_unique_name(
                    f"{dynamic_dim_name}_unsqueezed"
                )
                unsqueeze_axes_const = s.get_constant_name(
                    np.array([0], dtype=np.int64)
                )
                s.add_node(
                    helper.make_node(
                        "Unsqueeze",
                        [dynamic_dim_name, unsqueeze_axes_const],
                        [unsqueezed_dim_name],
                    )
                )
                s.add_shape_info(unsqueezed_dim_name, (1,), np.int64)
                shape_components.append(unsqueezed_dim_name)
            else:
                raise TypeError(
                    f"Unexpected type in new_sizes for reshape: {type(dim)}"
                )

        if not shape_components:
            shape_name = s.get_constant_name(np.array([], dtype=np.int64))
        elif len(shape_components) == 1:
            shape_name = shape_components[0]
        else:
            shape_name = s.get_unique_name("shape_tensor")
            s.add_node(
                helper.make_node(
                    "Concat", inputs=shape_components, outputs=[shape_name], axis=0
                )
            )
            s.add_shape_info(shape_name, (len(shape_components),), np.int64)

        s.add_node(
            helper.make_node(
                "Reshape",
                inputs=[data_input_name, shape_name],
                outputs=[output_name],
                name=s.get_unique_name("reshape"),
            )
        )

        # --- THE FIX ---
        # Manually construct the correct symbolic output shape, creating new
        # symbols for derived dimensions.
        output_aval = output_var.aval
        input_dims = {d for d in data_input.aval.shape if not isinstance(d, int)}

        final_shape: List[Union[int, str]] = []
        for dim in output_aval.shape:
            if isinstance(dim, int):
                final_shape.append(dim)
            elif dim in input_dims:
                # This dimension is identical to an input dimension, reuse its symbol.
                final_shape.append(s._dim_to_symbol_safe(dim))
            else:
                # This is a new, derived dimension (e.g., B*4).
                # We must create a new symbolic name for it.
                new_symbol = s.get_unique_name("dim")
                final_shape.append(new_symbol)
                # Update the converter's mapping so this new symbol is consistently used
                # if the same derived dimension object appears again.
                s.builder.var_to_symbol_map[dim] = new_symbol
                s.builder.var_to_symbol_map[str(dim)] = new_symbol

        s.add_shape_info(output_name, tuple(final_shape), output_aval.dtype)

    @staticmethod
    def patch_info():
        def _creator(orig_fn):
            ReshapePlugin._ORIG_CALL = orig_fn

            def patched_reshape(operand, new_sizes, dimensions=None, **kwargs):
                """
                Patched version of reshape that binds the primitive directly.
                """
                return reshape_p.bind(
                    operand,
                    new_sizes=new_sizes,
                    dimensions=dimensions,
                    sharding=None,
                )

            return patched_reshape

        return {
            "patch_targets": [lax],
            "target_attribute": "reshape",
            "patch_function": _creator,
        }


# Register the abstract evaluation rule with the primitive
lax.reshape_p.def_abstract_eval(ReshapePlugin.abstract_eval)
