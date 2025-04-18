from typing import TYPE_CHECKING

from jax import core
from jax import numpy as jnp
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the Concat primitive
jnp.concat_p = Primitive("jnp.concat")
jnp.concat_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=jnp.concat_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.concat.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="concat",
    testcases=[
        {
            "testcase": "concat",
            "callable": lambda a, b: jnp.concat((a, b), axis=0),
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "concat_abstract_middle_dim",
            "callable": lambda a, b: jnp.concatenate((a, b), axis=1),
            "input_shapes": [("B", 1, 8), ("B", 10, 8)],
            "expected_output_shapes": [("B", 11, 8)],
        },
    ],
)
class ConcatPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.concatenate to ONNX.  Note:  jax.numpy.concat
    is an alias for jax.numpy.concatenate.
    """

    @staticmethod
    def abstract_eval(*arrays, axis):
        import numpy as np

        if not arrays:
            raise ValueError("concatenate requires at least one input array.")

        actual_arrays = [
            a for a in arrays if hasattr(a, "shape") and hasattr(a, "dtype")
        ]
        if not actual_arrays:
            raise ValueError(
                "concatenate requires at least one array input with shape and dtype."
            )

        first_aval = actual_arrays[0]

        try:
            output_dtype = first_aval.dtype
            if not isinstance(output_dtype, (np.dtype, type(None))):
                if hasattr(output_dtype, "numpy_dtype"):
                    output_dtype = output_dtype.numpy_dtype
                else:
                    output_dtype = np.dtype("float32")
            elif output_dtype is None:
                output_dtype = np.dtype("float32")
        except Exception:
            output_dtype = np.dtype("float32")

        if not isinstance(output_dtype, np.dtype):
            try:
                output_dtype = np.dtype(output_dtype)
            except TypeError:
                output_dtype = np.dtype("float32")

        rank = first_aval.ndim
        for i, aval in enumerate(actual_arrays[1:], 1):
            if aval.ndim != rank:
                all_shapes = [a.shape for a in actual_arrays]
                raise TypeError(
                    f"Concatenate inputs must have same rank ({rank}). Got shapes {all_shapes} at index {i}"
                )

        if isinstance(axis, core.Tracer):
            raise TypeError(
                "Axis for concatenate cannot be a tracer during abstract evaluation."
            )
        if not isinstance(axis, (int, np.integer)):
            raise TypeError(f"Axis must be an integer, got {type(axis)}")
        if not -rank <= axis < rank:
            raise ValueError(f"Axis {axis} out of range for rank {rank}")
        axis = axis % rank

        def get_dim_size(aval, dim_idx):
            try:
                dim = aval.shape[dim_idx]
                if isinstance(dim, (int, np.integer)) and dim >= 0:
                    return int(dim)
                else:
                    return -1
            except AttributeError:
                return -1

        output_shape_list = []
        for i in range(rank):
            dims_at_i = [get_dim_size(aval, i) for aval in actual_arrays]
            if i == axis:
                # Axis to be concatenated: if any dim is unknown, result is unknown
                output_shape_list.append(-1 if -1 in dims_at_i else sum(dims_at_i))
            else:
                # Other dims: optimistic if at least one is concrete and all others are either same or unknown
                concrete = [d for d in dims_at_i if d != -1]
                if not concrete:
                    output_shape_list.append(-1)
                elif all(d == concrete[0] or d == -1 for d in dims_at_i):
                    output_shape_list.append(concrete[0])
                else:
                    raise TypeError(
                        f"Concat incompatible dim values at axis {i}: {dims_at_i}"
                    )

        final_output_shape_tuple = tuple(output_shape_list)
        return core.ShapedArray(final_output_shape_tuple, output_dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        axis = params.get("axis", 0)
        input_names = [s.get_name(var) for var in node_inputs]
        output_name = s.get_name(node_outputs[0])

        concat_node = helper.make_node(
            "Concat",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("concat"),
            axis=axis,
        )
        s.add_node(concat_node)

    @staticmethod
    def _concat(arrays, axis):
        return jnp.concat_p.bind(*arrays, axis=axis)

    @staticmethod
    def get_monkey_patch():
        def patched_concat(arrays, axis):
            return ConcatPlugin._concat(arrays, axis)

        return patched_concat

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: ConcatPlugin.get_monkey_patch(),
            "target_attribute": "concatenate",
        }


jnp.concat_p.def_abstract_eval(ConcatPlugin.abstract_eval)
