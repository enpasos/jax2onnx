from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
import contextlib
from typing import TYPE_CHECKING, Tuple, Union, Optional

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


# Define a new primitive for squeeze.
jnp.squeeze_p = Primitive("jnp.squeeze")


def get_primitive():
    return jnp.squeeze_p


def _squeeze_abstract_eval(x, axes: Optional[Tuple[int, ...]]):
    """
    Compute the output shape for squeeze.
    If no axes are provided, squeeze all dimensions that are 1.
    If axes are provided, only remove those axes for which the dimension
    is concrete and equal to 1; if the dimension is dynamic (a string), skip
    the check.
    """
    x_shape = list(x.shape)
    if axes is None:
        # Squeeze all dimensions that are known to be 1 (skip dynamic ones).
        new_shape = tuple(
            dim for dim in x_shape if not (isinstance(dim, int) and dim == 1)
        )
    else:
        # Normalize negative axes.
        normalized_axes = [axis if axis >= 0 else axis + len(x_shape) for axis in axes]
        # Check that concrete dimensions are 1.
        for axis in normalized_axes:
            if axis >= len(x_shape):
                raise ValueError(f"Invalid axis {axis} for shape {x_shape}")
            # Only check if the dimension is a concrete integer.
            if isinstance(x_shape[axis], int) and x_shape[axis] != 1:
                raise ValueError(
                    f"Cannot squeeze dimension {axis} of shape {x_shape}: size is not 1."
                )
        new_shape = tuple(
            dim for i, dim in enumerate(x_shape) if i not in normalized_axes
        )
    return core.ShapedArray(new_shape, x.dtype)


jnp.squeeze_p.def_abstract_eval(_squeeze_abstract_eval)
jnp.squeeze_p.multiple_results = False


def _get_monkey_patch():
    def squeeze(a, axis: Optional[Union[int, Tuple[int, ...]]] = None):
        if axis is None:
            axes = tuple(
                i for i, dim in enumerate(a.shape) if isinstance(dim, int) and dim == 1
            )
        elif isinstance(axis, int):
            axes = (axis,)
        else:
            axes = tuple(axis)
        return jnp.squeeze_p.bind(a, axes=axes)

    return squeeze


@contextlib.contextmanager
def temporary_patch():
    original_fn = jnp.squeeze
    jnp.squeeze = _get_monkey_patch()
    try:
        yield
    finally:
        jnp.squeeze = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_squeeze(node_inputs, node_outputs, params):
        # Retrieve the axes parameter (should be provided as a tuple of ints).
        axes = params["axes"]
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])
        input_shape = node_inputs[0].aval.shape

        # Normalize axes: convert any negative axes to positive.
        valid_axes = [axis if axis >= 0 else axis + len(input_shape) for axis in axes]

        # Create an initializer for the axes (ONNX expects these as a tensor).
        axes_name = s.get_unique_name("squeeze_axes")
        s.add_initializer(name=axes_name, vals=valid_axes)

        squeeze_node = helper.make_node(
            "Squeeze",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=s.get_unique_name("squeeze"),
        )
        s.add_node(squeeze_node)

        # Compute output shape for shape inference by removing squeezed axes.
        output_shape = tuple(
            dim for i, dim in enumerate(input_shape) if i not in valid_axes
        )
        s.add_shape_info(output_name, output_shape)

    return handle_squeeze


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "jnp.squeeze",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.squeeze.html",
        "onnx": [
            {
                "component": "Squeeze",
                "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.jnp",
        "testcases": [
            {
                "testcase": "squeeze_single_dim",
                "callable": lambda a: jnp.squeeze(a, axis=0),
                "input_shapes": [(1, 49, 10)],
            },
            {
                "testcase": "squeeze_multiple_dims",
                "callable": lambda a: jnp.squeeze(a, axis=(0, 2)),
                "input_shapes": [(1, 49, 1, 10)],
            },
            {
                "testcase": "squeeze_vit_output",
                "callable": lambda a: jnp.squeeze(a, axis=1),
                "input_shapes": [(1, 1, 10)],
            },
            {
                "testcase": "squeeze_dynamic_batch",
                "callable": lambda a: jnp.squeeze(a, axis=1),
                "input_shapes": [("B", 1, 10)],
            },
            {
                "testcase": "squeeze_all_dims",
                "callable": lambda a: jnp.squeeze(a),
                "input_shapes": [(1, 1, 1)],
            },
            {
                "testcase": "squeeze_negative_axis",
                "callable": lambda a: jnp.squeeze(a, axis=-1),
                "input_shapes": [(1, 49, 1)],
            },
            {
                "testcase": "squeeze_negative_axis_tuple",
                "callable": lambda a: jnp.squeeze(a, axis=(-1, -3)),
                "input_shapes": [(1, 49, 1)],
            },
            {
                "testcase": "squeeze_dynamic_and_negative_axis",
                "callable": lambda a: jnp.squeeze(a, axis=(-1, -3)),
                "input_shapes": [(1, "B", 1)],
            },
        ],
    }
