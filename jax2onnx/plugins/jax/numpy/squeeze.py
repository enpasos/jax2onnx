from typing import TYPE_CHECKING

from jax import core
from jax import numpy as jnp
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

import numpy as np

# Define a new primitive for squeeze
jnp.squeeze_p = Primitive("jnp.squeeze")
jnp.squeeze_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jnp.squeeze_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.squeeze.html",
    onnx=[
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="squeeze",
    testcases=[
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
)
class SqueezePlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.squeeze to ONNX.
    """

    @staticmethod
    def _squeeze_abstract_eval(x, axes: tuple[int, ...] | None):
        """
        Compute the output shape for squeeze.
        - If no axes are provided, squeeze all dimensions that are 1.
        - If axes are provided, only remove dimensions that are concretely 1.
        - Dynamic dimensions (strings) are not squeezed.
        """
        x_shape = list(x.shape)
        if axes is None:
            new_shape = tuple(
                dim for dim in x_shape if not (isinstance(dim, int) and dim == 1)
            )
        else:
            normalized_axes = [
                axis if axis >= 0 else axis + len(x_shape) for axis in axes
            ]
            for axis in normalized_axes:
                if axis >= len(x_shape):
                    raise ValueError(f"Invalid axis {axis} for shape {x_shape}")
                if isinstance(x_shape[axis], int) and x_shape[axis] != 1:
                    raise ValueError(
                        f"Cannot squeeze dimension {axis} of shape {x_shape}: size is not 1."
                    )
            new_shape = tuple(
                dim for i, dim in enumerate(x_shape) if i not in normalized_axes
            )
        return core.ShapedArray(new_shape, x.dtype)

    @staticmethod
    def abstract_eval(x, axes: tuple[int, ...] | None):
        return SqueezePlugin._squeeze_abstract_eval(x, axes)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles ONNX conversion for jnp.squeeze."""
        axes = params["axes"]
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])
        input_shape = node_inputs[0].aval.shape

        valid_axes = [axis if axis >= 0 else axis + len(input_shape) for axis in axes]

        # Create an initializer for axes (ONNX expects these as a tensor)
        axes_name = s.get_unique_name("squeeze_axes")
        s.add_initializer(name=axes_name, vals=np.array(valid_axes, dtype=np.int64))

        squeeze_node = helper.make_node(
            "Squeeze",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            name=s.get_unique_name("squeeze"),
        )
        s.add_node(squeeze_node)

        output_shape = tuple(
            dim
            for i, dim in enumerate(input_shape)
            if i not in valid_axes
            or (isinstance(dim, str))  # Keep dynamic dims even if in valid_axes
        )

        s.add_shape_info(output_name, output_shape)

    @staticmethod
    def _squeeze(a, axis: int | tuple[int, ...] | None = None):
        """Defines the primitive binding for Squeeze."""
        if axis is None:
            axes = tuple(
                i for i, dim in enumerate(a.shape) if isinstance(dim, int) and dim == 1
            )
        elif isinstance(axis, int):
            axes = (axis,)
        else:
            axes = tuple(axis)
        return jnp.squeeze_p.bind(a, axes=axes)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Squeeze."""

        def patched_squeeze(a, axis: int | tuple[int, ...] | None = None):
            return SqueezePlugin._squeeze(a, axis)

        return patched_squeeze

    @staticmethod
    def patch_info():
        """Provides patching information for Squeeze."""
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: SqueezePlugin.get_monkey_patch(),
            "target_attribute": "squeeze",
        }


# Register abstract evaluation function
jnp.squeeze_p.def_abstract_eval(SqueezePlugin.abstract_eval)
