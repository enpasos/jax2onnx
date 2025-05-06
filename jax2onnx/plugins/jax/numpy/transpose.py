from collections.abc import Sequence
from typing import TYPE_CHECKING

from jax import core
from jax import numpy as jnp
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the Transpose primitive
jnp.transpose_p = Primitive("jnp.transpose")
jnp.transpose_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=jnp.transpose_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html",
    onnx=[
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="transpose",
    testcases=[
        {
            "testcase": "transpose_basic",
            "callable": lambda a: jnp.transpose(a, axes=(1, 0)),
            "input_shapes": [(2, 3)],
        },
        {
            "testcase": "transpose_reverse",
            "callable": lambda a: jnp.transpose(a, axes=(2, 1, 0)),
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "transpose_4d",
            "callable": lambda a: jnp.transpose(a, axes=(0, 2, 3, 1)),
            "input_shapes": [("B", 2, 3, 4)],
        },
        {
            "testcase": "transpose_square_matrix",
            "callable": lambda a: jnp.transpose(a, axes=(1, 0)),
            "input_shapes": [(5, 5)],
        },
        {
            "testcase": "transpose_high_dim",
            "callable": lambda a: jnp.transpose(a, axes=(4, 3, 2, 1, 0)),
            "input_shapes": [(2, 3, 4, 5, 6)],
        },
        {
            "testcase": "transpose_no_axes",  # Test case for default axes (reversal)
            "callable": lambda a: jnp.transpose(a),
            "input_shapes": [(2, 3, 4)],
        },
        {
            "testcase": "transpose_3d",
            "callable": lambda a: jnp.transpose(a, axes=(0, 2, 1)),
            "input_shapes": [("B", 3, 4)],  # Dynamic batch dimension
        },
    ],
)
class TransposePlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.transpose to ONNX.
    """

    @staticmethod
    def _transpose_abstract_eval(x, axes: tuple[int, ...] | None):
        """Computes the output shape for jnp.transpose, robust to tracers."""
        x_shape = list(x.shape)
        if axes is None:
            axes = tuple(reversed(range(len(x_shape))))
        if len(axes) != len(x_shape):
            raise ValueError(
                f"Axes length {len(axes)} does not match input rank {len(x_shape)}"
            )

        def safe_dim(dim):
            # Accept int, str, or use -1 for tracers/unhashable
            try:
                hash(dim)
                return dim
            except Exception:
                return -1

        output_shape = tuple(safe_dim(x_shape[i]) for i in axes)
        return core.ShapedArray(output_shape, x.dtype)

    @staticmethod
    def abstract_eval(x, axes: tuple[int, ...] | None):
        return TransposePlugin._transpose_abstract_eval(x, axes)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles ONNX conversion for jnp.transpose."""
        axes = params["axes"]
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])
        input_shape = node_inputs[0].aval.shape

        # If axes is None, default to reversing the axes.
        if axes is None:
            axes = tuple(reversed(range(len(input_shape))))
        else:
            axes = tuple(axes)  # Ensure axes is a tuple.

        transpose_node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[output_name],
            perm=list(axes),  # ONNX expects a list
            name=s.get_unique_name("transpose"),
        )
        s.add_node(transpose_node)

        output_shape = tuple(input_shape[i] for i in axes)
        s.add_shape_info(output_name, output_shape)

    @staticmethod
    def _transpose(a, axes: Sequence[int] | int | None = None):
        """Defines the primitive binding for Transpose."""
        n = len(a.shape)
        if axes is None:
            axes = tuple(reversed(range(n)))
        elif isinstance(axes, int):
            axes = (axes,) + tuple(i for i in range(n) if i != axes)
        else:
            axes = tuple(axes)
        if len(axes) != n:
            raise ValueError(f"Axes length {len(axes)} does not match input rank {n}")
        return jnp.transpose_p.bind(a, axes=axes)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Transpose."""

        def patched_transpose(a, axes: Sequence[int] | int | None = None):
            return TransposePlugin._transpose(a, axes)

        return patched_transpose

    @staticmethod
    def patch_info():
        """Provides patching information for Transpose."""
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: TransposePlugin.get_monkey_patch(),
            "target_attribute": "transpose",
        }


# Register abstract evaluation function
jnp.transpose_p.def_abstract_eval(TransposePlugin.abstract_eval)
