from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
from typing import TYPE_CHECKING, Tuple, Union, Sequence
import numpy as np

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define a new primitive for tile
jnp.tile_p = Primitive("jnp.tile")


def get_primitive():
    """Returns the jnp.tile primitive."""
    return jnp.tile_p


def tile_abstract_eval(x, repeats: Sequence[int]):
    """Compute the output shape for tile."""
    x_shape = x.shape
    if len(repeats) != len(x_shape):
        if len(repeats) < len(x_shape):
            repeats = (1,) * (len(x_shape) - len(repeats)) + tuple(repeats)
        else:
            x_shape = (1,) * (len(repeats) - len(x_shape)) + x_shape

    output_shape = tuple(s * r for s, r in zip(x_shape, repeats))
    return core.ShapedArray(output_shape, x.dtype)


# Register abstract evaluation function
jnp.tile_p.def_abstract_eval(tile_abstract_eval)


def tile(a, reps: Union[int, Sequence[int]]):
    """Defines the primitive binding for Tile."""
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    return jnp.tile_p.bind(a, repeats=tup)


def patch_info():
    """Provides patching information for Tile."""
    return {
        "patch_targets": [jnp],
        "patch_function": lambda _: tile,
        "target_attribute": "tile",
    }


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_tile(node_inputs, node_outputs, params):
        """Handles ONNX conversion for jnp.tile."""
        repeats = params["repeats"]
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])
        input_shape = node_inputs[0].aval.shape

        # ONNX requires repeats as an initializer.
        repeats_name = s.get_unique_name("tile_repeats")

        # If repeats has more dimensions than input, reshape input first
        actual_input_name = input_name
        if len(repeats) > len(input_shape):
            # Add leading dimensions of size 1
            reshaped_input_shape = (1,) * (
                len(repeats) - len(input_shape)
            ) + input_shape

            # Create shape tensor for reshaping
            shape_name = s.get_unique_name("reshape_shape")
            s.add_initializer(
                name=shape_name, vals=np.array(reshaped_input_shape, dtype=np.int64)
            )

            # Create reshape node
            reshaped_name = s.get_unique_name("reshaped_input")
            reshape_node = helper.make_node(
                "Reshape",
                inputs=[input_name, shape_name],
                outputs=[reshaped_name],
                name=s.get_unique_name("reshape"),
            )
            s.add_node(reshape_node)

            # Update input name and shape for tile operation
            actual_input_name = reshaped_name
            input_shape = reshaped_input_shape
        elif len(repeats) < len(input_shape):
            # Pad repeats to match input rank if needed, prepending 1s
            repeats = (1,) * (len(input_shape) - len(repeats)) + tuple(repeats)

        # Add repeats as initializer
        s.add_initializer(name=repeats_name, vals=np.array(repeats, dtype=np.int64))

        # Create tile node
        tile_node = helper.make_node(
            "Tile",
            inputs=[actual_input_name, repeats_name],
            outputs=[output_name],
            name=s.get_unique_name("tile"),
        )
        s.add_node(tile_node)

        # Calculate output shape
        output_shape = tuple(s * r for s, r in zip(input_shape, repeats))
        s.add_shape_info(output_name, output_shape)

    return handle_tile


def get_metadata() -> dict:
    """Returns metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "jnp.tile",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tile.html",
        "onnx": [
            {
                "component": "Tile",
                "doc": "https://onnx.ai/onnx/operators/onnx__Tile.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.jnp",
        "testcases": [
            {
                "testcase": "tile_a",
                "callable": lambda a: jnp.tile(a, (1, 2)),
                "input_shapes": [(2, 3)],
            },
            {
                "testcase": "tile_b",
                "callable": lambda a: jnp.tile(a, (1, 2, 1)),
                "input_shapes": [(1, 5, 5)],
            },
            {
                "testcase": "tile_c",
                "callable": lambda a: jnp.tile(a, (1, 4)),
                "input_shapes": [(3, 3)],
            },
            {
                "testcase": "tile_d",
                "callable": lambda a: jnp.tile(a, 2),  # Scalar reps
                "input_shapes": [(3, 3)],
            },
            {
                "testcase": "tile_dynamic",
                "callable": lambda a: jnp.tile(a, (2, 1)),  # Repeat batch dimension
                "input_shapes": [("B", 3)],
            },
            {  # Test case to check padding.
                "testcase": "tile_pad",
                "callable": lambda a: jnp.tile(a, (2, 3, 4)),
                "input_shapes": [(4, 5)],
            },
        ],
    }
