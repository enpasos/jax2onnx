from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

import numpy as np

# Define a new primitive for tile.
jnp.tile_p = Primitive("jnp.tile")


def get_primitive():
    return jnp.tile_p


def _tile_abstract_eval(x, repeats):
    x_shape = x.shape
    if len(repeats) != len(x_shape):
        if len(repeats) < len(x_shape):
            repeats = (1,) * (len(x_shape) - len(repeats)) + tuple(repeats)
        else:
            x_shape = (1,) * (len(repeats) - len(x_shape)) + x_shape
            # raise ValueError(
            #    f"repeats length {len(repeats)} does not match input rank {len(x_shape)}"
            # )
    output_shape = tuple(s * r for s, r in zip(x_shape, repeats))
    return core.ShapedArray(output_shape, x.dtype)


jnp.tile_p.def_abstract_eval(_tile_abstract_eval)
jnp.tile_p.multiple_results = False


def _get_monkey_patch():
    def tile(a, reps):
        try:
            tup = tuple(reps)
        except TypeError:
            tup = (reps,)
        return jnp.tile_p.bind(a, repeats=tup)

    return tile


@contextlib.contextmanager
def temporary_patch():
    original_fn = jnp.tile
    jnp.tile = _get_monkey_patch()
    try:
        yield
    finally:
        jnp.tile = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_tile(node_inputs, node_outputs, params):
        repeats = params["repeats"]
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])
        input_shape = node_inputs[0].aval.shape

        # ONNX requires repeats to be an initializer.
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
