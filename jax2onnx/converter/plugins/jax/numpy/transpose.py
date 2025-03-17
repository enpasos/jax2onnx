from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
import contextlib
from typing import TYPE_CHECKING, Tuple, Union, Sequence, Optional

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


# Define a new primitive for transpose.
jnp.transpose_p = Primitive("jnp.transpose")


def get_primitive():
    return jnp.transpose_p


def _transpose_abstract_eval(x, axes: Optional[Tuple[int, ...]]):
    x_shape = list(x.shape)
    if axes is None:
        axes = tuple(reversed(range(len(x_shape))))
    if len(axes) != len(x_shape):
        raise ValueError(
            f"Axes length {len(axes)} does not match input rank {len(x_shape)}"
        )

    output_shape = [x_shape[i] for i in axes]
    return core.ShapedArray(tuple(output_shape), x.dtype)


jnp.transpose_p.def_abstract_eval(_transpose_abstract_eval)
jnp.transpose_p.multiple_results = False


def _get_monkey_patch():
    def transpose(a, axes: Optional[Union[Sequence[int], int]] = None):
        n = len(a.shape)
        if axes is None:
            axes = tuple(reversed(range(n)))
        elif isinstance(axes, int):
            # Correctly handle the single-integer case.
            axes = (axes,) + tuple(i for i in range(n) if i != axes)
        else:
            axes = tuple(axes)
        if len(axes) != n:
            raise ValueError(f"Axes length {len(axes)} does not match input rank {n}")
        return jnp.transpose_p.bind(a, axes=axes)

    return transpose


@contextlib.contextmanager
def temporary_patch():
    original_fn = jnp.transpose
    jnp.transpose = _get_monkey_patch()
    try:
        yield
    finally:
        jnp.transpose = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_transpose(node_inputs, node_outputs, params):
        axes = params["axes"]
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])
        input_shape = node_inputs[0].aval.shape

        # If axes is None, default to reversing the axes.
        if axes is None:
            axes = tuple(reversed(range(len(input_shape))))
        elif isinstance(axes, int):
            n = len(input_shape)
            axes = (axes,) + tuple(i for i in range(n) if i != axes)
        else:
            axes = tuple(axes)  # Ensure axes is a tuple

        # Normalize negative axes. No need, it is done in the abstract eval
        # normalized_axes = tuple(axis if axis >= 0 else axis + len(input_shape) for axis in axes)

        transpose_node = helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[output_name],
            perm=list(axes),  # ONNX expects a list
            name=s.get_unique_name("transpose"),
        )
        s.add_node(transpose_node)

        output_shape = tuple(input_shape[i] for i in axes)
        s.add_shape_info(output_name, output_shape)  # Use s.builder

    return handle_transpose


def get_metadata() -> dict:
    return {
        "jaxpr_primitive": "jnp.transpose",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.transpose.html",
        "onnx": [
            {
                "component": "Transpose",
                "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.jnp",
        "testcases": [
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
                "input_shapes": [(1, 2, 3, 4)],
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
                "testcase": "transpose_dynamic",
                "callable": lambda a: jnp.transpose(a, axes=(0, 2, 1)),
                "input_shapes": [("B", 3, 4)],  # Dynamic batch dimension
            },
        ],
    }
