from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
from typing import TYPE_CHECKING, Tuple, List, Union

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define the MatMul primitive
jnp.matmul_p = Primitive("jnp.matmul")


def get_primitive():
    """Returns the jnp.matmul primitive."""
    return jnp.matmul_p


def get_dynamic_output_shape(
    a_shape: Tuple[Union[int, str], ...], b_shape: Tuple[Union[int, str], ...]
) -> Tuple[Union[int, str], ...]:
    """Calculates the output shape of jnp.matmul while handling dynamic dimensions."""
    a_rank, b_rank = len(a_shape), len(b_shape)

    if a_rank == 1 and b_rank == 1:
        if (
            a_shape[0] == b_shape[0]
            or isinstance(a_shape[0], str)
            or isinstance(b_shape[0], str)
        ):
            return ()  # Scalar output
        raise ValueError("Incompatible shapes for matmul")

    a_shape_norm = a_shape if a_rank > 1 else (1,) + a_shape
    b_shape_norm = b_shape if b_rank > 1 else b_shape + (1,)

    a_rows, a_cols = a_shape_norm[-2], a_shape_norm[-1]
    b_rows, b_cols = b_shape_norm[-2], b_shape_norm[-1]

    if a_cols != b_rows and not (isinstance(a_cols, str) or isinstance(b_rows, str)):
        raise ValueError(f"Incompatible shapes for matmul: {a_shape} and {b_shape}")

    batch_dims: List[Union[int, str]] = []
    max_rank = max(a_rank, b_rank)
    for i in range(max_rank - 2):
        a_dim = a_shape[i] if i < a_rank - 2 else 1
        b_dim = b_shape[i] if i < b_rank - 2 else 1
        batch_dims.append(a_dim if a_dim != 1 else b_dim)

    output_shape = tuple(batch_dims) + (a_rows, b_cols)

    if a_rank == 1:
        output_shape = output_shape[1:]
    if b_rank == 1:
        output_shape = output_shape[:-1]

    return output_shape


def matmul_abstract_eval(a, b):
    """Abstract evaluation function for MatMul."""
    output_shape = get_dynamic_output_shape(a.shape, b.shape)
    return core.ShapedArray(output_shape, a.dtype)


# Register abstract evaluation function
jnp.matmul_p.def_abstract_eval(matmul_abstract_eval)


def matmul(a, b):
    """Defines the primitive binding for MatMul."""
    return jnp.matmul_p.bind(a, b)


def patch_info():
    """Provides patching information for MatMul."""
    return {
        "patch_targets": [jnp],
        "patch_function": lambda _: matmul,
        "target_attribute": "matmul",
    }


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_matmul(node_inputs, node_outputs, params):
        input_names = [s.get_name(var) for var in node_inputs]
        output_name = s.get_name(node_outputs[0])

        input_shapes = [inp.aval.shape for inp in node_inputs]
        output_shape = get_dynamic_output_shape(input_shapes[0], input_shapes[1])

        matmul_node = helper.make_node(
            "MatMul",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("matmul"),
        )
        s.add_node(matmul_node)
        s.add_shape_info(output_name, output_shape)

    return handle_matmul


def get_metadata() -> dict:
    """Returns metadata describing this plugin and its test cases."""
    return {
        "jaxpr_primitive": "jnp.matmul",
        "jax_doc": "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.matmul.html",
        "onnx": [
            {
                "component": "MatMul",
                "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
            }
        ],
        "since": "v0.1.0",
        "context": "plugins.jnp",
        "testcases": [
            {
                "testcase": "matmul_2d",
                "callable": lambda a, b: jnp.matmul(a, b),
                "input_shapes": [(3, 4), (4, 5)],
            },
            {
                "testcase": "matmul_1d_2d",
                "callable": lambda a, b: jnp.matmul(a, b),
                "input_shapes": [(4,), (4, 5)],
            },
            {
                "testcase": "matmul_2d_1d",
                "callable": lambda a, b: jnp.matmul(a, b),
                "input_shapes": [(3, 4), (4,)],
            },
            {
                "testcase": "matmul_dynamic",
                "callable": lambda a, b: jnp.matmul(a, b),
                "input_shapes": [("B", 3, 4), ("B", 4, 5)],
            },
            {
                "testcase": "matmul_dynamic_a",
                "callable": lambda a, b: jnp.matmul(a, b),
                "input_shapes": [("B", 3), (3, 4)],
            },
            {
                "testcase": "matmul_dynamic_b",
                "callable": lambda a, b: jnp.matmul(a, b),
                "input_shapes": [(3, "B"), ("B", 4)],
            },
            {
                "testcase": "matmul_1d",
                "callable": lambda a, b: jnp.matmul(a, b),
                "input_shapes": [(4,), (4,)],
            },
            {
                "testcase": "matmul_3d",
                "callable": lambda a, b: jnp.matmul(a, b),
                "input_shapes": [(2, 3, 4), (2, 4, 5)],
            },
        ],
    }
