from jax import core, numpy as jnp
from jax.extend.core import Primitive
from onnx import helper
import contextlib
from typing import TYPE_CHECKING, Tuple, List, Union

if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter

# Define a new primitive for matmul.
jnp.matmul_p = Primitive("jnp.matmul")


def get_primitive():
    return jnp.matmul_p


def _get_dynamic_output_shape(
    a_shape: Tuple[Union[int, str], ...], b_shape: Tuple[Union[int, str], ...]
) -> Tuple[Union[int, str], ...]:
    """Calculates the output shape of jnp.matmul, handling dynamic dimensions.

    This function mimics the shape inference logic of jnp.matmul, considering
    dynamic dimensions (represented by strings) in the input shapes.

    Args:
      a_shape: Shape of the first input tensor.
      b_shape: Shape of the second input tensor.

    Returns:
      The output shape, which may contain integers or strings.
    """
    a_rank = len(a_shape)
    b_rank = len(b_shape)

    # Handle vector-vector case
    if a_rank == 1 and b_rank == 1:
        if (
            a_shape[0] == b_shape[0]
            or isinstance(a_shape[0], str)
            or isinstance(b_shape[0], str)
        ):
            return ()  # Scalar output
        else:
            raise ValueError("Incompatible shapes for matmul")

    # Normalize shapes to at least 2D
    a_shape_norm = a_shape if a_rank > 1 else (1,) + a_shape
    b_shape_norm = b_shape if b_rank > 1 else b_shape + (1,)

    a_rows = a_shape_norm[-2]
    a_cols = a_shape_norm[-1]
    b_rows = b_shape_norm[-2]
    b_cols = b_shape_norm[-1]

    # Check compatibility and handle dynamic dimensions
    if a_cols != b_rows and not (isinstance(a_cols, str) or isinstance(b_rows, str)):
        raise ValueError(f"Incompatible shapes for matmul: {a_shape} and {b_shape}")

    # Determine batch dimensions.
    batch_dims: List[Union[int, str]] = []
    max_rank = max(a_rank, b_rank)
    for i in range(max_rank - 2):
        a_idx = a_rank - 3 - i
        b_idx = b_rank - 3 - i
        a_dim = a_shape[a_idx] if a_idx >= 0 else 1
        b_dim = b_shape[b_idx] if b_idx >= 0 else 1

        if a_dim == 1 and isinstance(b_dim, int):
            batch_dims.insert(0, b_dim)
        elif isinstance(a_dim, int) and b_dim == 1:
            batch_dims.insert(0, a_dim)
        elif a_dim == b_dim:
            batch_dims.insert(0, a_dim)
        elif isinstance(a_dim, str):
            batch_dims.insert(0, a_dim)
        elif isinstance(b_dim, str):
            batch_dims.insert(0, b_dim)
        else:
            raise ValueError(
                f"Incompatible batch dimensions for matmul: {a_shape} and {b_shape}"
            )

    output_shape = tuple(batch_dims) + (a_rows, b_cols)

    # If original ranks were 1, adjust the output shape.
    if a_rank == 1:
        output_shape = output_shape[1:]
    if b_rank == 1:
        output_shape = output_shape[:-1]

    return output_shape


def _get_monkey_patch():
    def matmul(a, b):
        def matmul_abstract_eval(a, b):
            output_shape = _get_dynamic_output_shape(a.shape, b.shape)
            return core.ShapedArray(output_shape, a.dtype)

        jnp.matmul_p.multiple_results = False
        jnp.matmul_p.def_abstract_eval(matmul_abstract_eval)
        return jnp.matmul_p.bind(a, b)

    return matmul


@contextlib.contextmanager
def temporary_patch():
    original_fn = jnp.matmul
    jnp.matmul = _get_monkey_patch()
    try:
        yield
    finally:
        jnp.matmul = original_fn


def get_handler(s: "Jaxpr2OnnxConverter"):
    def handle_matmul(node_inputs, node_outputs, params):
        input_names = [s.get_name(var) for var in node_inputs]
        output_name = s.get_name(node_outputs[0])

        input_shapes = [inp.aval.shape for inp in node_inputs]
        output_shape = _get_dynamic_output_shape(input_shapes[0], input_shapes[1])

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
                "testcase": "matmul_2d",  # 2D matrix multiplication
                "callable": lambda a, b: jnp.matmul(a, b),
                "input_shapes": [(3, 4), (4, 5)],
            },
            {
                "testcase": "matmul_1d_2d",  # Vector-matrix multiplication
                "callable": lambda a, b: jnp.matmul(a, b),
                "input_shapes": [(4,), (4, 5)],
            },
            {
                "testcase": "matmul_2d_1d",  # Matrix-vector multiplication
                "callable": lambda a, b: jnp.matmul(a, b),
                "input_shapes": [(3, 4), (4,)],
            },
            {
                "testcase": "matmul_dynamic",  # Dynamic batch size
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
                "testcase": "matmul_1d",  # 1D vector x 1D vector
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
