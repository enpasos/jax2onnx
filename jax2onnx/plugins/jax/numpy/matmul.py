from typing import TYPE_CHECKING

from jax import core
from jax import numpy as jnp
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# Define the MatMul primitive
jnp.matmul_p = Primitive("jnp.matmul")
jnp.matmul_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=jnp.matmul_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.matmul.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="matmul",
    testcases=[
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
)
class MatMulPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.matmul to ONNX.
    """

    @staticmethod
    def _get_dynamic_output_shape(
        a_shape: tuple[int | str, ...], b_shape: tuple[int | str, ...]
    ) -> tuple[int | str, ...]:
        """Calculates the output shape of jnp.matmul while handling dynamic dimensions and tracers."""

        def safe_eq(x, y):
            try:
                return x == y
            except Exception:
                return False

        def safe_is_one(x):
            try:
                return x == 1
            except Exception:
                return False

        a_rank, b_rank = len(a_shape), len(b_shape)

        if a_rank == 1 and b_rank == 1:
            if (
                safe_eq(a_shape[0], b_shape[0])
                or isinstance(a_shape[0], str)
                or isinstance(b_shape[0], str)
            ):
                return ()  # Scalar output
            raise ValueError("Incompatible shapes for matmul")

        a_shape_norm = a_shape if a_rank > 1 else (1,) + a_shape
        b_shape_norm = b_shape if b_rank > 1 else b_shape + (1,)

        a_rows, a_cols = a_shape_norm[-2], a_shape_norm[-1]
        b_rows, b_cols = b_shape_norm[-2], b_shape_norm[-1]

        if not (isinstance(a_cols, str) or isinstance(b_rows, str)):
            if not safe_eq(a_cols, b_rows):
                raise ValueError(
                    f"Incompatible shapes for matmul: {a_shape} and {b_shape}"
                )

        batch_dims: list[int | str] = []
        max_rank = max(a_rank, b_rank)
        for i in range(max_rank - 2):
            a_dim = a_shape[i] if i < a_rank - 2 else 1
            b_dim = b_shape[i] if i < b_rank - 2 else 1
            if isinstance(a_dim, str) or isinstance(b_dim, str):
                batch_dims.append(
                    a_dim if isinstance(b_dim, int) and safe_is_one(b_dim) else b_dim
                )
            else:
                batch_dims.append(a_dim if not safe_is_one(a_dim) else b_dim)

        output_shape = tuple(batch_dims) + (a_rows, b_cols)

        if a_rank == 1:
            output_shape = output_shape[1:]
        if b_rank == 1:
            output_shape = output_shape[:-1]

        return output_shape

    @staticmethod
    def abstract_eval(a, b):
        """Abstract evaluation function for MatMul, robust to tracers."""
        output_shape = MatMulPlugin._get_dynamic_output_shape(a.shape, b.shape)

        def safe_dim(dim):
            try:
                hash(dim)
                return dim
            except Exception:
                return -1

        output_shape_safe = tuple(safe_dim(d) for d in output_shape)
        return core.ShapedArray(output_shape_safe, a.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of MatMul to ONNX format."""
        input_names = [s.get_name(var) for var in node_inputs]
        output_name = s.get_name(node_outputs[0])

        input_shapes = [inp.aval.shape for inp in node_inputs]
        output_shape = MatMulPlugin._get_dynamic_output_shape(
            input_shapes[0], input_shapes[1]
        )

        matmul_node = helper.make_node(
            "MatMul",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("matmul"),
        )
        s.add_node(matmul_node)
        s.add_shape_info(
            output_name,
            tuple(int(dim) for dim in output_shape if isinstance(dim, (int, str))),
        )

    @staticmethod
    def _matmul(a, b):
        """Defines the primitive binding for MatMul."""
        return jnp.matmul_p.bind(a, b)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for MatMul."""

        def patched_matmul(a, b):
            return MatMulPlugin._matmul(a, b)

        return patched_matmul

    @staticmethod
    def patch_info():
        """Provides patching information for MatMul."""
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: MatMulPlugin.get_monkey_patch(),
            "target_attribute": "matmul",
        }


# Register abstract evaluation function
jnp.matmul_p.def_abstract_eval(MatMulPlugin.abstract_eval)
