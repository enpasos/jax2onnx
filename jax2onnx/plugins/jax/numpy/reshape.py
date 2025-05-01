from collections.abc import Sequence
from typing import TYPE_CHECKING

from jax import core
from jax import numpy as jnp
from jax.extend.core import Primitive
from onnx import helper, TensorProto  # <-- Add import

from jax2onnx.converter.dynamic_utils import encode_dims
from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

import numpy as np

# Define the reshape primitive
jnp.reshape_p = Primitive("jnp.reshape")
jnp.reshape_p.multiple_results = False  # Correct initialization


@register_primitive(
    jaxpr_primitive=jnp.reshape_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.reshape.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        }
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="reshape",
    testcases=[
        {
            "testcase": "reshape_1",
            "callable": lambda a: jnp.reshape(a, (2, 6)),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "reshape_2",
            "callable": lambda a: jnp.reshape(a, (-1, 2)),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "reshape_3",
            "callable": lambda a: jnp.reshape(a, (2, -1)),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "reshape_4",
            "callable": lambda a: jnp.reshape(a, (-1, 4)),
            "input_shapes": [("B", 3, 4)],
        },
        {
            "testcase": "reshape_to_scalar",
            "callable": lambda a: jnp.reshape(a, ()),
            "input_shapes": [(1,)],
        },
        {
            "testcase": "reshape_from_scalar",
            "callable": lambda a: jnp.reshape(a, (1,)),
            "input_shapes": [()],
        },
        {
            "testcase": "reshape_cnn",
            "callable": lambda x: jnp.reshape(x.shape[0], -1),
            "input_shapes": [("B", 64, 14, 14)],
        },
    ],
)
class ReshapePlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.numpy.reshape to ONNX.
    """

    @staticmethod
    def _process_newshape(newshape: Sequence[int | str]) -> list[int | str]:
        """Validates and processes the newshape argument for reshape."""
        if isinstance(newshape, (int, str)):
            newshape = [newshape]
        else:
            newshape = list(newshape)

        neg_one_count = sum(1 for dim in newshape if dim == -1)
        if neg_one_count > 1:
            raise ValueError("Only one dimension can be -1 (inferred).")

        return newshape

    @staticmethod
    def _get_dynamic_output_shape(
        input_shape: tuple[int | str, ...], newshape: Sequence[int | str]
    ) -> tuple[int | str, ...]:
        """Computes the output shape for jnp.reshape while handling dynamic dimensions and tracers."""
        newshape = ReshapePlugin._process_newshape(newshape)
        input_shape_list = list(input_shape)

        def safe_int(val):
            try:
                return int(val)
            except Exception:
                return 1  # Use 1 for symbolic/tracer dims in dummy shape

        dummy_input_shape = [safe_int(s) for s in input_shape_list]
        dummy_newshape = [safe_int(s) for s in newshape]

        if -1 in dummy_newshape:
            neg_one_index = dummy_newshape.index(-1)
            known_dims_product = np.prod([dim for dim in dummy_newshape if dim != -1])
            # Avoid ZeroDivisionError
            if known_dims_product == 0 and np.prod(dummy_input_shape) != 0:
                raise ValueError(
                    f"Cannot reshape array of shape {input_shape} into shape {newshape}"
                )
            try:
                inferred_dim = (
                    int(np.prod(dummy_input_shape) / known_dims_product)
                    if known_dims_product != 0
                    else 0
                )
            except Exception:
                inferred_dim = -1  # Use -1 if symbolic/tracer dims prevent computation
            dummy_newshape[neg_one_index] = inferred_dim

        try:
            if np.prod(dummy_input_shape) != np.prod(dummy_newshape):
                raise ValueError(
                    f"Cannot reshape array of shape {input_shape} into shape {newshape}"
                )
        except Exception:
            # If symbolic/tracer dims, skip this check
            pass

        output_shape = [
            orig if isinstance(orig, str) else dummy
            for orig, dummy in zip(newshape, dummy_newshape, strict=False)
        ]
        return tuple(output_shape)

    @staticmethod
    def abstract_eval(a, newshape):
        """Abstract evaluation function for Reshape."""
        newshape_processed = ReshapePlugin._process_newshape(newshape)
        output_shape = ReshapePlugin._get_dynamic_output_shape(
            a.shape, newshape_processed
        )
        return core.ShapedArray(tuple(output_shape), a.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of Reshape to ONNX format."""
        input_var = node_inputs[0]
        output_var = node_outputs[0]
        newshape = params["newshape"]

        input_name = s.get_name(input_var)
        # --- allocate an output name without registering wrong meta ---
        output_name = s.get_unique_name("reshape_out")
        s.var_to_name[output_var] = output_name

        input_shape = input_var.aval.shape
        # Note: output_shape calculation might need refinement for symbolic dims later
        output_shape = ReshapePlugin._get_dynamic_output_shape(input_shape, newshape)
        processed_newshape = ReshapePlugin._process_newshape(newshape)

        # --- Determine ONNX Input Type (Enum) ---
        # Use the JAX aval dtype – that is always correct and avoids the
        # int32 / int64 mix-ups we have seen with scalar constants
        input_dtype_enum = s._ensure_onnx_dtype(input_var.aval.dtype)

        # --- Determine Expected ONNX Output Type ---
        # ONNX Reshape output type matches the input data type
        onnx_output_dtype_enum = input_dtype_enum

        # --- Create Shape Tensor ---
        # Use encode_dims to handle potential symbolic dimensions ('B') in the shape
        shape_tensor_vals = encode_dims(processed_newshape)
        shape_tensor_name = s.get_constant_name(
            np.array(shape_tensor_vals, dtype=np.int64)
        )

        # --- Create Reshape Node ---
        reshape_node = helper.make_node(
            "Reshape",
            inputs=[input_name, shape_tensor_name],
            outputs=[output_name],
            name=s.get_unique_name("reshape"),
            allowzero=0,  # Set allowzero=0 based on NumPy/JAX behavior
        )
        s.add_node(reshape_node)

        # --- register *once* with the final, correct dtype ---
        s.add_shape_info(output_name, output_shape, onnx_output_dtype_enum)

    @staticmethod
    def _reshape(a, newshape, order="C"):
        """Defines the primitive binding for Reshape."""
        if order != "C":
            raise NotImplementedError("Only C-style reshape is supported.")
        return jnp.reshape_p.bind(a, newshape=newshape)

    @staticmethod
    def get_monkey_patch():
        """Provides patching information for Reshape."""

        def patched_reshape(a, newshape, order="C"):
            return ReshapePlugin._reshape(a, newshape, order)

        return patched_reshape

    @staticmethod
    def patch_info():
        """Provides patching information for Reshape."""
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: ReshapePlugin.get_monkey_patch(),
            "target_attribute": "reshape",
        }


# Register abstract evaluation function
jnp.reshape_p.def_abstract_eval(ReshapePlugin.abstract_eval)
