# file: jax2onnx/plugins/jax/lax/broadcast_in_dim.py


from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from onnx import helper, TensorProto

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # only for static type checkers
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# ---------------------------------------------------------------------
# 1. primitive alias
# ---------------------------------------------------------------------
broadcast_in_dim_p = lax.broadcast_in_dim_p
# ---------------------------------------------------------------------


@register_primitive(
    jaxpr_primitive=jax.lax.broadcast_in_dim_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/jax-primitives.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Expand",
            "doc": "https://onnx.ai/onnx/operators/onnx__Expand.html",
        },
        {  # Added Identity for completeness
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        },
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="broadcast_in_dim",
    testcases=[
        {
            "testcase": "broadcast_in_dim",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (3,), broadcast_dimensions=(0,)
            ),
            "input_shapes": [(3,)],
        },
        {
            "testcase": "broadcast_in_dim_2d_to_3d",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (2, 3, 4), broadcast_dimensions=(1, 2)
            ),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "broadcast_in_dim_scalar",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (2, 3, 4), broadcast_dimensions=()
            ),
            "input_shapes": [()],
            # switch to value-based numeric testing
            "input_values": [0.5],
        },
        {
            # ------------------------------------------------------------------
            # Re‑creates the "broadcast (1,1,D) → (B,1,D)" pattern that broke
            # when `shape` contained the symbolic batch dimension  B.
            # ------------------------------------------------------------------
            "testcase": "broadcast_in_dim_batch",
            "callable": lambda x: jnp.broadcast_to(  # ⤵ uses lax.broadcast_in_dim
                jnp.zeros((1, 1, x.shape[-1]), dtype=x.dtype),  #   token (1,1,D)
                (x.shape[0], 1, x.shape[-1]),  # → (B,1,D)
            ),
            "input_shapes": [
                ("B", 49, 256)
            ],  # Use a concrete batch for non-dynamic test
            "expected_output_shapes": [("B", 1, 256)],
        },
        # ------------------------------------------------------------------
        # dynamic-batch test: symbolic B
        {
            "testcase": "broadcast_in_dim_dynamic_B",
            "callable": lambda x: lax.broadcast_in_dim(
                0.5, shape=(x.shape[0], 3, 4), broadcast_dimensions=()
            ),
            "input_shapes": [("B",)],  # symbolic batch dim
            "post_check_onnx_graph": lambda m: (
                __import__("onnx").checker.check_model(m) or True
            ),
        },
    ],
)
class BroadcastInDimPlugin(PrimitiveLeafPlugin):
    """
    Plugin for converting jax.lax.broadcast_in_dim to ONNX.
    Handles static and dynamic shapes, and identity cases.
    """

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX broadcast_in_dim primitive."""
        input_var = node_inputs[0]
        input_name = s.get_name(input_var)
        input_aval = input_var.aval
        input_dtype = input_aval.dtype
        input_shape = input_aval.shape

        output_var = node_outputs[0]
        # *** Get a potentially NEW name for the output ***
        # Use get_var_name which handles potential renaming by the converter
        output_name_intended = s.get_var_name(output_var)

        output_aval = output_var.aval
        output_dtype = output_aval.dtype
        # Target shape from the JAX equation's output aval
        output_target_shape = output_aval.shape

        broadcast_dimensions = params["broadcast_dimensions"]
        # Target shape explicitly provided in params
        raw_target_shape_param = params["shape"]

        # --- Check for Identity Case ---
        is_identity = False
        if input_shape == output_target_shape and len(broadcast_dimensions) == len(
            input_shape
        ):
            # Check if broadcast_dimensions perfectly map input dims 0..N-1
            if tuple(sorted(broadcast_dimensions)) == tuple(range(len(input_shape))):
                is_identity = True

        if is_identity:
            # Ensure output name is different from input name for ONNX graph validity
            # If the converter assigned the same name, create a new one.
            if input_name == output_name_intended:
                output_name = s.get_unique_name(f"{input_name}_identity_out")
            else:
                output_name = output_name_intended  # Use converter's assigned name

            identity_node = helper.make_node(
                "Identity",
                inputs=[input_name],
                outputs=[output_name],
                name=s.get_unique_name(f"identity_{input_name}"),
            )
            s.add_node(identity_node)
            s.add_shape_info(output_name, output_target_shape, output_dtype)
            # No further nodes needed for identity case
            return
        else:
            # Not identity case, use the intended output name
            output_name = output_name_intended

        # --- Proceed with Reshape + Expand for non-identity cases ---

        # Calculate the intermediate shape for Reshape
        reshape_target_shape = []
        in_idx = 0
        for i in range(len(output_target_shape)):
            if i in broadcast_dimensions:
                if in_idx < len(input_shape):
                    dim = input_shape[in_idx]
                    reshape_target_shape.append(
                        dim if not isinstance(dim, int) else int(dim)
                    )
                    in_idx += 1
                else:
                    reshape_target_shape.append(1)  # Scalar input case
            else:
                reshape_target_shape.append(1)

        if not input_shape and len(reshape_target_shape) < len(output_target_shape):
            reshape_target_shape = [1] * len(output_target_shape)

        # Reshape needs a static shape input
        concrete_reshape_shape = []
        for dim in reshape_target_shape:
            # Use 1 if symbolic, actual value if int
            concrete_reshape_shape.append(1 if not isinstance(dim, int) else dim)

        reshape_shape_const_name = s.get_constant_name(
            np.array(concrete_reshape_shape, dtype=np.int64)
        )
        reshape_output_name = s.get_unique_name("reshape_output")

        node_reshape = helper.make_node(
            "Reshape",
            inputs=[input_name, reshape_shape_const_name],
            outputs=[reshape_output_name],
            name=s.get_unique_name("reshape_for_broadcast"),
        )
        s.add_node(node_reshape)
        # Use the potentially symbolic reshape_target_shape for metadata
        s.add_shape_info(reshape_output_name, tuple(reshape_target_shape), input_dtype)

        # --- Create Expand node ---
        # Use the shape from params['shape'] as it correctly reflects the target
        if all(isinstance(d, int) for d in raw_target_shape_param):
            target_shape_const_name = s.get_constant_name(
                np.array(raw_target_shape_param, dtype=np.int64)
            )
        else:
            # Dynamic target shape construction (using _runtime_dim helper)
            def _runtime_dim(sym_dim):
                origin_info = s.symbolic_dim_to_origin.get(sym_dim)
                if origin_info is None:
                    origin_info = s.symbolic_dim_to_origin.get(str(sym_dim))

                if origin_info is None:
                    raise RuntimeError(
                        f"Could not locate origin for symbolic dimension {sym_dim!r}."
                    )

                t_name, axis = origin_info
                t_shape = None

                input_vi = next(
                    (vi for vi in s.builder.inputs if vi.name == t_name), None
                )
                if input_vi:
                    shape_proto = input_vi.type.tensor_type.shape
                    t_shape = []
                    for d in shape_proto.dim:
                        t_shape.append(
                            d.dim_value if d.HasField("dim_value") else d.dim_param
                        )
                    t_shape = tuple(t_shape)
                elif t_name in s.builder.value_info_metadata:
                    meta_shape, _ = s.builder.value_info_metadata[t_name]
                    t_shape = tuple(meta_shape)  # Ensure tuple
                else:
                    raise RuntimeError(
                        f"Could not find shape info for tensor '{t_name}'."
                    )

                if not (0 <= axis < len(t_shape)):
                    raise ValueError(
                        f"Axis {axis} out of bounds for tensor '{t_name}' shape {t_shape}"
                    )

                shape_node_out = s.get_unique_name(f"shape_of_{t_name}")
                s.add_node(helper.make_node("Shape", [t_name], [shape_node_out]))
                s.add_shape_info(shape_node_out, (len(t_shape),), TensorProto.INT64)

                gather_idx_const = s.get_constant_name(np.array(axis, dtype=np.int64))
                gather_node_out = s.get_unique_name(f"dim_{t_name}_{axis}")
                s.add_node(
                    helper.make_node(
                        "Gather",
                        [shape_node_out, gather_idx_const],
                        [gather_node_out],
                        axis=0,
                    )
                )
                s.add_shape_info(gather_node_out, (), TensorProto.INT64)

                unsqueeze_axes_const = s.get_constant_name(
                    np.array([0], dtype=np.int64)
                )
                unsqueeze_node_out = s.get_unique_name(f"{gather_node_out}_1d")
                s.add_node(
                    helper.make_node(
                        "Unsqueeze",
                        [gather_node_out, unsqueeze_axes_const],
                        [unsqueeze_node_out],
                    )
                )
                s.add_shape_info(unsqueeze_node_out, (1,), TensorProto.INT64)
                return unsqueeze_node_out

            pieces: list[str] = []
            for dim in raw_target_shape_param:
                if isinstance(dim, int):
                    pieces.append(s.get_constant_name(np.array([dim], dtype=np.int64)))
                else:
                    pieces.append(_runtime_dim(dim))

            target_shape_tensor_name = s.get_unique_name("target_expand_shape")
            s.add_node(
                helper.make_node("Concat", pieces, [target_shape_tensor_name], axis=0)
            )
            s.add_shape_info(
                target_shape_tensor_name,
                (len(raw_target_shape_param),),
                TensorProto.INT64,
            )
            target_shape_const_name = target_shape_tensor_name  # Use dynamic tensor

        # Final Expand node
        node_expand = helper.make_node(
            "Expand",
            # Input is the output of Reshape, shape is static const or dynamic tensor
            inputs=[reshape_output_name, target_shape_const_name],
            outputs=[output_name],  # Final graph output name
            name=s.get_unique_name("expand"),
        )
        s.add_node(node_expand)
        # Add metadata for the final output, using the JAX aval's shape
        s.add_shape_info(output_name, output_target_shape, output_dtype)
