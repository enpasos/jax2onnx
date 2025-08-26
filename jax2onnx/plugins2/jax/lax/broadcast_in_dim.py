# file: jax2onnx/plugins2/jax/lax/broadcast_in_dim.py

from typing import TYPE_CHECKING, Optional
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import onnx_ir as ir
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import IRBuildContext  # for hints

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
    context="primitives2.lax",
    component="broadcast_in_dim",
    testcases=[
        {
            "testcase": "broadcast_in_dim",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (3,), broadcast_dimensions=(0,)
            ),
            "input_shapes": [(3,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "broadcast_in_dim_2d_to_3d",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (2, 3, 4), broadcast_dimensions=(1, 2)
            ),
            "input_shapes": [(3, 4)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "broadcast_in_dim_scalar",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                x, (2, 3, 4), broadcast_dimensions=()
            ),
            "input_shapes": [()],
            # switch to value-based numeric testing
            "input_values": [0.5],
            "use_onnx_ir": True,
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
    Lower jax.lax.broadcast_in_dim(x, shape, broadcast_dimensions) to:
        (optional) Reshape(x, reshape_shape) -> Expand(…, target_shape)
    where reshape_shape inserts 1s in the non-mapped result axes.
    """

    def to_onnx(self, *_, **__):  # pragma: no cover
        # IR-only path in converter2.
        raise NotImplementedError

    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        shape = tuple(eqn.params["shape"])
        bdims = tuple(eqn.params["broadcast_dimensions"])

        # Target shape constant (INT64)
        try:
            tgt_np = np.asarray(shape, dtype=np.int64)
        except Exception as e:  # pragma: no cover
            raise NotImplementedError(
                "broadcast_in_dim with non-integer (symbolic) target dims is not supported yet on IR path."
            ) from e

        tgt_shape_val = ir.Value(
            name=ctx.fresh_name("bcast_target_shape"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((len(shape),)),
            const_value=ir.tensor(tgt_np),
        )
        ctx._initializers.append(tgt_shape_val)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("bcast_in"))

        # If operand is a scalar, we can skip the Reshape and go straight to Expand.
        op_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        need_reshape = len(op_shape) > 0 and len(shape) != len(op_shape)

        if need_reshape:
            # Build reshape_shape by placing operand dims into their mapped result axes, 1 elsewhere.
            rrank = len(shape)
            reshape_dims: list[int] = [1] * rrank
            for i, r_axis in enumerate(bdims):
                # guard if aval dims are unknown (shouldn't happen on tests)
                dim = int(op_shape[i]) if i < len(op_shape) and isinstance(op_shape[i], (int, np.integer)) else 1
                reshape_dims[r_axis] = dim

            rs_np = np.asarray(reshape_dims, dtype=np.int64)
            rs_val = ir.Value(
                name=ctx.fresh_name("bcast_reshape_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((rrank,)),
                const_value=ir.tensor(rs_np),
            )
            ctx._initializers.append(rs_val)

            reshaped_val = ir.Value(
                name=ctx.fresh_name("bcast_reshape_out"),
                type=x_val.type,                       # same dtype
                shape=ir.Shape(tuple(reshape_dims)),   # rank == result rank
            )
            reshape_node = ir.Node(
                op_type="Reshape",
                domain="",
                inputs=[x_val, rs_val],
                outputs=[reshaped_val],
                name=ctx.fresh_name("Reshape"),
            )
            ctx.add_node(reshape_node)
            expand_input = reshaped_val
        else:
            expand_input = x_val  # scalar or already aligned

        # Final expanded tensor should match the outvar's jax aval; let ctx create it.
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("bcast_out"))
        expand_node = ir.Node(
            op_type="Expand",
            domain="",
            inputs=[expand_input, tgt_shape_val],
            outputs=[out_val],
            name=ctx.fresh_name("Expand"),
        )
        ctx.add_node(expand_node)
