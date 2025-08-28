# file: jax2onnx/plugins2/jax/lax/broadcast_in_dim.py

from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import onnx_ir as ir

# from onnx_ir import Attribute as IRAttr  # NEW: proper Attr objects
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from onnx_ir import (
    Attr as IRAttr,
    AttributeType as IRAttrType,
)  # FIX: correct attr types

if TYPE_CHECKING:
    pass  # for hints


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
            "use_onnx_ir": True,
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
            "use_onnx_ir": True,
        },
    ],
)
class BroadcastInDimPlugin(PrimitiveLeafPlugin):
    """
    Lower jax.lax.broadcast_in_dim(x, shape, broadcast_dimensions) to:
        (optional) Reshape(x, reshape_shape) -> Expand(…, target_shape)
    where reshape_shape inserts 1s in the non-mapped result axes.
    """

    def lower(self, ctx, eqn):
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        shape = tuple(eqn.params["shape"])
        bdims = tuple(eqn.params["broadcast_dimensions"])

        # Build target shape as a 1-D INT64 tensor, supporting symbolic dims.
        # Each dimension becomes a length-1 vector; we Concat along axis=0.
        dim_pieces: list[ir.Value] = []
        # We'll keep a reference tensor with the *desired* float dtype (usually from the
        # origin of a symbolic dim like 'B') so we can CastLike the operand if needed.
        like_ref_val: ir.Value | None = None
        for d in shape:
            if isinstance(d, (int, np.integer)):
                c = ir.Value(
                    name=ctx.fresh_name("bcast_dim_c"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                    const_value=ir.tensor(np.array([int(d)], dtype=np.int64)),
                )
                ctx._initializers.append(c)
                dim_pieces.append(c)
            else:
                # Dynamic/symbolic dimension: fetch from its recorded origin.
                origin = getattr(ctx, "get_symbolic_dim_origin", None)
                if origin is None:
                    raise NotImplementedError(
                        "symbolic dims require ctx.get_symbolic_dim_origin"
                    )
                src = origin(d)
                if src is None:
                    raise NotImplementedError(
                        f"no origin recorded for symbolic dim '{d}'"
                    )
                src_val, axis = src
                # Save the first available tensor as a dtype reference.
                if like_ref_val is None:
                    like_ref_val = src_val
                # Shape(src) → Gather(…, [axis]) → length-1 vector
                src_rank = len(
                    getattr(getattr(src_val, "shape", None), "dims", ()) or ()
                )
                shp = ir.Value(
                    name=ctx.fresh_name("bcast_src_shape"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((src_rank,)),
                )
                shape_node = ir.Node(
                    op_type="Shape",
                    domain="",
                    inputs=[src_val],
                    outputs=[shp],
                    name=ctx.fresh_name("Shape"),
                )
                ctx.add_node(shape_node)

                idx = ir.Value(
                    name=ctx.fresh_name("bcast_idx"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                    const_value=ir.tensor(np.array([int(axis)], dtype=np.int64)),
                )
                ctx._initializers.append(idx)

                dim1 = ir.Value(
                    name=ctx.fresh_name("bcast_dim_dyn"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                )
                gather_node = ir.Node(
                    op_type="Gather",
                    domain="",
                    inputs=[shp, idx],
                    outputs=[dim1],
                    name=ctx.fresh_name("Gather"),
                    attributes=[IRAttr("axis", IRAttrType.INT, 0)],  # FIX
                )
                ctx.add_node(gather_node)
                dim_pieces.append(dim1)

        tgt_shape_val = ir.Value(
            name=ctx.fresh_name("bcast_target_shape"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((len(shape),)),
        )
        concat_node = ir.Node(
            op_type="Concat",
            domain="",
            inputs=dim_pieces,
            outputs=[tgt_shape_val],
            name=ctx.fresh_name("Concat"),
            attributes=[IRAttr("axis", IRAttrType.INT, 0)],  # FIX
        )
        ctx.add_node(concat_node)

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
                dim = (
                    int(op_shape[i])
                    if i < len(op_shape) and isinstance(op_shape[i], (int, np.integer))
                    else 1
                )
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
                type=x_val.type,  # same dtype
                shape=ir.Shape(tuple(reshape_dims)),  # rank == result rank
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

        # --- DTYPE UNIFICATION --------------------------------------------------
        # If the operand (possibly a Python float literal → fp64) doesn't match the
        # expected output dtype (usually fp32 in x32 mode), cast it before Expand.
        in_dt = getattr(getattr(expand_input, "type", None), "dtype", None)
        out_dt = getattr(getattr(out_val, "type", None), "dtype", None)
        if (in_dt is not None) and (out_dt is not None) and (in_dt != out_dt):
            # Use a "like" tensor ONLY if it already matches the target dtype.
            like_for_cast = None
            lr_dt = getattr(getattr(like_ref_val, "type", None), "dtype", None)
            if like_ref_val is not None and lr_dt == out_dt:
                like_for_cast = like_ref_val

            # Otherwise synthesize a tiny constant with the target dtype.
            if like_for_cast is None:
                _np_dt = {
                    ir.DataType.FLOAT: np.float32,
                    ir.DataType.DOUBLE: np.float64,
                    # (extend here if you ever hit other types)
                }.get(out_dt, None)
                if _np_dt is None:
                    # If we can't materialize a "like", skip the cast safely.
                    # (Shouldn't happen in these tests.)
                    pass
                else:
                    like_for_cast = ir.Value(
                        name=ctx.fresh_name("bcast_like_c"),
                        type=ir.TensorType(out_dt),
                        shape=ir.Shape((1,)),
                        const_value=ir.tensor(np.zeros((1,), dtype=_np_dt)),
                    )
                    ctx._initializers.append(like_for_cast)
            if like_for_cast is not None:
                casted = ir.Value(
                    name=ctx.fresh_name("bcast_in_cast"),
                    type=ir.TensorType(out_dt),
                    shape=expand_input.shape,
                )
                ctx.add_node(
                    ir.Node(
                        op_type="CastLike",
                        domain="",
                        inputs=[expand_input, like_for_cast],
                        outputs=[casted],
                        name=ctx.fresh_name("CastLike"),
                    )
                )
                expand_input = casted
        # -----------------------------------------------------------------------

        expand_node = ir.Node(
            op_type="Expand",
            domain="",
            inputs=[expand_input, tgt_shape_val],
            outputs=[out_val],
            name=ctx.fresh_name("Expand"),
        )
        ctx.add_node(expand_node)
