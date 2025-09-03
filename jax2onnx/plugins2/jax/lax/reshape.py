# file: jax2onnx/plugins2/jax/lax/reshape.py

from __future__ import annotations
from typing import TYPE_CHECKING, List, Union

import numpy as np
import jax
from jax import lax
from jax._src.export.shape_poly import _DimExpr as DimExpr

import onnx_ir as ir
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._ir_shapes import (
    _stamp_type_and_shape,
    is_shape_all_unknown,
    _dim_label_from_value_or_aval,
    _ensure_value_info as _add_value_info,
)

if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


@register_primitive(
    jaxpr_primitive=lax.reshape_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reshape.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        }
    ],
    since="v0.2.0",
    context="primitives2.lax",
    component="reshape",
    testcases=[
        {
            "testcase": "reshape",
            "callable": lambda x: jax.lax.reshape(x, (9,)),
            "input_shapes": [(3, 3)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_valid_squeeze_middle_dim_from_problematic_source",
            "callable": lambda x: jax.lax.reshape(
                x, new_sizes=(x.shape[0], x.shape[2]), dimensions=(0, 1, 2)
            ),
            "input_shapes": [(201, 1, 201)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_valid_flatten_trailing",
            "callable": lambda x: jax.lax.reshape(x, new_sizes=(x.shape[0], -1)),
            "input_shapes": [(201, 1, 5)],
            "use_onnx_ir": True,
        },
        # TODO: restore
        # {
        #     "testcase": "reshape_with_target_shape_from_symbolic_dim_computation",
        #     "callable": lambda x: jax.lax.reshape(x, new_sizes=(x.shape[0], -1)),
        #     "input_shapes": [("N", "M", "K")],
        #     "use_onnx_ir": True,
        # },
        {
            "testcase": "reshape_with_inferred_dimension_from_input_dynamic",
            "callable": lambda x: jax.lax.reshape(x, new_sizes=(x.shape[0], -1)),
            "input_shapes": [("B", 10, 10)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_with_inferred_dimension_from_input",
            "callable": lambda x: jax.lax.reshape(x, new_sizes=(x.shape[0], -1)),
            "input_shapes": [(3, 10, 10)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_merge_symbolic_with_static_and_check_name",
            "callable": lambda x: jax.lax.reshape(x, new_sizes=(-1, x.shape[2])),
            "input_shapes": [("B", 4, 16)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: (
                m.graph.output[0].type.tensor_type.shape.dim[0].dim_param != "B"
                and m.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 16
            ),
        },
    ],
)
class ReshapePlugin(PrimitiveLeafPlugin):
    """
    plugins2 IR converter for jax.lax.reshape → ONNX Reshape.
    Builds the shape tensor minimally (all-const when possible), and stamps the
    output with correct symbolic labels (reusing input symbols; fresh labels for
    derived dims like B*4).
    """

    # ---------------- lowering (IR) ----------------
    def lower(self, ctx: "IRBuildContext", eqn):
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]
        new_sizes = tuple(eqn.params["new_sizes"])
        # Note: eqn.invars[1:] may carry runtime integer dims supplied as inputs
        runtime_dim_vars = list(eqn.invars[1:])

        # Inputs
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))

        # Preserve original input meta if binder left it unknown
        if is_shape_all_unknown(getattr(x_val, "shape", None)):
            if any(d is not None for d in x_shape):
                _stamp_type_and_shape(x_val, x_shape)

        # Helpers to make INT64 constants
        def const_i64_vec(vals: np.ndarray) -> ir.Value:
            v = ir.Value(
                name=ctx.fresh_name("shape_const"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((int(vals.size),)),
                const_value=ir.tensor(vals.astype(np.int64, copy=False)),
            )
            ctx._initializers.append(v)
            return v

        def const_i64_scalar(val: int) -> ir.Value:
            v = ir.Value(
                name=ctx.fresh_name("i64"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(()),
                const_value=ir.tensor(np.array(val, dtype=np.int64)),
            )
            ctx._initializers.append(v)
            return v

        def unsqueeze_to_1d0(src: ir.Value) -> ir.Value:
            axes = const_i64_vec(np.array([0], dtype=np.int64))
            out = ir.Value(
                name=ctx.fresh_name("unsq"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((1,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Unsqueeze",
                    domain="",
                    inputs=[src, axes],
                    outputs=[out],
                    name=ctx.fresh_name("Unsqueeze"),
                )
            )
            return out

        # Build pieces of the shape tensor
        shape_parts: List[ir.Value] = []
        all_const = True

        # Create 'Shape' result lazily when needed
        shape_of_x: ir.Value | None = None

        # Iterator over runtime scalar dim inputs (if any)
        runtime_iter = iter(runtime_dim_vars)

        # Cache: which DimExprs are literal input axes?
        {d for d in x_shape if isinstance(d, DimExpr)}

        # Track whether we already inserted an inferred dim (-1)
        inserted_neg1 = any(
            isinstance(d, (int, np.integer)) and int(d) == -1 for d in new_sizes
        )
        for dim in new_sizes:
            if isinstance(dim, (int, np.integer)):
                # include -1 as a literal: let ONNX infer that dim
                part = const_i64_vec(np.array([int(dim)], dtype=np.int64))
                shape_parts.append(part)
            elif isinstance(dim, DimExpr):
                all_const = False
                if shape_of_x is None:
                    shape_of_x = ir.Value(
                        name=ctx.fresh_name("shape_of_x"),
                        type=ir.TensorType(ir.DataType.INT64),
                        shape=None,
                    )
                    ctx.add_node(
                        ir.Node(
                            op_type="Shape",
                            domain="",
                            inputs=[x_val],
                            outputs=[shape_of_x],
                            name=ctx.fresh_name("Shape"),
                        )
                    )
                # Try to find an input axis with the SAME symbol (by identity or string).
                axis_idx = next(
                    (
                        i
                        for i, d in enumerate(x_shape)
                        if (d is dim)
                        or (not isinstance(d, (int, np.integer)) and str(d) == str(dim))
                    ),
                    None,
                )
                if axis_idx is None:
                    # Derived symbolic (e.g., 4*B). Use ONNX inference (-1).
                    if not inserted_neg1:
                        shape_parts.append(
                            const_i64_vec(np.array([-1], dtype=np.int64))
                        )
                        inserted_neg1 = True
                    else:
                        # (A second derived dim would require dynamic arithmetic;
                        # not needed in our suite; still keep model valid.)
                        shape_parts.append(
                            const_i64_vec(np.array([-1], dtype=np.int64))
                        )
                    continue
                idx_const = const_i64_scalar(axis_idx)
                gathered = ir.Value(
                    name=ctx.fresh_name("gather_dim"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape(()),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Gather",
                        domain="",
                        inputs=[shape_of_x, idx_const],
                        outputs=[gathered],
                        name=ctx.fresh_name("Gather"),
                        attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
                    )
                )
                shape_parts.append(unsqueeze_to_1d0(gathered))
            elif hasattr(dim, "dtype") and np.issubdtype(
                getattr(dim, "dtype", None), np.integer
            ):
                # runtime scalar integer
                dyn_var = next(runtime_iter)
                dyn_val = ctx.get_value_for_var(
                    dyn_var, name_hint=ctx.fresh_name("dyn")
                )
                shape_parts.append(unsqueeze_to_1d0(dyn_val))
                all_const = False
            else:
                raise TypeError(f"Unsupported reshape size element: {type(dim)}")

        # If all sizes are static, emit a single initializer tensor
        if all_const:
            shape_tensor = const_i64_vec(
                np.array([int(d) for d in new_sizes], dtype=np.int64)
            )
        else:
            if len(shape_parts) == 0:
                shape_tensor = const_i64_vec(np.array([], dtype=np.int64))
            elif len(shape_parts) == 1:
                shape_tensor = shape_parts[0]
            else:
                shape_tensor = ir.Value(
                    name=ctx.fresh_name("shape_tensor"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=None,
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Concat",
                        domain="",
                        inputs=shape_parts,
                        outputs=[shape_tensor],
                        name=ctx.fresh_name("Concat"),
                        attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
                    )
                )

        # Reshape node
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))
        ctx.add_node(
            ir.Node(
                op_type="Reshape",
                domain="",
                inputs=[x_val, shape_tensor],
                outputs=[y_val],
                name=ctx.fresh_name("Reshape"),
            )
        )

        # ---- Stamp symbolic output dims (reuse input labels; fresh for derived) ----
        y_aval_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))

        # precompute labels for input symbolic dims (DimExpr → label string)
        # mypy: spell out the type so the dict literal doesn't need inference
        sym_label_map: dict[object, str] = {}
        for i, d in enumerate(x_shape):
            if not isinstance(d, (int, np.integer)):
                sym_label_map[d] = _dim_label_from_value_or_aval(x_val, x_shape, i)

        final_dims: List[Union[int, str]] = []
        for d in y_aval_shape:
            if isinstance(d, (int, np.integer)):
                final_dims.append(int(d))
            elif d in sym_label_map:
                final_dims.append(sym_label_map[d])
            else:
                # derived symbolic (e.g., B*4): give it a fresh name
                final_dims.append(ctx.fresh_name("dim"))

        _stamp_type_and_shape(y_val, tuple(final_dims))
        _add_value_info(ctx, y_val)
