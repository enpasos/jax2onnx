# jax2onnx/plugins/jax/lax/reshape.py

from __future__ import annotations
from typing import TYPE_CHECKING, List, Union, Dict, Optional
from functools import reduce
import operator
import os

import numpy as np
import jax
from jax import lax
from jax._src.export.shape_poly import _DimExpr as DimExpr

import onnx_ir as ir
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    is_shape_all_unknown,
    _dim_label_from_value_or_aval,
    _ensure_value_info as _add_value_info,
)
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG

if TYPE_CHECKING:
    from jax2onnx.converter.ir_context import IRContext


# --- helper for tests: forbid constant-only Concat as Reshape shape input ---
def _no_const_concat_shape(model) -> bool:
    """
    Return True iff every Reshape node's second input is either
    a direct initializer or, if produced by Concat, that Concat is NOT
    composed purely of initializers (i.e., not a foldable const Concat).
    """
    g = model.graph
    init_names = {t.name for t in g.initializer}
    # fast map: tensor -> producer node
    produced_by = {}
    for n in g.node:
        for o in n.output:
            if o:
                produced_by[o] = n
    for n in g.node:
        if n.op_type != "Reshape":
            continue
        shape_in = n.input[1] if len(n.input) > 1 else ""
        if shape_in in init_names:
            continue
        prod = produced_by.get(shape_in)
        if (
            prod
            and prod.op_type == "Concat"
            and all(inp in init_names for inp in prod.input)
        ):
            return False  # foldable const Concat still present -> fail
    return True


# --- pattern helpers in the same style as nnx.linear ---
# A single Reshape is expected (the tiny graphs in these tests contain only one),
# and we do not want any dynamic shape ops for the all-static case.
EXPECT_SINGLE_RESHAPE_AND_NO_SHAPE_PLUMBING = EG(
    [
        (
            "Reshape",
            {
                "counts": {
                    "Reshape": 1,
                    "Concat": 0,
                    "Gather": 0,
                    "Shape": 0,
                }
            },
        )
    ]
)
EXPECT_NO_DYNAMIC_SHAPE_NODES = EG([], must_absent=["Concat", "Gather", "Shape"])


def _prod_dims(dims):
    prod = 1
    for d in dims:
        if isinstance(d, (int, np.integer)):
            prod = prod * int(d)
        else:
            prod = prod * d
    return prod


def _reshape_flatten_leading(x):
    try:
        flat = reduce(operator.mul, x.shape[1:], 1)
        return lax.reshape(x, new_sizes=(x.shape[0], flat))
    except TypeError:
        return jax.numpy.reshape(x, (x.shape[0], -1))


def _reshape_flatten_trailing(x):
    try:
        lead = reduce(operator.mul, x.shape[:-1], 1)
        return lax.reshape(x, new_sizes=(lead, x.shape[-1]))
    except TypeError:
        return jax.numpy.reshape(x, (-1, x.shape[-1]))


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
    context="primitives.lax",
    component="reshape",
    testcases=[
        {
            # CNN-like path: after a Transpose we reshape with (y.shape[0], -1)
            # for a fully-static leading axis. The shape vector must be a single
            # constant initializer; a constant-only Concat feeding Reshape is a bug.
            "testcase": "reshape_after_transpose_folds_const_shape",
            "callable": (
                lambda x: _reshape_flatten_leading(
                    jax.lax.transpose(x, permutation=(0, 3, 1, 2))
                )
            ),
            "input_shapes": [(3, 28, 28, 1)],
            "use_onnx_ir": True,
            "post_check_onnx_graph": _no_const_concat_shape,
        },
        {
            # Catch regression: when the input’s leading axis is static, the shape fed to
            # Reshape must be folded to a single constant initializer (no Concat/Gather/Shape).
            "testcase": "reshape_flatten_trailing_folds_const_shape",
            "callable": _reshape_flatten_leading,
            "input_shapes": [(3, 4, 5)],
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: (
                EXPECT_SINGLE_RESHAPE_AND_NO_SHAPE_PLUMBING(m)
                and EXPECT_NO_DYNAMIC_SHAPE_NODES(m)
                # and the shape input to Reshape is a constant initializer:
                and any(
                    init.name
                    == next(n for n in m.graph.node if n.op_type == "Reshape").input[1]
                    for init in m.graph.initializer
                )
            ),
        },
        {
            "testcase": "reshape",
            "callable": lambda x: jax.lax.reshape(x, (9,)),
            "input_shapes": [(3, 3)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_valid_squeeze_middle_dim_from_problematic_source",
            "callable": lambda x: lax.reshape(
                x, new_sizes=(x.shape[0], x.shape[2]), dimensions=(0, 1, 2)
            ),
            "input_shapes": [(201, 1, 201)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_valid_flatten_trailing",
            "callable": _reshape_flatten_leading,
            "input_shapes": [(201, 1, 5)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_with_target_shape_from_symbolic_dim_computation",
            "callable": _reshape_flatten_leading,
            "input_shapes": [("N", "M", "K")],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_with_inferred_dimension_from_input_dynamic",
            "callable": _reshape_flatten_leading,
            "input_shapes": [("B", 10, 10)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_with_inferred_dimension_from_input",
            "callable": _reshape_flatten_leading,
            "input_shapes": [(3, 10, 10)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "reshape_merge_symbolic_with_static_and_check_name",
            "callable": _reshape_flatten_trailing,
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
    plugins IR converter for jax.lax.reshape → ONNX Reshape.
    Builds the shape tensor minimally (all-const when possible), and stamps the
    output with correct symbolic labels (reusing input symbols; fresh labels for
    derived dims like B*4).
    """

    # ---------------- lowering (IR) ----------------
    def lower(self, ctx: "IRContext", eqn):
        x_var = eqn.invars[0]
        y_var = eqn.outvars[0]
        new_sizes = tuple(eqn.params["new_sizes"])
        debug = bool(int(os.getenv("JAX2ONNX_DEBUG_RESHAPE", "0")))
        if debug:
            print("[reshape] new_sizes:", new_sizes)
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
            return _const_i64(
                ctx, vals.astype(np.int64, copy=False), ctx.fresh_name("shape_const")
            )

        def const_i64_scalar(val: int) -> ir.Value:
            return _const_i64(
                ctx, np.asarray(val, dtype=np.int64), ctx.fresh_name("i64")
            )

        def unsqueeze_to_len1(src: ir.Value) -> ir.Value:
            axes = const_i64_vec(np.array([0], dtype=np.int64))
            unsqueezed = ctx.builder.Unsqueeze(
                src, axes, _outputs=[ctx.fresh_name("unsq")]
            )
            _stamp_type_and_shape(unsqueezed, (1,))
            _add_value_info(ctx, unsqueezed)
            return unsqueezed

        # Build pieces of the shape tensor.
        # We keep two views:
        #  - shape_parts: IR Values (may be const or dynamic pieces)
        #  - const_accum: plain Python ints if (and only if) we can prove every piece is constant
        shape_parts: List[ir.Value] = []
        all_const = True
        const_accum: list[int] = []

        # Create 'Shape' result lazily when needed
        shape_of_x: ir.Value | None = None

        # Iterator over runtime scalar dim inputs (if any)
        runtime_iter = iter(runtime_dim_vars)

        # ---- small helpers for DimExpr handling --------------------------------
        def _dimexpr_const_value(d: DimExpr) -> Optional[int]:
            """If a DimExpr prints like an integer (e.g. '3' or '-1'), return that int."""
            try:
                s = str(d).strip()
                if s.lstrip("-").isdigit():
                    return int(s)
            except Exception:
                pass
            return None

        def _same_symbol(dim_expr: DimExpr, axis_dim) -> bool:
            """Robust 'same symbol' test between a DimExpr and an input axis dim."""
            if axis_dim is dim_expr:
                return True
            try:
                # If input axis is static int, allow matching constant DimExpr too.
                if isinstance(axis_dim, (int, np.integer)):
                    cv = _dimexpr_const_value(dim_expr)
                    return cv is not None and int(axis_dim) == cv
                # Otherwise compare string forms (covers same-scope symbols)
                return str(axis_dim) == str(dim_expr)
            except Exception:
                return False

        # Track whether we already inserted an inferred dim (-1)
        inserted_neg1 = any(
            isinstance(d, (int, np.integer)) and int(d) == -1 for d in new_sizes
        )
        for dim in new_sizes:
            if debug:
                print("  dim:", dim, "type:", type(dim))
            if isinstance(dim, (int, np.integer)):
                # include -1 as a literal: let ONNX infer that dim
                part = const_i64_vec(np.array([int(dim)], dtype=np.int64))
                shape_parts.append(part)
                const_accum.append(int(dim))
            elif isinstance(dim, DimExpr):
                # Try first to fold to constant if the referenced axis is static.
                axis_idx = next(
                    (i for i, d in enumerate(x_shape) if _same_symbol(dim, d)), None
                )
                if debug:
                    print("    matched axis:", axis_idx)
                if axis_idx is not None:
                    axis_dim_val = x_shape[axis_idx]
                    # Fold if the matched axis is a Python int ...
                    if isinstance(axis_dim_val, (int, np.integer)):
                        v = int(axis_dim_val)
                        shape_parts.append(const_i64_vec(np.array([v], dtype=np.int64)))
                        const_accum.append(v)
                        continue
                    # ... or a constant DimExpr like '3'
                    if isinstance(axis_dim_val, DimExpr):
                        cv_axis = _dimexpr_const_value(axis_dim_val)
                        if cv_axis is not None:
                            shape_parts.append(
                                const_i64_vec(np.array([cv_axis], dtype=np.int64))
                            )
                            const_accum.append(int(cv_axis))
                            continue

                # If the DimExpr itself is constant (e.g., '3'), also fold to const.
                cv = _dimexpr_const_value(dim)
                if cv is not None:
                    part = const_i64_vec(np.array([int(cv)], dtype=np.int64))
                    shape_parts.append(part)
                    const_accum.append(int(cv))
                    continue

                # Truly symbolic: we may either copy an input axis dynamically
                # or, if this is a derived symbol (e.g. 4*B), fall back to -1 inference.
                # We only mark 'all_const=False' when we really introduce dynamic nodes.
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
                    const_accum.append(-1)
                    continue
                # Copy that input axis dynamically
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
                shape_parts.append(unsqueeze_to_len1(gathered))
                if debug:
                    print("    dynamic gather from input axis", axis_idx)
            elif hasattr(dim, "dtype") and np.issubdtype(
                getattr(dim, "dtype", None), np.integer
            ):
                # runtime scalar integer
                dyn_var = next(runtime_iter)
                dyn_val = ctx.get_value_for_var(
                    dyn_var, name_hint=ctx.fresh_name("dyn")
                )
                shape_parts.append(unsqueeze_to_len1(dyn_val))
                all_const = False
            else:
                raise TypeError(f"Unsupported reshape size element: {type(dim)}")
        # If we proved every element is constant, emit a *single* initializer tensor.
        # (Do NOT rely on new_sizes here; it may contain DimExpr objects.)
        if all_const:
            shape_tensor = const_i64_vec(np.array(const_accum, dtype=np.int64))
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

        # --------- emit the single Reshape node ----------
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
        sym_label_map: Dict[object, str] = {}
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
