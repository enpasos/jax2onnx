# jax2onnx/plugins/jax/lax/select_n.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _cast_to_i64, _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
import jax.numpy as jnp

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.select_n_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.select_n.html",
    onnx=[
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        },
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="select_n",
    testcases=[
        {
            "testcase": "select_n_bool_predicate_two_cases_float",
            "callable": lambda pred, on_false, on_true: jax.lax.select_n(
                pred, on_false, on_true
            ),
            "input_values": [
                jnp.array([True, False, True], dtype=jnp.bool_),
                jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
                jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),
            ],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [np.float32],
        },
        {
            "testcase": "select_n_bool_predicate_two_cases_int",
            "callable": lambda pred, on_false, on_true: jax.lax.select_n(
                pred, on_false, on_true
            ),
            "input_values": [
                jnp.array([[True, False], [False, True]], dtype=jnp.bool_),
                jnp.array([[10, 20], [30, 40]], dtype=jnp.int32),
                jnp.array([[50, 60], [70, 80]], dtype=jnp.int32),
            ],
            "expected_output_shapes": [(2, 2)],
            "expected_output_dtypes": [np.int32],
        },
        {
            "testcase": "select_n_bool_predicate_scalar_broadcast",
            "callable": lambda pred, on_false, on_true: jax.lax.select_n(
                pred, on_false, on_true
            ),
            "input_values": [
                jnp.array(True, dtype=jnp.bool_),
                jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
                jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32),
            ],
            "expected_output_shapes": [(3,)],
            "expected_output_dtypes": [np.float32],
        },
        {
            "testcase": "select_n_int_indices_three_cases",
            "callable": lambda indices, c0, c1, c2: jax.lax.select_n(
                indices, c0, c1, c2
            ),
            "input_values": [
                jnp.array([0, 1, 2, 0], dtype=jnp.int32),
                jnp.array([10, 11, 12, 13], dtype=jnp.float32),
                jnp.array([20, 21, 22, 23], dtype=jnp.float32),
                jnp.array([30, 31, 32, 33], dtype=jnp.float32),
            ],
            "expected_output_shapes": [(4,)],
            "expected_output_dtypes": [np.float32],
        },
        {
            "testcase": "select_n_int_indices_four_cases",
            "callable": lambda indices, c0, c1, c2, c3: jax.lax.select_n(
                indices, c0, c1, c2, c3
            ),
            "input_values": [
                jnp.array([0, 1, 2, 3, 1, 0], dtype=jnp.int32),
                jnp.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=jnp.float32),
                jnp.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5], dtype=jnp.float32),
                jnp.array([3.0, 3.1, 3.2, 3.3, 3.4, 3.5], dtype=jnp.float32),
                jnp.array([4.0, 4.1, 4.2, 4.3, 4.4, 4.5], dtype=jnp.float32),
            ],
            "expected_output_shapes": [(6,)],
            "expected_output_dtypes": [np.float32],
        },
    ],
)
class SelectNPlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``lax.select_n`` using ``Where`` cascades."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        invars = list(eqn.invars)
        out_var = eqn.outvars[0]
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("select_out"))

        if len(invars) < 2:
            raise ValueError("select_n requires at least one choice")

        selector_var = invars[0]
        case_vars = invars[1:]

        selector_val = ctx.get_value_for_var(
            selector_var, name_hint=ctx.fresh_name("select_idx")
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))

        # --- Boolean two-case path -------------------------------------------------
        if len(case_vars) == 2 and np.issubdtype(selector_var.aval.dtype, np.bool_):
            cond_val = selector_val
            if selector_var.aval.dtype != np.bool_:
                cond_cast = ir.Value(
                    name=ctx.fresh_name("select_cond_bool"),
                    type=ir.TensorType(ir.DataType.BOOL),
                    shape=cond_val.shape,
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Cast",
                        domain="",
                        inputs=[cond_val],
                        outputs=[cond_cast],
                        name=ctx.fresh_name("Cast"),
                        attributes=[
                            IRAttr("to", IRAttrType.INT, int(ir.DataType.BOOL.value))
                        ],
                    )
                )
                cond_val = cond_cast
            true_val = ctx.get_value_for_var(
                case_vars[1], name_hint=ctx.fresh_name("select_true")
            )
            false_val = ctx.get_value_for_var(
                case_vars[0], name_hint=ctx.fresh_name("select_false")
            )
            ctx.add_node(
                ir.Node(
                    op_type="Where",
                    domain="",
                    inputs=[cond_val, true_val, false_val],
                    outputs=[out_val],
                    name=ctx.fresh_name("Where"),
                )
            )
            out_val.type = ir.TensorType(true_val.type.dtype)
            out_val.dtype = true_val.type.dtype
            _stamp_type_and_shape(out_val, out_shape)
            _ensure_value_info(ctx, out_val)
            return

        # --- Integer selector path -------------------------------------------------
        selector_i64 = selector_val
        if not np.issubdtype(selector_var.aval.dtype, np.integer):
            raise TypeError(
                "select_n with more than two cases requires integer selector"
            )
        if selector_var.aval.dtype != np.int64:
            selector_i64 = _cast_to_i64(ctx, selector_val, ctx.fresh_name("select_i64"))

        current_val = ctx.get_value_for_var(
            case_vars[0], name_hint=ctx.fresh_name("select_case0")
        )

        if len(case_vars) == 1:
            ctx.add_node(
                ir.Node(
                    op_type="Identity",
                    domain="",
                    inputs=[current_val],
                    outputs=[out_val],
                    name=ctx.fresh_name("Identity"),
                )
            )
            out_val.type = ir.TensorType(current_val.type.dtype)
            out_val.dtype = current_val.type.dtype
            _stamp_type_and_shape(out_val, out_shape)
            _ensure_value_info(ctx, out_val)
            return

        result_val = current_val
        for idx, case_var in enumerate(case_vars[1:], start=1):
            case_val = ctx.get_value_for_var(
                case_var, name_hint=ctx.fresh_name(f"select_case{idx}")
            )

            const_idx = _const_i64(
                ctx, np.asarray(idx, dtype=np.int64), ctx.fresh_name("select_const")
            )
            eq_val = ir.Value(
                name=ctx.fresh_name("select_eq"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=selector_i64.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Equal",
                    domain="",
                    inputs=[selector_i64, const_idx],
                    outputs=[eq_val],
                    name=ctx.fresh_name("Equal"),
                )
            )
            _stamp_type_and_shape(
                eq_val, tuple(getattr(selector_var.aval, "shape", ()))
            )
            _ensure_value_info(ctx, eq_val)

            where_out = ir.Value(
                name=ctx.fresh_name("select_where_out"),
                type=ir.TensorType(case_val.type.dtype),
                shape=case_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Where",
                    domain="",
                    inputs=[eq_val, case_val, result_val],
                    outputs=[where_out],
                    name=ctx.fresh_name("Where"),
                )
            )
            _stamp_type_and_shape(where_out, out_shape)
            _ensure_value_info(ctx, where_out)
            result_val = where_out

        ctx.add_node(
            ir.Node(
                op_type="Identity",
                domain="",
                inputs=[result_val],
                outputs=[out_val],
                name=ctx.fresh_name("Identity"),
            )
        )
        out_val.type = ir.TensorType(result_val.type.dtype)
        out_val.dtype = result_val.type.dtype
        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)
