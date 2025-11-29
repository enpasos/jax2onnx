# jax2onnx/plugins/jax/lax/select_n.py

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.jax.lax._index_utils import _cast_to_i64, _const_i64
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
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
            "post_check_onnx_graph": EG(
                ["Where:3"],
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                ["Where:2x2"],
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                ["Where:3"],
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                ["Cast:4 -> Equal:4 -> Where:4"],
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                ["Cast:6 -> Equal:6 -> Where:6"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class SelectNPlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``lax.select_n`` using ``Where`` cascades."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for lax.select_n lowering"
            )

        invars = list(eqn.invars)
        out_var = eqn.outvars[0]
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("select_out")
        )

        if len(invars) < 2:
            raise ValueError("select_n requires at least one choice")

        selector_var = invars[0]
        case_vars = invars[1:]

        selector_val = ctx.get_value_for_var(
            selector_var, name_hint=ctx.fresh_name("select_idx")
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        selector_shape = tuple(getattr(selector_var.aval, "shape", ()))

        def _ensure_tensor_metadata(
            value: ir.Value, shape: tuple[int | str, ...], dtype: ir.DataType | None
        ) -> None:
            if dtype is not None:
                value.type = ir.TensorType(dtype)
            _stamp_type_and_shape(value, shape)
            _ensure_value_metadata(ctx, value)

        # --- Boolean two-case path -------------------------------------------------
        if len(case_vars) == 2 and np.issubdtype(selector_var.aval.dtype, np.bool_):
            cond_val = selector_val
            if selector_var.aval.dtype != np.bool_:
                cond_val = builder.Cast(
                    cond_val,
                    _outputs=[ctx.fresh_name("select_cond_bool")],
                    to=int(ir.DataType.BOOL.value),
                )
                _ensure_tensor_metadata(cond_val, selector_shape, ir.DataType.BOOL)

            true_val = ctx.get_value_for_var(
                case_vars[1], name_hint=ctx.fresh_name("select_true")
            )
            false_val = ctx.get_value_for_var(
                case_vars[0], name_hint=ctx.fresh_name("select_false")
            )

            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Where")
            where_out = builder.Where(
                cond_val,
                true_val,
                false_val,
                _outputs=[desired_name],
            )
            result_dtype = getattr(getattr(true_val, "type", None), "dtype", None)
            _ensure_tensor_metadata(where_out, out_shape, result_dtype)
            ctx.bind_value_for_var(out_var, where_out)
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
        current_dtype = getattr(getattr(current_val, "type", None), "dtype", None)

        if len(case_vars) == 1:
            identity_out = builder.Identity(
                current_val,
                _outputs=[ctx.fresh_name("select_identity")],
            )
            _ensure_tensor_metadata(identity_out, out_shape, current_dtype)
            ctx.bind_value_for_var(out_var, identity_out)
            return

        result_val = current_val
        for idx, case_var in enumerate(case_vars[1:], start=1):
            case_val = ctx.get_value_for_var(
                case_var, name_hint=ctx.fresh_name(f"select_case{idx}")
            )
            case_dtype = getattr(
                getattr(case_val, "type", None), "dtype", current_dtype
            )

            const_idx = _const_i64(
                ctx, np.asarray(idx, dtype=np.int64), ctx.fresh_name("select_const")
            )
            eq_val = builder.Equal(
                selector_i64,
                const_idx,
                _outputs=[ctx.fresh_name("select_eq")],
            )
            _ensure_tensor_metadata(eq_val, selector_shape, ir.DataType.BOOL)

            result_val = builder.Where(
                eq_val,
                case_val,
                result_val,
                _outputs=[ctx.fresh_name("select_where")],
            )
            _ensure_tensor_metadata(result_val, out_shape, case_dtype)
            current_dtype = case_dtype

        ctx.bind_value_for_var(out_var, result_val)
