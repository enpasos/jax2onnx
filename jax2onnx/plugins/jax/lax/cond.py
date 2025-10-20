# jax2onnx/plugins/jax/lax/cond.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Tuple

import jax
import numpy as np
import jax.numpy as jnp
import onnx_ir as ir

from jax2onnx.converter.ir_clone import clone_graph
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins.jax.lax._control_flow_utils import (
    lower_jaxpr_eqns,
    make_subgraph_context,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _unwrap_closed_jaxpr(jaxpr_like: Any) -> Tuple[Any, Iterable[Any]]:
    if hasattr(jaxpr_like, "jaxpr") and hasattr(jaxpr_like, "consts"):
        return jaxpr_like.jaxpr, getattr(jaxpr_like, "consts")
    return jaxpr_like, ()


def _extract_branches(
    params: dict[str, Any],
) -> Tuple[Tuple[Any, Iterable[Any]], Tuple[Any, Iterable[Any]]]:
    if "branches" in params:
        false_closed, true_closed = params["branches"]
    else:
        true_closed = params["true_jaxpr"]
        false_closed = params["false_jaxpr"]
    true_jaxpr, true_consts = _unwrap_closed_jaxpr(true_closed)
    false_jaxpr, false_consts = _unwrap_closed_jaxpr(false_closed)
    return (true_jaxpr, true_consts), (false_jaxpr, false_consts)


@register_primitive(
    jaxpr_primitive=jax.lax.cond_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html",
    onnx=[
        {
            "component": "If",
            "doc": "https://onnx.ai/onnx/operators/onnx__If.html",
        }
    ],
    since="v0.5.1",
    context="primitives.lax",
    component="cond",
    testcases=[
        {
            "testcase": "cond_scalar",
            "callable": lambda: jax.lax.cond(
                True,
                lambda x: x + 1,
                lambda x: x - 1,
                np.int32(3),
            ),
            "input_shapes": [],
            "expected_output_shapes": [()],
            "post_check_onnx_graph": EG(
                ["Cast -> If"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cond_multiple_operands_in_tuple",
            "callable": lambda: jax.lax.cond(
                True,
                lambda tup: tup[0] + tup[1] - tup[2],
                lambda tup: tup[0] - tup[1] + tup[2],
                (np.array(10, np.int32), np.array(5, np.int32), np.array(2, np.int32)),
            ),
            "input_shapes": [],
            "expected_output_shapes": [()],
            "post_check_onnx_graph": EG(
                ["Cast -> If"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cond_my_new_complex_scenario",
            "callable": lambda op1, op2: jax.lax.cond(
                jnp.all(op1 > 0),
                lambda t: (t[0] * 2 + t[1], jnp.sum(t[0], axis=(-2, -1))),
                lambda t: (t[0] - t[1] * 2, jnp.sum(t[0], axis=(-2, -1))),
                (op1, op2),
            ),
            "input_shapes": [(11, 3, 4), (3, 4)],
            "input_dtypes": [np.float32, np.float32],
            "expected_output_shapes": [(11, 3, 4), (11,)],
            "expected_output_dtypes": [np.float32, np.float32],
            "post_check_onnx_graph": EG(
                [
                    "Greater:11x3x4 -> Cast:11x3x4 -> ReduceMin -> Cast -> If",
                ],
            ),
        },
        {
            "testcase": "cond_nested_conditional",
            "callable": lambda x, y, z_pred: jax.lax.cond(
                x > 5,
                lambda op: jax.lax.cond(
                    z_pred,
                    lambda inner_op: inner_op * 10,
                    lambda inner_op: inner_op - 10,
                    op,
                ),
                lambda op: op + 100,
                y,
            ),
            "input_shapes": [(), (), ()],
            "input_dtypes": [np.int32, np.float32, np.bool_],
            "expected_output_shapes": [()],
            "post_check_onnx_graph": EG(["Greater -> If"]),
        },
        {
            "testcase": "cond_variables",
            "callable": lambda x, y: jax.lax.cond(
                x > 5,
                lambda op: op - 100,
                lambda op: op + 100,
                y,
            ),
            "input_shapes": [(), ()],
            "input_dtypes": [np.int32, np.float32],
            "expected_output_shapes": [()],
            "post_check_onnx_graph": EG(["Greater -> If"]),
        },
        {
            "testcase": "cond_internal_constant_f64",
            "callable": lambda: jax.lax.cond(
                False,
                lambda x: x * 2.0,
                lambda x: x + 1.0,
                jnp.zeros((2, 4), dtype=jnp.float64),
            ),
            "input_shapes": [],
            "expected_output_shapes": [(2, 4)],
            "expected_output_dtypes": [np.float64],
            "enable_double_precision": True,
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast -> If:2x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "cond_passthrough_identity",
            "callable": lambda pred, x, y: jax.lax.cond(
                pred,
                lambda op_x, op_y: jax.lax.add(op_x, op_y),
                lambda op_x, op_y: op_x,
                x,
                y,
            ),
            "input_values": [
                np.array(True),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([4.0, 5.0, 6.0], dtype=np.float32),
            ],
            "post_check_onnx_graph": EG(["If:3"]),
        },
        {
            "testcase": "cond_with_scatter",
            "callable": lambda operand, updates: jax.lax.cond(
                True,
                lambda op, upd: jax.lax.scatter_add(
                    op,
                    jnp.array([[1], [3]], dtype=jnp.int32),
                    upd,
                    jax.lax.ScatterDimensionNumbers(
                        update_window_dims=(1,),
                        inserted_window_dims=(0,),
                        scatter_dims_to_operand_dims=(0,),
                    ),
                ),
                lambda op, upd: op,
                operand,
                updates,
            ),
            "input_values": [
                np.ones((5, 3), dtype=np.float32),
                np.ones((2, 3), dtype=np.float32) * 9,
            ],
            "post_check_onnx_graph": EG(["Cast -> If:5x3"]),
        },
    ],
)
class CondPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.cond`` to ONNX ``If``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        cond_var, *operand_vars = eqn.invars
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for lax.cond lowering"
            )

        cond_val = ctx.get_value_for_var(
            cond_var, name_hint=ctx.fresh_name("cond_pred")
        )
        needs_cast = True
        aval_dtype = getattr(getattr(cond_var, "aval", None), "dtype", None)
        if aval_dtype is not None and np.issubdtype(np.dtype(aval_dtype), np.bool_):
            needs_cast = False
        elif (
            hasattr(cond_val, "type")
            and getattr(cond_val.type, "dtype", None) == ir.DataType.BOOL
        ):
            needs_cast = False

        if needs_cast:
            cond_val = builder.Cast(
                cond_val,
                _outputs=[ctx.fresh_name("cond_pred_bool")],
                to=int(ir.DataType.BOOL.value),
            )
            cond_val.type = ir.TensorType(ir.DataType.BOOL)
            _stamp_type_and_shape(
                cond_val, tuple(getattr(getattr(cond_var, "aval", None), "shape", ()))
            )
            _ensure_value_metadata(ctx, cond_val)
        else:
            _stamp_type_and_shape(
                cond_val, tuple(getattr(getattr(cond_var, "aval", None), "shape", ()))
            )
            _ensure_value_metadata(ctx, cond_val)

        branch_input_vals = [
            ctx.get_value_for_var(v, name_hint=ctx.fresh_name("cond_operand"))
            for v in operand_vars
        ]
        for var, val in zip(operand_vars, branch_input_vals):
            _stamp_type_and_shape(
                val, tuple(getattr(getattr(var, "aval", None), "shape", ()))
            )
            _ensure_value_metadata(ctx, val)

        (true_jaxpr, true_consts), (false_jaxpr, false_consts) = _extract_branches(
            eqn.params
        )

        then_graph = self._build_branch_graph(
            ctx,
            true_jaxpr,
            true_consts,
            branch_input_vals,
            prefix="cond_then",
        )
        else_graph = self._build_branch_graph(
            ctx,
            false_jaxpr,
            false_consts,
            branch_input_vals,
            prefix="cond_else",
        )

        output_names = [ctx.fresh_name("cond_out") for _ in eqn.outvars]
        if_outputs = builder.If(
            cond_val,
            then_branch=then_graph,
            else_branch=else_graph,
            _outputs=output_names,
        )

        if not isinstance(if_outputs, tuple):
            if_outputs = (if_outputs,)

        enable_dp = getattr(builder, "enable_double_precision", False)
        for var, out_val in zip(eqn.outvars, if_outputs):
            aval = getattr(var, "aval", None)
            dtype_enum = None
            if aval is not None:
                aval_dtype = getattr(aval, "dtype", None)
                if aval_dtype is not None:
                    try:
                        dtype_enum = _dtype_to_ir(np.dtype(aval_dtype), enable_dp)
                    except TypeError:
                        dtype_enum = None
            if dtype_enum is not None:
                out_val.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(out_val, tuple(getattr(aval, "shape", ()) or ()))
            _ensure_value_metadata(ctx, out_val)
            ctx.bind_value_for_var(var, out_val)

    def _build_branch_graph(
        self,
        ctx: "IRContext",
        branch_jaxpr,
        consts: Iterable[Any],
        operand_vals: list[ir.Value],
        *,
        prefix: str,
    ) -> ir.Graph:
        branch_ctx = make_subgraph_context(ctx, prefix=prefix)

        # Bind constants inside the branch context.
        for const_var, const_value in zip(branch_jaxpr.constvars, consts):
            branch_ctx.bind_const_for_var(const_var, np.asarray(const_value))

        builder = getattr(branch_ctx, "builder", None)
        if builder is None:
            raise AttributeError("Branch context missing builder for lax.cond")

        branch_inputs: list[ir.Value] = []
        for outer_val, inner_var in zip(operand_vals, branch_jaxpr.invars):
            capture = builder.Identity(
                outer_val,
                _outputs=[branch_ctx.fresh_name("cond_capture")],
            )
            orig_type = getattr(outer_val, "type", None)
            if orig_type is not None:
                capture.type = orig_type
            _stamp_type_and_shape(
                capture, tuple(getattr(getattr(inner_var, "aval", None), "shape", ()))
            )
            _ensure_value_metadata(branch_ctx, capture)
            branch_ctx.bind_value_for_var(inner_var, capture)
            branch_inputs.append(capture)

        lower_jaxpr_eqns(branch_ctx, branch_jaxpr)

        input_names = {val.name for val in branch_inputs}
        branch_outputs: list[ir.Value] = []
        for out_var in branch_jaxpr.outvars:
            val = branch_ctx.get_value_for_var(out_var)
            if val.name in input_names:
                orig_type = getattr(val, "type", None)
                val = builder.Identity(
                    val,
                    _outputs=[branch_ctx.fresh_name(f"{val.name}_identity")],
                )
                if orig_type is not None:
                    val.type = orig_type
            branch_outputs.append(val)
            _stamp_type_and_shape(val, tuple(getattr(out_var.aval, "shape", ())))
            _ensure_value_metadata(branch_ctx, val)

        branch_ctx.builder.outputs = branch_outputs

        branch_graph = clone_graph(branch_ctx.builder.graph)
        branch_graph.name = ctx.fresh_name(prefix)
        branch_graph.inputs.clear()
        opset_imports = dict(branch_graph.opset_imports)
        opset_imports.setdefault("", getattr(ctx.builder, "opset", 21))
        branch_graph.opset_imports.clear()
        branch_graph.opset_imports.update(opset_imports)
        return branch_graph
