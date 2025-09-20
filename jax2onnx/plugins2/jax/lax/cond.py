from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Tuple

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2.jax.lax._control_flow_utils import (
    lower_jaxpr_eqns,
    make_subgraph_context,
)
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


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


def _clone_value_for_subgraph(parent_val: ir.Value) -> ir.Value:
    return ir.Value(
        name=parent_val.name,
        type=parent_val.type,
        shape=parent_val.shape,
    )


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
    context="primitives2.lax",
    component="cond",
    testcases=[
        {
            "testcase": "cond_scalar_bool",
            "callable": lambda x: jax.lax.cond(
                True,
                lambda v: v + 1,
                lambda v: v - 1,
                x,
            ),
            "input_shapes": [()],
            "use_onnx_ir": True,
        },
    ],
)
class CondPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.cond`` to ONNX ``If``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        cond_var, *operand_vars = eqn.invars
        cond_val = ctx.get_value_for_var(
            cond_var, name_hint=ctx.fresh_name("cond_pred")
        )

        cond_shape = getattr(cond_val, "shape", ir.Shape(()))
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
            bool_val = ir.Value(
                name=ctx.fresh_name("cond_pred_bool"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=cond_shape,
            )
            cast_attr = IRAttr("to", IRAttrType.INT, int(ir.DataType.BOOL.value))
            ctx.add_node(
                ir.Node(
                    op_type="Cast",
                    domain="",
                    inputs=[cond_val],
                    outputs=[bool_val],
                    name=ctx.fresh_name("Cast"),
                    attributes=[cast_attr],
                )
            )
            _stamp_type_and_shape(
                bool_val, tuple(getattr(getattr(cond_var, "aval", None), "shape", ()))
            )
            _ensure_value_info(ctx, bool_val)
            cond_val = bool_val
        else:
            _stamp_type_and_shape(
                cond_val, tuple(getattr(getattr(cond_var, "aval", None), "shape", ()))
            )
            _ensure_value_info(ctx, cond_val)

        branch_input_vals = [
            ctx.get_value_for_var(v, name_hint=ctx.fresh_name("cond_operand"))
            for v in operand_vars
        ]

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

        out_vals = [
            ctx.get_value_for_var(var, name_hint=ctx.fresh_name("cond_out"))
            for var in eqn.outvars
        ]

        if_node = ir.Node(
            op_type="If",
            domain="",
            inputs=[cond_val],
            outputs=out_vals,
            name=ctx.fresh_name("If"),
            attributes=[
                IRAttr("then_branch", IRAttrType.GRAPH, then_graph),
                IRAttr("else_branch", IRAttrType.GRAPH, else_graph),
            ],
        )
        ctx.add_node(if_node)

        for var, val in zip(eqn.outvars, out_vals):
            _stamp_type_and_shape(val, tuple(getattr(var.aval, "shape", ())))
            _ensure_value_info(ctx, val)

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

        branch_inputs: list[ir.Value] = []
        for outer_val, inner_var in zip(operand_vals, branch_jaxpr.invars):
            # ONNX If subgraphs do not receive explicit inputs. Instead they
            # "capture" values from the parent scope by name. Mirror the
            # parent value metadata but keep graph inputs empty so ORT does
            # not expect additional feeds.
            cloned = _clone_value_for_subgraph(outer_val)
            branch_ctx.bind_value_for_var(inner_var, cloned)
            branch_inputs.append(cloned)

        lower_jaxpr_eqns(branch_ctx, branch_jaxpr)

        input_names = {val.name for val in branch_inputs}
        branch_outputs: list[ir.Value] = []
        for out_var in branch_jaxpr.outvars:
            val = branch_ctx.get_value_for_var(out_var)
            if val.name in input_names:
                new_val = ir.Value(
                    name=branch_ctx.fresh_name(f"{val.name}_identity"),
                    type=val.type,
                    shape=val.shape,
                )
                branch_ctx.add_node(
                    ir.Node(
                        op_type="Identity",
                        domain="",
                        inputs=[val],
                        outputs=[new_val],
                        name=branch_ctx.fresh_name("Identity"),
                    )
                )
                val = new_val
            branch_outputs.append(val)
            _stamp_type_and_shape(val, tuple(getattr(out_var.aval, "shape", ())))
            _ensure_value_info(branch_ctx, val)

        branch_ctx.builder.outputs = branch_outputs

        return ir.Graph(
            inputs=[],
            outputs=list(branch_outputs),
            nodes=list(branch_ctx.builder.nodes),
            initializers=list(branch_ctx.builder.initializers),
            name=ctx.fresh_name(prefix),
            opset_imports={"": getattr(ctx.builder, "opset", 21)},
        )
