from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import jax
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType, Shape as IRShape

from jax2onnx.converter2.ir_builder import _dtype_to_ir
from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2.jax.lax._control_flow_utils import (
    lower_jaxpr_eqns,
    make_subgraph_context,
)
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


def _unwrap_closed_jaxpr(jaxpr_like):
    if hasattr(jaxpr_like, "jaxpr") and hasattr(jaxpr_like, "consts"):
        return jaxpr_like.jaxpr, tuple(jaxpr_like.consts)
    return jaxpr_like, ()


def _bind_closed_jaxpr_consts(ctx: "IRContext", jaxpr, consts: Iterable[object]):
    for var, const in zip(getattr(jaxpr, "constvars", ()), consts):
        ctx.bind_const_for_var(var, np.asarray(const))


def _clone_value_for_subgraph(v: ir.Value) -> ir.Value:
    dtype = getattr(getattr(v, "type", None), "dtype", None)
    shape = getattr(v, "shape", IRShape(()))
    tensor_type = ir.TensorType(dtype) if dtype is not None else None
    return ir.Value(name=v.name, type=tensor_type, shape=shape)


def _ensure_bool_value(
    ctx: "IRContext", value: ir.Value, *, name_hint: str
) -> ir.Value:
    dtype = getattr(getattr(value, "type", None), "dtype", None)
    if dtype == ir.DataType.BOOL:
        return value
    shape_obj = getattr(value, "shape", IRShape(()))
    bool_val = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(ir.DataType.BOOL),
        shape=shape_obj,
    )
    ctx.add_node(
        ir.Node(
            op_type="Cast",
            domain="",
            inputs=[value],
            outputs=[bool_val],
            name=ctx.fresh_name("Cast"),
            attributes=[IRAttr("to", IRAttrType.INT, int(ir.DataType.BOOL.value))],
        )
    )
    dims = getattr(shape_obj, "dims", None)
    if dims is None:
        try:
            dims = list(shape_obj)
        except Exception:
            dims = ()
    _stamp_type_and_shape(bool_val, tuple(dims or ()))
    _ensure_value_info(ctx, bool_val)
    return bool_val


def _evaluate_closed_jaxpr(
    ctx: "IRContext", closed_jaxpr, inputs: Sequence[ir.Value], *, prefix: str
) -> list[ir.Value]:
    jaxpr, consts = _unwrap_closed_jaxpr(closed_jaxpr)
    _bind_closed_jaxpr_consts(ctx, jaxpr, consts)
    for var, val in zip(jaxpr.invars, inputs):
        ctx.bind_value_for_var(var, val)
    lower_jaxpr_eqns(ctx, jaxpr)
    outputs: list[ir.Value] = []
    for out_var in jaxpr.outvars:
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name(prefix))
        aval = getattr(out_var, "aval", None)
        if aval is not None:
            shape = tuple(getattr(aval, "shape", ()))
            dtype_enum = _dtype_to_ir(
                np.dtype(getattr(aval, "dtype", np.float32)),
                ctx.builder.enable_double_precision,
            )
            out_val.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(out_val, shape)
            _ensure_value_info(ctx, out_val)
        outputs.append(out_val)
    return outputs


def _build_loop_body_graph(
    ctx: "IRContext",
    cond_cj,
    body_cj,
    state_template: Sequence[ir.Value],
) -> ir.Graph:
    body_ctx = make_subgraph_context(ctx, prefix="while_body")

    iter_input = ir.Value(
        name=body_ctx.fresh_name("loop_iter"),
        type=ir.TensorType(ir.DataType.INT64),
        shape=IRShape(()),
    )
    cond_input = ir.Value(
        name=body_ctx.fresh_name("loop_cond_in"),
        type=ir.TensorType(ir.DataType.BOOL),
        shape=IRShape(()),
    )
    body_ctx.builder.inputs.extend([iter_input, cond_input])

    body_jaxpr, body_consts = _unwrap_closed_jaxpr(body_cj)
    _bind_closed_jaxpr_consts(body_ctx, body_jaxpr, body_consts)

    cond_jaxpr, cond_consts = _unwrap_closed_jaxpr(cond_cj)
    _bind_closed_jaxpr_consts(body_ctx, cond_jaxpr, cond_consts)

    state_inputs: list[ir.Value] = []
    for var, tmpl_val in zip(body_jaxpr.invars, state_template):
        cloned = _clone_value_for_subgraph(tmpl_val)
        cloned.name = body_ctx.fresh_name("loop_state_in")
        body_ctx.builder.inputs.append(cloned)
        body_ctx.bind_value_for_var(var, cloned)
        state_inputs.append(cloned)

    lower_jaxpr_eqns(body_ctx, body_jaxpr)

    state_outputs = [
        body_ctx.get_value_for_var(
            out_var, name_hint=body_ctx.fresh_name("loop_state_out")
        )
        for out_var in body_jaxpr.outvars
    ]
    for out_var, out_val in zip(body_jaxpr.outvars, state_outputs):
        aval = getattr(out_var, "aval", None)
        if aval is not None:
            shape = tuple(getattr(aval, "shape", ()))
            dtype_enum = _dtype_to_ir(
                np.dtype(getattr(aval, "dtype", np.float32)),
                ctx.builder.enable_double_precision,
            )
            out_val.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(out_val, shape)
            _ensure_value_info(body_ctx, out_val)

    for var, val in zip(cond_jaxpr.invars, state_outputs):
        body_ctx.bind_value_for_var(var, val)

    lower_jaxpr_eqns(body_ctx, cond_jaxpr)
    cond_out = body_ctx.get_value_for_var(
        cond_jaxpr.outvars[0], name_hint=body_ctx.fresh_name("loop_cond_out")
    )
    cond_out = _ensure_bool_value(body_ctx, cond_out, name_hint="loop_cond_bool")

    body_ctx.builder.outputs = [cond_out] + state_outputs

    return ir.Graph(
        inputs=list(body_ctx.builder.inputs),
        outputs=list(body_ctx.builder.outputs),
        nodes=list(body_ctx.builder.nodes),
        initializers=list(body_ctx.builder.initializers),
        name=ctx.fresh_name("while_body"),
        opset_imports={"": getattr(ctx.builder, "opset", 21)},
    )


@register_primitive(
    jaxpr_primitive=jax.lax.while_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html",
    onnx=[
        {
            "component": "Loop",
            "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html",
        }
    ],
    since="v0.5.1",
    context="primitives2.lax",
    component="while_loop",
    testcases=[
        {
            "testcase": "while_scalar_counter",
            "callable": lambda x: jax.lax.while_loop(
                lambda v: v < 5, lambda v: v + 1, x
            ),
            "input_shapes": [()],
            "use_onnx_ir": True,
        },
        {
            "testcase": "while_tuple_state",
            "callable": lambda a, b: jax.lax.while_loop(
                lambda s: s[0] < 3,
                lambda s: (s[0] + 1, s[1] + 2),
                (a, b),
            ),
            "input_shapes": [(), ()],
            "input_dtypes": [np.int32, np.int32],
            "use_onnx_ir": True,
        },
    ],
)
class WhileLoopPlugin(PrimitiveLeafPlugin):
    """IR-first lowering for ``lax.while_loop`` via ONNX ``Loop``."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        params = getattr(eqn, "params", {})
        cond_cj = params.get("cond_jaxpr") or params.get("cond_fun")
        body_cj = params.get("body_jaxpr") or params.get("body_fun")
        if cond_cj is None or body_cj is None:
            raise NotImplementedError("while_loop lowering requires cond/body jaxpr")

        state_vals = [
            ctx.get_value_for_var(var, name_hint=ctx.fresh_name("while_state"))
            for var in eqn.invars
        ]

        cond_init = _evaluate_closed_jaxpr(
            ctx, cond_cj, state_vals, prefix="while_cond"
        )
        if not cond_init:
            raise RuntimeError("while_loop cond_jaxpr produced no outputs")
        cond_init_val = _ensure_bool_value(
            ctx, cond_init[0], name_hint="while_cond_bool"
        )

        body_graph = _build_loop_body_graph(ctx, cond_cj, body_cj, state_vals)

        trip_count = ctx.builder.add_initializer_from_scalar(
            ctx.fresh_name("while_trip_count"),
            np.asarray(np.iinfo(np.int64).max, dtype=np.int64),
        )

        loop_inputs = [trip_count, cond_init_val] + state_vals

        out_vals = [
            ctx.get_value_for_var(var, name_hint=ctx.fresh_name("while_out"))
            for var in eqn.outvars
        ]

        loop_node = ir.Node(
            op_type="Loop",
            domain="",
            inputs=loop_inputs,
            outputs=out_vals,
            name=ctx.fresh_name("Loop"),
            attributes=[IRAttr("body", IRAttrType.GRAPH, body_graph)],
        )
        ctx.add_node(loop_node)

        for var, val in zip(eqn.outvars, out_vals):
            aval = getattr(var, "aval", None)
            if aval is None:
                continue
            shape = tuple(getattr(aval, "shape", ()))
            dtype = _dtype_to_ir(
                np.dtype(getattr(aval, "dtype", np.float32)),
                ctx.builder.enable_double_precision,
            )
            val.type = ir.TensorType(dtype)
            _stamp_type_and_shape(val, shape)
            _ensure_value_info(ctx, val)
