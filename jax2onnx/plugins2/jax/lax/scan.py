from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import core as jax_core

try:
    from jax.extend import core as jax_core_ext  # type: ignore
except ImportError:  # pragma: no cover - older jax
    jax_core_ext = None
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2.jax.lax._control_flow_utils import (
    lower_jaxpr_eqns,
    make_subgraph_context,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from jax2onnx.converter2.ir_context import IRContext


def _maybe_var_type(mod):
    if mod is None:
        return None
    try:
        return getattr(mod, "Var")
    except AttributeError:
        return None


_JAX_VAR_TYPES = tuple(
    t
    for t in (_maybe_var_type(jax_core), _maybe_var_type(jax_core_ext))
    if t is not None
)


def _is_jax_var(obj) -> bool:
    return bool(_JAX_VAR_TYPES) and isinstance(obj, _JAX_VAR_TYPES)


@register_primitive(
    jaxpr_primitive=jax.lax.scan_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html",
    onnx=[
        {
            "component": "Scan",
            "doc": "https://onnx.ai/onnx/operators/onnx__Scan.html",
        }
    ],
    since="v0.5.1",
    context="primitives2.lax",
    component="scan",
    testcases=[
        {
            "testcase": "scan_identity_slice_helper",
            "callable": lambda x: jax.lax.scan(
                lambda c, xt: (c, jnp.squeeze(xt[None, ...][0:1, :, :, :], axis=0)),
                jnp.zeros(x.shape[1:], dtype=x.dtype),
                x,
            )[1],
            "input_shapes": [(2, 3, 4, 5)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "scan_cumsum",
            "callable": lambda xs: jax.lax.scan(
                lambda c, x: (c + x, c + x),
                jnp.zeros((), dtype=xs.dtype),
                xs,
            )[1],
            "input_shapes": [(5,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "scan_carry_only",
            "callable": lambda xs: jax.lax.scan(
                lambda c, x: (c + x, c),
                jnp.zeros((), dtype=xs.dtype),
                xs,
            )[0],
            "input_shapes": [(3,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "scan_multiple_sequences",
            "callable": lambda xs, ys: jax.lax.scan(
                lambda c, xy: (c + xy[0] * xy[1], c + xy[0]),
                jnp.zeros((), dtype=xs.dtype),
                (xs, ys),
            )[1],
            "input_shapes": [(4,), (4,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "scan_multiple_carry",
            "callable": lambda xs: jax.lax.scan(
                lambda carry, x: (
                    (carry[0] + x, carry[1] * x),
                    carry[0] + carry[1],
                ),
                (
                    jnp.zeros((), dtype=xs.dtype),
                    jnp.ones((), dtype=xs.dtype),
                ),
                xs,
            )[1],
            "input_shapes": [(3,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "scan_matrix_carry_multidim_xs",
            "callable": lambda init_carry, xs_seq: jax.lax.scan(
                lambda c_mat, x_slice: (c_mat + x_slice, jnp.sum(c_mat + x_slice)),
                init_carry,
                xs_seq,
            )[1],
            "input_shapes": [(3, 2), (5, 3, 2)],
            "use_onnx_ir": True,
        },
    ],
)
class ScanPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: IRContext, eqn):  # type: ignore[name-defined]
        params = eqn.params
        closed_jaxpr = params["jaxpr"]
        num_carry = int(params.get("num_carry", 0))
        num_consts = int(params.get("num_consts", 0) or 0)
        if params.get("reverse", False):
            raise NotImplementedError("Reverse scan is not supported in IR pipeline.")
        if params.get("unroll", 1) != 1:
            raise NotImplementedError(
                "Scan with unroll > 1 is not supported in IR pipeline."
            )

        # Determine number of scanned inputs.
        jaxpr = closed_jaxpr.jaxpr
        total_invars = len(jaxpr.invars)
        num_scan = int(params.get("num_xs", total_invars - num_carry - num_consts))
        if num_scan < 0 or (num_carry + num_scan + num_consts) != total_invars:
            raise ValueError(
                "Inconsistent Scan arity: expected consts/carry/scan to match jaxpr invars"
            )
        length = params.get("length")

        # Build a child IR context for the Scan body graph.
        body_ctx = make_subgraph_context(ctx, prefix="scan_body")

        # Bind consts inside the body context.
        for cv, cval in zip(jaxpr.constvars, closed_jaxpr.consts):
            body_ctx.bind_const_for_var(cv, np.asarray(cval))

        # Body inputs follow JAX order: [consts..., carry..., scan_inputs...]
        body_inputs: list[ir.Value] = []
        for idx, var in enumerate(jaxpr.invars):
            val = body_ctx.add_input_for_invar(var, idx)
            body_inputs.append(val)

        # Lower body equations using the shared plugin registry.
        lower_jaxpr_eqns(body_ctx, jaxpr)

        # Collect body outputs: const passthrough → carry outputs → stacked outputs.
        body_outputs: list[ir.Value] = []

        # 1) Const passthrough: maintain ordering, cast via Identity to avoid aliasing inputs.
        for ci in range(num_consts):
            inp_val = body_inputs[ci]
            out_val = ir.Value(
                name=body_ctx.fresh_name("scan_const_out"),
                type=inp_val.type,
                shape=inp_val.shape,
            )
            body_ctx.add_node(
                ir.Node(
                    op_type="Identity",
                    domain="",
                    inputs=[inp_val],
                    outputs=[out_val],
                    name=body_ctx.fresh_name("Identity"),
                )
            )
            body_outputs.append(out_val)

        # 2) Carry outputs (if any)
        for out_var in jaxpr.outvars[:num_carry]:
            body_outputs.append(body_ctx.get_value_for_var(out_var))

        # 3) Stacked y outputs
        for out_var in jaxpr.outvars[num_carry:]:
            body_outputs.append(body_ctx.get_value_for_var(out_var))

        body_ctx.builder.outputs = body_outputs

        dummy_scan_init: ir.Value | None = None
        if num_scan == 0:
            if not isinstance(length, (int, np.integer)):
                raise NotImplementedError(
                    "Scan without xs and non-constant length is not supported yet"
                )
            trip_count = int(length)
            dummy_seq = np.zeros((trip_count,), dtype=np.int64)
            dummy_scan_init = ir.Value(
                name=ctx.fresh_name("scan_dummy"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((trip_count,)),
                const_value=ir.tensor(dummy_seq),
            )
            ctx.builder.initializers.append(dummy_scan_init)

            dummy_iter = ir.Value(
                name=body_ctx.fresh_name("scan_dummy_in"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape(()),
            )
            body_ctx.builder.inputs.append(dummy_iter)
            _stamp_type_and_shape(dummy_iter, ())
            num_scan = 1

        body_graph = ir.Graph(
            inputs=list(body_ctx.builder.inputs),
            outputs=list(body_ctx.builder.outputs),
            nodes=list(body_ctx.builder.nodes),
            initializers=list(body_ctx.builder.initializers),
            name=ctx.fresh_name("scan_body"),
            opset_imports={"": getattr(ctx.builder, "opset", 21)},
        )

        # Bind node inputs/outputs to the outer context.
        node_inputs = [ctx.get_value_for_var(v) for v in eqn.invars]
        if dummy_scan_init is not None:
            node_inputs.append(dummy_scan_init)

        num_consts + num_carry
        node_outputs: list[ir.Value] = []

        # const passthrough outputs are not exposed by JAX (DropVar); keep dummies to
        # satisfy ONNX state arity.
        for idx, body_out in enumerate(body_outputs[:num_consts]):
            node_outputs.append(
                ir.Value(
                    name=ctx.fresh_name("scan_const_unused"),
                    type=body_out.type,
                    shape=body_out.shape,
                )
            )

        # Carry outputs align with the leading eqn outvars (if any).
        for i in range(num_carry):
            out_var = eqn.outvars[i] if i < len(eqn.outvars) else None
            body_out = body_outputs[num_consts + i]
            if out_var is not None and _is_jax_var(out_var):
                node_outputs.append(ctx.get_value_for_var(out_var))
            else:
                node_outputs.append(
                    ir.Value(
                        name=ctx.fresh_name("scan_state_unused"),
                        type=body_out.type,
                        shape=body_out.shape,
                    )
                )

        # Stacked outputs consume the remaining eqn outvars.
        for out_var in eqn.outvars[num_carry:]:
            node_outputs.append(
                ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("scan_out"))
            )

        attrs = [
            IRAttr("body", IRAttrType.GRAPH, body_graph),
            IRAttr("num_scan_inputs", IRAttrType.INT, int(num_scan)),
        ]
        if num_scan > 0:
            attrs.append(IRAttr("scan_input_axes", IRAttrType.INTS, [0] * num_scan))
        num_scan_outputs = max(0, len(eqn.outvars) - num_carry)
        if num_scan_outputs > 0:
            attrs.append(
                IRAttr("scan_output_axes", IRAttrType.INTS, [0] * num_scan_outputs)
            )

        scan_node = ir.Node(
            op_type="Scan",
            domain="",
            inputs=node_inputs,
            outputs=node_outputs,
            name=ctx.fresh_name("Scan"),
            attributes=attrs,
        )
        ctx.add_node(scan_node)

        for idx, var in enumerate(eqn.outvars):
            val = node_outputs[num_consts + idx]
            aval = getattr(var, "aval", None)
            if aval is not None:
                _stamp_type_and_shape(val, getattr(aval, "shape", ()))
