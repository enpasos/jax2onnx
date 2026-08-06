from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import numpy as np
import onnx_ir as ir
from equinox._unvmap import unvmap_all_p, unvmap_any_p, unvmap_max_p
from equinox.internal._nontraceable import nonbatchable_p

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.jax.lax._control_flow_utils import (
    builder_cast,
    builder_identity,
)
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _lower_unvmap_reduction(
    ctx: LoweringContextProtocol,
    eqn: Any,
    *,
    op_type: str,
    boolean: bool,
) -> None:
    x_var = eqn.invars[0]
    out_var = eqn.outvars[0]
    x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("unvmap_in"))
    input_rank = len(tuple(getattr(x_var.aval, "shape", ())))

    reduction_input = x_val
    if boolean:
        reduction_input = builder_cast(
            ctx,
            x_val,
            ir.DataType.INT64,
            name_hint="unvmap_bool_to_i64",
        )

    axes = _const_i64(ctx, list(range(input_rank)), "unvmap_axes")
    reduction = cast(
        ir.Value,
        getattr(ctx.builder, op_type)(
            reduction_input,
            axes,
            keepdims=0,
            _outputs=[ctx.fresh_name(op_type)],
        ),
    )
    reduction.type = reduction_input.type
    _stamp_type_and_shape(reduction, ())
    _ensure_value_metadata(ctx, reduction)

    result = reduction
    if boolean:
        result = builder_cast(
            ctx,
            reduction,
            ir.DataType.BOOL,
            name_hint="unvmap_i64_to_bool",
        )

    out_dtype = _dtype_to_ir(
        np.dtype(out_var.aval.dtype), ctx.builder.enable_double_precision
    )
    result.type = ir.TensorType(out_dtype)
    _stamp_type_and_shape(result, ())
    _ensure_value_metadata(ctx, result)
    ctx.bind_value_for_var(out_var, result)


@register_primitive(
    jaxpr_primitive=unvmap_any_p.name,
    jax_doc="https://docs.kidger.site/equinox/api/internal/",
    onnx=[
        {
            "component": "ReduceMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
        }
    ],
    since="0.15.0",
    context="primitives.eqx",
    component="unvmap_any",
    testcases=[
        {
            "testcase": "eqx_unvmap_any",
            "callable": eqx.internal.unvmap_any,
            "input_shapes": [(2, 3)],
            "input_dtypes": [np.bool_],
            "post_check_onnx_graph": expect_graph(["Cast -> ReduceMax -> Cast"]),
        }
    ],
)
class UnvmapAnyPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        _lower_unvmap_reduction(ctx, eqn, op_type="ReduceMax", boolean=True)


@register_primitive(
    jaxpr_primitive=unvmap_all_p.name,
    jax_doc="https://docs.kidger.site/equinox/api/internal/",
    onnx=[
        {
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        }
    ],
    since="0.15.0",
    context="primitives.eqx",
    component="unvmap_all",
    testcases=[
        {
            "testcase": "eqx_unvmap_all",
            "callable": eqx.internal.unvmap_all,
            "input_shapes": [(2, 3)],
            "input_dtypes": [np.bool_],
            "post_check_onnx_graph": expect_graph(["Cast -> ReduceMin -> Cast"]),
        }
    ],
)
class UnvmapAllPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        _lower_unvmap_reduction(ctx, eqn, op_type="ReduceMin", boolean=True)


@register_primitive(
    jaxpr_primitive=unvmap_max_p.name,
    jax_doc="https://docs.kidger.site/equinox/api/internal/",
    onnx=[
        {
            "component": "ReduceMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
        }
    ],
    since="0.15.0",
    context="primitives.eqx",
    component="unvmap_max",
    testcases=[
        {
            "testcase": "eqx_unvmap_max",
            "callable": eqx.internal.unvmap_max,
            "input_shapes": [(2, 3)],
            "input_dtypes": [np.int32],
            "post_check_onnx_graph": expect_graph(["ReduceMax"]),
        }
    ],
)
class UnvmapMaxPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        _lower_unvmap_reduction(ctx, eqn, op_type="ReduceMax", boolean=False)


@register_primitive(
    jaxpr_primitive=nonbatchable_p.name,
    jax_doc="https://docs.kidger.site/equinox/api/internal/",
    onnx=[
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        }
    ],
    since="0.15.0",
    context="primitives.eqx",
    component="nonbatchable",
    testcases=[
        {
            "testcase": "eqx_nonbatchable",
            "callable": lambda x: eqx.internal.nonbatchable(x),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": expect_graph(["Identity:2x3"]),
        }
    ],
)
class NonbatchablePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        x_val = ctx.get_value_for_var(
            x_var, name_hint=ctx.fresh_name("nonbatchable_in")
        )
        result = builder_identity(ctx, x_val, name_hint="nonbatchable_out")
        ctx.bind_value_for_var(out_var, result)
