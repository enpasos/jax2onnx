# jax2onnx/plugins/jax/lax/reduce_sum.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.ir_utils import const_value_to_numpy
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.lax._reduce_utils import lower_reduction
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _const_scalar(value: Any) -> float | int | None:
    arr = const_value_to_numpy(value)
    if arr is None:
        return None
    if arr.shape != ():
        return None
    scalar = arr.item()
    if isinstance(scalar, (float, np.floating)):
        return float(scalar)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    return None


def _all_axes(x: Any) -> tuple[int, ...]:
    return tuple(range(x.ndim))


@register_primitive(
    jaxpr_primitive=jax.lax.reduce_sum_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_sum.html",
    onnx=[
        {
            "component": "ReduceL1",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceL1.html",
        },
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {
            "component": "ReduceSumSquare",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html",
        },
    ],
    since="0.2.0",
    context="primitives.lax",
    component="reduce_sum",
    testcases=[
        {
            "testcase": "reduce_sum",
            "callable": lambda x: jax.lax.reduce_sum(x, axes=_all_axes(x)),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["ReduceSum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_sum_allaxes",
            "callable": lambda x: jax.lax.reduce_sum(x, axes=_all_axes(x)),
            "input_shapes": [(2, 3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceSum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_sum_dtype",
            "callable": lambda x: jax.lax.reduce_sum(
                jax.lax.convert_element_type(x, jnp.float32),
                axes=_all_axes(x),
            ),
            "input_values": [np.arange(6, dtype=np.float32).reshape(2, 3)],
            "post_check_onnx_graph": EG(
                ["ReduceSum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_sum_dtype_f64",
            "callable": lambda x: jax.lax.reduce_sum(
                jax.lax.convert_element_type(x, jnp.float64),
                axes=_all_axes(x),
            ),
            "input_values": [np.arange(6, dtype=np.float64).reshape(2, 3)],
            "post_check_onnx_graph": EG(
                ["ReduceSum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_sum_uint32_axis",
            "callable": lambda x: jax.lax.reduce_sum(x, axes=(1,)),
            "input_values": [np.arange(8, dtype=np.uint32).reshape(1, 8)],
            "expected_output_dtypes": [np.uint32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast:1x8 -> ReduceSum:1 -> Cast:1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_sum_keepdims",
            "callable": lambda x: jax.lax.broadcast_in_dim(
                jax.lax.reduce_sum(x, axes=(1,)),
                shape=(x.shape[0], 1),
                broadcast_dimensions=(0,),
            ),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["ReduceSum:3 -> Reshape:3x1 -> Expand:3x1"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_sum_no_axes",
            "callable": lambda x: jax.lax.reduce_sum(x, axes=()),
            "input_values": [np.asarray([1.0, 2.0, -3.0], dtype=np.float32)],
            "post_check_onnx_graph": EG(
                ["Identity:3"],
                no_unused_inputs=True,
                must_absent=["ReduceSum"],
            ),
        },
        {
            "testcase": "reduce_sum_of_abs_axis1",
            "callable": lambda x: jax.lax.reduce_sum(jax.lax.abs(x), axes=(1,)),
            "input_values": [np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ReduceL1:2",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "reduce_sum_of_square_axis1",
            "callable": lambda x: jax.lax.reduce_sum(jax.lax.mul(x, x), axes=(1,)),
            "input_values": [np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "ReduceSumSquare:2",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ReduceSumPlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``lax.reduce_sum`` via ONNX ReduceSum."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        operand_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        axes = params.get("axes")
        keepdims = bool(params.get("keepdims", False))
        dtype = params.get("dtype")

        # Keep the legacy lowering path when dtype conversion is requested.
        if dtype is not None:
            lower_reduction(ctx, eqn, op_type="ReduceSum", allow_dtype_param=True)
            return

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("reducesum_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("reducesum_out")
        )

        producer_getter = getattr(operand_val, "producer", lambda: None)
        producer = producer_getter() if callable(producer_getter) else None
        producer_op_type = str(getattr(producer, "op_type", ""))
        producer_inputs = tuple(getattr(producer, "inputs", ()))

        target_base: Any | None = None
        op_name: str | None = None
        if producer_op_type == "Abs" and producer_inputs:
            target_base = producer_inputs[0]
            op_name = "ReduceL1"
        elif producer_op_type == "Mul" and len(producer_inputs) >= 2:
            lhs, rhs = producer_inputs[:2]
            same_input = lhs is rhs or (
                getattr(lhs, "name", None) is not None
                and getattr(lhs, "name", None) == getattr(rhs, "name", None)
            )
            if same_input:
                target_base = lhs
                op_name = "ReduceSumSquare"
        elif producer_op_type == "Pow" and len(producer_inputs) >= 2:
            base, exponent = producer_inputs[:2]
            exponent_scalar = _const_scalar(exponent)
            if exponent_scalar is not None and np.allclose(exponent_scalar, 2):
                target_base = base
                op_name = "ReduceSumSquare"

        if op_name is None or target_base is None:
            lower_reduction(ctx, eqn, op_type="ReduceSum", allow_dtype_param=True)
            return

        rank = len(tuple(getattr(operand_var.aval, "shape", ())))
        axes_norm: list[int] | None = None
        if axes is not None:
            axes_norm = []
            for ax in cast("tuple[Any, ...]", axes):
                ax_i = int(ax)
                if ax_i < 0:
                    ax_i += rank
                axes_norm.append(ax_i)

        inputs = [target_base]
        if axes_norm is not None:
            axes_const = _const_i64(ctx, list(axes_norm), f"{op_name.lower()}_axes")
            inputs.append(axes_const)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(op_name)
        out_producer = getattr(out_spec, "producer", lambda: None)
        if callable(out_producer) and out_producer() is not None:
            desired_name = ctx.fresh_name(op_name)

        if op_name == "ReduceL1":
            result = ctx.builder.ReduceL1(
                *inputs,
                keepdims=1 if keepdims else 0,
                _outputs=[desired_name],
            )
        else:
            result = ctx.builder.ReduceSumSquare(
                *inputs,
                keepdims=1 if keepdims else 0,
                _outputs=[desired_name],
            )

        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
