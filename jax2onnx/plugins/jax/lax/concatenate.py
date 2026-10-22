# jax2onnx/plugins/jax/lax/concatenate.py

from __future__ import annotations

from typing import Any, Sequence, cast

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import _Site, _tensor, _validate_tensor_layout
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx._compat.jax import is_unpartitioned_prng_placement
from jax2onnx.plugins._complex_utils import coerce_dim_values
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._precision_utils import promote_float32_operand
from jax2onnx.plugins._loop_extent_meta import (
    get_axis0_override,
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


def _normalize_axis(axis: int, rank: int) -> int:
    if rank == 0:
        return 0
    ax = int(axis)
    return ax % rank if ax < 0 else ax


def _promote_dtype(dtypes: Sequence[np.dtype]) -> np.dtype:
    result = dtypes[0]
    for dt in dtypes[1:]:
        result = np.promote_types(result, dt)
    return result


def _cast_value(
    ctx: LoweringContextProtocol,
    value: ir.Value,
    target: ir.DataType,
    shape: tuple[int | str, ...],
) -> ir.Value:
    current = getattr(getattr(value, "type", None), "dtype", None)
    if current == target:
        return value
    cast_val = cast(
        ir.Value,
        ctx.builder.Cast(
            value,
            _outputs=[ctx.fresh_name("concat_cast")],
            to=int(target.value),
        ),
    )
    cast_val.type = ir.TensorType(target)
    _stamp_type_and_shape(cast_val, shape)
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


@register_native_primitive(
    jaxpr_primitive=jax.lax.concatenate_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
    ],
    since="0.2.0",
    context="primitives.lax",
    component="concatenate",
    testcases=[
        {
            "testcase": "concatenate",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=0),
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Concat:6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_axis1",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=1),
            "input_shapes": [("B", 3), ("B", 4)],
            "post_check_onnx_graph": EG(
                ["Concat:Bx7"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_axis0",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=0),
            "input_shapes": [(7, 3), (4, 3)],
            "post_check_onnx_graph": EG(
                ["Concat:11x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_3d",
            "callable": lambda a, b: jax.lax.concatenate((a, b), dimension=1),
            "input_shapes": [(2, 3, 4), (2, 5, 4)],
            "post_check_onnx_graph": EG(
                ["Concat:2x8x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "concatenate_internal_int32_then_cast_to_f32_zeroarg",
            "callable": (
                lambda: jax.lax.concatenate(
                    (
                        jax.numpy.array([1], dtype=jax.numpy.int32),
                        jax.numpy.array([2], dtype=jax.numpy.int32),
                    ),
                    dimension=0,
                ).astype(jax.numpy.float32)
            ),
            "expected_output_shapes": [(2,)],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Concat:2 -> Cast:2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class ConcatenatePlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        if ctx.precision_policy is PrecisionPolicy.SOURCE:
            self._lower_source_native(ctx, eqn)
            return
        params = getattr(eqn, "params", {})
        axis = int(params.get("dimension", 0))

        in_vars: list[Any] = list(eqn.invars)
        out_var = eqn.outvars[0]

        shapes = [tuple(getattr(v.aval, "shape", ())) for v in in_vars]
        dtypes: list[np.dtype[Any]] = [
            np.dtype(getattr(v.aval, "dtype", np.float32)) for v in in_vars
        ]
        target_dtype: np.dtype[Any] = np.dtype(
            getattr(out_var.aval, "dtype", np.float32)
        )

        rank = len(shapes[0]) if shapes else 0
        norm_axis = _normalize_axis(axis, rank)
        out_spec = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("concat_out")
        )
        target_enum = out_spec.dtype
        if target_enum is None:
            raise TypeError(f"Unsupported concatenate output dtype '{target_dtype}'")

        inputs: list[ir.Value] = []
        for var, shape, dtype in zip(in_vars, shapes, dtypes):
            val = ctx.get_value_for_var(var, name_hint=ctx.fresh_name("concat_in"))
            val = promote_float32_operand(
                ctx,
                val,
                target_dtype=target_enum,
                name_hint="concat_operand_double",
            )
            if np.dtype(dtype) != target_dtype:
                val = _cast_value(ctx, val, target_enum, shape)
            inputs.append(val)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Concat")
        producer = getattr(out_spec, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Concat")

        result = cast(
            ir.Value,
            ctx.builder.Concat(
                *inputs,
                axis=int(norm_axis),
                _outputs=[desired_name],
            ),
        )
        result.type = ir.TensorType(target_enum)
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        for inp in inputs:
            propagate_axis0_override(inp, result)
        if norm_axis == 0:
            out_axis0 = (
                int(out_shape[0])
                if out_shape
                and isinstance(out_shape[0], (int, np.integer))
                and int(out_shape[0]) >= 0
                else None
            )
            candidate_overrides: list[int] = []
            for inp in inputs:
                axis0_candidate = get_axis0_override(inp)
                if (
                    isinstance(axis0_candidate, (int, np.integer))
                    and int(axis0_candidate) > 1
                ):
                    candidate_overrides.append(int(axis0_candidate))
            override = max(candidate_overrides, default=None)
            if override is None and isinstance(out_axis0, int) and out_axis0 > 1:
                override = out_axis0
            if override is not None:
                set_axis0_override(result, override)
        ctx.bind_value_for_var(out_var, result)

    def _lower_source_native(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        site = _Site("native Concatenate emission", ctx.opset, eqn.primitive.name)
        site.require(
            eqn.primitive is jax.lax.concatenate_p and not eqn.effects,
            "Concatenate emission needs the pure canonical native primitive",
            "the existing native lax.concatenate owner without a synthetic jnp carrier",
            category="lowering-defect",
        )
        inputs = [_tensor(v, site) for v in eqn.invars]
        outputs = [_tensor(v, site) for v in eqn.outvars]
        _validate_tensor_layout(eqn, inputs, outputs, site)
        for variable in (*eqn.invars, *eqn.outvars):
            site.require(
                is_unpartitioned_prng_placement(
                    variable.aval.sharding, len(variable.aval.shape)
                ),
                "native Concatenate placement needs an exact adapter",
                "ordinary absent/empty unpartitioned native placement",
            )
        values = [ctx.require_value_for_var(v) for v in eqn.invars]
        for variable, value in zip(eqn.invars, values, strict=True):
            if not representation_for_aval(variable.aval).matches_ir_value(value):
                raise TypeError(
                    "SOURCE Concatenate physical input differs from its native declaration"
                )
        out_var = eqn.outvars[0]
        representation = representation_for_aval(out_var.aval)
        output = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("concat_out")
        )
        result = ctx.builder.op(
            "Concat",
            values,
            {"axis": eqn.params["dimension"]},
            output=output,
        )
        result.type = ir.TensorType(representation.ir_dtype)
        _stamp_type_and_shape(result, coerce_dim_values(representation.physical_shape))
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
