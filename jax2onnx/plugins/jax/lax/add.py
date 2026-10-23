# jax2onnx/plugins/jax/lax/add.py

from typing import Any, Optional, cast

import onnx_ir as ir
import jax
import numpy as np
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    _Site,
    _tensor,
    _validate_complex_add,
    _validate_equation,
)
from jax2onnx.converter.source_native_contract import validate_native_results
from jax2onnx.converter.source_representation import representation_for_aval
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._axis0_utils import (
    maybe_expand_binary_axis0,
    stamp_axis0_binary_result,
)
from jax2onnx.plugins._loop_extent_meta import (
    propagate_axis0_override,
    set_axis0_override,
)
from jax2onnx.plugins._complex_utils import (
    COMPLEX_DTYPES,
    cast_real_tensor,
    ensure_packed_real_pair,
    is_packed_complex_tensor,
    resolve_common_real_dtype,
    coerce_dim_values,
    _shape_tuple,
)
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._precision_utils import promote_float32_operand
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


def _lower_complex_add(
    ctx: LoweringContextProtocol,
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    output_name: str,
) -> tuple[ir.Value, ir.DataType]:
    lhs_packed, lhs_dtype = ensure_packed_real_pair(ctx, lhs, name_hint="add_lhs")
    rhs_packed, rhs_dtype = ensure_packed_real_pair(ctx, rhs, name_hint="add_rhs")

    target_dtype = resolve_common_real_dtype(lhs_dtype, rhs_dtype)
    lhs_ready = (
        lhs_packed
        if lhs_packed.dtype == target_dtype
        else cast_real_tensor(ctx, lhs_packed, target_dtype, name_hint="add_lhs_cast")
    )
    rhs_ready = (
        rhs_packed
        if rhs_packed.dtype == target_dtype
        else cast_real_tensor(ctx, rhs_packed, target_dtype, name_hint="add_rhs_cast")
    )

    result = cast(
        ir.Value,
        ctx.builder.Add(
            lhs_ready,
            rhs_ready,
            _outputs=[output_name],
        ),
    )
    result.type = ir.TensorType(target_dtype)
    result.dtype = target_dtype
    dims = coerce_dim_values(_shape_tuple(lhs_ready))
    _stamp_type_and_shape(result, dims)
    _ensure_value_metadata(ctx, result)
    return result, target_dtype


def _lower_source_complex_add(ctx: LoweringContextProtocol, eqn: Any) -> None:
    site = _Site("complex Add lowering", ctx.builder.opset, eqn.primitive.name)
    _validate_complex_add(
        eqn,
        [_tensor(var, site) for var in eqn.invars],
        [_tensor(var, site) for var in eqn.outvars],
        site,
    )
    operands = [ctx.require_value_for_var(var) for var in eqn.invars]
    for var, value in zip(eqn.invars, operands, strict=True):
        if not representation_for_aval(var.aval).matches_ir_value(value):
            raise TypeError(
                "SOURCE complex Add physical storage differs from its declared real/imaginary pair"
            )
    out_var = eqn.outvars[0]
    representation = representation_for_aval(out_var.aval)
    output = ctx.allocate_value_for_var(
        out_var, name_hint=ctx.fresh_name("source_complex_add")
    )
    result, dtype = _lower_complex_add(
        ctx, *operands, output_name=output.name or ctx.fresh_name("source_complex_add")
    )
    if dtype != representation.ir_dtype:
        raise TypeError("SOURCE complex Add changed its declared component width")
    # The ordinary helper follows the left operand, while this native Add may
    # broadcast. Derive the complete pair result from its proved declaration;
    # logical-axis stamping would discard the physical channel dimension.
    for value in (output, result):
        value.type = ir.TensorType(representation.ir_dtype)
        _stamp_type_and_shape(value, coerce_dim_values(representation.physical_shape))
        _ensure_value_metadata(ctx, value)
    ctx.bind_value_for_var(out_var, result)


def lower_add(ctx: LoweringContextProtocol, eqn: Any) -> None:
    """Shared lowering routine for lax/jnp Add plugins."""

    if ctx.precision_policy is PrecisionPolicy.SOURCE and any(
        np.dtype(var.aval.dtype).name in {"complex64", "complex128"}
        for var in (*eqn.invars, *eqn.outvars)
    ):
        _lower_source_complex_add(ctx, eqn)
        return

    if (
        ctx.precision_policy is PrecisionPolicy.SOURCE
        and _lower_source_static_empty_add(ctx, eqn)
    ):
        return

    x_var, y_var = eqn.invars
    out_var = eqn.outvars[0]

    prefer_dt: Optional[np.dtype] = np.dtype(getattr(x_var.aval, "dtype", np.float32))

    a_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("add_lhs"))
    b_val = ctx.get_value_for_var(
        y_var, name_hint=ctx.fresh_name("add_rhs"), prefer_np_dtype=prefer_dt
    )
    out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("add_out"))
    a_val, b_val, override = maybe_expand_binary_axis0(
        ctx, a_val, b_val, out_spec, out_var
    )
    output_name = out_spec.name or ctx.fresh_name("add_out")
    ir.convenience.rename_values(out_spec, output_name)

    def _is_complex_var(var: object) -> bool:
        aval_dtype = getattr(getattr(var, "aval", None), "dtype", None)
        if aval_dtype is None:
            return False
        try:
            return np.issubdtype(np.dtype(aval_dtype), np.complexfloating)
        except TypeError:
            return False

    complex_var_hint = (
        _is_complex_var(out_var) or _is_complex_var(x_var) or _is_complex_var(y_var)
    )
    complex_dtype_hint = a_val.dtype in COMPLEX_DTYPES or b_val.dtype in COMPLEX_DTYPES
    packed_hint = False
    if complex_var_hint or complex_dtype_hint:
        packed_hint = is_packed_complex_tensor(a_val) or is_packed_complex_tensor(b_val)
    complex_route = complex_var_hint or complex_dtype_hint or packed_hint

    if complex_route:
        result, result_dtype = _lower_complex_add(
            ctx,
            a_val,
            b_val,
            output_name=output_name,
        )
        out_spec.type = ir.TensorType(result_dtype)
        out_spec.dtype = result_dtype
        if getattr(result, "shape", None) is not None:
            out_spec.shape = result.shape
        stamp_axis0_binary_result(result, out_var, out_spec, override)
        if override is not None:
            set_axis0_override(result, override)
        propagate_axis0_override(a_val, result)
        propagate_axis0_override(b_val, result)
        ctx.bind_value_for_var(out_var, result)
        try:
            outputs = ctx.builder.outputs
        except AttributeError:
            outputs = []
        result_name = getattr(result, "name", None)
        for out_val in outputs:
            if out_val is result or getattr(out_val, "name", None) == result_name:
                out_val.type = ir.TensorType(result_dtype)
                out_val.dtype = result_dtype
                if getattr(result, "shape", None) is not None:
                    out_val.shape = result.shape
                _ensure_value_metadata(ctx, out_val)
                break
        return

    a_val = promote_float32_operand(
        ctx, a_val, target_dtype=out_spec.dtype, name_hint="add_lhs_double"
    )
    b_val = promote_float32_operand(
        ctx, b_val, target_dtype=out_spec.dtype, name_hint="add_rhs_double"
    )
    result = cast(
        ir.Value,
        ctx.builder.Add(a_val, b_val, _outputs=[output_name]),
    )
    if getattr(out_spec, "type", None) is not None:
        result.type = out_spec.type
    stamp_axis0_binary_result(result, out_var, out_spec, override)
    if override is not None:
        set_axis0_override(result, override)
    propagate_axis0_override(a_val, result)
    propagate_axis0_override(b_val, result)
    ctx.bind_value_for_var(out_var, result)


def _lower_source_static_empty_add(ctx: LoweringContextProtocol, eqn: Any) -> bool:
    """Preserve a proved static empty result without target empty-Add folding.

    This representation changes no input binding, port, stored payload or native
    equation. Symbolic zero tails remain under their existing owner gates.
    """
    if len(eqn.outvars) != 1:
        return False
    shape = eqn.outvars[0].aval.shape
    if not (
        all(
            isinstance(dim, (int, np.integer)) and not isinstance(dim, (bool, np.bool_))
            for dim in shape
        )
        and 0 in shape
    ):
        return False
    site = _Site("static empty Add lowering", ctx.builder.opset, eqn.primitive.name)
    # Reuse the existing owner admission before the exact original native rule;
    # the frozen evaluator does not itself grant new dtype or shape support.
    _validate_equation(eqn, site, set(), frozenset())
    validate_native_results(eqn, site, primitive=jax.lax.add_p, parameters=set())
    operands = [ctx.require_value_for_var(var) for var in eqn.invars]
    for var, value in zip(eqn.invars, operands, strict=True):
        if not representation_for_aval(var.aval).matches_ir_value(value):
            raise ValueError(
                "SOURCE empty Add physical operand dtype/shape differs from aval"
            )
    output_representation = representation_for_aval(eqn.outvars[0].aval)
    selected = next(
        (value for value in operands if output_representation.matches_ir_value(value)),
        None,
    )
    output = ctx.allocate_value_for_var(eqn.outvars[0], name_hint="source_empty_add")
    if not output_representation.matches_ir_value(output):
        raise ValueError(
            "SOURCE empty Add physical result dtype/shape differs from aval"
        )
    if selected is not None:
        result = ctx.builder.op("Identity", [selected], output=output)
    else:
        # Complementary singleton broadcasts may have no input of the complete
        # output shape. There are still no elements to compute or approximate.
        extent = _const_i64(ctx, np.asarray(shape, dtype=np.int64), "empty_add_shape")
        zero = ir.tensor(np.zeros((1,), dtype=eqn.outvars[0].aval.dtype))
        result = ctx.builder.op(
            "ConstantOfShape",
            [extent],
            {"value": ir.AttrTensor("value", zero)},
            output=output,
        )
    ctx.bind_value_for_var(eqn.outvars[0], result)
    return True


@register_native_primitive(
    jaxpr_primitive=jax.lax.add_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.add.html",
    onnx=[
        {
            "component": "Add",
            "doc": "https://onnx.ai/onnx/operators/onnx__Add.html",
        }
    ],
    since="0.2.0",
    context="primitives.lax",
    component="add",
    testcases=[
        {
            "testcase": "add",
            "callable": lambda x1, x2: x1 + x2,
            "input_shapes": [(3,), (3,)],
            "post_check_onnx_graph": EG(
                ["Add:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "add_const",
            "callable": lambda x: x + 1.0,
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Add:3",
                        "inputs": {1: {"const": 1.0}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "add_complex64",
            "callable": lambda x, y: x + y,
            "input_values": [
                np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64),
                np.array([-1.5 + 0.5j, 2.0 + 1.25j], dtype=np.complex64),
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                ["Add:2x2"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class AddPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        lower_add(ctx, eqn)
