from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType
from jax import core

from jax2onnx.converter2.ir_builder import _dtype_to_ir
from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.jax.lax._index_utils import _scalar_i64
from jax2onnx.plugins2.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_LINSPACE_PRIM = make_jnp_primitive("jax.numpy.linspace")


def _normalize_dtype(result_dtype: np.dtype, enable_double: bool) -> np.dtype:
    if np.issubdtype(result_dtype, np.floating):
        return result_dtype
    return np.dtype(np.float64 if enable_double else np.float32)


def _infer_result_dtype(
    start_aval: core.AbstractValue,
    stop_aval: core.AbstractValue,
    dtype_param,
    enable_double: bool,
) -> np.dtype:
    if dtype_param is not None:
        return np.dtype(dtype_param)

    cand = []
    for aval in (start_aval, stop_aval):
        aval_dtype = getattr(aval, "dtype", None)
        if aval_dtype is not None:
            cand.append(np.dtype(aval_dtype))
    if cand:
        result = np.result_type(*cand)
    else:
        result = np.dtype(np.float64 if enable_double else np.float32)

    if not np.issubdtype(result, np.floating):
        result = np.dtype(np.float64 if enable_double else np.float32)
    return result


def _maybe_cast(
    ctx: "IRContext",
    value: ir.Value,
    target_enum: ir.DataType,
    name_hint: str,
) -> ir.Value:
    const_arr = getattr(value, "const_value", None)
    if const_arr is not None:
        np_arr = np.asarray(const_arr)
        enum_to_np = {
            ir.DataType.FLOAT: np.float32,
            ir.DataType.INT32: np.int32,
            ir.DataType.INT64: np.int64,
            ir.DataType.INT16: np.int16,
            ir.DataType.INT8: np.int8,
            ir.DataType.BOOL: np.bool_,
        }
        double_enum = getattr(ir.DataType, "DOUBLE", None)
        if double_enum is not None:
            enum_to_np[double_enum] = np.float64
        uint8_enum = getattr(ir.DataType, "UINT8", None)
        if uint8_enum is not None:
            enum_to_np[uint8_enum] = np.uint8
        target_dtype = enum_to_np.get(target_enum)
        if target_dtype is not None and np_arr.dtype != target_dtype:
            new_val = ctx.bind_const_for_var(
                object(), np_arr.astype(target_dtype, copy=False)
            )
            _stamp_type_and_shape(new_val, tuple(int(d) for d in np_arr.shape))
            _ensure_value_info(ctx, new_val)
            return new_val
    cur_type = getattr(value, "type", None)
    cur_dtype = getattr(cur_type, "dtype", None)
    if cur_dtype == target_enum:
        return value
    cast_val = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(target_enum),
        shape=value.shape,
    )
    ctx.add_node(
        ir.Node(
            op_type="Cast",
            domain="",
            inputs=[value],
            outputs=[cast_val],
            name=ctx.fresh_name("Cast"),
            attributes=[IRAttr("to", IRAttrType.INT, int(target_enum.value))],
        )
    )
    shape_obj = getattr(value, "shape", None)
    dims = None
    if shape_obj is not None:
        dims = getattr(shape_obj, "dims", None)
        if dims is None:
            try:
                dims = tuple(shape_obj)
            except TypeError:
                dims = None
    _stamp_type_and_shape(cast_val, tuple(dims) if dims is not None else ())
    _ensure_value_info(ctx, cast_val)
    return cast_val


@register_primitive(
    jaxpr_primitive=_LINSPACE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linspace.html",
    onnx=[
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        },
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    ],
    since="v0.9.0",
    context="primitives2.jnp",
    component="linspace",
    testcases=[
        {
            "testcase": "linspace_basic_f32",
            "callable": lambda: jnp.linspace(0.0, 10.0, num=5, dtype=jnp.float32),
            "input_values": [],
            "expected_output_shapes": [(5,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "linspace_endpoint_false_i32",
            "callable": lambda: jnp.linspace(
                0, 4, num=4, endpoint=False, dtype=jnp.int32
            ),
            "input_values": [],
            "expected_output_shapes": [(4,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "linspace_num_zero",
            "callable": lambda: jnp.linspace(0.0, 10.0, num=0),
            "input_values": [],
            "expected_output_shapes": [(0,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "linspace_num_one",
            "callable": lambda: jnp.linspace(3.0, 10.0, num=1, dtype=jnp.float64),
            "input_values": [],
            "expected_output_shapes": [(1,)],
            "use_onnx_ir": True,
        },
    ],
)
class JnpLinspacePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _LINSPACE_PRIM
    _FUNC_NAME: ClassVar[str] = "linspace"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        start,
        stop,
        *,
        num: int = 50,
        endpoint: bool = True,
        retstep: bool = False,
        dtype=None,
        axis: int = 0,
    ):
        if retstep:
            raise NotImplementedError("jnp.linspace with retstep=True is not supported")
        if axis != 0:
            raise NotImplementedError("jnp.linspace currently supports axis=0 only")
        num_int = int(num)
        if num_int < 0:
            raise ValueError("num must be non-negative")
        enable_x64 = bool(jax.config.jax_enable_x64)
        result_dtype = _infer_result_dtype(start, stop, dtype, enable_x64)
        return core.ShapedArray((num_int,), result_dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        params = getattr(eqn, "params", {})
        num = int(params.get("num", 50))
        if num < 0:
            raise ValueError("jnp.linspace lowering received negative num")
        endpoint = bool(params.get("endpoint", True))
        retstep = bool(params.get("retstep", False))
        if retstep:
            raise NotImplementedError("jnp.linspace with retstep=True is not supported")
        axis = int(params.get("axis", 0))
        if axis != 0:
            raise NotImplementedError("jnp.linspace currently supports axis=0 only")

        start_var, stop_var = eqn.invars
        start_shape = tuple(getattr(start_var.aval, "shape", ()))
        stop_shape = tuple(getattr(stop_var.aval, "shape", ()))
        if start_shape != () or stop_shape != ():
            raise NotImplementedError(
                "jnp.linspace supports scalar start/stop only in IR path"
            )

        out_var = eqn.outvars[0]
        out_dtype = np.dtype(getattr(out_var.aval, "dtype", np.float32))
        target_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)

        work_dtype = _normalize_dtype(out_dtype, ctx.builder.enable_double_precision)
        work_enum = _dtype_to_ir(work_dtype, ctx.builder.enable_double_precision)

        start_val = ctx.get_value_for_var(
            start_var,
            name_hint=ctx.fresh_name("linspace_start"),
            prefer_np_dtype=work_dtype,
        )
        stop_val = ctx.get_value_for_var(
            stop_var,
            name_hint=ctx.fresh_name("linspace_stop"),
            prefer_np_dtype=work_dtype,
        )

        start_val = _maybe_cast(ctx, start_val, work_enum, "linspace_start_cast")
        stop_val = _maybe_cast(ctx, stop_val, work_enum, "linspace_stop_cast")

        idx_start = _scalar_i64(ctx, 0, "linspace_idx_start")
        idx_limit = _scalar_i64(ctx, num, "linspace_idx_limit")
        idx_delta = _scalar_i64(ctx, 1, "linspace_idx_step")
        indices = ir.Value(
            name=ctx.fresh_name("linspace_indices"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((num,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Range",
                domain="",
                inputs=[idx_start, idx_limit, idx_delta],
                outputs=[indices],
                name=ctx.fresh_name("Range"),
            )
        )
        _stamp_type_and_shape(indices, (num,))
        _ensure_value_info(ctx, indices)

        indices_cast = ir.Value(
            name=ctx.fresh_name("linspace_indices_cast"),
            type=ir.TensorType(work_enum),
            shape=indices.shape,
        )
        ctx.add_node(
            ir.Node(
                op_type="Cast",
                domain="",
                inputs=[indices],
                outputs=[indices_cast],
                name=ctx.fresh_name("Cast"),
                attributes=[IRAttr("to", IRAttrType.INT, int(work_enum.value))],
            )
        )
        _stamp_type_and_shape(indices_cast, (num,))
        _ensure_value_info(ctx, indices_cast)

        diff = ir.Value(
            name=ctx.fresh_name("linspace_diff"),
            type=ir.TensorType(work_enum),
            shape=start_val.shape,
        )
        ctx.add_node(
            ir.Node(
                op_type="Sub",
                domain="",
                inputs=[stop_val, start_val],
                outputs=[diff],
                name=ctx.fresh_name("Sub"),
            )
        )
        _stamp_type_and_shape(diff, ())
        _ensure_value_info(ctx, diff)

        denom = num - 1 if endpoint else num
        if denom <= 0:
            denom = 1
        denom_i64 = _scalar_i64(ctx, denom, "linspace_denom_i64")
        denom_val = ctx.cast_like(denom_i64, indices_cast, name_hint="linspace_denom")

        step = ir.Value(
            name=ctx.fresh_name("linspace_step"),
            type=ir.TensorType(work_enum),
            shape=diff.shape,
        )
        ctx.add_node(
            ir.Node(
                op_type="Div",
                domain="",
                inputs=[diff, denom_val],
                outputs=[step],
                name=ctx.fresh_name("Div"),
            )
        )
        _stamp_type_and_shape(step, ())
        _ensure_value_info(ctx, step)

        scaled = ir.Value(
            name=ctx.fresh_name("linspace_scaled"),
            type=ir.TensorType(work_enum),
            shape=indices_cast.shape,
        )
        ctx.add_node(
            ir.Node(
                op_type="Mul",
                domain="",
                inputs=[indices_cast, step],
                outputs=[scaled],
                name=ctx.fresh_name("Mul"),
            )
        )
        _stamp_type_and_shape(scaled, (num,))
        _ensure_value_info(ctx, scaled)

        linspace_vals = ir.Value(
            name=ctx.fresh_name("linspace_vals"),
            type=ir.TensorType(work_enum),
            shape=scaled.shape,
        )
        ctx.add_node(
            ir.Node(
                op_type="Add",
                domain="",
                inputs=[scaled, start_val],
                outputs=[linspace_vals],
                name=ctx.fresh_name("Add"),
            )
        )
        _stamp_type_and_shape(linspace_vals, (num,))
        _ensure_value_info(ctx, linspace_vals)

        out_val = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("linspace_out"), prefer_np_dtype=out_dtype
        )
        if target_enum != work_enum:
            cast_out = ir.Value(
                name=ctx.fresh_name("linspace_out_cast"),
                type=ir.TensorType(target_enum),
                shape=linspace_vals.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Cast",
                    domain="",
                    inputs=[linspace_vals],
                    outputs=[cast_out],
                    name=ctx.fresh_name("Cast"),
                    attributes=[IRAttr("to", IRAttrType.INT, int(target_enum.value))],
                )
            )
            _stamp_type_and_shape(cast_out, (num,))
            _ensure_value_info(ctx, cast_out)
            final_val = cast_out
        else:
            final_val = linspace_vals

        ctx.add_node(
            ir.Node(
                op_type="Identity",
                domain="",
                inputs=[final_val],
                outputs=[out_val],
                name=ctx.fresh_name("Identity"),
            )
        )
        _stamp_type_and_shape(out_val, (num,))
        out_val.type = ir.TensorType(target_enum)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.linspace not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                start,
                stop,
                num=50,
                *,
                endpoint=True,
                retstep=False,
                dtype=None,
                axis=0,
            ):
                if retstep:
                    return orig(
                        start,
                        stop,
                        num=num,
                        endpoint=endpoint,
                        retstep=retstep,
                        dtype=dtype,
                        axis=axis,
                    )
                axis_int = int(axis)
                if axis_int != 0:
                    return orig(
                        start,
                        stop,
                        num=num,
                        endpoint=endpoint,
                        retstep=retstep,
                        dtype=dtype,
                        axis=axis,
                    )
                num_int = int(num)
                if num_int < 0:
                    raise ValueError("num must be non-negative")
                start_arr = jnp.asarray(start)
                stop_arr = jnp.asarray(stop)
                if start_arr.ndim != 0 or stop_arr.ndim != 0:
                    return orig(
                        start,
                        stop,
                        num=num,
                        endpoint=endpoint,
                        retstep=retstep,
                        dtype=dtype,
                        axis=axis,
                    )
                return cls._PRIM.bind(
                    start_arr,
                    stop_arr,
                    num=num_int,
                    endpoint=bool(endpoint),
                    retstep=False,
                    dtype=dtype,
                    axis=axis_int,
                )

            return _patched

        return [
            AssignSpec(
                "jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpLinspacePlugin._PRIM.def_impl
def _linspace_impl(
    start, stop, *_, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    orig = get_orig_impl(JnpLinspacePlugin._PRIM, JnpLinspacePlugin._FUNC_NAME)
    return orig(
        start,
        stop,
        num=num,
        endpoint=endpoint,
        retstep=retstep,
        dtype=dtype,
        axis=axis,
    )


JnpLinspacePlugin._PRIM.def_abstract_eval(JnpLinspacePlugin.abstract_eval)
