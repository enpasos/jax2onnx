# jax2onnx/plugins/jax/numpy/linspace.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import core
from jax.interpreters import batching

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.lax._index_utils import _scalar_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_LINSPACE_PRIM: Final = make_jnp_primitive("jax.numpy.linspace")


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
            ir.DataType.DOUBLE: np.float64,
            ir.DataType.INT32: np.int32,
            ir.DataType.INT64: np.int64,
            ir.DataType.INT16: np.int16,
            ir.DataType.INT8: np.int8,
            ir.DataType.UINT8: np.uint8,
            ir.DataType.BOOL: np.bool_,
        }
        target_dtype = enum_to_np.get(target_enum)
        if target_dtype is not None and np_arr.dtype != target_dtype:
            new_val = ctx.bind_const_for_var(
                object(), np_arr.astype(target_dtype, copy=False)
            )
            _stamp_type_and_shape(new_val, tuple(int(d) for d in np_arr.shape))
            _ensure_value_metadata(ctx, new_val)
            return new_val
    cur_type = getattr(value, "type", None)
    cur_dtype = getattr(cur_type, "dtype", None)
    if cur_dtype == target_enum:
        return value
    cast_name = ctx.fresh_name(name_hint)
    cast_val = ctx.builder.Cast(
        value,
        _outputs=[cast_name],
        to=int(target_enum.value),
    )
    cast_val.type = ir.TensorType(target_enum)
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
    _ensure_value_metadata(ctx, cast_val)
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
    since="v0.5.2",
    context="primitives.jnp",
    component="linspace",
    testcases=[
        {
            "testcase": "linspace_static_basic",
            "callable": lambda: jnp.linspace(0.0, 5.0, num=6),
            "input_values": [],
            "expected_output_shapes": [(6,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "Range:6 -> Cast:6 -> Mul:6 -> Add:6",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linspace_static_endpoint_false",
            "callable": lambda: jnp.linspace(0.0, 1.0, num=5, endpoint=False),
            "input_values": [],
            "expected_output_shapes": [(5,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "Range:5 -> Cast:5 -> Mul:5 -> Add:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linspace_static_int_inputs_default_dtype",
            "callable": lambda: jnp.linspace(0, 4, num=5),
            "input_values": [],
            "expected_output_shapes": [(5,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "Range:5 -> Cast:5 -> Mul:5 -> Add:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linspace_basic_f32",
            "callable": lambda: jnp.linspace(0.0, 10.0, num=5, dtype=jnp.float32),
            "input_values": [],
            "expected_output_shapes": [(5,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "Range:5 -> Cast:5 -> Mul:5 -> Add:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linspace_endpoint_false_i32",
            "callable": lambda: jnp.linspace(
                0, 4, num=4, endpoint=False, dtype=jnp.int32
            ),
            "input_values": [],
            "expected_output_shapes": [(4,)],
            "post_check_onnx_graph": EG(
                ["Range:4 -> Cast:4 -> Mul:4 -> Add:4 -> Cast:4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linspace_num_zero",
            "callable": lambda: jnp.linspace(0.0, 10.0, num=0),
            "input_values": [],
            "expected_output_shapes": [(0,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.0}},
                        "path": "Range:0 -> Cast:0 -> Mul:0 -> Add:0",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linspace_num_one",
            "callable": lambda: jnp.linspace(3.0, 10.0, num=1, dtype=jnp.float64),
            "input_values": [],
            "expected_output_shapes": [(1,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 3.0}},
                        "path": "Range:1 -> Cast:1 -> Mul:1 -> Add:1",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linspace_static_num_0",
            "callable": lambda: jnp.linspace(1.0, 2.0, num=0),
            "input_values": [],
            "expected_output_shapes": [(0,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 1.0}},
                        "path": "Range:0 -> Cast:0 -> Mul:0 -> Add:0",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linspace_static_num_1",
            "callable": lambda: jnp.linspace(7.0, 9.0, num=1),
            "input_values": [],
            "expected_output_shapes": [(1,)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 7.0}},
                        "path": "Range:1 -> Cast:1 -> Mul:1 -> Add:1",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "linspace_vmap_batching",
            "callable": lambda x: jax.vmap(lambda t: jnp.linspace(0.0, t, num=4))(x),
            "input_shapes": [(3,)],
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
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for linspace lowering"
            )

        start_val = _maybe_cast(ctx, start_val, work_enum, "linspace_start_cast")
        stop_val = _maybe_cast(ctx, stop_val, work_enum, "linspace_stop_cast")
        _stamp_type_and_shape(start_val, ())
        _stamp_type_and_shape(stop_val, ())
        _ensure_value_metadata(ctx, start_val)
        _ensure_value_metadata(ctx, stop_val)

        idx_start = _scalar_i64(ctx, 0, "linspace_idx_start")
        idx_limit = _scalar_i64(ctx, num, "linspace_idx_limit")
        idx_delta = _scalar_i64(ctx, 1, "linspace_idx_step")
        for scalar_val in (idx_start, idx_limit, idx_delta):
            _stamp_type_and_shape(scalar_val, ())
            _ensure_value_metadata(ctx, scalar_val)
        indices = builder.Range(
            idx_start,
            idx_limit,
            idx_delta,
            _outputs=[ctx.fresh_name("linspace_indices")],
        )
        indices.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(indices, (num,))
        _ensure_value_metadata(ctx, indices)

        indices_cast = builder.Cast(
            indices,
            _outputs=[ctx.fresh_name("linspace_indices_cast")],
            to=int(work_enum.value),
        )
        indices_cast.type = ir.TensorType(work_enum)
        _stamp_type_and_shape(indices_cast, (num,))
        _ensure_value_metadata(ctx, indices_cast)

        diff = builder.Sub(
            stop_val,
            start_val,
            _outputs=[ctx.fresh_name("linspace_diff")],
        )
        diff.type = ir.TensorType(work_enum)
        _stamp_type_and_shape(diff, ())
        _ensure_value_metadata(ctx, diff)

        denom = num - 1 if endpoint else num
        if denom <= 0:
            denom = 1
        denom_i64 = _scalar_i64(ctx, denom, "linspace_denom_i64")
        _stamp_type_and_shape(denom_i64, ())
        _ensure_value_metadata(ctx, denom_i64)
        denom_val = builder.CastLike(
            denom_i64,
            indices_cast,
            _outputs=[ctx.fresh_name("linspace_denom")],
        )
        denom_val.type = indices_cast.type
        _stamp_type_and_shape(denom_val, ())
        _ensure_value_metadata(ctx, denom_val)

        step = builder.Div(
            diff,
            denom_val,
            _outputs=[ctx.fresh_name("linspace_step")],
        )
        step.type = ir.TensorType(work_enum)
        _stamp_type_and_shape(step, ())
        _ensure_value_metadata(ctx, step)

        scaled = builder.Mul(
            indices_cast,
            step,
            _outputs=[ctx.fresh_name("linspace_scaled")],
        )
        scaled.type = ir.TensorType(work_enum)
        _stamp_type_and_shape(scaled, (num,))
        _ensure_value_metadata(ctx, scaled)

        linspace_vals = builder.Add(
            scaled,
            start_val,
            _outputs=[ctx.fresh_name("linspace_vals")],
        )
        linspace_vals.type = ir.TensorType(work_enum)
        _stamp_type_and_shape(linspace_vals, (num,))
        _ensure_value_metadata(ctx, linspace_vals)

        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("linspace_out"), prefer_np_dtype=out_dtype
        )
        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("linspace_out")
        if target_enum != work_enum:
            final_val = builder.Cast(
                linspace_vals,
                _outputs=[ctx.fresh_name("linspace_out_cast")],
                to=int(target_enum.value),
            )
            final_val.type = ir.TensorType(target_enum)
            _stamp_type_and_shape(final_val, (num,))
            _ensure_value_metadata(ctx, final_val)
        else:
            final_val = linspace_vals
        final_val.name = out_name
        _stamp_type_and_shape(final_val, (num,))
        _ensure_value_metadata(ctx, final_val)
        ctx.bind_value_for_var(out_var, final_val)

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


BatchDim = int | type(batching.not_mapped)


def _linspace_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype=None,
    axis: int = 0,
) -> tuple[jax.Array, BatchDim]:
    start, stop = batched_args
    start_bdim, stop_bdim = batch_dims
    mapped = [
        (arg, bd)
        for arg, bd in zip(batched_args, batch_dims)
        if bd is not batching.not_mapped
    ]
    if not mapped:
        out = JnpLinspacePlugin._PRIM.bind(
            start,
            stop,
            num=num,
            endpoint=endpoint,
            retstep=retstep,
            dtype=dtype,
            axis=axis,
        )
        return out, batching.not_mapped

    sample_arg, sample_bd = mapped[0]
    batch_size = sample_arg.shape[sample_bd]

    if start_bdim is not batching.not_mapped:
        start = batching.bdim_at_front(start, start_bdim, batch_size)
    if stop_bdim is not batching.not_mapped:
        stop = batching.bdim_at_front(stop, stop_bdim, batch_size)

    in_axes = (
        0 if start_bdim is not batching.not_mapped else None,
        0 if stop_bdim is not batching.not_mapped else None,
    )
    orig = get_orig_impl(JnpLinspacePlugin._PRIM, JnpLinspacePlugin._FUNC_NAME)

    def _call_single(s, t):
        return orig(
            s,
            t,
            num=num,
            endpoint=endpoint,
            retstep=retstep,
            dtype=dtype,
            axis=axis,
        )

    result = jax.vmap(_call_single, in_axes=in_axes)(start, stop)
    return result, 0


batching.primitive_batchers[JnpLinspacePlugin._PRIM] = _linspace_batch_rule
