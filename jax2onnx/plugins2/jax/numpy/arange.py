from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType
from jax import core
from jax.extend.core import Literal as JaxLiteral

from jax2onnx.converter2.ir_builder import _dtype_to_ir
from jax2onnx.plugins2._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_DYNAMIC_DIM_LABEL = "JAX2ONNX_DYNAMIC_DIM_SENTINEL"


class _DynamicDimSentinel:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - simple helper
        return _DYNAMIC_DIM_LABEL

    __str__ = __repr__

    def __hash__(self) -> int:
        return hash(_DYNAMIC_DIM_LABEL)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _DynamicDimSentinel)

    def dimension_as_value(self):  # pragma: no cover - compatibility helper
        msg = "Dynamic dimension sentinel cannot be materialized as a value"
        raise TypeError(msg)


DATA_DEPENDENT_DYNAMIC_DIM = _DynamicDimSentinel()


_ARANGE_PRIM = make_jnp_primitive("jax.numpy.arange")


def _as_scalar(aval) -> float | int | None:
    """Best-effort extraction of a scalar literal from an abstract value."""
    if aval is None:
        return None
    val = getattr(aval, "val", None)
    if val is None:
        return None
    arr = np.asarray(val)
    if arr.shape:
        return None
    return arr.item()


def _all_scalars(avals: Sequence[core.AbstractValue]):
    scalars: list[float | int] = []
    for aval in avals:
        if isinstance(aval, JaxLiteral):
            arr = np.asarray(aval.val)
            if arr.shape:
                return None
            scalars.append(arr.item())
            continue
        scalar = _as_scalar(aval)
        if scalar is None:
            return None
        scalars.append(scalar)
    return scalars


def _normalize_args(values: Sequence[float | int]) -> tuple[float, float, float]:
    if len(values) == 1:
        return 0.0, float(values[0]), 1.0
    if len(values) == 2:
        return float(values[0]), float(values[1]), 1.0
    if len(values) == 3:
        start, stop, step = values
        return float(start), float(stop), float(step)
    raise TypeError(
        f"jnp.arange expects 1 to 3 positional arguments, got {len(values)}"
    )


def _determine_length(values: Sequence[float | int]) -> int | None:
    start, stop, step = _normalize_args(values)
    if step == 0:
        return None
    try:
        if len(values) == 1:
            arr = np.arange(stop)
        elif len(values) == 2:
            arr = np.arange(start, stop)
        else:
            arr = np.arange(start, stop, step)
    except Exception:
        return None
    return int(arr.size)


def _resolve_result_dtype(
    avals: Sequence[core.AbstractValue], dtype_param, enable_x64: bool
) -> np.dtype:
    if dtype_param is not None:
        requested = np.dtype(dtype_param)
        if np.issubdtype(requested, np.floating):
            return np.dtype(np.float64 if enable_x64 else np.float32)
        return requested

    float_detected = False
    for aval in avals:
        aval_dtype = getattr(aval, "dtype", None)
        if aval_dtype is not None and np.issubdtype(np.dtype(aval_dtype), np.floating):
            float_detected = True
            break
        scalar = _as_scalar(aval)
        if isinstance(scalar, (float, np.floating)):
            float_detected = True
            break
    if float_detected:
        return np.dtype(np.float64 if enable_x64 else np.float32)
    return np.dtype(np.int64 if enable_x64 else np.int32)


def _maybe_cast_value(
    ctx: IRContext,
    value: ir.Value,
    target_enum: ir.DataType,
    name_hint: str,
) -> ir.Value:
    current_type = getattr(value, "type", None)
    current_dtype = getattr(current_type, "dtype", None)
    if current_dtype == target_enum:
        return value
    cast = ir.Value(
        name=ctx.fresh_name(name_hint),
        type=ir.TensorType(target_enum),
        shape=value.shape,
    )
    ctx.add_node(
        ir.Node(
            op_type="Cast",
            domain="",
            inputs=[value],
            outputs=[cast],
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
    _stamp_type_and_shape(cast, tuple(dims) if dims is not None else ())
    _ensure_value_info(ctx, cast)
    return cast


def _with_ir_shape_dims(shape: Sequence[object]) -> tuple[object, ...]:
    return tuple(
        _DYNAMIC_DIM_LABEL if dim is DATA_DEPENDENT_DYNAMIC_DIM else dim
        for dim in shape
    )


@register_primitive(
    jaxpr_primitive=_ARANGE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.arange.html",
    onnx=[
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        }
    ],
    since="v0.9.0",
    context="primitives2.jnp",
    component="arange",
    testcases=[
        {
            "testcase": "arange_data_dependent_indices",
            "callable": lambda x: jnp.arange(x.shape[1]),
            "input_shapes": [(3, 10)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_stop_only_concrete_input_val",
            "callable": lambda stop: jnp.arange(stop, dtype=jnp.float32),
            "input_values": [np.array(5.0, dtype=np.float32)],
            "expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_start_stop_concrete_input_val",
            "callable": lambda start, stop: jnp.arange(start, stop, dtype=jnp.float32),
            "input_values": [
                np.array(2.0, dtype=np.float32),
                np.array(7.0, dtype=np.float32),
            ],
            "expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_start_stop_step_concrete_input_val",
            "callable": lambda start, stop, step: jnp.arange(
                start, stop, step, dtype=jnp.float32
            ),
            "input_values": [
                np.array(1.0, dtype=np.float32),
                np.array(7.0, dtype=np.float32),
                np.array(2.0, dtype=np.float32),
            ],
            "expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_float_concrete_input_val",
            "callable": lambda start, stop, step: jnp.arange(
                start, stop, step, dtype=jnp.float32
            ),
            "input_values": [
                np.array(1.0, dtype=np.float32),
                np.array(4.5, dtype=np.float32),
                np.array(0.5, dtype=np.float32),
            ],
            "expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_stop_only_int",
            "callable": lambda: jnp.arange(5),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_stop_only_float",
            "callable": lambda: jnp.arange(5.0),
            "input_values": [],
            "expected_output_shapes": [(5,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_start_stop_int",
            "callable": lambda: jnp.arange(2, 7),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_start_stop_step_int",
            "callable": lambda: jnp.arange(1, 10, 2),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_empty_result_pos_step",
            "callable": lambda: jnp.arange(5, 2, 1),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_empty_result_neg_step",
            "callable": lambda: jnp.arange(2, 5, -1),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_negative_step",
            "callable": lambda: jnp.arange(5, 0, -1),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_float_step_explicit_dtype",
            "callable": lambda: jnp.arange(1.0, 2.0, 0.25, dtype=jnp.float32),
            "input_values": [],
            "expected_output_shapes": [(4,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_float_step_inferred_dtype",
            "callable": lambda: jnp.arange(0.0, 1.0, 0.3),
            "input_values": [],
            "expected_output_shapes": [(4,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_stop_zero",
            "callable": lambda: jnp.arange(0),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_start_equals_stop",
            "callable": lambda: jnp.arange(5, 5, 1),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
        {
            "testcase": "arange_static_large_numbers_int",
            "callable": lambda: jnp.arange(1000, 1010, 1, dtype=jnp.int32),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "use_onnx_ir": True,
        },
    ],
)
class JnpArangePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ARANGE_PRIM
    _FUNC_NAME: ClassVar[str] = "arange"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(*in_avals, dtype=None):
        enable_x64 = bool(jax.config.jax_enable_x64)
        result_dtype = _resolve_result_dtype(in_avals, dtype, enable_x64)
        scalars = _all_scalars(in_avals)
        if scalars is None:
            shape = (DATA_DEPENDENT_DYNAMIC_DIM,)
        else:
            length = _determine_length(scalars)
            shape = DATA_DEPENDENT_DYNAMIC_DIM if length is None else int(length)
            shape = (shape,)
        return jax.core.ShapedArray(shape, result_dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        params = getattr(eqn, "params", {})
        dtype_param = params.get("dtype")
        avals: list[core.AbstractValue] = []
        for var in eqn.invars:
            if isinstance(var, JaxLiteral):
                avals.append(var.aval)
            else:
                avals.append(getattr(var, "aval", None))
        enable_x64 = bool(ctx.builder.enable_double_precision)
        result_dtype = _resolve_result_dtype(avals, dtype_param, enable_x64)
        target_enum = _dtype_to_ir(result_dtype, ctx.builder.enable_double_precision)

        literal_args: list[float | int] | None = []
        input_vals: list[ir.Value] = []
        for idx, var in enumerate(eqn.invars):
            val = ctx.get_value_for_var(
                var,
                name_hint=ctx.fresh_name("arange_arg"),
                prefer_np_dtype=result_dtype,
            )
            input_vals.append(
                _maybe_cast_value(ctx, val, target_enum, f"arange_cast_{idx}")
            )
            if literal_args is not None and isinstance(var, JaxLiteral):
                literal_args.append(np.asarray(var.val).item())
            else:
                literal_args = None

        def _const(value: float | int, tag: str) -> ir.Value:
            arr = np.asarray(value, dtype=result_dtype)
            return ctx.bind_const_for_var(object(), arr)

        if len(input_vals) == 1:
            start_val = _const(0, "start")
            limit_val = input_vals[0]
            delta_val = _const(1, "delta")
        elif len(input_vals) == 2:
            start_val, limit_val = input_vals
            delta_val = _const(1, "delta")
        elif len(input_vals) == 3:
            start_val, limit_val, delta_val = input_vals
        else:
            raise TypeError(
                f"jnp.arange lowering expects 1 to 3 inputs, got {len(input_vals)}"
            )

        start_val = _maybe_cast_value(ctx, start_val, target_enum, "arange_start")
        limit_val = _maybe_cast_value(ctx, limit_val, target_enum, "arange_limit")
        delta_val = _maybe_cast_value(ctx, delta_val, target_enum, "arange_delta")

        out_var = eqn.outvars[0]
        out_val = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("arange_out"),
            prefer_np_dtype=result_dtype,
        )
        out_val.type = ir.TensorType(target_enum)

        ctx.add_node(
            ir.Node(
                op_type="Range",
                domain="",
                inputs=[start_val, limit_val, delta_val],
                outputs=[out_val],
                name=ctx.fresh_name("Range"),
            )
        )

        length_hint = None
        if literal_args is not None:
            try:
                length_hint = _determine_length(literal_args)
            except Exception:
                length_hint = None

        if length_hint is not None:
            final_shape: tuple[object, ...] = (int(length_hint),)
        else:
            out_shape = tuple(getattr(out_var.aval, "shape", ()))
            final_shape = _with_ir_shape_dims(out_shape)

        _stamp_type_and_shape(out_val, final_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.arange not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(*args, **kwargs):
                dtype = kwargs.pop("dtype", None)
                if kwargs:
                    raise TypeError(
                        f"Unsupported keyword arguments for jnp.arange: {tuple(kwargs.keys())}"
                    )
                num_args = len(args)
                if not (1 <= num_args <= 3):
                    return orig(*args, dtype=dtype)
                return cls._PRIM.bind(*args[:num_args], dtype=dtype)

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


@JnpArangePlugin._PRIM.def_impl
def _arange_impl(*args, dtype=None):
    orig = get_orig_impl(JnpArangePlugin._PRIM, JnpArangePlugin._FUNC_NAME)
    return orig(*args, dtype=dtype)


JnpArangePlugin._PRIM.def_abstract_eval(JnpArangePlugin.abstract_eval)
