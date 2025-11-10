# jax2onnx/plugins/jax/numpy/arange.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Final, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import core
from jax.extend.core import Literal as JaxLiteral

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_DYNAMIC_DIM_LABEL: Final[str] = "JAX2ONNX_DYNAMIC_DIM_SENTINEL"
_JNP_ARANGE_ORIG: Final = jnp.arange


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


DATA_DEPENDENT_DYNAMIC_DIM: Final[_DynamicDimSentinel] = _DynamicDimSentinel()


_ARANGE_PRIM: Final = make_jnp_primitive("jax.numpy.arange")


def _as_scalar(aval: core.AbstractValue | JaxLiteral | None) -> float | int | None:
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


def _all_scalars(
    avals: Sequence[core.AbstractValue | JaxLiteral | None],
) -> list[float | int] | None:
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
    avals: Sequence[core.AbstractValue | JaxLiteral | None],
    dtype_param: np.dtype[Any] | type | None,
    enable_x64: bool,
) -> np.dtype[Any]:
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
    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError("IR build context missing builder for arange lowering")
    cast = builder.Cast(
        value,
        _outputs=[ctx.fresh_name(name_hint)],
        to=int(target_enum.value),
    )
    cast.type = ir.TensorType(target_enum)
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
    _ensure_value_metadata(ctx, cast)
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
    since="v0.5.2",
    context="primitives.jnp",
    component="arange",
    testcases=[
        {
            "testcase": "arange_data_dependent_indices",
            "callable": lambda x: jnp.arange(x.shape[1]),
            "input_shapes": [(3, 10)],
            "input_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 0.0},
                            1: {"const": 10.0},
                            2: {"const": 1.0},
                        },
                        "path": "Range:10",
                    }
                ]
            ),
        },
        {
            "testcase": "arange_stop_only_concrete_input_val",
            "callable": lambda stop: jnp.arange(stop, dtype=jnp.float32),
            "input_values": [np.array(5.0, dtype=np.float32)],
            "expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 0.0},
                            2: {"const": 1.0},
                        },
                        "path": "Range:B",
                    }
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_start_stop_concrete_input_val",
            "callable": lambda start, stop: jnp.arange(start, stop, dtype=jnp.float32),
            "input_values": [
                np.array(2.0, dtype=np.float32),
                np.array(7.0, dtype=np.float32),
            ],
            "expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            2: {"const": 1.0},
                        },
                        "path": "Range:B",
                    }
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                ["Range:B"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
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
            "post_check_onnx_graph": EG(
                ["Range:B"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_stop_only_int",
            "callable": lambda: jnp.arange(5),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 0.0},
                            1: {"const": 5.0},
                            2: {"const": 1.0},
                        },
                        "path": "Range:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_stop_only_float",
            "callable": lambda: jnp.arange(5.0),
            "input_values": [],
            "expected_output_shapes": [(5,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 0.0},
                            1: {"const": 5.0},
                            2: {"const": 1.0},
                        },
                        "path": "Range:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_start_stop_int",
            "callable": lambda: jnp.arange(2, 7),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 2.0},
                            1: {"const": 7.0},
                            2: {"const": 1.0},
                        },
                        "path": "Range:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_start_stop_step_int",
            "callable": lambda: jnp.arange(1, 10, 2),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 1.0},
                            1: {"const": 10.0},
                            2: {"const": 2.0},
                        },
                        "path": "Range:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_empty_result_pos_step",
            "callable": lambda: jnp.arange(5, 2, 1),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 5.0},
                            1: {"const": 2.0},
                            2: {"const": 1.0},
                        },
                        "path": "Range:0",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_empty_result_neg_step",
            "callable": lambda: jnp.arange(2, 5, -1),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 2.0},
                            1: {"const": 5.0},
                            2: {"const": -1.0},
                        },
                        "path": "Range:0",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_negative_step",
            "callable": lambda: jnp.arange(5, 0, -1),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 5.0},
                            1: {"const": 0.0},
                            2: {"const": -1.0},
                        },
                        "path": "Range:5",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_float_step_explicit_dtype",
            "callable": lambda: jnp.arange(1.0, 2.0, 0.25, dtype=jnp.float32),
            "input_values": [],
            "expected_output_shapes": [(4,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 1.0},
                            1: {"const": 2.0},
                            2: {"const": 0.25},
                        },
                        "path": "Range:4",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_float_step_inferred_dtype",
            "callable": lambda: jnp.arange(0.0, 1.0, 0.3),
            "input_values": [],
            "expected_output_shapes": [(4,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 0.0},
                            1: {"const": 1.0},
                            2: {"const": 0.3},
                        },
                        "path": "Range:4",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_stop_zero",
            "callable": lambda: jnp.arange(0),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 0.0},
                            1: {"const": 0.0},
                            2: {"const": 1.0},
                        },
                        "path": "Range:0",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_start_equals_stop",
            "callable": lambda: jnp.arange(5, 5, 1),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 5.0},
                            1: {"const": 5.0},
                            2: {"const": 1.0},
                        },
                        "path": "Range:0",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "arange_static_large_numbers_int",
            "callable": lambda: jnp.arange(1000, 1010, 1, dtype=jnp.int32),
            "input_values": [],
            "x64_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "x32_expected_output_shapes": [(_DYNAMIC_DIM_LABEL,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {
                            0: {"const": 1000.0},
                            1: {"const": 1010.0},
                            2: {"const": 1.0},
                        },
                        "path": "Range:10",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpArangePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _ARANGE_PRIM
    _FUNC_NAME: ClassVar[str] = "arange"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        *in_avals: core.AbstractValue,
        dtype: np.dtype[Any] | type | None = None,
    ) -> core.ShapedArray:
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

    def lower(self, ctx: "IRContext", eqn: core.JaxprEqn) -> None:
        params = getattr(eqn, "params", {})
        dtype_param = params.get("dtype")
        avals: list[core.AbstractValue | JaxLiteral | None] = []
        for var in eqn.invars:
            if isinstance(var, JaxLiteral):
                avals.append(var.aval)
            else:
                avals.append(getattr(var, "aval", None))
        enable_x64 = bool(ctx.builder.enable_double_precision)
        result_dtype: np.dtype[Any] = _resolve_result_dtype(
            avals, dtype_param, enable_x64
        )
        target_enum: ir.DataType = _dtype_to_ir(
            result_dtype, ctx.builder.enable_double_precision
        )

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
                literal_value = cast(float | int, np.asarray(var.val).item())
                literal_args.append(literal_value)
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

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder for arange lowering")

        out_var = eqn.outvars[0]
        out_spec = ctx.get_value_for_var(
            out_var,
            name_hint=ctx.fresh_name("arange_out"),
            prefer_np_dtype=result_dtype,
        )
        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("arange_out")
        result = builder.Range(
            start_val,
            limit_val,
            delta_val,
            _outputs=[out_name],
        )
        result.type = ir.TensorType(target_enum)

        length_hint: int | None = None
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

        _stamp_type_and_shape(result, final_shape)
        _ensure_value_metadata(ctx, result)
        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.arange not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(*args: Any, **kwargs: Any) -> jax.Array:
                dtype_param: np.dtype[Any] | type | None = kwargs.pop("dtype", None)

                start_kw = kwargs.pop("start", None)
                stop_kw = kwargs.pop("stop", None)
                step_kw = kwargs.pop("step", None)
                if kwargs:
                    raise TypeError(
                        f"Unsupported keyword arguments for jnp.arange: {tuple(kwargs.keys())}"
                    )

                num_args = len(args)
                if num_args > 3:
                    return orig(
                        *args,
                        dtype=dtype_param,
                        start=start_kw,
                        stop=stop_kw,
                        step=step_kw,
                    )

                if num_args == 0:
                    if stop_kw is None:
                        raise TypeError(
                            "jnp.arange requires 'stop' when using keyword arguments."
                        )
                    start = 0 if start_kw is None else start_kw
                    stop = stop_kw
                    step = 1 if step_kw is None else step_kw
                elif num_args == 1:
                    if start_kw is not None or stop_kw is not None:
                        return orig(
                            *args,
                            dtype=dtype_param,
                            start=start_kw,
                            stop=stop_kw,
                            step=step_kw,
                        )
                    start = 0
                    stop = args[0]
                    step = 1 if step_kw is None else step_kw
                elif num_args == 2:
                    if start_kw is not None or stop_kw is not None:
                        return orig(
                            *args,
                            dtype=dtype_param,
                            start=start_kw,
                            stop=stop_kw,
                            step=step_kw,
                        )
                    start, stop = args
                    step = 1 if step_kw is None else step_kw
                else:  # num_args == 3
                    if any(v is not None for v in (start_kw, stop_kw, step_kw)):
                        return orig(
                            *args,
                            dtype=dtype_param,
                            start=start_kw,
                            stop=stop_kw,
                            step=step_kw,
                        )
                    start, stop, step = args

                values: list[Any] = [start, stop]
                if step_kw is not None or num_args == 3 or step != 1:
                    values.append(step)
                return cls._PRIM.bind(*values, dtype=dtype_param)

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
def _arange_impl(*args: Any, dtype: np.dtype[Any] | type | None = None) -> jax.Array:
    try:
        orig = get_orig_impl(JnpArangePlugin._PRIM, JnpArangePlugin._FUNC_NAME)
    except RuntimeError:
        orig = _JNP_ARANGE_ORIG
    return orig(*args, dtype=dtype)


JnpArangePlugin._PRIM.def_abstract_eval(JnpArangePlugin.abstract_eval)
