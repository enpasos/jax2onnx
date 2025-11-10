# jax2onnx/plugins/_axis0_utils.py

from __future__ import annotations

from typing import Any, Final
from types import SimpleNamespace

import os

import numpy as np

import onnx_ir as ir

from jax2onnx.plugins._ir_shapes import (
    _ensure_value_metadata,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins._loop_extent_meta import get_axis0_override, set_axis0_override
from jax2onnx.plugins.jax.lax._index_utils import _const_i64

_FLOAT16: Final[ir.DataType] = ir.DataType.FLOAT16
_BFLOAT16: Final[ir.DataType] = ir.DataType.BFLOAT16
_DOUBLE: Final[ir.DataType] = ir.DataType.DOUBLE
_COMPLEX64: Final[ir.DataType] = ir.DataType.COMPLEX64
_COMPLEX128: Final[ir.DataType] = ir.DataType.COMPLEX128
_UINT8: Final[ir.DataType] = ir.DataType.UINT8
_UINT16: Final[ir.DataType] = ir.DataType.UINT16
_UINT32: Final[ir.DataType] = ir.DataType.UINT32
_UINT64: Final[ir.DataType] = ir.DataType.UINT64
_NP_BFLOAT16: Final[np.dtype[Any]] = np.dtype(
    np.bfloat16 if hasattr(np, "bfloat16") else np.float16
)

_IR_TO_NP_DTYPE: dict[ir.DataType | None, np.dtype[Any]] = {
    _FLOAT16: np.dtype(np.float16),
    _BFLOAT16: np.dtype(_NP_BFLOAT16),
    ir.DataType.FLOAT: np.dtype(np.float32),
    _DOUBLE: np.dtype(np.float64),
    _COMPLEX64: np.dtype(np.complex64),
    _COMPLEX128: np.dtype(np.complex128),
    ir.DataType.INT8: np.dtype(np.int8),
    ir.DataType.INT16: np.dtype(np.int16),
    ir.DataType.INT32: np.dtype(np.int32),
    ir.DataType.INT64: np.dtype(np.int64),
    _UINT8: np.dtype(np.uint8),
    _UINT16: np.dtype(np.uint16),
    _UINT32: np.dtype(np.uint32),
    _UINT64: np.dtype(np.uint64),
    ir.DataType.BOOL: np.dtype(np.bool_),
}


def _axis0_debug_enabled() -> bool:
    flag = os.environ.get("J2O_DEBUG_AXIS0", "")
    if not flag:
        return False
    return flag.strip().lower() in ("1", "true", "yes", "on")


def _axis0_debug(message: str) -> None:
    if _axis0_debug_enabled():
        print(f"[axis0-debug] {message}", flush=True)


def _value_name(value: Any) -> str | None:
    if value is None or not hasattr(value, "name"):
        return None
    name = value.name
    return str(name) if name is not None else None


def _shape_dims(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not hasattr(value, "shape"):
        return ()
    shape = value.shape
    if not hasattr(shape, "dims"):
        return ()
    dims = shape.dims
    if dims:
        return tuple(dims)
    return ()


def _aval_shape_tuple(var: Any) -> tuple[Any, ...]:
    if var is None:
        return ()
    if not hasattr(var, "aval"):
        return ()
    aval = var.aval
    shape = aval.shape if hasattr(aval, "shape") else ()
    if shape:
        return tuple(shape)
    return ()


def _np_dtype_for_enum(enum: Any) -> np.dtype[Any] | None:
    if isinstance(enum, np.dtype):
        return enum
    if isinstance(enum, ir.DataType):
        return _IR_TO_NP_DTYPE.get(enum)
    if isinstance(enum, (int, np.integer)):
        try:
            return _IR_TO_NP_DTYPE.get(ir.DataType(int(enum)))
        except Exception:
            return None
    return None


def _static_dim_as_int(dim: Any) -> int | None:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    value = dim.value if hasattr(dim, "value") else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return int(dim)
    except Exception:
        if _axis0_debug_enabled() and dim is not None:
            _axis0_debug(
                f"static_dim_as_int failed type={type(dim).__name__} repr={dim!r}"
            )
        return None


def _pad_axis0_to_extent(
    ctx: Any,
    value: Any,
    *,
    override: int,
    dims: list[Any],
    reference: Any | None,
) -> Any | None:
    if not dims:
        return None
    dim0 = dims[0]
    dim0_int = _static_dim_as_int(dim0)
    if dim0_int is None:
        _axis0_debug(
            "ensure_axis0_extent unable to read static dim0 "
            f"type={type(dim0)} value={_value_name(value)} override={override}"
        )
        return None
    pad_amount = override - dim0_int
    if pad_amount <= 0:
        return None

    value_type = value.type if hasattr(value, "type") else None
    value_dtype = (
        value_type.dtype
        if value_type is not None and hasattr(value_type, "dtype")
        else None
    )
    dtype_enum = value_dtype
    if dtype_enum is None and reference is not None:
        ref_type = reference.type if hasattr(reference, "type") else None
        if ref_type is not None and hasattr(ref_type, "dtype"):
            dtype_enum = ref_type.dtype
    np_dtype = _np_dtype_for_enum(dtype_enum) or np.dtype(np.float32)

    zero_scalar = np.zeros((), dtype=np_dtype)
    zero_init = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name("axis0_pad_zero"),
        array=zero_scalar,
    )
    if value_dtype is not None:
        zero_init.type = ir.TensorType(value_dtype)
    _stamp_type_and_shape(zero_init, ())
    _ensure_value_metadata(ctx, zero_init)

    rank = len(dims)
    pads_before = [0] * rank
    pads_after = [0] * rank
    pads_after[0] = pad_amount
    pads_vec = pads_before + pads_after

    pads_val = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name("axis0_pad_spec"),
        array=np.asarray(pads_vec, dtype=np.int64),
    )
    pads_val.type = ir.TensorType(ir.DataType.INT64)
    dynamic_pad_dims = tuple(None for _ in pads_vec)
    _stamp_type_and_shape(pads_val, dynamic_pad_dims)
    _ensure_value_metadata(ctx, pads_val)

    pad_inputs = [value, pads_val, zero_init]

    padded = ctx.builder.Pad(
        *pad_inputs,
        mode="constant",
        _outputs=[ctx.fresh_name("axis0_pad")],
    )
    if value_type is not None:
        padded.type = value_type
    new_dims = list(dims)
    new_dims[0] = override
    try:
        stamped_dims = tuple(_to_ir_dim_for_shape(dim) for dim in new_dims)
        _stamp_type_and_shape(padded, stamped_dims)
    except Exception:
        _axis0_debug(
            "ensure_axis0_extent concat stamp failed "
            f"value={_value_name(padded)} dims={new_dims}"
        )
    _ensure_value_metadata(ctx, padded)
    set_axis0_override(padded, override)
    _axis0_debug(
        "ensure_axis0_extent padded "
        f"value={_value_name(padded)} "
        f"original_dim={dim0_int} override={override}"
    )
    if _axis0_debug_enabled():
        shape_dims = _shape_dims(padded)
        shape_desc = [
            (
                _static_dim_as_int(dim)
                if _static_dim_as_int(dim) is not None
                else repr(dim)
            )
            for dim in shape_dims
        ]
        _axis0_debug(
            "ensure_axis0_extent padded dims "
            f"value={_value_name(padded)} dims={shape_desc}"
        )
    return padded


def ensure_axis0_extent(
    ctx: Any, value: Any, override: int | None, reference: Any | None = None
) -> Any:
    if override is None or override <= 1:
        _axis0_debug(
            f"ensure_axis0_extent skip override={override} value={_value_name(value)}"
        )
        return value

    dims_tuple = _shape_dims(value)
    dims = list(dims_tuple) if dims_tuple else None
    if (not dims or len(dims) == 0) and reference is not None:
        ref_dims = _shape_dims(reference)
        if ref_dims and len(ref_dims) > 0:
            dims = list(ref_dims)
    if not dims or len(dims) == 0:
        _axis0_debug(
            f"ensure_axis0_extent no dims override={override} value={_value_name(value)}"
        )
        return value
    if not dims_tuple or len(dims_tuple) == 0:
        try:
            stamp_dims = tuple(_to_ir_dim_for_shape(dim) for dim in dims)
            _stamp_type_and_shape(value, stamp_dims)
        except Exception:
            _axis0_debug(
                f"ensure_axis0_extent failed to stamp input shape value={_value_name(value)}"
            )
    dim0 = dims[0]
    existing = get_axis0_override(value)
    if isinstance(existing, (int, np.integer)) and int(existing) == override:
        dim0_int = _static_dim_as_int(dim0)
        if dim0_int is None or dim0_int != override:
            padded = _pad_axis0_to_extent(
                ctx,
                value,
                override=override,
                dims=list(dims) if dims is not None else [],
                reference=reference,
            )
            if padded is not None:
                return padded
            try:
                new_dims = list(dims)
                new_dims[0] = override
                stamped_dims = tuple(_to_ir_dim_for_shape(dim) for dim in new_dims)
                _stamp_type_and_shape(value, stamped_dims)
                _ensure_value_metadata(ctx, value)
            except Exception:
                _axis0_debug(
                    "ensure_axis0_extent failed to restamp existing override "
                    f"value={_value_name(value)}"
                )
        _axis0_debug(
            f"ensure_axis0_extent existing override matches override={override} value={_value_name(value)}"
        )
        return value

    dim0_int = _static_dim_as_int(dim0)
    if dim0_int is not None:
        if dim0_int == override:
            set_axis0_override(value, override)
            _axis0_debug(
                f"ensure_axis0_extent dim0 already {override} value={_value_name(value)}"
            )
            return value
        if dim0_int > override:
            _axis0_debug(
                f"ensure_axis0_extent dim0={dim0_int} incompatible with override={override} value={_value_name(value)}"
            )
            return value
        if dim0_int < override:
            padded = _pad_axis0_to_extent(
                ctx,
                value,
                override=override,
                dims=dims,
                reference=reference,
            )
            if padded is not None:
                return padded
            _axis0_debug(
                "ensure_axis0_extent unable to pad "
                f"value={_value_name(value)} "
                f"dim0={dim0_int} override={override}"
            )
        set_axis0_override(value, override)
        _axis0_debug(
            "ensure_axis0_extent metadata override only "
            f"value={_value_name(value)} override={override}"
        )
        try:
            new_dims = list(dims)
            new_dims[0] = override
            stamped_dims = tuple(_to_ir_dim_for_shape(dim) for dim in new_dims)
            _stamp_type_and_shape(value, stamped_dims)
            _ensure_value_metadata(ctx, value)
        except Exception:
            _axis0_debug(
                "ensure_axis0_extent failed to stamp override-only shape "
                f"value={_value_name(value)}"
            )
        return value
    else:
        _axis0_debug(
            f"ensure_axis0_extent non-static dim0 override={override} value={_value_name(value)}"
        )

    rank = len(dims)
    override_vec = _const_i64(
        ctx,
        np.asarray([override], dtype=np.int64),
        ctx.fresh_name("axis0_override"),
    )

    _axis0_debug(
        "ensure_axis0_extent expanding "
        f"value={_value_name(value)} override={override} dims={dims}"
    )
    if rank > 1:
        shape_tensor = ctx.builder.Shape(
            value,
            _outputs=[ctx.fresh_name("axis0_shape")],
        )
        shape_tensor.dtype = ir.DataType.INT64
        _stamp_type_and_shape(shape_tensor, (rank,))
        _ensure_value_metadata(ctx, shape_tensor)

        starts = _const_i64(
            ctx,
            np.asarray([1], dtype=np.int64),
            ctx.fresh_name("axis0_tail_starts"),
        )
        ends = _const_i64(
            ctx,
            np.asarray([np.iinfo(np.int64).max], dtype=np.int64),
            ctx.fresh_name("axis0_tail_ends"),
        )
        axes = _const_i64(
            ctx,
            np.asarray([0], dtype=np.int64),
            ctx.fresh_name("axis0_tail_axes"),
        )
        steps = _const_i64(
            ctx,
            np.asarray([1], dtype=np.int64),
            ctx.fresh_name("axis0_tail_steps"),
        )
        tail_shape = ctx.builder.Slice(
            shape_tensor,
            starts,
            ends,
            axes,
            steps,
            _outputs=[ctx.fresh_name("axis0_shape_tail")],
        )
        tail_shape.dtype = ir.DataType.INT64
        _stamp_type_and_shape(tail_shape, (max(rank - 1, 0),))
        _ensure_value_metadata(ctx, tail_shape)

        target_shape = ctx.builder.Concat(
            override_vec,
            tail_shape,
            axis=0,
            _outputs=[ctx.fresh_name("axis0_target_shape")],
        )
        target_shape.dtype = ir.DataType.INT64
        _stamp_type_and_shape(target_shape, (rank,))
        _ensure_value_metadata(ctx, target_shape)
    else:
        target_shape = override_vec
        _ensure_value_metadata(ctx, target_shape)

    expanded = ctx.builder.Expand(
        value,
        target_shape,
        _outputs=[ctx.fresh_name("axis0_expand")],
    )
    if hasattr(value, "type") and value.type is not None:
        expanded.type = value.type
    try:
        original_dims = list(dims)
        if original_dims:
            original_dims[0] = override
            _stamp_type_and_shape(expanded, tuple(original_dims))
    except Exception:
        _axis0_debug(
            f"ensure_axis0_extent failed to stamp shape for value={_value_name(expanded)}"
        )
    _ensure_value_metadata(ctx, expanded)
    set_axis0_override(expanded, override)
    _axis0_debug(
        f"ensure_axis0_extent produced expand value={_value_name(expanded)} override={override}"
    )
    return expanded


def maybe_expand_binary_axis0(
    ctx: Any,
    lhs: Any,
    rhs: Any,
    out_val: Any,
    out_var: Any | None = None,
) -> tuple[Any, Any, int | None]:
    override_sources = [
        get_axis0_override(lhs),
        get_axis0_override(rhs),
        get_axis0_override(out_val),
        (
            ctx._static_loop_extent_axis0
            if hasattr(ctx, "_static_loop_extent_axis0")
            else None
        ),
    ]
    override_candidates = [
        int(val)
        for val in override_sources
        if isinstance(val, (int, np.integer)) and int(val) > 1
    ]
    override = max(override_candidates, default=None)

    _axis0_debug(
        "maybe_expand_binary_axis0 "
        f"lhs={_value_name(lhs)} rhs={_value_name(rhs)} "
        f"out={_value_name(out_val)} override_candidates={override_candidates} "
        f"selected={override}"
    )

    if override is None:
        out_override = get_axis0_override(out_val)
        if isinstance(out_override, (int, np.integer)) and int(out_override) > 1:
            override = int(out_override)
        else:
            return lhs, rhs, None

    lhs_shape0 = _shape_dims(lhs)
    rhs_shape0 = _shape_dims(rhs)
    lhs0 = lhs_shape0[0] if len(lhs_shape0) > 0 else None
    rhs0 = rhs_shape0[0] if len(rhs_shape0) > 0 else None
    lhs0_int = _static_dim_as_int(lhs0)
    rhs0_int = _static_dim_as_int(rhs0)

    def _needs_expand(dim_int: int | None, rank: int) -> bool:
        if override is None or override <= 1:
            return False
        if rank == 0:
            return True
        if dim_int is None:
            return True
        return dim_int != override

    lhs_needs = override is not None and _needs_expand(lhs0_int, len(lhs_shape0))
    rhs_needs = override is not None and _needs_expand(rhs0_int, len(rhs_shape0))

    if _axis0_debug_enabled() and override is not None and override > 1:
        lhs_dims_desc = [
            (
                _static_dim_as_int(dim)
                if _static_dim_as_int(dim) is not None
                else repr(dim)
            )
            for dim in lhs_shape0
        ]
        rhs_dims_desc = [
            (
                _static_dim_as_int(dim)
                if _static_dim_as_int(dim) is not None
                else repr(dim)
            )
            for dim in rhs_shape0
        ]
        _axis0_debug(
            "maybe_expand_binary_axis0 analysis "
            f"lhs={_value_name(lhs)} "
            f"rhs={_value_name(rhs)} "
            f"override={override} "
            f"lhs_rank={len(lhs_shape0)} lhs_dim={lhs0_int} lhs_needs={lhs_needs} lhs_dims={lhs_dims_desc} "
            f"rhs_rank={len(rhs_shape0)} rhs_dim={rhs0_int} rhs_needs={rhs_needs} rhs_dims={rhs_dims_desc}"
        )

    if lhs_needs:
        lhs = ensure_axis0_extent(ctx, lhs, override, reference=rhs or out_val)
    if rhs_needs:
        rhs = ensure_axis0_extent(ctx, rhs, override, reference=lhs or out_val)

    lhs_override = get_axis0_override(lhs)
    rhs_override = get_axis0_override(rhs)
    if lhs_override == override or rhs_override == override:
        return lhs, rhs, override

    fallback_lhs = ensure_axis0_extent(ctx, lhs, override, reference=out_val)
    fallback_rhs = ensure_axis0_extent(ctx, rhs, override, reference=out_val)
    lhs2_override = get_axis0_override(fallback_lhs)
    rhs2_override = get_axis0_override(fallback_rhs)
    if lhs2_override == override or rhs2_override == override:
        _axis0_debug(
            "maybe_expand_binary_axis0 override salvaged via out_spec "
            f"lhs_override={lhs2_override} rhs_override={rhs2_override} "
        )
        return fallback_lhs, fallback_rhs, override

    if out_var is not None:
        out_shape = _aval_shape_tuple(out_var)
        if out_shape:
            fake_ref = SimpleNamespace(shape=SimpleNamespace(dims=out_shape))
            lhs_alt = ensure_axis0_extent(ctx, lhs, override, reference=fake_ref)
            rhs_alt = ensure_axis0_extent(ctx, rhs, override, reference=fake_ref)
            lhs3_override = get_axis0_override(lhs_alt)
            rhs3_override = get_axis0_override(rhs_alt)
            if lhs3_override == override or rhs3_override == override:
                _axis0_debug(
                    "maybe_expand_binary_axis0 override forced via target shape "
                    f"lhs_override={lhs3_override} rhs_override={rhs3_override} "
                )
                return lhs_alt, rhs_alt, override
    out_shape = ()
    if out_var is not None:
        out_shape = _aval_shape_tuple(out_var)
    _axis0_debug(
        "maybe_expand_binary_axis0 override dropped "
        f"lhs_override={lhs_override} rhs_override={rhs_override} "
        f"selected={override} out_shape={out_shape}"
    )
    return lhs, rhs, None


def stamp_axis0_binary_result(
    result: Any, out_var: Any, out_spec: Any, override: int | None
) -> None:
    out_shape = _aval_shape_tuple(out_var)
    if override is not None and out_shape:
        out_shape = (override,) + out_shape[1:]
    if out_shape:
        dims = tuple(_to_ir_dim_for_shape(dim) for dim in out_shape)
        _stamp_type_and_shape(result, dims)
    elif hasattr(out_spec, "shape") and out_spec.shape is not None:
        result.shape = out_spec.shape
