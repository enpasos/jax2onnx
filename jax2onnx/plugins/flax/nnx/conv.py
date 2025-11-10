# jax2onnx/plugins/flax/nnx/conv.py

from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Callable,
    Final,
    Sequence,
    Tuple,
    Any,
    overload,
    cast,
)
import logging

import numpy as np
import math
import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins._ir_shapes import (
    _as_ir_dim_label,
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from functools import reduce
from operator import mul

if TYPE_CHECKING:
    from jax2onnx.converter.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore

logger: logging.Logger = logging.getLogger("jax2onnx.plugins.flax.nnx.conv")

# --- graph pattern expectations for tests ---
# Single-path: Transpose -> Conv -> Transpose
EXPECT_TCT: Final = EG(
    [
        (
            "Transpose -> Conv -> Transpose",
            {
                "counts": {
                    "Transpose": 2,
                    "Conv": 1,
                    "Reshape": 0,
                    "Add": 0,
                    "CastLike": 0,
                    "Shape": 0,
                    "Gather": 0,
                    "Unsqueeze": 0,
                    "Concat": 0,
                }
            },
        )
    ]
)
# Flattened-path (mixed-dim inputs):
# Reshape -> Transpose -> Conv -> Transpose -> Reshape
EXPECT_FLATTEN_TCT: Final = EG(
    [
        (
            "Reshape -> Transpose -> Conv -> Transpose -> Reshape",
            {
                "counts": {
                    "Reshape": 2,
                    "Transpose": 2,
                    "Conv": 1,
                    "Add": 0,
                    "CastLike": 0,
                    "Shape": 0,
                    "Gather": 0,
                    "Unsqueeze": 0,
                    "Concat": 0,
                }
            },
        )
    ]
)


# ---------- helper: normalize value metadata so exported graphs show shapes ----------
def _np_dtype_of(var, fallback=np.float32):
    dt = getattr(getattr(var, "aval", None), "dtype", None)
    try:
        return np.dtype(dt) if dt is not None else np.dtype(fallback)
    except Exception:
        return np.dtype(fallback)


def _is_concrete_shape(shape) -> bool:
    try:
        return all(isinstance(d, (int, np.integer)) for d in shape)
    except Exception:
        return False


def _annotate_value(ctx, val: ir.Value, dtype, shape) -> None:
    """Ensure the IR Value carries dtype/shape so the exported graph emits ValueInfoProto."""
    if not shape:
        return
    dims = []
    for dim in shape:
        label = _as_ir_dim_label(dim)
        if label is not None:
            dims.append(label)
        else:
            dims.append(dim)
    dims_tuple = tuple(dims)
    try:
        _stamp_type_and_shape(val, dims_tuple)
    except Exception:
        pass
    try:
        if getattr(val, "type", None) is None:
            val.type = ir.TensorType(_ir_dtype_from_numpy(dtype))
    except Exception:
        pass
    if ctx is not None:
        try:
            _ensure_value_metadata(ctx, val)
        except Exception:
            pass


def _calc_out_spatial(
    in_sp: Sequence[int],
    k_sp: Sequence[int],
    strides: Sequence[int],
    dilations: Sequence[int],
    padding,
) -> Tuple[int, ...]:
    """
    ONNX-style output size:
      floor((in + pad_beg + pad_end - eff_kernel)/stride + 1)
    For SAME_* we use ceil(in/stride).
    """
    auto = None
    pads_beg = [0] * len(in_sp)
    pads_end = [0] * len(in_sp)
    if isinstance(padding, str):
        p = padding.upper()
        auto = "SAME" if p.startswith("SAME") else "VALID"
    else:
        pads_beg = [int(p[0]) for p in padding]
        pads_end = [int(p[1]) for p in padding]

    outs = []
    for i in range(len(in_sp)):
        s_in = int(in_sp[i])
        k_eff = (int(k_sp[i]) - 1) * int(dilations[i]) + 1
        if auto == "SAME":
            outs.append(int(math.ceil(s_in / float(strides[i]))))
        else:
            total_pad = pads_beg[i] + pads_end[i]
            # Clamp at zero to avoid negative spatial sizes for VALID when k_eff is large
            calc = math.floor((s_in + total_pad - k_eff) / float(strides[i]) + 1)
            outs.append(int(max(0, calc)))
    return tuple(outs)


def _all_ones(seq: Sequence[int]) -> bool:
    """True if all elements are 1."""
    return all(int(v) == 1 for v in seq)


def _same_upper_pads_static(
    in_sp: Sequence[int],
    k_sp: Sequence[int],
    strides: Sequence[int],
    dilations: Sequence[int],
) -> tuple[list[int], list[int]]:
    """
    Compute SAME_UPPER padding:
      k_eff = (k-1)*d + 1
      out   = ceil(in/stride)
      total = max(0, (out-1)*stride + k_eff - in)
      beg   = total//2 ; end = total - beg
    """
    pads_beg, pads_end = [], []
    for s_in, k, st, dil in zip(in_sp, k_sp, strides, dilations):
        k_eff = (int(k) - 1) * int(dil) + 1
        out_sz = int(math.ceil(int(s_in) / float(st)))
        total = max(0, (out_sz - 1) * int(st) + k_eff - int(s_in))
        beg = total // 2
        end = total - beg
        pads_beg.append(int(beg))
        pads_end.append(int(end))
    return pads_beg, pads_end


def _ir_dtype_from_numpy(dt):
    dt = np.dtype(dt)
    # Common dtypes we need for these tests
    if dt == np.dtype("float32"):
        return ir.DataType.FLOAT
    if dt == np.dtype("float64"):
        return ir.DataType.DOUBLE
    if dt == np.dtype("int64"):
        return ir.DataType.INT64
    if dt == np.dtype("int32"):
        return ir.DataType.INT32
    return ir.DataType.FLOAT


def _const_from_array(ctx, arr: np.ndarray, name: str | None = None) -> ir.Value:
    builder = getattr(ctx, "builder", None)
    base_name = name or "const"
    name_hint = ctx.fresh_name(base_name) if hasattr(ctx, "fresh_name") else base_name
    np_arr = np.asarray(arr)

    builder_mode = (
        bool(getattr(builder, "_function_mode", False))
        if builder is not None
        else False
    )
    inside_function = bool(
        getattr(ctx, "_inside_function_scope", False)
        or getattr(ctx, "_function_mode", False)
    )

    if builder is not None and not inside_function and not builder_mode:
        add_initializer = getattr(builder, "add_initializer_from_array", None)
        if callable(add_initializer):
            return add_initializer(name_hint, np_arr)

    value = ir.Value(
        name=name_hint,
        type=ir.TensorType(_ir_dtype_from_numpy(np_arr.dtype)),
        shape=ir.Shape(tuple(int(d) for d in np_arr.shape)),
        const_value=ir.tensor(np_arr),
    )

    handler = getattr(ctx, "_handle_initializer_append", None)
    if callable(handler):
        handler(value)
        return value

    init_list = getattr(ctx, "_initializers", None)
    if init_list is not None and hasattr(init_list, "append"):
        init_list.append(value)
        return value

    if builder is not None:
        builder_inits = getattr(builder, "initializers", None)
        if isinstance(builder_inits, list):
            builder_inits.append(value)

    return value


def _require_builder(ctx):
    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError("IR build context missing builder")
    return builder


def _to_numpy(tensor_obj) -> np.ndarray | None:
    # Try several common IR flavors
    for meth in ("numpy", "to_numpy", "to_array"):
        fn = getattr(tensor_obj, meth, None)
        if callable(fn):
            try:
                return np.asarray(fn())
            except Exception:
                pass
    for attr in ("array", "ndarray", "value"):
        val = getattr(tensor_obj, attr, None)
        if val is not None:
            try:
                return np.asarray(val)
            except Exception:
                pass
    try:
        return np.asarray(tensor_obj)
    except Exception:
        return None


def _maybe_fold_param_transpose(
    ctx, val: ir.Value, perm: Sequence[int], name: str = "folded_param"
) -> ir.Value:
    const = getattr(val, "const_value", None)
    if const is None:
        # Not a constant initializer; fall back to runtime Transpose.
        return _transpose(ctx, val, perm)
    arr = _to_numpy(const)
    if arr is None:
        # Couldn't read data -> safe fallback
        return _transpose(ctx, val, perm)
    arr_t = np.transpose(arr, tuple(int(p) for p in perm))
    return _const_from_array(ctx, arr_t, name)


# ---------- typing helpers ----------
@overload
def _to_int_tuple(x: int | np.integer, rank: int) -> Tuple[int, ...]: ...
@overload
def _to_int_tuple(x: Sequence[int], rank: int) -> Tuple[int, ...]: ...
def _to_int_tuple(x, rank):
    """Normalize an int-or-sequence hyperparameter to a tuple of length `rank`."""
    if isinstance(x, (int, np.integer)):
        return (int(x),) * int(rank)
    return tuple(int(v) for v in x)


def _shape_of(ctx, x: ir.Value) -> ir.Value:
    builder = _require_builder(ctx)
    result = builder.Shape(
        x,
        _outputs=[ctx.fresh_name("Shape")],
    )
    result.type = ir.TensorType(ir.DataType.INT64)
    return result


def _gather(ctx, data: ir.Value, indices: ir.Value, axis: int = 0) -> ir.Value:
    builder = _require_builder(ctx)
    result = builder.Gather(
        data,
        indices,
        axis=int(axis),
        _outputs=[ctx.fresh_name("Gather")],
    )
    dtype = getattr(getattr(data, "type", None), "dtype", None)
    if dtype is not None:
        result.type = ir.TensorType(dtype)
    return result


def _unsqueeze(ctx, x: ir.Value, axes: Sequence[int]) -> ir.Value:
    """
    ONNX Unsqueeze (opset >= 13): axes is a second input (INT64 1-D), not an attribute.
    """
    axes_v = _const_i64(
        ctx, np.asarray([int(a) for a in axes], dtype=np.int64), name_hint="unsq_axes"
    )
    builder = _require_builder(ctx)
    result = builder.Unsqueeze(
        x,
        axes_v,
        _outputs=[ctx.fresh_name("Unsqueeze")],
    )
    dtype = getattr(getattr(x, "type", None), "dtype", None)
    if dtype is not None:
        result.type = ir.TensorType(dtype)
    return result


def _concat0(ctx, parts: Sequence[ir.Value]) -> ir.Value:
    # Normalize and avoid creating identity Concat nodes
    _parts = [p for p in parts if p is not None]
    if len(_parts) == 0:
        # Explicit empty int64 vector (rare but safe fallback)
        return _const_i64(ctx, np.asarray([], dtype=np.int64), name_hint="empty_i64")
    if len(_parts) == 1:
        # Do not emit a 1-input Concat (it shows up as a spurious node in the graph)
        return _parts[0]

    builder = _require_builder(ctx)
    result = builder.Concat(
        *_parts,
        axis=0,
        _outputs=[ctx.fresh_name("Concat")],
    )
    dtype = getattr(getattr(_parts[0], "type", None), "dtype", None)
    if dtype is not None:
        result.type = ir.TensorType(dtype)
    return result


def _transpose(ctx, x: ir.Value, perm: Sequence[int]) -> ir.Value:
    perm_tuple = tuple(int(p) for p in perm)
    builder = _require_builder(ctx)
    result = builder.Transpose(
        x,
        _outputs=[ctx.fresh_name("Transpose")],
        perm=perm_tuple,
    )
    dtype = getattr(getattr(x, "type", None), "dtype", None)
    if dtype is not None:
        result.type = ir.TensorType(dtype)
    return result


def _reshape(ctx, x: ir.Value, shape: ir.Value) -> ir.Value:
    builder = _require_builder(ctx)
    result = builder.Reshape(
        x,
        shape,
        _outputs=[ctx.fresh_name("Reshape")],
    )
    dtype = getattr(getattr(x, "type", None), "dtype", None)
    if dtype is not None:
        result.type = ir.TensorType(dtype)
    return result


def _conv(
    ctx, x: ir.Value, w: ir.Value, b: ir.Value | None, attrs: dict[str, Any]
) -> ir.Value:
    inputs = [x, w] + ([b] if b is not None else [])
    builder = _require_builder(ctx)
    kwargs: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, np.ndarray):
            kwargs[key] = tuple(int(v) for v in value.tolist())
        elif isinstance(value, (list, tuple)):
            if key in {"auto_pad"}:
                kwargs[key] = value
            else:
                kwargs[key] = tuple(int(v) for v in value)
        elif isinstance(value, np.integer):
            kwargs[key] = int(value)
        elif isinstance(value, np.floating):
            kwargs[key] = float(value)
        else:
            kwargs[key] = value
    result = builder.Conv(
        *inputs,
        _outputs=[ctx.fresh_name("Conv")],
        **kwargs,
    )
    dtype = getattr(getattr(x, "type", None), "dtype", None)
    if dtype is not None:
        result.type = ir.TensorType(dtype)
    return result


@register_primitive(
    jaxpr_primitive="nnx.conv",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Conv",
    onnx=[
        {"component": "Conv", "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html"},
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "CastLike",
            "doc": "https://onnx.ai/onnx/operators/onnx__CastLike.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="conv",
    testcases=[
        {
            "testcase": "conv_basic_bias",
            "callable": construct_and_call(
                nnx.Conv,
                in_features=3,
                out_features=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 28, 28, 3)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [("B", 28, 28, 16)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_basic_bias_2",
            "callable": construct_and_call(
                nnx.Conv,
                1,
                32,
                kernel_size=(3, 3),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 28, 28, 1)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 28, 28, 32)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_basic_bias_3",
            "callable": construct_and_call(
                nnx.Conv,
                in_features=1,
                out_features=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 28, 28, 1)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(3, 28, 28, 32)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_stride2_bias",
            "callable": construct_and_call(
                nnx.Conv,
                in_features=32,
                out_features=64,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                use_bias=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 28, 28, 32)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(3, 14, 14, 64)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_no_bias",
            "callable": construct_and_call(
                nnx.Conv,
                in_features=3,
                out_features=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=False,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 28, 28, 3)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [("B", 28, 28, 16)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_valid_padding",
            "callable": construct_and_call(
                nnx.Conv,
                in_features=3,
                out_features=8,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding="VALID",
                use_bias=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 32, 32, 3)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 14, 14, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_stride1",
            "callable": construct_and_call(
                nnx.Conv,
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 16, 16, 3)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 16, 16, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_stride2",
            "callable": construct_and_call(
                nnx.Conv,
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                use_bias=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 16, 16, 3)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 8, 8, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_different_kernel",
            "callable": construct_and_call(
                nnx.Conv,
                in_features=3,
                out_features=8,
                kernel_size=(1, 5),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 16, 16, 3)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 16, 16, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_float64",
            "callable": construct_and_call(
                nnx.Conv,
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                dtype=np.float64,
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 16, 16, 3)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 16, 16, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_single_batch",
            "callable": construct_and_call(
                nnx.Conv,
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 16, 16, 3)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(1, 16, 16, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_large_batch",
            "callable": construct_and_call(
                nnx.Conv,
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(32, 16, 16, 3)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(32, 16, 16, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_1d",
            "callable": construct_and_call(
                nnx.Conv,
                28,
                4,
                kernel_size=(3,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 28, 28)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 28, 4)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_1d_more_1d_inputs",
            "callable": construct_and_call(
                nnx.Conv,
                28,
                4,
                kernel_size=(3,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 4, 4, 28)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 4, 4, 4)],
            "post_check_onnx_graph": EXPECT_FLATTEN_TCT,
        },
        {
            "testcase": "conv_1d_more_2d_inputs",
            "callable": construct_and_call(
                nnx.Conv,
                28,
                4,
                kernel_size=(3,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 4, 4, 8, 28)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 4, 4, 8, 4)],
            "post_check_onnx_graph": EXPECT_FLATTEN_TCT,
        },
        {
            "testcase": "conv_1d_large_kernel",
            "callable": construct_and_call(
                nnx.Conv,
                16,
                8,
                kernel_size=(5,),
                strides=(2,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(4, 32, 16)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(4, 16, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_1d_dilation",
            "callable": construct_and_call(
                nnx.Conv,
                8,
                16,
                kernel_size=(3,),
                kernel_dilation=(2,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 24, 8)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 24, 16)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_1d_stride_dilation",
            "callable": construct_and_call(
                nnx.Conv,
                12,
                6,
                kernel_size=(7,),
                strides=(3,),
                kernel_dilation=(2,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 40, 12)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(3, 14, 6)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_2d_asymmetric_kernel",
            "callable": construct_and_call(
                nnx.Conv,
                4,
                8,
                kernel_size=(2, 5),
                strides=(1, 2),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 20, 20, 4)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 20, 10, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_2d_asymmetric_stride",
            "callable": construct_and_call(
                nnx.Conv,
                6,
                12,
                kernel_size=(3, 3),
                strides=(1, 3),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 18, 24, 6)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 18, 8, 12)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_2d_asymmetric_dilation",
            "callable": construct_and_call(
                nnx.Conv,
                3,
                9,
                kernel_size=(3, 3),
                kernel_dilation=(1, 2),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 16, 16, 3)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 16, 16, 9)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_2d_large_dilation",
            "callable": construct_and_call(
                nnx.Conv,
                8,
                16,
                kernel_size=(3, 3),
                kernel_dilation=(3, 3),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 32, 32, 8)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 32, 32, 16)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_2d_large_stride",
            "callable": construct_and_call(
                nnx.Conv,
                4,
                8,
                kernel_size=(5, 5),
                strides=(4, 4),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 32, 32, 4)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 8, 8, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_2d_mixed_params",
            "callable": construct_and_call(
                nnx.Conv,
                5,
                10,
                kernel_size=(4, 6),
                strides=(2, 3),
                kernel_dilation=(2, 1),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 24, 30, 5)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(3, 12, 10, 10)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_2d_same_padding_mixed_dilation",
            "callable": construct_and_call(
                nnx.Conv,
                10,
                20,
                kernel_size=(3, 7),
                strides=(1, 2),
                kernel_dilation=(4, 1),
                padding="SAME",
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 18, 28, 10)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 18, 14, 20)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_3d_basic",
            "callable": construct_and_call(
                nnx.Conv,
                2,
                4,
                kernel_size=(3, 3, 3),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 8, 8, 8, 2)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 8, 8, 8, 4)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_3d_stride",
            "callable": construct_and_call(
                nnx.Conv,
                4,
                8,
                kernel_size=(3, 3, 3),
                strides=(2, 2, 2),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 16, 16, 16, 4)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 8, 8, 8, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_3d_asymmetric",
            "callable": construct_and_call(
                nnx.Conv,
                3,
                6,
                kernel_size=(2, 3, 4),
                strides=(1, 2, 1),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 12, 14, 16, 3)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 12, 7, 16, 6)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_3d_dilation",
            "callable": construct_and_call(
                nnx.Conv,
                2,
                4,
                kernel_size=(3, 3, 3),
                kernel_dilation=(2, 1, 2),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 16, 12, 16, 2)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 16, 12, 16, 4)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_2d_small_input",
            "callable": construct_and_call(
                nnx.Conv,
                1,
                4,
                kernel_size=(2, 2),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(1, 4, 4, 1)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(1, 4, 4, 4)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_2d_many_channels",
            "callable": construct_and_call(
                nnx.Conv,
                64,
                128,
                kernel_size=(3, 3),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 16, 16, 64)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 16, 16, 128)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_1d_wide_input",
            "callable": construct_and_call(
                nnx.Conv,
                8,
                16,
                kernel_size=(7,),
                strides=(1,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 128, 8)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 128, 16)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_2d_kernel_1x1",
            "callable": construct_and_call(
                nnx.Conv,
                16,
                32,
                kernel_size=(1, 1),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(4, 14, 14, 16)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(4, 14, 14, 32)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_1d_kernel_1",
            "callable": construct_and_call(
                nnx.Conv,
                8,
                16,
                kernel_size=(1,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 20, 8)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(3, 20, 16)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_2d_group_conv",
            "callable": construct_and_call(
                nnx.Conv,
                16,
                32,
                kernel_size=(3, 3),
                feature_group_count=4,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 14, 14, 16)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 14, 14, 32)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_1d_group_conv_more_dims",
            "callable": construct_and_call(
                nnx.Conv,
                12,
                24,
                kernel_size=(5,),
                feature_group_count=3,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 8, 6, 12)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 8, 6, 24)],
            "post_check_onnx_graph": EXPECT_FLATTEN_TCT,
        },
        {
            "testcase": "conv_2d_depthwise",
            "callable": construct_and_call(
                nnx.Conv,
                8,
                8,
                kernel_size=(3, 3),
                feature_group_count=8,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 16, 16, 8)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 16, 16, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_1d_complex_on_4d",
            "callable": construct_and_call(
                nnx.Conv,
                20,
                40,
                kernel_size=(7,),
                strides=(2,),
                kernel_dilation=(3,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 8, 10, 20)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(3, 8, 5, 40)],
            "post_check_onnx_graph": EXPECT_FLATTEN_TCT,
        },
        {
            "testcase": "conv_2d_complex_on_5d",
            "callable": construct_and_call(
                nnx.Conv,
                15,
                30,
                kernel_size=(3, 5),
                strides=(1, 2),
                kernel_dilation=(2, 1),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 6, 8, 12, 15)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 6, 8, 6, 30)],
            "post_check_onnx_graph": EXPECT_FLATTEN_TCT,
        },
        {
            "testcase": "conv_2d_asymmetric_on_5d",
            "callable": construct_and_call(
                nnx.Conv,
                14,
                28,
                kernel_size=(2, 8),
                strides=(3, 1),
                kernel_dilation=(1, 2),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 6, 9, 15, 14)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 6, 3, 15, 28)],
            "post_check_onnx_graph": EXPECT_FLATTEN_TCT,
        },
        {
            "testcase": "conv_1d_high_dilation_on_3d",
            "callable": construct_and_call(
                nnx.Conv,
                24,
                12,
                kernel_size=(9,),
                strides=(3,),
                kernel_dilation=(4,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 16, 24)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 6, 12)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_1d_large_kernel_on_4d",
            "callable": construct_and_call(
                nnx.Conv,
                25,
                50,
                kernel_size=(11,),
                strides=(4,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 8, 12, 25)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 8, 3, 50)],
            "post_check_onnx_graph": EXPECT_FLATTEN_TCT,
        },
        {
            "testcase": "conv_2d_group_stride_dilation",
            "callable": construct_and_call(
                nnx.Conv,
                32,
                64,
                kernel_size=(5, 3),
                strides=(2, 1),
                kernel_dilation=(1, 3),
                feature_group_count=8,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 20, 24, 32)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 10, 24, 64)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_1d_group_on_higher_dim",
            "callable": construct_and_call(
                nnx.Conv,
                18,
                36,
                kernel_size=(4,),
                strides=(2,),
                kernel_dilation=(2,),
                feature_group_count=6,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 5, 7, 9, 18)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(3, 5, 7, 5, 36)],
            "post_check_onnx_graph": EXPECT_FLATTEN_TCT,
        },
        {
            "testcase": "conv_1d_same_padding_on_3d",
            "callable": construct_and_call(
                nnx.Conv,
                16,
                8,
                kernel_size=(5,),
                strides=(2,),
                kernel_dilation=(3,),
                padding="SAME",
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 12, 16)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 6, 8)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_3d_group_complex",
            "callable": construct_and_call(
                nnx.Conv,
                24,
                48,
                kernel_size=(2, 3, 4),
                strides=(1, 2, 1),
                kernel_dilation=(2, 1, 3),
                feature_group_count=8,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 10, 14, 18, 24)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 10, 7, 18, 48)],
            "post_check_onnx_graph": EXPECT_TCT,
        },
        {
            "testcase": "conv_1d_unit_group_on_multi_dim",
            "callable": construct_and_call(
                nnx.Conv,
                21,
                21,
                kernel_size=(6,),
                strides=(3,),
                kernel_dilation=(2,),
                feature_group_count=21,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 7, 11, 13, 21)],
            "run_only_f32_variant": True,
            "expected_output_shapes": [(2, 7, 11, 5, 21)],
            "post_check_onnx_graph": EXPECT_FLATTEN_TCT,
        },
    ],
)
class ConvPlugin(PrimitiveLeafPlugin):
    """
    IR-only plugin for flax.nnx.Conv → ONNX Conv.
    Assumes NHWC input/output in user space; converts to NCHx… internally.
    Supports 1D/2D/3D, SAME/VALID or explicit pads, strides/dilations, and groups.
    Also supports "mixed-dimension" inputs (e.g., 1D conv on higher-rank NHWC)
    by flattening non-participating spatial dims into batch and reshaping back.
    """

    _PRIM: ClassVar[Primitive] = Primitive("nnx.conv")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- monkey patch factory (needed by binding_specs/patch_info) ----------
    @staticmethod
    def _make_patch(orig_fn: Callable):
        """
        Replace nnx.Conv.__call__ with a shim that binds our Primitive so
        the jaxpr contains only 'nnx.conv' instead of raw lax convs/reshapes.
        """
        # If we are re-patching (common under reloads), unwrap to the true original
        # so that abstract_eval doesn't accidentally call our shim and recurse.
        real_orig = getattr(orig_fn, "__j2o_nnx_conv_original__", orig_fn)
        ConvPlugin._ORIGINAL_CALL = real_orig
        prim = ConvPlugin._PRIM

        def patched(self, x):
            # Pull config from the nnx.Conv instance
            strides = getattr(self, "strides", 1)
            padding = getattr(self, "padding", "VALID")
            dilations = getattr(self, "kernel_dilation", 1)
            groups = getattr(self, "feature_group_count", 1)
            use_bias = bool(getattr(self, "use_bias", True))
            kernel = self.kernel.value
            # Keep arity stable, but avoid creating broadcast nodes when bias is disabled.
            if use_bias:
                if (
                    getattr(self, "bias", None) is not None
                    and getattr(self.bias, "value", None) is not None
                ):
                    bias = self.bias.value
                else:
                    # vector bias expected by use_bias=True path
                    bias = jnp.zeros((kernel.shape[-1],), dtype=kernel.dtype)
            else:
                # scalar literal (no broadcast_in_dim in the JAXPR)
                bias = jnp.asarray(0, dtype=kernel.dtype)

            return prim.bind(
                x,
                kernel,
                bias,
                use_bias=use_bias,
                strides=strides,
                padding=padding,
                dilations=dilations,
                dimension_numbers=getattr(self, "dimension_numbers", None),
                feature_group_count=int(groups),
            )

        # Mark the shim so a subsequent patch can recover the original.
        # mypy: 'patched' is typed as a plain Callable, so attribute writes need an Any cast
        patched_any: Any = patched
        setattr(patched_any, "__j2o_nnx_conv_shim__", True)
        setattr(patched_any, "__j2o_nnx_conv_original__", real_orig)
        return patched

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(
        x,
        kernel,
        bias,
        *,
        use_bias: bool,
        strides: Sequence[int] | int = 1,
        padding: str | Sequence[Tuple[int, int]] = "VALID",
        dilations: Sequence[int] | int = 1,
        dimension_numbers=None,
        feature_group_count: int = 1,
    ):
        # Compute shapes using ONLY lax.* — never call (patched) nnx.Conv.__call__ here.
        # Calling the original would re-enter prim.bind and recurse.
        x_shape = tuple(x.shape)
        k_shape = tuple(kernel.shape)
        x_dtype = getattr(x, "dtype", jnp.float32)
        k_dtype = getattr(kernel, "dtype", x_dtype)

        conv_spatial = max(len(k_shape) - 2, 1)
        x_spatial = max(len(x_shape) - 2, 1)

        # Normalize params to tuples of length conv_spatial.
        s = (
            (int(strides),) * conv_spatial
            if isinstance(strides, int)
            else tuple(int(v) for v in strides)
        )
        d = (
            (int(dilations),) * conv_spatial
            if isinstance(dilations, int)
            else tuple(int(v) for v in dilations)
        )

        # If kernel has fewer spatial dims than input, flatten the extras into batch.
        extra = max(0, x_spatial - conv_spatial)
        leading = x_shape[: 1 + extra]  # (N, *extra)
        part = x_shape[1 + extra : 1 + x_spatial]  # participating spatial dims
        # IMPORTANT: keep symbolic dims symbolic (do not cast to int)
        n_flat = reduce(mul, leading, 1) if leading else 1
        a_flat_shape = (n_flat, *part, x_shape[-1])  # N' + participating + C

        # Build general layouts (1D/2D/3D): N{W|HW|DHW}C and {W|HW|DHW}IO.
        spatial_map = {1: "W", 2: "HW", 3: "DHW"}
        try:
            layout_token = spatial_map[conv_spatial]
        except KeyError:
            raise NotImplementedError(
                f"conv with {conv_spatial}D kernels is not supported"
            )

        in_layout = "N" + layout_token + "C"
        ker_layout = layout_token + "IO"
        out_layout = in_layout
        dn = lax.conv_dimension_numbers(
            a_flat_shape, k_shape, (in_layout, ker_layout, out_layout)
        )

        def _pure(a, w, b):
            y = lax.conv_general_dilated(
                a,
                w,
                window_strides=tuple(s),
                padding=padding,
                lhs_dilation=None,
                rhs_dilation=tuple(d),
                dimension_numbers=dn,
                feature_group_count=feature_group_count,
            )
            return y if not use_bias else y + b

        a_s = jax.ShapeDtypeStruct(a_flat_shape, x_dtype)
        w_s = jax.ShapeDtypeStruct(k_shape, k_dtype)
        # If bias is unused, this tracer is ignored by _pure (safe in eval_shape).
        b_s = jax.ShapeDtypeStruct(tuple(bias.shape), getattr(bias, "dtype", k_dtype))
        y_flat = jax.eval_shape(_pure, a_s, w_s, b_s)
        y_flat = jax.tree_util.tree_leaves(y_flat)[0]

        # Unflatten leading dims back to (N, *extra, ...).
        out_spatial = y_flat.shape[1 : 1 + conv_spatial]
        y_shape = (*leading, *out_spatial, k_shape[-1])
        return jax.core.ShapedArray(y_shape, y_flat.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: "IRBuildContext", eqn):
        x_var, k_var, b_var = eqn.invars[:3]
        y_var = eqn.outvars[0]

        builder = _require_builder(ctx)

        params = eqn.params
        use_bias = bool(params.get("use_bias", True))
        strides_param = params.get("strides", 1)
        padding_param = params.get("padding", "VALID")
        dilations_param = params.get("dilations", 1)
        groups = int(params.get("feature_group_count", 1))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        k_shape = tuple(getattr(getattr(k_var, "aval", None), "shape", ()))
        y_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()) or ())
        len(y_shape)
        x_dtype_np = _np_dtype_of(x_var)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        k_val = ctx.get_value_for_var(k_var, name_hint=ctx.fresh_name("kernel"))
        b_val = (
            ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("bias"))
            if use_bias
            else None
        )

        # Normalize params
        conv_spatial = max(len(k_shape) - 2, 1)
        x_spatial = max(len(x_shape) - 2, 1)
        if isinstance(strides_param, int):
            strides = (int(strides_param),) * conv_spatial
        else:
            strides = tuple(int(s) for s in strides_param)
        if isinstance(dilations_param, int):
            dilations = (int(dilations_param),) * conv_spatial
        else:
            dilations = tuple(int(d) for d in dilations_param)

        # Flatten N + extra spatial dims into batch when input has more spatial dims than kernel
        need_flatten = conv_spatial < x_spatial
        x_pre = x_val
        extra = x_spatial - conv_spatial
        if need_flatten:
            if _is_concrete_shape(x_shape):
                # STATIC reshape shape at convert-time:
                # x: (N, extra..., part..., C) -> (N*prod(extra...), part..., C)
                n_flat_static = int(np.prod([int(d) for d in x_shape[: 1 + extra]]))
                part_dims = [
                    int(d) for d in x_shape[1 + extra : 1 + extra + conv_spatial]
                ]
                ch = int(x_shape[-1])
                reshape_in = _const_i64(
                    ctx,
                    np.asarray([n_flat_static, *part_dims, ch], dtype=np.int64),
                    name_hint="reshape_in_static",
                )
                x_pre = _reshape(ctx, x_val, reshape_in)
            else:
                # Dynamic fallback when ranks/dims are symbolic
                # x: (N, extra..., part..., C) -> (N*prod(extra...), part..., C)
                sh_x = _shape_of(ctx, x_val)
                # indices for N and "extra" dims
                idx_nextra = _const_i64(
                    ctx,
                    np.arange(0, 1 + extra, dtype=np.int64),
                    name_hint="idx_nextra",
                )
                ne_dims = _gather(ctx, sh_x, idx_nextra, axis=0)  # 1+extra
                # opset >=13: 'axes' is an input (not an attribute). Reduce over all axes; keepdims=0.
                n_flat_val: ir.Value = cast(
                    ir.Value,
                    builder.ReduceProd(
                        ne_dims,
                        keepdims=0,
                        _outputs=[ctx.fresh_name("ReduceProd")],
                    ),
                )
                n_flat_val.type = ir.TensorType(ir.DataType.INT64)
                _stamp_type_and_shape(n_flat_val, ())
                _ensure_value_metadata(ctx, n_flat_val)
                n_flat_1d = _unsqueeze(ctx, n_flat_val, [0])  # [1]

                # participating spatial dims (the last conv_spatial dims before channel)
                idx_part = _const_i64(
                    ctx,
                    np.arange(1 + extra, 1 + extra + conv_spatial, dtype=np.int64),
                    name_hint="idx_part",
                )
                part_dims = _gather(ctx, sh_x, idx_part, axis=0)  # [conv_spatial]

                # channel dim (last)
                idx_ch = _const_i64(
                    ctx,
                    np.array([len(x_shape) - 1], dtype=np.int64),
                    name_hint="idx_ch",
                )
                ch_dim = _gather(ctx, sh_x, idx_ch, axis=0)  # [1]

                new_shape = _concat0(
                    ctx, [n_flat_1d, part_dims, ch_dim]
                )  # [conv_spatial+2]
                x_pre = _reshape(ctx, x_val, new_shape)
            x_spatial = conv_spatial  # effective for following layout logic

        # NHWC -> NCH... and kernel (spatial..., I, O) -> (O, I, spatial...)
        rank_after = x_spatial + 2
        perm = (0, rank_after - 1, *range(1, rank_after - 1))
        x_nchw = _transpose(ctx, x_pre, perm)
        # Annotate Transpose output (preserve batch/channel labels when available)
        if not need_flatten and x_shape:
            in_sp = x_shape[1 : 1 + x_spatial]
            batch_dim = _dim_label_from_value_or_aval(x_val, x_shape, 0)
            if batch_dim is None:
                batch_dim = x_shape[0]
            channel_dim = _dim_label_from_value_or_aval(
                x_val, x_shape, len(x_shape) - 1
            )
            if channel_dim is None:
                channel_dim = x_shape[-1]
            spatial_dims = []
            for axis in range(1, 1 + x_spatial):
                label = _dim_label_from_value_or_aval(x_val, x_shape, axis)
                spatial_dims.append(label if label is not None else x_shape[axis])
            _annotate_value(
                ctx,
                x_nchw,
                x_dtype_np,
                (batch_dim, channel_dim, *spatial_dims),
            )

        # Match param dtypes to activation statically (no CastLike nodes)
        k_val = cast_param_like(ctx, k_val, x_val)
        if use_bias:
            b_val = cast_param_like(ctx, b_val, x_val)

        # Fold kernel transpose into the initializer when possible
        k_rank = len(k_shape)
        k_perm = (
            k_rank - 1,
            k_rank - 2,
            *range(0, k_rank - 2),
        )  # (spatial..., I, O) -> (O, I, spatial...)
        k_oih = _maybe_fold_param_transpose(ctx, k_val, k_perm, name="kernel_oih")

        # Conv attributes
        conv_attrs: dict[str, Any] = {"strides": strides, "dilations": dilations}
        if groups and groups != 1:
            conv_attrs["group"] = int(groups)

        if isinstance(padding_param, str):
            pad_mode = padding_param.upper()
            if pad_mode == "VALID":
                # default pads=0; no attribute needed
                pass
            elif pad_mode == "SAME":
                if _all_ones(dilations):
                    # Legal to use auto_pad when dilation == 1
                    conv_attrs["auto_pad"] = "SAME_UPPER"
                else:
                    # SAME_* with dilation != 1 is not allowed by some runtimes (ORT).
                    # If spatial dims are concrete, compute explicit SAME_UPPER pads.
                    part_in_sp = (
                        x_shape[1 + extra : 1 + extra + conv_spatial]
                        if need_flatten
                        else x_shape[1 : 1 + conv_spatial]
                    )
                    if _is_concrete_shape(part_in_sp) and _is_concrete_shape(
                        k_shape[:conv_spatial]
                    ):
                        beg, end = _same_upper_pads_static(
                            [int(d) for d in part_in_sp],
                            [int(k) for k in k_shape[:conv_spatial]],
                            [int(s) for s in strides],
                            [int(d) for d in dilations],
                        )
                        conv_attrs["pads"] = (*beg, *end)
                    else:
                        # No safe static pads available; fall back to VALID semantics.
                        pass
            else:
                # Unknown string -> treat as VALID
                pass
        else:
            # explicit pads: ((l,h), ... ) -> [l..., h...]
            beg = [int(p[0]) for p in padding_param]
            end = [int(p[1]) for p in padding_param]
            conv_attrs["pads"] = (*beg, *end)

        # 3) Use Conv's bias input directly (no Add/Reshape, no CastLike)
        y = _conv(ctx, x_nchw, k_oih, b_val if use_bias else None, conv_attrs)
        # Try to annotate Conv's NCH... output when possible
        if not need_flatten and x_shape and k_shape:
            in_sp = x_shape[1 : 1 + conv_spatial]
            k_sp = k_shape[:conv_spatial]
            try:
                in_sp_ints = [int(d) for d in in_sp]
                k_sp_ints = [int(d) for d in k_sp]
                strides_ints = [int(s) for s in strides]
                dilations_ints = [int(d) for d in dilations]
                out_sp = _calc_out_spatial(
                    in_sp_ints, k_sp_ints, strides_ints, dilations_ints, padding_param
                )
            except Exception:
                out_sp = None
            if out_sp is not None:
                batch_dim = _dim_label_from_value_or_aval(x_val, x_shape, 0)
                if batch_dim is None:
                    batch_dim = x_shape[0]
                _annotate_value(
                    ctx,
                    y,
                    x_dtype_np,
                    (batch_dim, int(k_shape[-1]), *out_sp),
                )

        # Back to NH...C
        post_perm = (0, *range(2, x_spatial + 2), 1)  # (N,C,S...) -> (N,S...,C)
        final_value: ir.Value

        if need_flatten:
            # Prefer STATIC reshape back when concrete dims are available.
            y_nhwc_tmp = _transpose(ctx, y, post_perm)
            if _is_concrete_shape(x_shape) and _is_concrete_shape(k_shape):
                part_in_sp = x_shape[1 + extra : 1 + extra + conv_spatial]
                out_sp = _calc_out_spatial(
                    [int(d) for d in part_in_sp],
                    [int(k) for k in k_shape[:conv_spatial]],
                    [int(s) for s in strides],
                    [int(d) for d in dilations],
                    padding_param,
                )
                tgt_list = [*x_shape[: 1 + extra], *out_sp, int(k_shape[-1])]
                tgt = _const_i64(
                    ctx,
                    np.asarray(tgt_list, dtype=np.int64),
                    name_hint="reshape_out_static",
                )
                final_value = _reshape(ctx, y_nhwc_tmp, tgt)
                _stamp_type_and_shape(final_value, tuple(tgt_list))
                _ensure_value_metadata(ctx, final_value)
            else:
                # Dynamic fallback: Recover (N, extra..., out_spatial..., C_out)
                sh_x = _shape_of(ctx, x_val)  # [N, extra..., part..., C]
                sh_y = _shape_of(ctx, y_nhwc_tmp)  # [N', out sp..., C_out]

                # N and extra dims from original input
                idx_nextra = _const_i64(
                    ctx,
                    np.arange(0, 1 + extra, dtype=np.int64),
                    name_hint="idx_nextra_back",
                )
                n_extras = _gather(ctx, sh_x, idx_nextra, axis=0)  # [1+extra]

                # out spatial dims from y_nhwc (skip batch N')
                idx_outsp = _const_i64(
                    ctx,
                    np.arange(1, 1 + conv_spatial, dtype=np.int64),
                    name_hint="idx_outsp",
                )
                out_sp_dyn = _gather(ctx, sh_y, idx_outsp, axis=0)  # [conv_spatial]

                # output channel (last dim of y_nhwc) -> index is conv_spatial + 1
                idx_outc = _const_i64(
                    ctx,
                    np.array([conv_spatial + 1], dtype=np.int64),
                    name_hint="idx_outc",
                )
                out_c = _gather(ctx, sh_y, idx_outc, axis=0)  # [1]

                tgt = _concat0(
                    ctx, [n_extras, out_sp_dyn, out_c]
                )  # [1+extra + conv_spatial + 1]
                final_value = _reshape(ctx, y_nhwc_tmp, tgt)
                _ensure_value_metadata(ctx, final_value)
        else:
            final_value = _transpose(ctx, y, post_perm)
            if x_shape and k_shape:
                in_sp = x_shape[1 : 1 + conv_spatial]
                k_sp = k_shape[:conv_spatial]
                try:
                    in_sp_ints = [int(d) for d in in_sp]
                    k_sp_ints = [int(d) for d in k_sp]
                    strides_ints = [int(s) for s in strides]
                    dilations_ints = [int(d) for d in dilations]
                    out_sp = _calc_out_spatial(
                        in_sp_ints,
                        k_sp_ints,
                        strides_ints,
                        dilations_ints,
                        padding_param,
                    )
                except Exception:
                    out_sp = None
                if out_sp is not None:
                    batch_dim = _dim_label_from_value_or_aval(x_val, x_shape, 0)
                    if batch_dim is None:
                        batch_dim = x_shape[0]
                    _annotate_value(
                        ctx,
                        final_value,
                        x_dtype_np,
                        (batch_dim, *out_sp, int(k_shape[-1])),
                    )

        bind_value = getattr(ctx, "bind_value_for_var", None)
        if not callable(bind_value):
            raise AttributeError("IR build context missing bind_value_for_var")
        bind_value(y_var, final_value)
        return final_value

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.nnx", "conv_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx.Conv",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    # --- Back-compat for patching in environments that use `patch_info()` ---
    @staticmethod
    def patch_info():
        """
        Some runners still call `patch_info()` instead of `binding_specs()`.
        Provide a shim that applies the same monkey patch at activation time.
        """

        def _wrapper(orig):
            return ConvPlugin._make_patch(orig)

        return {
            "patch_targets": [nnx.Conv],
            "patch_function": _wrapper,
            "target_attribute": "__call__",
            # Expose the primitive on `flax.nnx.conv_p` too (old-world compat)
            "extra_assignments": [("flax.nnx", "conv_p", ConvPlugin._PRIM)],
        }


@ConvPlugin._PRIM.def_impl
def _impl(
    x,
    kernel,
    bias,
    *,
    use_bias,
    strides,
    padding,
    dilations,
    dimension_numbers,
    feature_group_count,
):
    # Fallback: assume NHWC → output NHWC with same batch & rank, compute via lax directly.
    # (We only need the shape/dtype here.)
    num_spatial: int = max(kernel.ndim - 2, 1)
    # Normalize to tuples without reassigning the params (keeps types stable for mypy)
    win_strides: Tuple[int, ...] = _to_int_tuple(strides, num_spatial)  # type: ignore[arg-type]
    rhs_dils: Tuple[int, ...] = _to_int_tuple(dilations, num_spatial)  # type: ignore[arg-type]
    # Build correct layouts for 1D/2D/3D: N{W|HW|DHW}C and {W|HW|DHW}IO
    layout_token: str = {1: "W", 2: "HW", 3: "DHW"}[num_spatial]
    in_layout = "N" + layout_token + "C"
    ker_layout = layout_token + "IO"
    out_layout = in_layout
    dn = lax.conv_dimension_numbers(
        x.shape, kernel.shape, (in_layout, ker_layout, out_layout)
    )
    y = (
        lax.conv_general_dilated(
            x,
            kernel,
            window_strides=tuple(win_strides),
            padding=padding,
            lhs_dilation=None,
            rhs_dilation=tuple(rhs_dils),
            dimension_numbers=dn,
            feature_group_count=feature_group_count,
        )
        if not use_bias
        else lax.conv_general_dilated(
            x,
            kernel,
            window_strides=tuple(win_strides),
            padding=padding,
            lhs_dilation=None,
            rhs_dilation=tuple(rhs_dils),
            dimension_numbers=dn,
            feature_group_count=feature_group_count,
        )
        + bias
    )
    return y


# Ensure environments that import this module outside the world activator
# still see the primitive and abstract eval properly bound.
try:
    # Make sure `flax.nnx.conv_p` exists and points at this primitive
    if not hasattr(nnx, "conv_p") or getattr(nnx, "conv_p") is not ConvPlugin._PRIM:
        setattr(nnx, "conv_p", ConvPlugin._PRIM)
    # (Re)bind abstract eval — def_abstract_eval overwrites any prior binding.
    ConvPlugin.ensure_abstract_eval_bound()
except Exception as _e:
    # Do not fail import-time; activation will call `ensure_abstract_eval_bound` again.
    logger.debug("conv plugin eager init skipped: %s", _e)

    def patch_info():
        """
        Some runners still call `patch_info()` instead of `binding_specs()`.
        Provide a shim that applies the same monkey patch at activation time.
        """

        def _wrapper(orig):
            return ConvPlugin._make_patch(orig)

        return {
            "patch_targets": [nnx.Conv],
            "patch_function": _wrapper,
            "target_attribute": "__call__",
            # Expose the primitive on `flax.nnx.conv_p` too (old-world compat)
            "extra_assignments": [("flax.nnx", "conv_p", ConvPlugin._PRIM)],
        }
