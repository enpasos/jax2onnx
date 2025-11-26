# jax2onnx/plugins/jax/lax/scan.py

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Final, Union

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import core as jax_core
from jax import lax

from jax2onnx.converter.ir_clone import clone_graph
from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._loop_extent_meta import set_axis0_override
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.lax._control_flow_utils import (
    builder_cast,
    builder_identity,
    builder_loop,
    clone_input_for_subgraph,
    clone_value_for_subgraph,
    create_loop_header_inputs,
    lower_jaxpr_eqns,
    make_subgraph_context,
    relax_value_to_rank_only,
)
from jax2onnx.plugins.jax.lax._index_utils import (
    _gather_int_scalar,
    _scalar_i64,
    _shape_of,
    _const_i64,
    _unsqueeze_scalar,
)

import jax.extend.core as jax_core_ext

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from jax2onnx.converter.ir_context import IRContext


def _jaxpr_contains_scatter(jpr_like: Any) -> bool:
    if hasattr(jpr_like, "eqns"):
        for eqn in jpr_like.eqns:
            prim_name = getattr(eqn.primitive, "name", "")
            if prim_name.startswith("scatter"):
                return True
            for val in getattr(eqn, "params", {}).values():
                if _jaxpr_contains_scatter(val):
                    return True
    if hasattr(jpr_like, "jaxpr"):
        return _jaxpr_contains_scatter(jpr_like.jaxpr)
    if isinstance(jpr_like, (tuple, list)):
        return any(_jaxpr_contains_scatter(item) for item in jpr_like)
    return False


def _static_scatter_extent(jpr_like: Any) -> int | None:
    if hasattr(jpr_like, "eqns"):
        for eqn in jpr_like.eqns:
            prim_name = getattr(eqn.primitive, "name", "")
            if prim_name.startswith("scatter") and len(getattr(eqn, "invars", ())) >= 3:
                updates_var = eqn.invars[2]
                aval = getattr(updates_var, "aval", None)
                shape = getattr(aval, "shape", None)
                if shape and len(shape) > 0:
                    dim0 = shape[0]
                    if isinstance(dim0, (int, np.integer)):
                        return int(dim0)
            for val in getattr(eqn, "params", {}).values():
                extent = _static_scatter_extent(val)
                if extent is not None:
                    return extent
    if hasattr(jpr_like, "jaxpr"):
        return _static_scatter_extent(jpr_like.jaxpr)
    if isinstance(jpr_like, (tuple, list)):
        for item in jpr_like:
            extent = _static_scatter_extent(item)
            if extent is not None:
                return extent
    return None


def _dtype_enum_for_var(var, enable_double: bool) -> ir.DataType | None:
    aval = getattr(var, "aval", None)
    if aval is None:
        return None
    aval_dtype = getattr(aval, "dtype", None)
    if aval_dtype is None:
        return None
    try:
        np_dtype = np.dtype(aval_dtype)
    except TypeError:
        return None
    if np.issubdtype(np_dtype, np.floating):
        return _dtype_to_ir(np_dtype, enable_double)
    return _dtype_to_ir(np_dtype, enable_double)


def _set_value_dtype_from_var(ctx, value: ir.Value, var) -> None:
    aval = getattr(var, "aval", None)
    aval_dtype = getattr(aval, "dtype", None)
    if aval_dtype is None:
        return
    try:
        np_dtype = np.dtype(aval_dtype)
    except TypeError:
        return
    promote_flag = getattr(ctx.builder, "enable_double_precision", False)
    if np.issubdtype(np_dtype, np.floating):
        promote_flag = False
    dtype_enum = _dtype_to_ir(np_dtype, promote_flag)
    try:
        value.type = ir.TensorType(dtype_enum, getattr(value, "shape", None))
    except Exception:
        value.type = ir.TensorType(dtype_enum)
    _ensure_value_metadata(ctx, value)


def _maybe_cast_value(
    ctx, value: ir.Value, target_enum: ir.DataType, *, force: bool = False
) -> ir.Value:
    current_enum = getattr(getattr(value, "type", None), "dtype", None)
    if not force and (current_enum is None or current_enum == target_enum):
        return value
    cast_val = builder_cast(
        ctx,
        value,
        target_enum,
        name_hint="scan_dtype_fix",
    )
    _ensure_value_metadata(ctx, cast_val)
    return cast_val


def _maybe_var_type(mod):
    if mod is None:
        return None
    try:
        return getattr(mod, "Var")
    except AttributeError:
        return None


_JAX_VAR_TYPES: Final[tuple[type, ...]] = tuple(
    t
    for t in (
        _maybe_var_type(jax_core),
        _maybe_var_type(jax_core_ext),
    )
    if t is not None
)


def _maybe_dropvar_type(mod):
    if mod is None:
        return None
    try:
        return getattr(mod, "DropVar")
    except AttributeError:
        return None


_DROP_VAR_TYPES: Final[tuple[type, ...]] = tuple(
    t
    for t in (
        _maybe_dropvar_type(jax_core),
        _maybe_dropvar_type(jax_core_ext),
    )
    if t is not None
)


def _is_dropvar(obj) -> bool:
    return bool(_DROP_VAR_TYPES) and isinstance(obj, _DROP_VAR_TYPES)


def _is_jax_var(obj) -> bool:
    return (
        bool(_JAX_VAR_TYPES)
        and isinstance(obj, _JAX_VAR_TYPES)
        and not _is_dropvar(obj)
    )


def scan_fn(x):
    def body(carry, _):
        carry = carry + 1
        return carry, carry

    _, ys = lax.scan(body, x, None, length=5)
    return ys


def _scan_jit_no_xs() -> jax.Array:
    def simulate():
        def step_fn(carry, _):
            return carry + 1, carry * 2

        _, ys = lax.scan(step_fn, 0, xs=None, length=10)
        return ys

    return jax.jit(simulate)()


def _nested_scan_len_mismatch_f32():
    xs_outer = jnp.arange(100, dtype=jnp.float32)

    def inner(carry, x):
        x = x.astype(carry.dtype)
        _, ys = lax.scan(lambda c, _: (c + x, c + x), carry, xs=None, length=5)
        return carry + x, ys[-1]

    _, ys = lax.scan(inner, jnp.asarray(0.0, dtype=jnp.float32), xs_outer)
    return ys


def _nested_scan_len_mismatch_f64():
    xs_outer = jnp.arange(100, dtype=jnp.float64)

    def inner(carry, x):
        x = x.astype(carry.dtype)
        _, ys = lax.scan(lambda c, _: (c + x, c + x), carry, xs=None, length=5)
        return carry + x, ys[-1]

    _, ys = lax.scan(inner, jnp.asarray(0.0, dtype=jnp.float64), xs_outer)
    return ys


def _two_scans_diff_len_f32():
    xs_small = jnp.asarray(np.arange(5, dtype=np.float32))
    xs_big = jnp.asarray(np.arange(100, dtype=np.float32))
    fill_small = jnp.asarray(np.full(xs_small.shape, 0.1, dtype=np.float32))
    fill_big = jnp.asarray(np.full(xs_big.shape, 0.1, dtype=np.float32))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_small, fill_small)
    )
    _, y2 = lax.scan(lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_big, fill_big))
    return y1, y2


def _two_scans_diff_len_f64():
    xs_small = jnp.asarray(np.arange(5, dtype=np.float64))
    xs_big = jnp.asarray(np.arange(100, dtype=np.float64))
    fill_small = jnp.asarray(np.full(xs_small.shape, 0.1, dtype=np.float64))
    fill_big = jnp.asarray(np.full(xs_big.shape, 0.1, dtype=np.float64))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_small, fill_small)
    )
    _, y2 = lax.scan(lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_big, fill_big))
    return y1, y2


def _two_scans_len_mismatch_broadcast_f32():
    xs_small = jnp.asarray(np.arange(5, dtype=np.float32))
    xs_big = jnp.asarray(np.arange(100, dtype=np.float32))
    fill = jnp.asarray(np.full(xs_big.shape, 0.1, dtype=np.float32))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_small, jnp.broadcast_to(0.1, xs_small.shape)),
    )
    _, y2 = lax.scan(lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_big, fill))
    return y1, y2


def _two_scans_len_mismatch_broadcast_f64():
    xs_small = jnp.asarray(np.arange(5, dtype=np.float64))
    xs_big = jnp.asarray(np.arange(100, dtype=np.float64))
    fill = jnp.asarray(np.full(xs_big.shape, 0.1, dtype=np.float64))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_small, jnp.broadcast_to(0.1, xs_small.shape)),
    )
    _, y2 = lax.scan(lambda c, xs: (c + xs[0] + xs[1], c), 0.0, xs=(xs_big, fill))
    return y1, y2


def _two_scans_diff_len_with_broadcast_f32():
    xs_small = jnp.asarray(np.arange(5, dtype=np.float32))
    xs_big = jnp.asarray(np.arange(100, dtype=np.float32))

    _, y1 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_small, jnp.broadcast_to(0.1, xs_small.shape)),
    )
    _, y2 = lax.scan(
        lambda c, xs: (c + xs[0] + xs[1], c),
        0.0,
        xs=(xs_big, jnp.full_like(xs_big, 0.1)),
    )
    return y1, y2


@register_primitive(
    jaxpr_primitive=jax.lax.scan_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html",
    onnx=[
        {
            "component": "Scan",
            "doc": "https://onnx.ai/onnx/operators/onnx__Scan.html",
        }
    ],
    since="v0.5.1",
    context="primitives.lax",
    component="scan",
    testcases=[
        {
            "testcase": "scan_identity_slice_helper",
            "callable": lambda x: jax.lax.scan(
                lambda c, xt: (c, jnp.squeeze(xt[None, ...][0:1, :, :, :], axis=0)),
                jnp.zeros(x.shape[1:], dtype=x.dtype),
                x,
            )[1],
            "input_shapes": [(2, 3, 4, 5)],
            "post_check_onnx_graph": EG(
                ["Loop"],
                search_functions=True,
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scan_cumsum",
            "callable": lambda xs: jax.lax.scan(
                lambda c, x: (c + x, c + x),
                jnp.zeros((), dtype=xs.dtype),
                xs,
            )[1],
            "input_shapes": [(5,)],
            "post_check_onnx_graph": EG(
                ["Loop"],
                search_functions=True,
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scan_carry_only",
            "callable": lambda xs: jax.lax.scan(
                lambda c, x: (c + x, c),
                jnp.zeros((), dtype=xs.dtype),
                xs,
            )[0],
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "path": "Loop",
                        "inputs": {
                            0: {"const": 3.0},
                            1: {"const_bool": True},
                            2: {"const": 0.0},
                        },
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scan_multiple_sequences",
            "callable": lambda xs, ys: jax.lax.scan(
                lambda c, xy: (c + xy[0] * xy[1], c + xy[0]),
                jnp.zeros((), dtype=xs.dtype),
                (xs, ys),
            )[1],
            "input_shapes": [(4,), (4,)],
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
        },
        {
            "testcase": "scan_multiple_carry",
            "callable": lambda xs: jax.lax.scan(
                lambda carry, x: (
                    (carry[0] + x, carry[1] * x),
                    carry[0] + carry[1],
                ),
                (
                    jnp.zeros((), dtype=xs.dtype),
                    jnp.ones((), dtype=xs.dtype),
                ),
                xs,
            )[1],
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
        },
        {
            "testcase": "scan_matrix_carry_multidim_xs",
            "callable": lambda init_carry, xs_seq: jax.lax.scan(
                lambda c_mat, x_slice: (c_mat + x_slice, jnp.sum(c_mat + x_slice)),
                init_carry,
                xs_seq,
            )[1],
            "input_shapes": [(3, 2), (5, 3, 2)],
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
        },
        {
            "testcase": "scan_no_xs",
            "callable": lambda x: jax.lax.scan(
                lambda carry, _: (carry + 1, carry), x, xs=None, length=5
            )[1],
            "input_shapes": [()],
            "input_dtypes": [jnp.float32],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
        },
        {
            "testcase": "scan_fn",
            "callable": scan_fn,
            "input_values": [jnp.array(0.0, dtype=jnp.float32)],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
        },
        {
            "testcase": "scan_jit_no_xs",
            "callable": _scan_jit_no_xs,
            "input_shapes": [],
            "expected_output_shapes": [(10,)],
            "expected_output_dtypes": [jnp.int32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
        },
        {
            "testcase": "scan_jit_no_xs_f64",
            "callable": _scan_jit_no_xs,
            "input_shapes": [],
            "expected_output_shapes": [(10,)],
            "expected_output_dtypes": [jnp.int64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
        },
        {
            "testcase": "scan_captured_scalar",
            "callable": (
                lambda dt=jnp.asarray(0.1, dtype=jnp.float32): (
                    jax.lax.scan(
                        lambda carry, _: (carry + dt, carry + dt),
                        jnp.asarray(0.0, dtype=jnp.float32),
                        xs=None,
                        length=3,
                    )[1]
                )
            ),
            "input_shapes": [],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
        },
        {
            "testcase": "scan_captured_scalar_f64",
            "callable": (
                lambda dt=jnp.asarray(0.1, dtype=jnp.float64): (
                    jax.lax.scan(
                        lambda carry, _: (carry + dt, carry + dt),
                        jnp.asarray(0.0, dtype=jnp.float64),
                        xs=None,
                        length=3,
                    )[1]
                )
            ),
            "input_shapes": [],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "expected_output_dtypes": [jnp.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
        },
        {
            "testcase": "scan_rank0_sequence_vectorized",
            "callable": (
                lambda xs_vec=jnp.arange(4, dtype=jnp.float32): jax.lax.scan(
                    lambda carry, xs: (carry + xs[0] + xs[1], carry),
                    0.0,
                    xs=(xs_vec, jnp.full(xs_vec.shape, 0.1, dtype=jnp.float32)),
                )[1]
            ),
            "input_shapes": [],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
        },
        {
            "testcase": "scan_rank0_sequence_vectorized_f64",
            "callable": (
                lambda xs_vec=jnp.arange(4, dtype=jnp.float64): jax.lax.scan(
                    lambda carry, xs: (carry + xs[0] + xs[1], carry),
                    0.0,
                    xs=(xs_vec, jnp.full(xs_vec.shape, 0.1, dtype=jnp.float64)),
                )[1]
            ),
            "input_shapes": [],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "expected_output_dtypes": [jnp.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
        },
        {
            "testcase": "scan_two_diff_lengths",
            "callable": _two_scans_diff_len_f32,
            "input_shapes": [],
            "expected_output_shapes": [
                ("JAX2ONNX_DYNAMIC_DIM_SENTINEL",),
                ("JAX2ONNX_DYNAMIC_DIM_SENTINEL",),
            ],
            "expected_output_dtypes": [jnp.float32, jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Loop",
                    "Loop",
                ],
                no_unused_inputs=True,
            ),
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_two_diff_lengths_f64",
            "callable": _two_scans_diff_len_f64,
            "input_shapes": [],
            "expected_output_shapes": [
                ("JAX2ONNX_DYNAMIC_DIM_SENTINEL",),
                ("JAX2ONNX_DYNAMIC_DIM_SENTINEL",),
            ],
            "expected_output_dtypes": [jnp.float64, jnp.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Loop",
                    "Loop",
                ],
                no_unused_inputs=True,
            ),
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_two_diff_lengths_broadcast",
            "callable": _two_scans_len_mismatch_broadcast_f32,
            "input_shapes": [],
            "expected_output_shapes": [
                ("JAX2ONNX_DYNAMIC_DIM_SENTINEL",),
                ("JAX2ONNX_DYNAMIC_DIM_SENTINEL",),
            ],
            "expected_output_dtypes": [jnp.float32, jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Loop",
                    "Loop",
                ],
                no_unused_inputs=True,
            ),
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_two_diff_lengths_broadcast_f64",
            "callable": _two_scans_len_mismatch_broadcast_f64,
            "input_shapes": [],
            "expected_output_shapes": [
                ("JAX2ONNX_DYNAMIC_DIM_SENTINEL",),
                ("JAX2ONNX_DYNAMIC_DIM_SENTINEL",),
            ],
            "expected_output_dtypes": [jnp.float64, jnp.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Loop",
                    "Loop",
                ],
                no_unused_inputs=True,
            ),
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_two_diff_lengths_with_broadcast",
            "callable": _two_scans_diff_len_with_broadcast_f32,
            "input_shapes": [],
            "expected_output_shapes": [
                ("JAX2ONNX_DYNAMIC_DIM_SENTINEL",),
                ("JAX2ONNX_DYNAMIC_DIM_SENTINEL",),
            ],
            "expected_output_dtypes": [jnp.float32, jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Loop",
                    "Loop",
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "scan_nested_len_mismatch",
            "callable": _nested_scan_len_mismatch_f32,
            "input_shapes": [],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Range -> Shape -> Gather -> Squeeze -> Loop"],
                no_unused_inputs=True,
            ),
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_nested_len_mismatch_f64",
            "callable": _nested_scan_len_mismatch_f64,
            "input_shapes": [],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "expected_output_dtypes": [jnp.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Range -> Shape -> Gather -> Squeeze -> Loop"],
                no_unused_inputs=True,
            ),
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_captured_scalar_with_xs",
            "callable": (
                lambda xs, dt=jnp.asarray(0.1, dtype=jnp.float32): (
                    jax.lax.scan(
                        lambda carry, x: (carry + dt * x, carry + dt * x),
                        jnp.asarray(0.0, dtype=jnp.float32),
                        xs,
                    )[1]
                )
            ),
            "input_shapes": [(8,)],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL",)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
            "check_onnx_load": True,
        },
        {
            "testcase": "scan_captured_vector_with_xs_f64",
            "callable": (
                lambda xs, dt=jnp.asarray([0.1, -0.2], dtype=jnp.float64): (
                    jax.lax.scan(
                        lambda carry, x: (carry + dt * x, carry + dt * x),
                        jnp.zeros((2,), dtype=jnp.float64),
                        xs,
                    )[1]
                )
            ),
            "input_shapes": [(5, 2)],
            "expected_output_shapes": [("JAX2ONNX_DYNAMIC_DIM_SENTINEL", 2)],
            "expected_output_dtypes": [jnp.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                ["Loop"], search_functions=True, no_unused_inputs=True
            ),
            "check_onnx_load": True,
        },
    ],
)
class ScanPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: IRContext, eqn):  # type: ignore[name-defined]
        params = eqn.params
        closed_jaxpr = params["jaxpr"]
        num_carry = int(params.get("num_carry", 0))
        num_consts = int(params.get("num_consts", 0) or 0)
        if params.get("reverse", False):
            raise NotImplementedError("Reverse scan is not supported in IR pipeline.")
        if params.get("unroll", 1) != 1:
            raise NotImplementedError(
                "Scan with unroll > 1 is not supported in IR pipeline."
            )

        jaxpr = closed_jaxpr.jaxpr
        total_invars = len(jaxpr.invars)
        num_scan = int(params.get("num_xs", total_invars - num_carry - num_consts))
        if num_scan < 0 or (num_carry + num_scan + num_consts) != total_invars:
            raise ValueError(
                "Inconsistent Scan arity: expected consts/carry/scan to match jaxpr invars"
            )
        length = params.get("length")

        if num_scan == 0:
            node_outputs = self._lower_without_scan_inputs(
                ctx, eqn, closed_jaxpr, num_carry, num_consts, length
            )
        else:
            node_outputs = self._lower_with_scan_inputs(
                ctx, eqn, closed_jaxpr, num_carry, num_consts, num_scan, length
            )

        carry_outvars = list(closed_jaxpr.jaxpr.outvars[:num_carry])
        scan_outvars = list(closed_jaxpr.jaxpr.outvars[num_carry:])
        output_indices: list[int] = []
        eqn_idx = 0
        for i, carry_var in enumerate(carry_outvars):
            if not _is_dropvar(carry_var):
                output_indices.append(num_consts + i)
                eqn_idx += 1
        for j, _ in enumerate(scan_outvars):
            output_indices.append(num_consts + num_carry + num_scan + j)

        carry_invars = eqn.invars[num_consts : num_consts + num_carry]
        scan_invars = eqn.invars[
            num_consts + num_carry : num_consts + num_carry + num_scan
        ]
        has_dynamic_inputs = any(_is_jax_var(v) for v in carry_invars + scan_invars)

        for idx, var in enumerate(eqn.outvars):
            out_idx = output_indices[idx]
            val = node_outputs[out_idx]
            aval = getattr(var, "aval", None)
            if aval is not None:
                aval_shape = tuple(getattr(aval, "shape", ()))
                if idx >= num_carry and aval_shape:
                    first_dim = aval_shape[0]
                    first_dim_resolved: Union[int, str]
                    builder_stacktrace_enabled = False
                    builder_obj = getattr(ctx, "builder", None)
                    if builder_obj is not None:
                        stacktrace_flag = getattr(
                            builder_obj, "stacktrace_metadata_enabled", None
                        )
                        if callable(stacktrace_flag):
                            builder_stacktrace_enabled = bool(stacktrace_flag())
                    if (
                        num_consts == 0
                        and not has_dynamic_inputs
                        and isinstance(first_dim, (int, np.integer))
                    ):
                        first_dim_resolved = int(first_dim)
                    elif (
                        num_consts == 0
                        and not has_dynamic_inputs
                        and isinstance(length, (int, np.integer))
                    ):
                        first_dim_resolved = int(length)
                    elif (
                        isinstance(first_dim, (int, np.integer))
                        and builder_stacktrace_enabled
                    ):
                        first_dim_resolved = int(first_dim)
                    else:
                        first_dim_resolved = _DYNAMIC_DIM_SENTINEL
                    desired_shape = (first_dim_resolved,) + aval_shape[1:]
                else:
                    desired_shape = aval_shape
                desired_np_dtype = np.dtype(getattr(aval, "dtype", np.float32))
                if np.issubdtype(desired_np_dtype, np.integer):
                    target_enum = (
                        ir.DataType.INT64
                        if ctx.builder.enable_double_precision
                        else ir.DataType.INT32
                    )
                    cast_val = builder_cast(
                        ctx,
                        val,
                        target_enum,
                        name_hint="scan_out_cast",
                    )
                    _ensure_value_metadata(ctx, cast_val)
                    node_outputs[out_idx] = cast_val
                    try:
                        ctx.builder._var2val[var] = cast_val
                    except TypeError:
                        pass
                    val = cast_val
                _stamp_type_and_shape(val, desired_shape)

    def _lower_without_scan_inputs(
        self,
        ctx: IRContext,  # type: ignore[name-defined]
        eqn,
        closed_jaxpr,
        num_carry: int,
        num_consts: int,
        length,
    ) -> list[ir.Value]:
        if not isinstance(length, (int, np.integer)):
            raise NotImplementedError(
                "Scan without xs requires a static length in the IR pipeline."
            )

        jaxpr = closed_jaxpr.jaxpr
        loop_ctx = make_subgraph_context(ctx, prefix="scan_loop")

        has_scatter = _jaxpr_contains_scatter(jaxpr)
        jaxpr = closed_jaxpr.jaxpr
        dtypes = [
            np.dtype(getattr(var.aval, "dtype"))
            for var in list(jaxpr.invars) + list(jaxpr.outvars)
            if hasattr(var, "aval") and getattr(var.aval, "dtype", None) is not None
        ]
        if (
            dtypes
            and any(dt == np.float32 for dt in dtypes)
            and not any(dt == np.float64 for dt in dtypes)
        ):
            setattr(loop_ctx, "_keep_function_float32", True)
        const_float32 = any(
            hasattr(var, "aval")
            and getattr(var.aval, "dtype", None) is not None
            and np.dtype(var.aval.dtype) == np.float32
            for var in jaxpr.invars[:num_consts]
        )
        if const_float32:
            setattr(loop_ctx, "_keep_function_float32", True)
        # Only retain float32 when there are no scanned inputs (handled in the no-xs code path).

        for cv, cval in zip(jaxpr.constvars, closed_jaxpr.consts):
            np_c = np.asarray(cval)
            aval = getattr(cv, "aval", None)
            if aval is not None:
                np_c = np_c.astype(getattr(aval, "dtype", np_c.dtype), copy=False)
            loop_ctx.bind_const_for_var(cv, np_c)

        iter_in, cond_in = create_loop_header_inputs(
            loop_ctx,
            prefix="scan_loop",
        )

        function_keep_float32 = getattr(loop_ctx, "_keep_function_float32", False)
        allow_double_consts = (
            ctx.builder.enable_double_precision and not function_keep_float32
        )

        state_inputs: list[ir.Value] = []
        for idx, var in enumerate(jaxpr.invars):
            state_input = loop_ctx.add_input_for_invar(var, idx + 2)
            if idx < num_consts:
                target_enum = _dtype_enum_for_var(var, allow_double_consts)
                casted = _maybe_cast_value(loop_ctx, state_input, target_enum)
                if casted is not state_input:
                    loop_ctx.bind_value_for_var(var, casted)
                    state_input = casted
            _set_value_dtype_from_var(loop_ctx, state_input, var)
            relax_value_to_rank_only(state_input)
            state_inputs.append(state_input)

        loop_extent_scalar = _scalar_i64(loop_ctx, int(length), "scan_loop_extent")
        loop_extent_vec = _unsqueeze_scalar(
            loop_ctx, loop_extent_scalar, 0, "scan_loop_extent_vec"
        )
        trip_count_val: ir.Value | None = None
        extent_hints = getattr(loop_ctx, "_loop_extent_hints", None)
        if not isinstance(extent_hints, dict):
            extent_hints = {}
            setattr(loop_ctx, "_loop_extent_hints", extent_hints)
        extent_hints.setdefault(0, []).append(loop_extent_vec)
        scatter_static_extent = _static_scatter_extent(jaxpr) if has_scatter else None
        if has_scatter:
            setattr(loop_ctx, "_loop_extent_hints_enabled", True)
        if os.environ.get("J2O_DEBUG_LOOP_HINTS") == "1":
            print("[loop_hint_static]", scatter_static_extent, flush=True)
        if scatter_static_extent is not None:
            setattr(loop_ctx, "_force_loop_extent_axis0", True)
            setattr(loop_ctx, "_static_loop_extent_axis0", int(scatter_static_extent))
            if not isinstance(extent_hints, dict):
                extent_hints = {}
                setattr(loop_ctx, "_loop_extent_hints", extent_hints)
            override_scalar = _scalar_i64(
                loop_ctx,
                int(scatter_static_extent),
                "scan_loop_extent_override",
            )
            override_vec = _unsqueeze_scalar(
                loop_ctx,
                override_scalar,
                0,
                "scan_loop_extent_override_vec",
            )
            axis_hints = extent_hints.setdefault(0, [])
            axis_hints.clear()
            axis_hints.append(override_vec)
        if os.environ.get("J2O_DEBUG_LOOP_DTYPES") == "1":
            print(
                "[scan_no_xs_state]",
                [
                    (
                        idx,
                        getattr(getattr(var, "aval", None), "dtype", None),
                        getattr(
                            getattr(state_inputs[idx], "type", None), "dtype", None
                        ),
                    )
                    for idx, var in enumerate(jaxpr.invars)
                ],
                flush=True,
            )

        keep_float32 = function_keep_float32
        allow_double_outputs = ctx.builder.enable_double_precision and not keep_float32

        lower_jaxpr_eqns(loop_ctx, jaxpr)

        body_outputs: list[ir.Value] = []

        cond_out = builder_identity(
            loop_ctx,
            cond_in,
            name_hint="loop_cond_out",
        )
        cond_out.type = ir.TensorType(ir.DataType.BOOL)
        body_outputs.append(cond_out)

        for i in range(num_consts):
            inp_val = state_inputs[i]
            out_val = builder_identity(
                loop_ctx,
                inp_val,
                name_hint="loop_const_out",
            )
            relax_value_to_rank_only(out_val)
            body_outputs.append(out_val)

        for out_var in jaxpr.outvars[:num_carry]:
            out_val = loop_ctx.get_value_for_var(out_var)
            if os.environ.get("J2O_DEBUG_LOOP_DTYPES") == "1":
                print(
                    "[scan_no_xs_out_carry]",
                    getattr(getattr(out_var, "aval", None), "dtype", None),
                    flush=True,
                )
            _set_value_dtype_from_var(loop_ctx, out_val, out_var)
            relax_value_to_rank_only(out_val)
            body_outputs.append(out_val)

        carry_outputs_start = 1 + num_consts
        for rel_idx, out_var in enumerate(jaxpr.outvars[:num_carry]):
            value = body_outputs[carry_outputs_start + rel_idx]
            target_enum = _dtype_enum_for_var(out_var, allow_double_outputs)
            if target_enum is None:
                continue
            casted = _maybe_cast_value(loop_ctx, value, target_enum, force=True)
            if casted is not value:
                body_outputs[carry_outputs_start + rel_idx] = casted
                loop_ctx.bind_value_for_var(out_var, casted)

        for out_var in jaxpr.outvars[num_carry:]:
            out_val = loop_ctx.get_value_for_var(out_var)
            if os.environ.get("J2O_DEBUG_LOOP_DTYPES") == "1":
                print(
                    "[scan_no_xs_out_seq]",
                    getattr(getattr(out_var, "aval", None), "dtype", None),
                    flush=True,
                )
            _set_value_dtype_from_var(loop_ctx, out_val, out_var)
            relax_value_to_rank_only(out_val)
            body_outputs.append(out_val)

        outputs_start_no_xs = 1 + num_consts + num_carry
        for rel_idx, out_var in enumerate(jaxpr.outvars[num_carry:]):
            value = body_outputs[outputs_start_no_xs + rel_idx]
            _set_value_dtype_from_var(loop_ctx, value, out_var)
            target_enum = _dtype_enum_for_var(out_var, allow_double_outputs)
            casted = _maybe_cast_value(loop_ctx, value, target_enum, force=True)
            if casted is not value:
                body_outputs[outputs_start_no_xs + rel_idx] = casted
                loop_ctx.bind_value_for_var(out_var, casted)
            if os.environ.get("J2O_DEBUG_LOOP_DTYPES") == "1":
                print(
                    "[scan_no_xs_body_output]",
                    getattr(getattr(out_var, "aval", None), "dtype", None),
                    getattr(getattr(casted, "type", None), "dtype", None),
                    flush=True,
                )
            if os.environ.get("J2O_DEBUG_LOOP_DTYPES") == "1":
                print(
                    "[scan_no_xs_body_output]",
                    getattr(getattr(out_var, "aval", None), "dtype", None),
                    getattr(getattr(value, "type", None), "dtype", None),
                    flush=True,
                )

        loop_ctx.builder.outputs = body_outputs

        body_graph = clone_graph(loop_ctx.builder.graph)
        body_graph.name = ctx.fresh_name("scan_loop_body")
        opset_imports = dict(body_graph.opset_imports)
        opset_imports.setdefault("", getattr(ctx.builder, "opset", 21))
        body_graph.opset_imports.clear()
        body_graph.opset_imports.update(opset_imports)

        if trip_count_val is None:
            trip_count = _scalar_i64(ctx, int(length), "scan_trip_count")
        else:
            trip_count = trip_count_val

        cond_init = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("scan_cond_init"),
            array=np.asarray(True, dtype=np.bool_),
        )

        node_inputs = [trip_count, cond_init]
        inbound_vals: list[ir.Value] = []
        for idx, v in enumerate(eqn.invars):
            top_val = ctx.get_value_for_var(v)
            if idx < len(state_inputs):
                state_enum = getattr(
                    getattr(state_inputs[idx], "type", None), "dtype", None
                )
                expected_enum = _dtype_enum_for_var(v, allow_double_consts)
                if expected_enum is None:
                    expected_enum = state_enum
                else:
                    aval_dtype = getattr(getattr(v, "aval", None), "dtype", None)
                    if aval_dtype is not None:
                        try:
                            np_dtype = np.dtype(aval_dtype)
                        except TypeError:
                            np_dtype = None
                        else:
                            if (
                                state_enum is not None
                                and expected_enum != state_enum
                                and ctx.builder.enable_double_precision
                                and np_dtype is not None
                                and np.issubdtype(np_dtype, np.floating)
                            ):
                                expected_enum = state_enum
                top_dtype = getattr(getattr(top_val, "type", None), "dtype", None)
                if (
                    expected_enum is not None
                    and top_dtype is not None
                    and expected_enum != top_dtype
                ):
                    top_val = builder_cast(
                        ctx,
                        top_val,
                        expected_enum,
                        name_hint="scan_state_cast",
                    )
                    _ensure_value_metadata(ctx, top_val)
            inbound_vals.append(top_val)
        node_inputs.extend(inbound_vals)

        node_outputs: list[ir.Value] = []

        const_body_outs = body_outputs[1 : 1 + num_consts]
        for body_out in const_body_outs:
            node_outputs.append(
                clone_value_for_subgraph(
                    ctx,
                    body_out,
                    name_hint="loop_const_unused",
                )
            )

        carry_body_outs = body_outputs[1 + num_consts : 1 + num_consts + num_carry]
        for idx, body_out in enumerate(carry_body_outs):
            out_var = eqn.outvars[idx] if idx < len(eqn.outvars) else None
            if out_var is not None and not _is_dropvar(out_var):
                top_val = ctx.get_value_for_var(out_var)
                expected_enum = _dtype_enum_for_var(out_var, allow_double_outputs)
                if expected_enum is None:
                    expected_enum = getattr(
                        getattr(body_out, "type", None), "dtype", None
                    )
                top_dtype = getattr(getattr(top_val, "type", None), "dtype", None)
                if (
                    expected_enum is not None
                    and top_dtype is not None
                    and expected_enum != top_dtype
                ):
                    replacement = ir.Value(
                        name=ctx.fresh_name("loop_out"),
                        type=ir.TensorType(expected_enum),
                        shape=body_out.shape,
                    )
                    ctx.bind_value_for_var(out_var, replacement)
                    node_outputs.append(replacement)
                else:
                    node_outputs.append(top_val)
            else:
                node_outputs.append(
                    clone_value_for_subgraph(
                        ctx,
                        body_out,
                        name_hint="loop_state_unused",
                    )
                )

        body_seq_outvars = list(jaxpr.outvars[num_carry:])
        seq_axis_overrides: list[int | None] = []
        ctx_override_extent = getattr(loop_ctx, "_static_loop_extent_axis0", None)
        for body_out_var in body_seq_outvars:
            aval = getattr(body_out_var, "aval", None)
            axis0_extent = None
            if aval is not None:
                shape = getattr(aval, "shape", ())
                if (
                    isinstance(shape, tuple)
                    and shape
                    and isinstance(shape[0], (int, np.integer))
                    and int(shape[0]) > 1
                ):
                    axis0_extent = int(shape[0])
            if (
                axis0_extent is None
                and isinstance(ctx_override_extent, (int, np.integer))
                and int(ctx_override_extent) > 1
            ):
                axis0_extent = int(ctx_override_extent)
            seq_axis_overrides.append(axis0_extent)

        for rel_idx, out_var in enumerate(eqn.outvars[num_carry:]):
            top_val = ctx.get_value_for_var(
                out_var, name_hint=ctx.fresh_name("loop_out")
            )
            override_extent = (
                seq_axis_overrides[rel_idx]
                if rel_idx < len(seq_axis_overrides)
                else None
            )
            if (
                isinstance(override_extent, (int, np.integer))
                and int(override_extent) > 1
            ):
                override_int = int(override_extent)
                set_axis0_override(top_val, override_int)
                aval_shape = getattr(getattr(out_var, "aval", None), "shape", None)
                if isinstance(aval_shape, tuple):
                    aval_shape_tuple = aval_shape
                else:
                    aval_shape_tuple = None
                _restamp_axis0(top_val, override_int, aval_shape_tuple)
            expected_enum = _dtype_enum_for_var(out_var, allow_double_outputs)
            if expected_enum is None:
                expected_enum = getattr(
                    getattr(
                        body_outputs[1 + num_consts + num_carry + rel_idx],
                        "type",
                        None,
                    ),
                    "dtype",
                    None,
                )
            top_dtype = getattr(getattr(top_val, "type", None), "dtype", None)
            if (
                expected_enum is not None
                and top_dtype is not None
                and expected_enum != top_dtype
            ):
                replacement = ir.Value(
                    name=ctx.fresh_name("loop_out"),
                    type=ir.TensorType(expected_enum),
                    shape=top_val.shape,
                )
                ctx.bind_value_for_var(out_var, replacement)
                node_outputs.append(replacement)
            else:
                node_outputs.append(top_val)

        output_names = [
            getattr(val, "name", None) or ctx.fresh_name("scan_out")
            for val in node_outputs
        ]

        loop_results = builder_loop(
            ctx,
            *node_inputs,
            body=body_graph,
            output_names=output_names,
        )

        if not isinstance(loop_results, tuple):
            loop_results = (loop_results,)

        for template, produced in zip(node_outputs, loop_results):
            tmpl_meta = getattr(template, "meta", None)
            prod_meta = getattr(produced, "meta", None)
            if tmpl_meta is not None and prod_meta is not None:
                for key, value in tmpl_meta.items():
                    prod_meta[key] = value
            tmpl_type = getattr(template, "type", None)
            if tmpl_type is not None:
                produced.type = tmpl_type
            tmpl_shape = getattr(template, "shape", None)
            if tmpl_shape is not None:
                produced.shape = tmpl_shape
            _ensure_value_metadata(ctx, produced)

        loop_override_extent = getattr(loop_ctx, "_static_loop_extent_axis0", None)
        if (
            getattr(loop_ctx, "_force_loop_extent_axis0", False)
            and isinstance(loop_override_extent, (int, np.integer))
            and int(loop_override_extent) > 1
        ):
            override_int = int(loop_override_extent)
            for produced in loop_results:
                set_axis0_override(produced, override_int)

        const_count = len(const_body_outs)
        carry_count = num_carry
        seq_count = 0

        carry_results = loop_results[const_count : const_count + carry_count]
        value_results = loop_results[const_count + carry_count + seq_count :]

        for idx, res in enumerate(carry_results):
            if idx >= len(eqn.outvars):
                break
            out_var = eqn.outvars[idx]
            if _is_dropvar(out_var):
                continue
            ctx.bind_value_for_var(out_var, res)

        for rel_idx, out_var in enumerate(eqn.outvars[num_carry:]):
            if rel_idx >= len(value_results):
                break
            override_extent = (
                seq_axis_overrides[rel_idx]
                if rel_idx < len(seq_axis_overrides)
                else None
            )
            if (
                override_extent is None
                and isinstance(ctx_override_extent, (int, np.integer))
                and int(ctx_override_extent) > 1
            ):
                override_extent = int(ctx_override_extent)
            if (
                isinstance(override_extent, (int, np.integer))
                and int(override_extent) > 1
            ):
                override_int = int(override_extent)
                set_axis0_override(value_results[rel_idx], override_int)
                aval_shape = getattr(
                    getattr(eqn.outvars[num_carry + rel_idx], "aval", None),
                    "shape",
                    None,
                )
                aval_shape_tuple = aval_shape if isinstance(aval_shape, tuple) else None
                _restamp_axis0(value_results[rel_idx], override_int, aval_shape_tuple)
            ctx.bind_value_for_var(out_var, value_results[rel_idx])

        return list(loop_results)

    def _lower_with_scan_inputs(
        self,
        ctx: IRContext,  # type: ignore[name-defined]
        eqn,
        closed_jaxpr,
        num_carry: int,
        num_consts: int,
        num_scan: int,
        length,
    ) -> list[ir.Value]:
        jaxpr = closed_jaxpr.jaxpr

        seq_invars = list(
            eqn.invars[num_consts + num_carry : num_consts + num_carry + num_scan]
        )
        if not seq_invars:
            raise ValueError("Expected at least one scanned input when num_scan > 0")

        if length is None:
            first_seq_aval = getattr(seq_invars[0], "aval", None)
            if first_seq_aval is None or not getattr(first_seq_aval, "shape", ()):  # type: ignore[arg-type]
                trip_count_int = None
            else:
                dim0 = first_seq_aval.shape[0]
                trip_count_int = (
                    int(dim0) if isinstance(dim0, (int, np.integer)) else None
                )
        elif isinstance(length, (int, np.integer)):
            trip_count_int = int(length)
        else:
            trip_count_int = None

        for seq_var in seq_invars:
            aval = getattr(seq_var, "aval", None)
            if aval is None or not getattr(aval, "shape", ()):  # type: ignore[arg-type]
                continue
            dim0 = aval.shape[0]
            if trip_count_int is not None and isinstance(dim0, (int, np.integer)):
                if int(dim0) != trip_count_int:
                    raise ValueError(
                        "All scanned inputs must share the same leading dimension"
                    )

        loop_ctx = make_subgraph_context(ctx, prefix="scan_loop")
        jaxpr = closed_jaxpr.jaxpr
        dtypes = [
            np.dtype(getattr(var.aval, "dtype"))
            for var in list(jaxpr.invars) + list(jaxpr.outvars)
            if hasattr(var, "aval") and getattr(var.aval, "dtype", None) is not None
        ]
        if (
            dtypes
            and any(dt == np.float32 for dt in dtypes)
            and not any(dt == np.float64 for dt in dtypes)
        ):
            setattr(loop_ctx, "_keep_function_float32", True)

        for cv, cval in zip(jaxpr.constvars, closed_jaxpr.consts):
            np_c = np.asarray(cval)
            aval = getattr(cv, "aval", None)
            if aval is not None:
                np_c = np_c.astype(getattr(aval, "dtype", np_c.dtype), copy=False)
            loop_ctx.bind_const_for_var(cv, np_c)

        iter_in, cond_in = create_loop_header_inputs(
            loop_ctx,
            prefix="scan_loop",
        )

        function_keep_float32 = getattr(loop_ctx, "_keep_function_float32", False)
        allow_double_consts = (
            ctx.builder.enable_double_precision and not function_keep_float32
        )

        has_scatter = _jaxpr_contains_scatter(jaxpr)

        state_inputs: list[ir.Value] = []
        for idx, var in enumerate(jaxpr.invars[: num_consts + num_carry]):
            state_input = loop_ctx.add_input_for_invar(var, idx + 2)
            if idx < num_consts:
                target_enum = _dtype_enum_for_var(var, allow_double_consts)
                casted = _maybe_cast_value(loop_ctx, state_input, target_enum)
                if casted is not state_input:
                    loop_ctx.bind_value_for_var(var, casted)
                    state_input = casted
            _set_value_dtype_from_var(loop_ctx, state_input, var)
            relax_value_to_rank_only(state_input)
            state_inputs.append(state_input)

        if os.environ.get("J2O_DEBUG_LOOP_DTYPES") == "1":
            print(
                "[scan_with_inputs_state]",
                [
                    (
                        idx,
                        getattr(getattr(var, "aval", None), "dtype", None),
                        getattr(
                            getattr(state_inputs[idx], "type", None), "dtype", None
                        ),
                    )
                    for idx, var in enumerate(jaxpr.invars[: num_consts + num_carry])
                ],
                flush=True,
            )

        sequence_states: list[ir.Value] = []
        for seq_eqn_var in seq_invars:
            outer_seq_val = ctx.get_value_for_var(seq_eqn_var)
            seq_state = clone_input_for_subgraph(
                loop_ctx,
                outer_seq_val,
                name_hint="scan_seq_state",
            )
            sequence_states.append(seq_state)

        extent_hints = getattr(loop_ctx, "_loop_extent_hints", None)
        if has_scatter and sequence_states:
            seq_state = sequence_states[0]
            seq_shape = _shape_of(loop_ctx, seq_state, "scan_seq_state_shape")
            loop_extent_scalar = _gather_int_scalar(
                loop_ctx, seq_shape, 0, "scan_loop_extent_seq"
            )
            loop_extent_vec = _unsqueeze_scalar(
                loop_ctx, loop_extent_scalar, 0, "scan_loop_extent_vec"
            )
            if not isinstance(extent_hints, dict):
                extent_hints = {}
                setattr(loop_ctx, "_loop_extent_hints", extent_hints)
            extent_hints.setdefault(0, []).append(loop_extent_vec)
            setattr(loop_ctx, "_loop_extent_hints_enabled", True)
        scatter_static_extent = _static_scatter_extent(jaxpr) if has_scatter else None
        if scatter_static_extent is not None:
            setattr(loop_ctx, "_force_loop_extent_axis0", True)
            setattr(loop_ctx, "_static_loop_extent_axis0", int(scatter_static_extent))
            if not isinstance(extent_hints, dict):
                extent_hints = {}
                setattr(loop_ctx, "_loop_extent_hints", extent_hints)
            override_scalar = _scalar_i64(
                loop_ctx,
                int(scatter_static_extent),
                "scan_loop_extent_override",
            )
            override_vec = _unsqueeze_scalar(
                loop_ctx,
                override_scalar,
                0,
                "scan_loop_extent_override_vec",
            )
            axis_hints = extent_hints.setdefault(0, [])
            axis_hints.clear()
            axis_hints.append(override_vec)
            trip_count_val = _scalar_i64(
                ctx,
                int(scatter_static_extent),
                "scan_trip_extent_static",
            )
        scan_input_vars = jaxpr.invars[
            num_consts + num_carry : num_consts + num_carry + num_scan
        ]
        keep_float32 = function_keep_float32
        allow_double_outputs = ctx.builder.enable_double_precision and not keep_float32
        for seq_state, per_step_var in zip(sequence_states, scan_input_vars):
            per_step_val = loop_ctx.get_value_for_var(
                per_step_var, name_hint=loop_ctx.fresh_name("scan_elem")
            )
            per_step_val = loop_ctx.builder.Gather(
                seq_state,
                iter_in,
                axis=0,
                _outputs=[loop_ctx.fresh_name("GatherScanInput")],
            )
            loop_ctx.bind_value_for_var(per_step_var, per_step_val)
            aval_dtype = getattr(getattr(per_step_var, "aval", None), "dtype", None)
            if aval_dtype is not None:
                try:
                    np_dtype = np.dtype(aval_dtype)
                except TypeError:
                    np_dtype = None
                if np_dtype is not None and np.issubdtype(np_dtype, np.floating):
                    target_enum = ir.DataType.DOUBLE
                    if keep_float32:
                        target_enum = ir.DataType.FLOAT
                    per_step_val.type = ir.TensorType(target_enum)
                    _ensure_value_metadata(loop_ctx, per_step_val)
                    loop_ctx.bind_value_for_var(per_step_var, per_step_val)
            if (
                scatter_static_extent is not None
                and isinstance(scatter_static_extent, (int, np.integer))
                and int(scatter_static_extent) > 1
            ):
                per_step_shape = getattr(
                    getattr(per_step_var, "aval", None), "shape", ()
                )
                rank = len(per_step_shape)
                if rank >= 1:
                    gather_shape = _shape_of(
                        loop_ctx, per_step_val, "scan_per_step_shape"
                    )
                    start_tail = _const_i64(
                        loop_ctx,
                        np.asarray([1], dtype=np.int64),
                        "scan_per_step_shape_start",
                    )
                    limit_tail = _const_i64(
                        loop_ctx,
                        np.asarray([rank], dtype=np.int64),
                        "scan_per_step_shape_limit",
                    )
                    axes_tail = _const_i64(
                        loop_ctx,
                        np.asarray([0], dtype=np.int64),
                        "scan_per_step_shape_axes",
                    )
                    tail_shape = loop_ctx.builder.Slice(
                        gather_shape,
                        start_tail,
                        limit_tail,
                        axes_tail,
                        _outputs=[loop_ctx.fresh_name("scan_per_step_tail_shape")],
                    )
                    override_scalar = _scalar_i64(
                        loop_ctx,
                        int(scatter_static_extent),
                        "scan_per_step_extent",
                    )
                    override_vec = _unsqueeze_scalar(
                        loop_ctx,
                        override_scalar,
                        0,
                        "scan_per_step_extent_vec",
                    )
                    target_shape = loop_ctx.builder.Concat(
                        override_vec,
                        tail_shape,
                        axis=0,
                        _outputs=[loop_ctx.fresh_name("scan_per_step_target_shape")],
                    )
                    expanded = loop_ctx.builder.Expand(
                        per_step_val,
                        target_shape,
                        _outputs=[loop_ctx.fresh_name("scan_per_step_expand")],
                    )
                    _ensure_value_metadata(loop_ctx, expanded)
                    loop_ctx.bind_value_for_var(per_step_var, expanded)

        lower_jaxpr_eqns(loop_ctx, jaxpr)

        body_outputs: list[ir.Value] = []

        cond_out = builder_identity(
            loop_ctx,
            cond_in,
            name_hint="loop_cond_out",
        )
        cond_out.type = ir.TensorType(ir.DataType.BOOL)
        body_outputs.append(cond_out)

        for i in range(num_consts):
            inp_val = state_inputs[i]
            out_val = builder_identity(
                loop_ctx,
                inp_val,
                name_hint="loop_const_out",
            )
            relax_value_to_rank_only(out_val)
            body_outputs.append(out_val)

        for out_var in jaxpr.outvars[:num_carry]:
            out_val = loop_ctx.get_value_for_var(out_var)
            if os.environ.get("J2O_DEBUG_LOOP_DTYPES") == "1":
                print(
                    "[scan_with_inputs_out_carry]",
                    getattr(getattr(out_var, "aval", None), "dtype", None),
                    flush=True,
                )
            _set_value_dtype_from_var(loop_ctx, out_val, out_var)
            relax_value_to_rank_only(out_val)
            body_outputs.append(out_val)

        carry_outputs_start = 1 + num_consts
        for rel_idx, out_var in enumerate(jaxpr.outvars[:num_carry]):
            value = body_outputs[carry_outputs_start + rel_idx]
            target_enum = _dtype_enum_for_var(out_var, allow_double_outputs)
            if target_enum is None:
                continue
            casted = _maybe_cast_value(loop_ctx, value, target_enum, force=True)
            if casted is not value:
                body_outputs[carry_outputs_start + rel_idx] = casted
                loop_ctx.bind_value_for_var(out_var, casted)

        for seq_state in sequence_states:
            seq_passthrough = builder_identity(
                loop_ctx,
                seq_state,
                name_hint="loop_seq_out",
            )
            body_outputs.append(seq_passthrough)

        for out_var in jaxpr.outvars[num_carry:]:
            out_val = loop_ctx.get_value_for_var(out_var)
            if os.environ.get("J2O_DEBUG_LOOP_DTYPES") == "1":
                print(
                    "[scan_with_inputs_out_seq]",
                    getattr(getattr(out_var, "aval", None), "dtype", None),
                    flush=True,
                )
            _set_value_dtype_from_var(loop_ctx, out_val, out_var)
            relax_value_to_rank_only(out_val)
            body_outputs.append(out_val)

        outputs_start_with_seq = 1 + num_consts + num_carry + len(sequence_states)
        for rel_idx, out_var in enumerate(jaxpr.outvars[num_carry:]):
            value = body_outputs[outputs_start_with_seq + rel_idx]
            _set_value_dtype_from_var(loop_ctx, value, out_var)
            target_enum = _dtype_enum_for_var(out_var, allow_double_outputs)
            casted = _maybe_cast_value(loop_ctx, value, target_enum, force=True)
            if casted is not value:
                body_outputs[outputs_start_with_seq + rel_idx] = casted
                loop_ctx.bind_value_for_var(out_var, casted)
                if os.environ.get("J2O_DEBUG_LOOP_DTYPES") == "1":
                    print(
                        "[scan_with_inputs_body_output]",
                        getattr(getattr(out_var, "aval", None), "dtype", None),
                        getattr(getattr(casted, "type", None), "dtype", None),
                        flush=True,
                    )

        loop_ctx.builder.outputs = body_outputs

        body_graph = clone_graph(loop_ctx.builder.graph)
        body_graph.name = ctx.fresh_name("scan_loop_body")
        opset_imports = dict(body_graph.opset_imports)
        opset_imports.setdefault("", getattr(ctx.builder, "opset", 21))
        body_graph.opset_imports.clear()
        body_graph.opset_imports.update(opset_imports)

        if trip_count_int is not None:
            trip_count_val = _scalar_i64(ctx, trip_count_int, "scan_trip_count")
        else:
            first_seq_val = ctx.get_value_for_var(seq_invars[0])
            shape_val = _shape_of(ctx, first_seq_val, "scan_seq_shape")
            trip_count_val = _gather_int_scalar(ctx, shape_val, 0, "scan_trip_dynamic")

        cond_init = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("scan_cond_init"),
            array=np.asarray(True, dtype=np.bool_),
        )

        node_inputs = [trip_count_val, cond_init]
        inbound_vals = []
        limit = num_consts + num_carry + num_scan
        for idx, v in enumerate(eqn.invars[:limit]):
            top_val = ctx.get_value_for_var(v)
            if idx < len(state_inputs):
                state_enum = getattr(
                    getattr(state_inputs[idx], "type", None), "dtype", None
                )
                expected_enum = _dtype_enum_for_var(v, allow_double_consts)
                if expected_enum is None:
                    expected_enum = state_enum
                else:
                    aval_dtype = getattr(getattr(v, "aval", None), "dtype", None)
                    if aval_dtype is not None:
                        try:
                            np_dtype = np.dtype(aval_dtype)
                        except TypeError:
                            np_dtype = None
                        else:
                            if (
                                state_enum is not None
                                and expected_enum != state_enum
                                and allow_double_consts
                                and np_dtype is not None
                                and np.issubdtype(np_dtype, np.floating)
                            ):
                                expected_enum = state_enum
                top_dtype = getattr(getattr(top_val, "type", None), "dtype", None)
                if (
                    expected_enum is not None
                    and top_dtype is not None
                    and expected_enum != top_dtype
                ):
                    top_val = builder_cast(
                        ctx,
                        top_val,
                        expected_enum,
                        name_hint="scan_state_cast",
                    )
                    _ensure_value_metadata(ctx, top_val)
            inbound_vals.append(top_val)
        node_inputs.extend(inbound_vals)

        node_outputs: list[ir.Value] = []

        const_body_outs = body_outputs[1 : 1 + num_consts]
        for body_out in const_body_outs:
            node_outputs.append(
                clone_value_for_subgraph(
                    ctx,
                    body_out,
                    name_hint="loop_const_unused",
                )
            )

        carry_body_start = 1 + num_consts
        for idx in range(num_carry):
            body_out = body_outputs[carry_body_start + idx]
            out_var = eqn.outvars[idx] if idx < len(eqn.outvars) else None
            if out_var is not None and not _is_dropvar(out_var):
                top_val = ctx.get_value_for_var(out_var)
                expected_enum = _dtype_enum_for_var(
                    out_var, ctx.builder.enable_double_precision
                )
                if expected_enum is None:
                    expected_enum = getattr(
                        getattr(body_out, "type", None), "dtype", None
                    )
                top_dtype = getattr(getattr(top_val, "type", None), "dtype", None)
                if (
                    expected_enum is not None
                    and top_dtype is not None
                    and expected_enum != top_dtype
                ):
                    replacement = ir.Value(
                        name=ctx.fresh_name("loop_out"),
                        type=ir.TensorType(expected_enum),
                        shape=body_out.shape,
                    )
                    ctx.bind_value_for_var(out_var, replacement)
                    node_outputs.append(replacement)
                else:
                    node_outputs.append(top_val)
            else:
                node_outputs.append(
                    clone_value_for_subgraph(
                        ctx,
                        body_out,
                        name_hint="loop_state_unused",
                    )
                )

        seq_body_start = carry_body_start + num_carry
        body_seq_outvars = list(jaxpr.outvars[num_carry:])
        seq_axis_overrides: list[int | None] = []
        ctx_override_extent = getattr(loop_ctx, "_static_loop_extent_axis0", None)
        for body_out_var in body_seq_outvars:
            aval = getattr(body_out_var, "aval", None)
            axis0_extent = None
            if aval is not None:
                shape = getattr(aval, "shape", ())
                if (
                    isinstance(shape, tuple)
                    and shape
                    and isinstance(shape[0], (int, np.integer))
                    and int(shape[0]) > 1
                ):
                    axis0_extent = int(shape[0])
            if (
                axis0_extent is None
                and isinstance(ctx_override_extent, (int, np.integer))
                and int(ctx_override_extent) > 1
            ):
                axis0_extent = int(ctx_override_extent)
            seq_axis_overrides.append(axis0_extent)

        for seq_idx in range(num_scan):
            body_out = body_outputs[seq_body_start + seq_idx]
            node_outputs.append(
                clone_value_for_subgraph(
                    ctx,
                    body_out,
                    name_hint="loop_seq_unused",
                )
            )

        for rel_idx, out_var in enumerate(eqn.outvars[num_carry:]):
            top_val = ctx.get_value_for_var(
                out_var, name_hint=ctx.fresh_name("loop_out")
            )
            body_out = body_outputs[seq_body_start + num_scan + rel_idx]
            override_extent = (
                seq_axis_overrides[rel_idx]
                if rel_idx < len(seq_axis_overrides)
                else None
            )
            if (
                override_extent is None
                and isinstance(ctx_override_extent, (int, np.integer))
                and int(ctx_override_extent) > 1
            ):
                override_extent = int(ctx_override_extent)
            if (
                isinstance(override_extent, (int, np.integer))
                and int(override_extent) > 1
            ):
                override_int = int(override_extent)
                set_axis0_override(top_val, override_int)
                aval_shape = getattr(getattr(out_var, "aval", None), "shape", None)
                aval_shape_tuple = aval_shape if isinstance(aval_shape, tuple) else None
                _restamp_axis0(top_val, override_int, aval_shape_tuple)
            expected_enum = _dtype_enum_for_var(
                out_var, ctx.builder.enable_double_precision
            )
            if expected_enum is None:
                expected_enum = getattr(getattr(body_out, "type", None), "dtype", None)
            top_dtype = getattr(getattr(top_val, "type", None), "dtype", None)
            if (
                expected_enum is not None
                and top_dtype is not None
                and expected_enum != top_dtype
            ):
                replacement = ir.Value(
                    name=ctx.fresh_name("loop_out"),
                    type=ir.TensorType(expected_enum),
                    shape=body_out.shape,
                )
                ctx.bind_value_for_var(out_var, replacement)
                node_outputs.append(replacement)
            else:
                node_outputs.append(top_val)

        output_names = [
            getattr(val, "name", None) or ctx.fresh_name("scan_out")
            for val in node_outputs
        ]

        loop_results = builder_loop(
            ctx,
            *node_inputs,
            body=body_graph,
            output_names=output_names,
        )

        if not isinstance(loop_results, tuple):
            loop_results = (loop_results,)

        for template, produced in zip(node_outputs, loop_results):
            tmpl_meta = getattr(template, "meta", None)
            prod_meta = getattr(produced, "meta", None)
            if tmpl_meta is not None and prod_meta is not None:
                for key, value in tmpl_meta.items():
                    prod_meta[key] = value
            tmpl_type = getattr(template, "type", None)
            if tmpl_type is not None:
                produced.type = tmpl_type
            tmpl_shape = getattr(template, "shape", None)
            if tmpl_shape is not None:
                produced.shape = tmpl_shape
            _ensure_value_metadata(ctx, produced)

        loop_override_extent = getattr(loop_ctx, "_static_loop_extent_axis0", None)
        if (
            getattr(loop_ctx, "_force_loop_extent_axis0", False)
            and isinstance(loop_override_extent, (int, np.integer))
            and int(loop_override_extent) > 1
        ):
            override_int = int(loop_override_extent)
            for produced in loop_results:
                set_axis0_override(produced, override_int)

        const_count = len(const_body_outs)
        carry_count = num_carry
        seq_count = num_scan

        carry_results = loop_results[const_count : const_count + carry_count]
        value_results = loop_results[const_count + carry_count + seq_count :]

        for idx, res in enumerate(carry_results):
            if idx >= len(eqn.outvars):
                break
            out_var = eqn.outvars[idx]
            if _is_dropvar(out_var):
                continue
            ctx.bind_value_for_var(out_var, res)

        for rel_idx, out_var in enumerate(eqn.outvars[num_carry:]):
            if rel_idx >= len(value_results):
                break
            override_extent = (
                seq_axis_overrides[rel_idx]
                if rel_idx < len(seq_axis_overrides)
                else None
            )
            if (
                override_extent is None
                and isinstance(ctx_override_extent, (int, np.integer))
                and int(ctx_override_extent) > 1
            ):
                override_extent = int(ctx_override_extent)
            if (
                isinstance(override_extent, (int, np.integer))
                and int(override_extent) > 1
            ):
                override_int = int(override_extent)
                set_axis0_override(value_results[rel_idx], override_int)
                aval_shape = getattr(
                    getattr(eqn.outvars[num_carry + rel_idx], "aval", None),
                    "shape",
                    None,
                )
                aval_shape_tuple = aval_shape if isinstance(aval_shape, tuple) else None
                _restamp_axis0(value_results[rel_idx], override_int, aval_shape_tuple)
            ctx.bind_value_for_var(out_var, value_results[rel_idx])

        return list(loop_results)


_DYNAMIC_DIM_SENTINEL: Final[str] = "JAX2ONNX_DYNAMIC_DIM_SENTINEL"


def _restamp_axis0(
    value: ir.Value, override: int | None, aval_shape: tuple | None
) -> None:
    if not isinstance(override, (int, np.integer)) or int(override) <= 1:
        return
    override_int = int(override)
    shape_obj = getattr(value, "shape", None)
    dims: list[object] = []
    if isinstance(shape_obj, ir.Shape):
        dims = list(shape_obj.dims)
    elif isinstance(shape_obj, (tuple, list)):
        dims = list(shape_obj)
    if not dims:
        if isinstance(aval_shape, tuple) and aval_shape:
            dims = list(aval_shape)
    if not dims:
        return
    if len(dims) == 0:
        dims = [None]
    dims = list(dims)
    dims[0] = override_int
    _stamp_type_and_shape(value, tuple(dims))
