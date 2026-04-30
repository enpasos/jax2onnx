# jax2onnx/plugins/jax/numpy/histogram2d.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final, Sequence, TypeAlias, cast

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import (
    DimInput,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.jax.numpy.histogram import (
    _all_static_ints,
    _as_dtype,
    _bool_initializer,
    _gather_edges,
    _num_elements,
    _reshape,
    _unsqueeze,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_HISTOGRAM2D_PRIM: Final = make_jnp_primitive("jax.numpy.histogram2d")
_HISTOGRAM2D_PRIM.multiple_results = True

BatchDim: TypeAlias = int | None


def _unsupported_params(
    *,
    bins: object,
    range: Sequence[None | ArrayLike | Sequence[ArrayLike]] | None,
    weights: ArrayLike | None,
    density: bool | None,
) -> str | None:
    if range is not None:
        return "jnp.histogram2d with 'range' is not supported for ONNX export"
    if weights is not None:
        return "jnp.histogram2d with 'weights' is not supported for ONNX export"
    if density not in (None, False):
        return "jnp.histogram2d with density=True is not supported for ONNX export"
    if not isinstance(bins, (tuple, list)) or len(bins) != 2:
        return "jnp.histogram2d lowering requires explicit 1-D x/y bin edge arrays"
    if any(isinstance(item, (int, np.integer)) for item in bins):
        return "jnp.histogram2d with scalar bin counts is not supported for ONNX export"
    return None


def _validate_edges_shape(shape: tuple[object, ...], *, axis_name: str) -> int:
    if len(shape) != 1:
        raise TypeError(f"jnp.histogram2d lowering requires 1-D {axis_name} edges")
    if not _all_static_ints(shape):
        raise TypeError(
            f"jnp.histogram2d lowering requires static {axis_name} edge length"
        )
    static_shape = cast(tuple[int | np.integer[Any]], shape)
    edge_count = int(static_shape[0])
    if edge_count < 2:
        raise ValueError(
            f"jnp.histogram2d lowering requires at least two {axis_name} edges"
        )
    return edge_count


def _abstract_eval_via_orig(
    prim: Any,
    func_name: str,
    x: core.AbstractValue,
    y: core.AbstractValue,
    x_edges: core.AbstractValue,
    y_edges: core.AbstractValue,
) -> tuple[core.ShapedArray, core.ShapedArray, core.ShapedArray]:
    x_shape = tuple(getattr(x, "shape", ()))
    y_shape = tuple(getattr(y, "shape", ()))
    x_edges_shape = tuple(getattr(x_edges, "shape", ()))
    y_edges_shape = tuple(getattr(y_edges, "shape", ()))
    if _num_elements(x_shape) != _num_elements(y_shape):
        raise TypeError("jnp.histogram2d lowering requires x and y with same size")
    _validate_edges_shape(x_edges_shape, axis_name="x")
    _validate_edges_shape(y_edges_shape, axis_name="y")

    x_spec = jax.ShapeDtypeStruct(x_shape, np.dtype(getattr(x, "dtype", np.float32)))
    y_spec = jax.ShapeDtypeStruct(y_shape, np.dtype(getattr(y, "dtype", np.float32)))
    x_edges_spec = jax.ShapeDtypeStruct(
        x_edges_shape,
        np.dtype(getattr(x_edges, "dtype", np.float32)),
    )
    y_edges_spec = jax.ShapeDtypeStruct(
        y_edges_shape,
        np.dtype(getattr(y_edges, "dtype", np.float32)),
    )
    orig = get_orig_impl(prim, func_name)
    hist, x_edges_out, y_edges_out = jax.eval_shape(
        lambda x_val, y_val, x_bins, y_bins: orig(
            x_val,
            y_val,
            bins=(x_bins, y_bins),
        ),
        x_spec,
        y_spec,
        x_edges_spec,
        y_edges_spec,
    )
    return (
        core.ShapedArray(
            tuple(getattr(hist, "shape", ())),
            np.dtype(getattr(hist, "dtype", np.int32)),
        ),
        core.ShapedArray(
            tuple(getattr(x_edges_out, "shape", ())),
            np.dtype(getattr(x_edges_out, "dtype", np.float32)),
        ),
        core.ShapedArray(
            tuple(getattr(y_edges_out, "shape", ())),
            np.dtype(getattr(y_edges_out, "dtype", np.float32)),
        ),
    )


def _interval_membership(
    ctx: LoweringContextProtocol,
    values: ir.Value,
    edges: ir.Value,
    *,
    bin_count: int,
    value_count: int,
    name_hint: str,
) -> ir.Value:
    low_edges = _gather_edges(
        ctx,
        edges,
        indices=np.arange(0, bin_count, dtype=np.int64),
        name_hint=f"{name_hint}_low_edges",
    )
    high_edges = _gather_edges(
        ctx,
        edges,
        indices=np.arange(1, bin_count + 1, dtype=np.int64),
        name_hint=f"{name_hint}_high_edges",
    )
    lows = _unsqueeze(
        ctx,
        low_edges,
        axis=1,
        shape=(bin_count, 1),
        name_hint=f"{name_hint}_lows_unsqueeze",
    )
    highs = _unsqueeze(
        ctx,
        high_edges,
        axis=1,
        shape=(bin_count, 1),
        name_hint=f"{name_hint}_highs_unsqueeze",
    )

    matrix_shape = (bin_count, value_count)
    lower_ok = ctx.builder.GreaterOrEqual(
        values,
        lows,
        _outputs=[ctx.fresh_name(f"{name_hint}_lower_ok")],
    )
    lower_ok.type = ir.TensorType(ir.DataType.BOOL)
    _stamp_type_and_shape(lower_ok, matrix_shape)
    _ensure_value_metadata(ctx, lower_ok)

    upper_open = ctx.builder.Less(
        values,
        highs,
        _outputs=[ctx.fresh_name(f"{name_hint}_upper_open")],
    )
    upper_open.type = ir.TensorType(ir.DataType.BOOL)
    _stamp_type_and_shape(upper_open, matrix_shape)
    _ensure_value_metadata(ctx, upper_open)

    upper_closed = ctx.builder.LessOrEqual(
        values,
        highs,
        _outputs=[ctx.fresh_name(f"{name_hint}_upper_closed")],
    )
    upper_closed.type = ir.TensorType(ir.DataType.BOOL)
    _stamp_type_and_shape(upper_closed, matrix_shape)
    _ensure_value_metadata(ctx, upper_closed)

    last_bin_mask: np.ndarray[Any, np.dtype[np.bool_]] = np.zeros(
        (bin_count, 1), dtype=np.bool_
    )
    last_bin_mask[-1, 0] = True
    last_bin = _bool_initializer(
        ctx,
        last_bin_mask,
        name_hint=f"{name_hint}_last_bin",
    )
    closed_last_bin = ctx.builder.And(
        last_bin,
        upper_closed,
        _outputs=[ctx.fresh_name(f"{name_hint}_closed_last_bin")],
    )
    closed_last_bin.type = ir.TensorType(ir.DataType.BOOL)
    _stamp_type_and_shape(closed_last_bin, matrix_shape)
    _ensure_value_metadata(ctx, closed_last_bin)

    upper_ok = ctx.builder.Or(
        upper_open,
        closed_last_bin,
        _outputs=[ctx.fresh_name(f"{name_hint}_upper_ok")],
    )
    upper_ok.type = ir.TensorType(ir.DataType.BOOL)
    _stamp_type_and_shape(upper_ok, matrix_shape)
    _ensure_value_metadata(ctx, upper_ok)

    in_bin: ir.Value = ctx.builder.And(
        lower_ok,
        upper_ok,
        _outputs=[ctx.fresh_name(f"{name_hint}_in_bin")],
    )
    in_bin.type = ir.TensorType(ir.DataType.BOOL)
    _stamp_type_and_shape(in_bin, matrix_shape)
    _ensure_value_metadata(ctx, in_bin)
    return in_bin


@register_primitive(
    jaxpr_primitive=_HISTOGRAM2D_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogram2d.html",
    onnx=[
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "GreaterOrEqual",
            "doc": "https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html",
        },
        {"component": "Less", "doc": "https://onnx.ai/onnx/operators/onnx__Less.html"},
        {
            "component": "LessOrEqual",
            "doc": "https://onnx.ai/onnx/operators/onnx__LessOrEqual.html",
        },
        {"component": "And", "doc": "https://onnx.ai/onnx/operators/onnx__And.html"},
        {"component": "Or", "doc": "https://onnx.ai/onnx/operators/onnx__Or.html"},
        {"component": "Cast", "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {
            "component": "Identity",
            "doc": "https://onnx.ai/onnx/operators/onnx__Identity.html",
        },
    ],
    since="0.13.0",
    context="primitives.jnp",
    component="histogram2d",
    testcases=[
        {
            "testcase": "jnp_histogram2d_explicit_bins",
            "callable": lambda x, y, x_edges, y_edges: jnp.histogram2d(
                x,
                y,
                bins=(x_edges, y_edges),
            ),
            "input_values": [
                np.asarray([0.0, 0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 3.0, 4.0], dtype=np.float32),
                np.asarray([0.0, 2.0, 4.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(3, 2), (4,), (3,)],
            "post_check_onnx_graph": EG(
                ["And:3x2x6 -> Cast:3x2x6 -> ReduceSum:3x2"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_histogram2d_out_of_range",
            "callable": lambda x, y, x_edges, y_edges: jnp.histogram2d(
                x,
                y,
                bins=(x_edges, y_edges),
            ),
            "input_values": [
                np.asarray([-1.0, 0.0, 1.0, 3.0, 5.0], dtype=np.float32),
                np.asarray([0.0, -1.0, 1.0, 4.0, 2.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32),
                np.asarray([0.0, 2.0, 4.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(3, 2), (4,), (3,)],
        },
        {
            "testcase": "jnp_histogram2d_integer_inputs",
            "callable": lambda x, y, x_edges, y_edges: jnp.histogram2d(
                x,
                y,
                bins=(x_edges, y_edges),
            ),
            "input_values": [
                np.asarray([0, 1, 2, 3], dtype=np.int32),
                np.asarray([0, 1, 2, 3], dtype=np.int32),
                np.asarray([0, 2, 4], dtype=np.int32),
                np.asarray([0, 1, 4], dtype=np.int32),
            ],
            "expected_output_shapes": [(2, 2), (3,), (3,)],
        },
    ],
)
class JnpHistogram2dPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _HISTOGRAM2D_PRIM
    _FUNC_NAME: ClassVar[str] = "histogram2d"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        y: core.AbstractValue,
        x_edges: core.AbstractValue,
        y_edges: core.AbstractValue,
        *,
        density: bool | None = None,
    ) -> tuple[core.ShapedArray, ...]:
        if density not in (None, False):
            raise NotImplementedError(
                "jnp.histogram2d with density=True is not supported for ONNX export"
            )
        return _abstract_eval_via_orig(
            JnpHistogram2dPlugin._PRIM,
            JnpHistogram2dPlugin._FUNC_NAME,
            x,
            y,
            x_edges,
            y_edges,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        x_var, y_var, x_edges_var, y_edges_var = eqn.invars
        hist_var, x_edges_out_var, y_edges_out_var = eqn.outvars
        params = getattr(eqn, "params", {})
        if params.get("density") not in (None, False):
            raise NotImplementedError(
                "jnp.histogram2d with density=True is not supported for ONNX export"
            )

        x_shape = tuple(getattr(x_var.aval, "shape", ()))
        y_shape = tuple(getattr(y_var.aval, "shape", ()))
        x_edges_shape = tuple(getattr(x_edges_var.aval, "shape", ()))
        y_edges_shape = tuple(getattr(y_edges_var.aval, "shape", ()))
        value_count = _num_elements(x_shape)
        if value_count != _num_elements(y_shape):
            raise TypeError("jnp.histogram2d lowering requires x and y with same size")
        x_edge_count = _validate_edges_shape(x_edges_shape, axis_name="x")
        y_edge_count = _validate_edges_shape(y_edges_shape, axis_name="y")
        x_bin_count = x_edge_count - 1
        y_bin_count = y_edge_count - 1

        hist_shape = tuple(getattr(hist_var.aval, "shape", ()))
        x_edges_out_shape = tuple(getattr(x_edges_out_var.aval, "shape", ()))
        y_edges_out_shape = tuple(getattr(y_edges_out_var.aval, "shape", ()))
        hist_dtype: np.dtype[Any] = np.dtype(getattr(hist_var.aval, "dtype", np.int32))
        x_edges_out_dtype: np.dtype[Any] = np.dtype(
            getattr(x_edges_out_var.aval, "dtype", np.float32)
        )
        y_edges_out_dtype: np.dtype[Any] = np.dtype(
            getattr(y_edges_out_var.aval, "dtype", np.float32)
        )
        x_dtype: np.dtype[Any] = np.dtype(getattr(x_var.aval, "dtype", np.float32))
        y_dtype: np.dtype[Any] = np.dtype(getattr(y_var.aval, "dtype", np.float32))
        x_edges_dtype: np.dtype[Any] = np.dtype(
            getattr(x_edges_var.aval, "dtype", x_edges_out_dtype)
        )
        y_edges_dtype: np.dtype[Any] = np.dtype(
            getattr(y_edges_var.aval, "dtype", y_edges_out_dtype)
        )
        compare_dtype: np.dtype[Any] = np.promote_types(
            np.promote_types(x_dtype, y_dtype),
            np.promote_types(x_edges_out_dtype, y_edges_out_dtype),
        )

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("hist2d_x"))
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("hist2d_y"))
        x_edges_val = ctx.get_value_for_var(
            x_edges_var,
            name_hint=ctx.fresh_name("hist2d_x_edges"),
        )
        y_edges_val = ctx.get_value_for_var(
            y_edges_var,
            name_hint=ctx.fresh_name("hist2d_y_edges"),
        )
        hist_spec = ctx.get_value_for_var(
            hist_var, name_hint=ctx.fresh_name("hist2d_hist")
        )
        x_edges_spec = ctx.get_value_for_var(
            x_edges_out_var, name_hint=ctx.fresh_name("hist2d_x_edges_out")
        )
        y_edges_spec = ctx.get_value_for_var(
            y_edges_out_var, name_hint=ctx.fresh_name("hist2d_y_edges_out")
        )

        x_edges_ready = _as_dtype(
            ctx,
            x_edges_val,
            from_dtype=x_edges_dtype,
            to_dtype=x_edges_out_dtype,
            shape=x_edges_shape,
            name_hint="hist2d_x_edges_cast",
        )
        y_edges_ready = _as_dtype(
            ctx,
            y_edges_val,
            from_dtype=y_edges_dtype,
            to_dtype=y_edges_out_dtype,
            shape=y_edges_shape,
            name_hint="hist2d_y_edges_cast",
        )
        x_ready = _as_dtype(
            ctx,
            x_val,
            from_dtype=x_dtype,
            to_dtype=compare_dtype,
            shape=x_shape,
            name_hint="hist2d_x_cast",
        )
        y_ready = _as_dtype(
            ctx,
            y_val,
            from_dtype=y_dtype,
            to_dtype=compare_dtype,
            shape=y_shape,
            name_hint="hist2d_y_cast",
        )
        x_compare_edges = _as_dtype(
            ctx,
            x_edges_ready,
            from_dtype=x_edges_out_dtype,
            to_dtype=compare_dtype,
            shape=x_edges_shape,
            name_hint="hist2d_x_compare_edges_cast",
        )
        y_compare_edges = _as_dtype(
            ctx,
            y_edges_ready,
            from_dtype=y_edges_out_dtype,
            to_dtype=compare_dtype,
            shape=y_edges_shape,
            name_hint="hist2d_y_compare_edges_cast",
        )

        flat_x = _reshape(ctx, x_ready, shape=(value_count,), name_hint="hist2d_x_flat")
        flat_y = _reshape(ctx, y_ready, shape=(value_count,), name_hint="hist2d_y_flat")
        x_values = _unsqueeze(
            ctx,
            flat_x,
            axis=0,
            shape=(1, value_count),
            name_hint="hist2d_x_values_unsqueeze",
        )
        y_values = _unsqueeze(
            ctx,
            flat_y,
            axis=0,
            shape=(1, value_count),
            name_hint="hist2d_y_values_unsqueeze",
        )

        x_membership = _interval_membership(
            ctx,
            x_values,
            x_compare_edges,
            bin_count=x_bin_count,
            value_count=value_count,
            name_hint="hist2d_x",
        )
        y_membership = _interval_membership(
            ctx,
            y_values,
            y_compare_edges,
            bin_count=y_bin_count,
            value_count=value_count,
            name_hint="hist2d_y",
        )
        x_membership_3d = _unsqueeze(
            ctx,
            x_membership,
            axis=1,
            shape=(x_bin_count, 1, value_count),
            name_hint="hist2d_x_membership_unsqueeze",
        )
        y_membership_3d = _unsqueeze(
            ctx,
            y_membership,
            axis=0,
            shape=(1, y_bin_count, value_count),
            name_hint="hist2d_y_membership_unsqueeze",
        )

        joint_shape = (x_bin_count, y_bin_count, value_count)
        joint = ctx.builder.And(
            x_membership_3d,
            y_membership_3d,
            _outputs=[ctx.fresh_name("hist2d_joint_membership")],
        )
        joint.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(joint, joint_shape)
        _ensure_value_metadata(ctx, joint)

        hist_enum = _dtype_to_ir(hist_dtype, ctx.builder.enable_double_precision)
        hits = ctx.builder.Cast(
            joint,
            to=int(hist_enum.value),
            _outputs=[ctx.fresh_name("hist2d_hits")],
        )
        hits.type = ir.TensorType(hist_enum)
        _stamp_type_and_shape(hits, joint_shape)
        _ensure_value_metadata(ctx, hits)

        reduce_axes = _const_i64(
            ctx,
            np.asarray([2], dtype=np.int64),
            "hist2d_reduce_axes",
        )
        hist_name = getattr(hist_spec, "name", None) or ctx.fresh_name("hist2d_hist")
        hist_producer = getattr(hist_spec, "producer", None)
        if callable(hist_producer) and hist_producer() is not None:
            hist_name = ctx.fresh_name("hist2d_hist")
        hist_result = ctx.builder.ReduceSum(
            hits,
            reduce_axes,
            keepdims=0,
            _outputs=[hist_name],
        )
        if getattr(hist_spec, "type", None) is not None:
            hist_result.type = hist_spec.type
        else:
            hist_result.type = ir.TensorType(hist_enum)
        if getattr(hist_spec, "shape", None) is not None:
            hist_result.shape = hist_spec.shape
        else:
            _stamp_type_and_shape(hist_result, hist_shape)
        _ensure_value_metadata(ctx, hist_result)

        x_edges_result = self._identity_output(
            ctx,
            x_edges_ready,
            out_spec=x_edges_spec,
            output_shape=x_edges_out_shape,
            output_dtype=x_edges_out_dtype,
            name_hint="hist2d_x_edges_out",
        )
        y_edges_result = self._identity_output(
            ctx,
            y_edges_ready,
            out_spec=y_edges_spec,
            output_shape=y_edges_out_shape,
            output_dtype=y_edges_out_dtype,
            name_hint="hist2d_y_edges_out",
        )

        ctx.bind_value_for_var(hist_var, hist_result)
        ctx.bind_value_for_var(x_edges_out_var, x_edges_result)
        ctx.bind_value_for_var(y_edges_out_var, y_edges_result)

    @staticmethod
    def _identity_output(
        ctx: LoweringContextProtocol,
        value: ir.Value,
        *,
        out_spec: ir.Value,
        output_shape: tuple[object, ...],
        output_dtype: np.dtype[Any],
        name_hint: str,
    ) -> ir.Value:
        output_name = getattr(out_spec, "name", None) or ctx.fresh_name(name_hint)
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            output_name = ctx.fresh_name(name_hint)
        result: ir.Value = ctx.builder.Identity(value, _outputs=[output_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = ir.TensorType(
                _dtype_to_ir(output_dtype, ctx.builder.enable_double_precision)
            )
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            _stamp_type_and_shape(result, cast(tuple[DimInput, ...], output_shape))
        _ensure_value_metadata(ctx, result)
        return result

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., tuple[jax.Array, jax.Array, jax.Array]] | None,
        ) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
            if orig is None:
                raise RuntimeError(
                    "Original jnp.histogram2d not found for monkey patching"
                )
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                x: ArrayLike,
                y: ArrayLike,
                bins: ArrayLike | list[ArrayLike] = 10,
                range: Sequence[None | ArrayLike | Sequence[ArrayLike]] | None = None,
                weights: ArrayLike | None = None,
                density: bool | None = None,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                unsupported = _unsupported_params(
                    bins=bins,
                    range=range,
                    weights=weights,
                    density=density,
                )
                if unsupported is not None:
                    raise NotImplementedError(unsupported)
                x_edges, y_edges = cast(Sequence[ArrayLike], bins)
                x_edges_array = jnp.asarray(x_edges)
                y_edges_array = jnp.asarray(y_edges)
                if x_edges_array.ndim != 1 or y_edges_array.ndim != 1:
                    raise NotImplementedError(
                        "jnp.histogram2d lowering requires explicit 1-D x/y bin edge arrays"
                    )
                return cast(
                    tuple[jax.Array, jax.Array, jax.Array],
                    cls._PRIM.bind(
                        jnp.asarray(x),
                        jnp.asarray(y),
                        x_edges_array,
                        y_edges_array,
                        density=density,
                    ),
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


@JnpHistogram2dPlugin._PRIM.def_impl
def _histogram2d_impl(
    x: object,
    y: object,
    x_edges: object,
    y_edges: object,
    *,
    density: bool | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    orig = get_orig_impl(JnpHistogram2dPlugin._PRIM, JnpHistogram2dPlugin._FUNC_NAME)
    return cast(
        tuple[jax.Array, jax.Array, jax.Array],
        orig(x, y, bins=(x_edges, y_edges), density=density),
    )


JnpHistogram2dPlugin._PRIM.def_abstract_eval(JnpHistogram2dPlugin.abstract_eval)


def _histogram2d_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    density: bool | None = None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[BatchDim, BatchDim, BatchDim]]:
    del density
    if any(dim is not None for dim in batch_dims):
        raise NotImplementedError("vmap over jnp.histogram2d is not supported")
    hist, x_edges, y_edges = JnpHistogram2dPlugin._PRIM.bind(*batched_args)
    return (hist, x_edges, y_edges), (None, None, None)


batching.primitive_batchers[JnpHistogram2dPlugin._PRIM] = _histogram2d_batch_rule
