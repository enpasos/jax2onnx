# jax2onnx/plugins/jax/numpy/histogram.py

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
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_HISTOGRAM_PRIM: Final = make_jnp_primitive("jax.numpy.histogram")
_HISTOGRAM_PRIM.multiple_results = True

BatchDim: TypeAlias = int | None


def _all_static_ints(shape: tuple[object, ...]) -> bool:
    return all(isinstance(dim, (int, np.integer)) for dim in shape)


def _num_elements(shape: tuple[object, ...]) -> int:
    if not _all_static_ints(shape):
        raise TypeError("jnp.histogram lowering requires static input shape")
    return int(np.prod(tuple(int(dim) for dim in shape), dtype=np.int64))


def _as_dtype(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    from_dtype: np.dtype[Any],
    to_dtype: np.dtype[Any],
    shape: tuple[object, ...],
    name_hint: str,
) -> ir.Value:
    target_enum = _dtype_to_ir(to_dtype, ctx.builder.enable_double_precision)
    if (
        from_dtype == to_dtype
        and getattr(getattr(val, "type", None), "dtype", None) == target_enum
    ):
        return val
    if (
        from_dtype == to_dtype
        and getattr(getattr(val, "type", None), "dtype", None) is None
    ):
        val.type = ir.TensorType(target_enum)
        _stamp_type_and_shape(val, shape)
        _ensure_value_metadata(ctx, val)
        return val
    result = ctx.builder.Cast(
        val,
        to=int(target_enum.value),
        _outputs=[ctx.fresh_name(name_hint)],
    )
    result.type = ir.TensorType(target_enum)
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _reshape(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    shape: tuple[int, ...],
    name_hint: str,
) -> ir.Value:
    shape_val = _const_i64(
        ctx,
        np.asarray(shape, dtype=np.int64),
        f"{name_hint}_shape",
    )
    result = ctx.builder.Reshape(
        val,
        shape_val,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    if getattr(val, "type", None) is not None:
        result.type = val.type
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _unsqueeze(
    ctx: LoweringContextProtocol,
    val: ir.Value,
    *,
    axis: int,
    shape: tuple[object, ...],
    name_hint: str,
) -> ir.Value:
    axes = _const_i64(ctx, np.asarray([axis], dtype=np.int64), f"{name_hint}_axes")
    result = ctx.builder.Unsqueeze(
        val,
        axes,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    if getattr(val, "type", None) is not None:
        result.type = val.type
    _stamp_type_and_shape(result, shape)
    _ensure_value_metadata(ctx, result)
    return result


def _gather_edges(
    ctx: LoweringContextProtocol,
    edges: ir.Value,
    *,
    indices: np.ndarray[Any, np.dtype[np.int64]],
    name_hint: str,
) -> ir.Value:
    idx = _const_i64(ctx, indices, f"{name_hint}_indices")
    result = ctx.builder.Gather(
        edges,
        idx,
        axis=0,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    if getattr(edges, "type", None) is not None:
        result.type = edges.type
    _stamp_type_and_shape(result, (int(indices.size),))
    _ensure_value_metadata(ctx, result)
    return result


def _bool_initializer(
    ctx: LoweringContextProtocol,
    array: np.ndarray[Any, np.dtype[np.bool_]],
    *,
    name_hint: str,
) -> ir.Value:
    result = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name(name_hint),
        array=array,
    )
    result.type = ir.TensorType(ir.DataType.BOOL)
    _stamp_type_and_shape(result, tuple(array.shape))
    _ensure_value_metadata(ctx, result)
    return result


def _unsupported_params(
    *,
    bins: object,
    range: Sequence[ArrayLike] | None,
    weights: ArrayLike | None,
    density: bool | None,
) -> str | None:
    if isinstance(bins, (int, np.integer)):
        return "jnp.histogram with scalar 'bins' is not supported for ONNX export"
    if range is not None:
        return "jnp.histogram with 'range' is not supported for ONNX export"
    if weights is not None:
        return "jnp.histogram with 'weights' is not supported for ONNX export"
    if density not in (None, False):
        return "jnp.histogram with density=True is not supported for ONNX export"
    return None


def _abstract_eval_via_orig(
    prim: Any,
    func_name: str,
    a: core.AbstractValue,
    bins: core.AbstractValue,
) -> tuple[core.ShapedArray, core.ShapedArray]:
    a_shape = tuple(getattr(a, "shape", ()))
    bins_shape = tuple(getattr(bins, "shape", ()))
    if len(bins_shape) != 1:
        raise TypeError("jnp.histogram lowering requires explicit 1-D bins")
    if not _all_static_ints(a_shape) or not _all_static_ints(bins_shape):
        raise TypeError("jnp.histogram lowering requires static input and bins shapes")
    if int(bins_shape[0]) < 2:
        raise ValueError("jnp.histogram lowering requires at least two bin edges")

    a_dtype: np.dtype[Any] = np.dtype(getattr(a, "dtype", np.float32))
    bins_dtype: np.dtype[Any] = np.dtype(getattr(bins, "dtype", np.float32))
    a_spec = jax.ShapeDtypeStruct(a_shape, a_dtype)
    bins_spec = jax.ShapeDtypeStruct(bins_shape, bins_dtype)
    orig = get_orig_impl(prim, func_name)
    hist, edges = jax.eval_shape(lambda x, b: orig(x, bins=b), a_spec, bins_spec)
    return (
        core.ShapedArray(
            tuple(getattr(hist, "shape", ())),
            np.dtype(getattr(hist, "dtype", np.float32)),
        ),
        core.ShapedArray(
            tuple(getattr(edges, "shape", ())),
            np.dtype(getattr(edges, "dtype", np.float32)),
        ),
    )


@register_primitive(
    jaxpr_primitive=_HISTOGRAM_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogram.html",
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
    component="histogram",
    testcases=[
        {
            "testcase": "jnp_histogram_explicit_bins",
            "callable": lambda a, bins: jnp.histogram(a, bins=bins),
            "input_values": [
                np.asarray([0.0, 0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 3.0, 4.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.float32, np.float32],
            "post_check_onnx_graph": EG(
                [
                    "GreaterOrEqual:3x6",
                    "Or:3x6",
                    "And:3x6 -> Cast:3x6 -> ReduceSum:3",
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_histogram_out_of_range",
            "callable": lambda a, bins: jnp.histogram(a, bins=bins),
            "input_values": [
                np.asarray([-1.0, 0.0, 1.0, 3.0, 5.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32),
            ],
            "expected_output_dtypes": [np.float32, np.float32],
        },
        {
            "testcase": "jnp_histogram_integer_inputs",
            "callable": lambda a, bins: jnp.histogram(a, bins=bins),
            "input_values": [
                np.asarray([0, 1, 2, 3], dtype=np.int32),
                np.asarray([0, 2, 4], dtype=np.int32),
            ],
            "expected_output_dtypes": [np.float32, np.float32],
        },
    ],
)
class JnpHistogramPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _HISTOGRAM_PRIM
    _FUNC_NAME: ClassVar[str] = "histogram"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        a: core.AbstractValue,
        bins: core.AbstractValue,
        *,
        density: bool | None = None,
    ) -> tuple[core.ShapedArray, ...]:
        if density not in (None, False):
            raise NotImplementedError(
                "jnp.histogram with density=True is not supported for ONNX export"
            )
        return _abstract_eval_via_orig(
            JnpHistogramPlugin._PRIM,
            JnpHistogramPlugin._FUNC_NAME,
            a,
            bins,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        a_var, bins_var = eqn.invars
        hist_var, edges_var = eqn.outvars
        params = getattr(eqn, "params", {})
        if params.get("density") not in (None, False):
            raise NotImplementedError(
                "jnp.histogram with density=True is not supported for ONNX export"
            )

        a_shape = tuple(getattr(a_var.aval, "shape", ()))
        bins_shape = tuple(getattr(bins_var.aval, "shape", ()))
        if len(bins_shape) != 1:
            raise TypeError("jnp.histogram lowering requires explicit 1-D bins")
        if not _all_static_ints(bins_shape):
            raise TypeError("jnp.histogram lowering requires static bins length")
        edges_len = int(bins_shape[0])
        if edges_len < 2:
            raise ValueError("jnp.histogram lowering requires at least two bin edges")
        bin_count = edges_len - 1
        value_count = _num_elements(a_shape)

        hist_shape = tuple(getattr(hist_var.aval, "shape", ()))
        edges_shape = tuple(getattr(edges_var.aval, "shape", ()))
        hist_dtype: np.dtype[Any] = np.dtype(
            getattr(hist_var.aval, "dtype", np.float32)
        )
        edge_dtype: np.dtype[Any] = np.dtype(
            getattr(edges_var.aval, "dtype", np.float32)
        )
        a_dtype: np.dtype[Any] = np.dtype(getattr(a_var.aval, "dtype", np.float32))
        bins_dtype: np.dtype[Any] = np.dtype(
            getattr(bins_var.aval, "dtype", edge_dtype)
        )
        compare_dtype: np.dtype[Any] = np.promote_types(a_dtype, edge_dtype)

        a_val = ctx.get_value_for_var(a_var, name_hint=ctx.fresh_name("histogram_a"))
        bins_val = ctx.get_value_for_var(
            bins_var,
            name_hint=ctx.fresh_name("histogram_bins"),
        )
        hist_spec = ctx.get_value_for_var(
            hist_var, name_hint=ctx.fresh_name("histogram_hist")
        )
        edges_spec = ctx.get_value_for_var(
            edges_var, name_hint=ctx.fresh_name("histogram_edges")
        )

        edges_ready = _as_dtype(
            ctx,
            bins_val,
            from_dtype=bins_dtype,
            to_dtype=edge_dtype,
            shape=bins_shape,
            name_hint="histogram_edges_cast",
        )
        a_ready = _as_dtype(
            ctx,
            a_val,
            from_dtype=a_dtype,
            to_dtype=compare_dtype,
            shape=a_shape,
            name_hint="histogram_a_cast",
        )
        compare_edges = _as_dtype(
            ctx,
            edges_ready,
            from_dtype=edge_dtype,
            to_dtype=compare_dtype,
            shape=bins_shape,
            name_hint="histogram_compare_edges_cast",
        )

        flat_a = _reshape(
            ctx,
            a_ready,
            shape=(value_count,),
            name_hint="histogram_flatten",
        )
        values = _unsqueeze(
            ctx,
            flat_a,
            axis=0,
            shape=(1, value_count),
            name_hint="histogram_values_unsqueeze",
        )

        low_edges = _gather_edges(
            ctx,
            compare_edges,
            indices=np.arange(0, bin_count, dtype=np.int64),
            name_hint="histogram_low_edges",
        )
        high_edges = _gather_edges(
            ctx,
            compare_edges,
            indices=np.arange(1, edges_len, dtype=np.int64),
            name_hint="histogram_high_edges",
        )
        lows = _unsqueeze(
            ctx,
            low_edges,
            axis=1,
            shape=(bin_count, 1),
            name_hint="histogram_lows_unsqueeze",
        )
        highs = _unsqueeze(
            ctx,
            high_edges,
            axis=1,
            shape=(bin_count, 1),
            name_hint="histogram_highs_unsqueeze",
        )

        matrix_shape = (bin_count, value_count)
        lower_ok = ctx.builder.GreaterOrEqual(
            values,
            lows,
            _outputs=[ctx.fresh_name("histogram_lower_ok")],
        )
        lower_ok.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(lower_ok, matrix_shape)
        _ensure_value_metadata(ctx, lower_ok)

        upper_open = ctx.builder.Less(
            values,
            highs,
            _outputs=[ctx.fresh_name("histogram_upper_open")],
        )
        upper_open.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(upper_open, matrix_shape)
        _ensure_value_metadata(ctx, upper_open)

        upper_closed = ctx.builder.LessOrEqual(
            values,
            highs,
            _outputs=[ctx.fresh_name("histogram_upper_closed")],
        )
        upper_closed.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(upper_closed, matrix_shape)
        _ensure_value_metadata(ctx, upper_closed)

        last_bin_mask = np.zeros((bin_count, 1), dtype=np.bool_)
        last_bin_mask[-1, 0] = True
        last_bin = _bool_initializer(ctx, last_bin_mask, name_hint="histogram_last_bin")
        closed_last_bin = ctx.builder.And(
            last_bin,
            upper_closed,
            _outputs=[ctx.fresh_name("histogram_closed_last_bin")],
        )
        closed_last_bin.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(closed_last_bin, matrix_shape)
        _ensure_value_metadata(ctx, closed_last_bin)

        upper_ok = ctx.builder.Or(
            upper_open,
            closed_last_bin,
            _outputs=[ctx.fresh_name("histogram_upper_ok")],
        )
        upper_ok.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(upper_ok, matrix_shape)
        _ensure_value_metadata(ctx, upper_ok)

        in_bin = ctx.builder.And(
            lower_ok,
            upper_ok,
            _outputs=[ctx.fresh_name("histogram_in_bin")],
        )
        in_bin.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(in_bin, matrix_shape)
        _ensure_value_metadata(ctx, in_bin)

        hist_enum = _dtype_to_ir(hist_dtype, ctx.builder.enable_double_precision)
        bin_hits = ctx.builder.Cast(
            in_bin,
            to=int(hist_enum.value),
            _outputs=[ctx.fresh_name("histogram_bin_hits")],
        )
        bin_hits.type = ir.TensorType(hist_enum)
        _stamp_type_and_shape(bin_hits, matrix_shape)
        _ensure_value_metadata(ctx, bin_hits)

        reduce_axes = _const_i64(
            ctx,
            np.asarray([1], dtype=np.int64),
            "histogram_reduce_axes",
        )
        hist_name = getattr(hist_spec, "name", None) or ctx.fresh_name("histogram_hist")
        hist_producer = getattr(hist_spec, "producer", None)
        if callable(hist_producer) and hist_producer() is not None:
            hist_name = ctx.fresh_name("histogram_hist")
        hist_result = ctx.builder.ReduceSum(
            bin_hits,
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

        edges_name = getattr(edges_spec, "name", None) or ctx.fresh_name(
            "histogram_edges"
        )
        edges_producer = getattr(edges_spec, "producer", None)
        if callable(edges_producer) and edges_producer() is not None:
            edges_name = ctx.fresh_name("histogram_edges")
        edges_result = ctx.builder.Identity(
            edges_ready,
            _outputs=[edges_name],
        )
        if getattr(edges_spec, "type", None) is not None:
            edges_result.type = edges_spec.type
        else:
            edges_result.type = ir.TensorType(
                _dtype_to_ir(edge_dtype, ctx.builder.enable_double_precision)
            )
        if getattr(edges_spec, "shape", None) is not None:
            edges_result.shape = edges_spec.shape
        else:
            _stamp_type_and_shape(edges_result, edges_shape)
        _ensure_value_metadata(ctx, edges_result)

        ctx.bind_value_for_var(hist_var, hist_result)
        ctx.bind_value_for_var(edges_var, edges_result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., tuple[jax.Array, jax.Array]] | None,
        ) -> Callable[..., tuple[jax.Array, jax.Array]]:
            if orig is None:
                raise RuntimeError(
                    "Original jnp.histogram not found for monkey patching"
                )
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                a: ArrayLike,
                bins: ArrayLike = 10,
                range: Sequence[ArrayLike] | None = None,
                weights: ArrayLike | None = None,
                density: bool | None = None,
            ) -> tuple[jax.Array, jax.Array]:
                unsupported = _unsupported_params(
                    bins=bins,
                    range=range,
                    weights=weights,
                    density=density,
                )
                if unsupported is not None:
                    raise NotImplementedError(unsupported)
                bins_array = jnp.asarray(bins)
                if bins_array.ndim != 1:
                    raise NotImplementedError(
                        "jnp.histogram lowering requires explicit 1-D bins"
                    )
                return cast(
                    tuple[jax.Array, jax.Array],
                    cls._PRIM.bind(
                        jnp.asarray(a),
                        bins_array,
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


@JnpHistogramPlugin._PRIM.def_impl
def _histogram_impl(
    a: object,
    bins: object,
    *,
    density: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    orig = get_orig_impl(JnpHistogramPlugin._PRIM, JnpHistogramPlugin._FUNC_NAME)
    return cast(tuple[jax.Array, jax.Array], orig(a, bins=bins, density=density))


JnpHistogramPlugin._PRIM.def_abstract_eval(JnpHistogramPlugin.abstract_eval)


def _histogram_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    density: bool | None = None,
) -> tuple[tuple[jax.Array, jax.Array], tuple[BatchDim, BatchDim]]:
    del density
    _a, _bins = batched_args
    a_bdim, bins_bdim = batch_dims
    if a_bdim is not None or bins_bdim is not None:
        raise NotImplementedError("vmap over jnp.histogram is not supported")
    hist, edges = JnpHistogramPlugin._PRIM.bind(*batched_args)
    return (hist, edges), (None, None)


batching.primitive_batchers[JnpHistogramPlugin._PRIM] = _histogram_batch_rule
