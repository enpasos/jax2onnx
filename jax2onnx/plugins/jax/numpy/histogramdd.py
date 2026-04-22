# jax2onnx/plugins/jax/numpy/histogramdd.py

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
from jax2onnx.plugins.jax.numpy.histogram import _all_static_ints, _as_dtype
from jax2onnx.plugins.jax.numpy.histogram2d import (
    _interval_membership,
    _validate_edges_shape,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_HISTOGRAMDD_PRIM: Final = make_jnp_primitive("jax.numpy.histogramdd")
_HISTOGRAMDD_PRIM.multiple_results = True

BatchDim: TypeAlias = int | None


def _unsupported_params(
    *,
    bins: object,
    range: Sequence[None | ArrayLike | Sequence[ArrayLike]] | None,
    weights: ArrayLike | None,
    density: bool | None,
) -> str | None:
    if range is not None:
        return "jnp.histogramdd with 'range' is not supported for ONNX export"
    if weights is not None:
        return "jnp.histogramdd with 'weights' is not supported for ONNX export"
    if density not in (None, False):
        return "jnp.histogramdd with density=True is not supported for ONNX export"
    if not isinstance(bins, (tuple, list)) or len(bins) != 2:
        return (
            "jnp.histogramdd lowering requires exactly two explicit 1-D bin edge arrays"
        )
    if any(isinstance(item, (int, np.integer)) for item in bins):
        return "jnp.histogramdd with scalar bin counts is not supported for ONNX export"
    return None


def _validate_sample_shape(shape: tuple[object, ...]) -> int:
    if len(shape) != 2:
        raise TypeError("jnp.histogramdd lowering requires sample shape (N, 2)")
    if not _all_static_ints(shape):
        raise TypeError("jnp.histogramdd lowering requires static sample shape")
    dim_count = int(shape[1])
    if dim_count != 2:
        raise TypeError("jnp.histogramdd lowering currently supports exactly 2 dims")
    return int(shape[0])


def _abstract_eval_via_orig(
    prim: Any,
    func_name: str,
    sample: core.AbstractValue,
    x_edges: core.AbstractValue,
    y_edges: core.AbstractValue,
) -> tuple[core.ShapedArray, core.ShapedArray, core.ShapedArray]:
    sample_shape = tuple(getattr(sample, "shape", ()))
    x_edges_shape = tuple(getattr(x_edges, "shape", ()))
    y_edges_shape = tuple(getattr(y_edges, "shape", ()))
    _validate_sample_shape(sample_shape)
    _validate_edges_shape(x_edges_shape, axis_name="x")
    _validate_edges_shape(y_edges_shape, axis_name="y")

    sample_spec = jax.ShapeDtypeStruct(
        sample_shape,
        np.dtype(getattr(sample, "dtype", np.float32)),
    )
    x_edges_spec = jax.ShapeDtypeStruct(
        x_edges_shape,
        np.dtype(getattr(x_edges, "dtype", np.float32)),
    )
    y_edges_spec = jax.ShapeDtypeStruct(
        y_edges_shape,
        np.dtype(getattr(y_edges, "dtype", np.float32)),
    )
    orig = get_orig_impl(prim, func_name)
    hist, edges = jax.eval_shape(
        lambda sample_val, x_bins, y_bins: orig(
            sample_val,
            bins=(x_bins, y_bins),
        ),
        sample_spec,
        x_edges_spec,
        y_edges_spec,
    )
    return (
        core.ShapedArray(
            tuple(getattr(hist, "shape", ())),
            np.dtype(getattr(hist, "dtype", np.int32)),
        ),
        core.ShapedArray(
            tuple(getattr(edges[0], "shape", ())),
            np.dtype(getattr(edges[0], "dtype", np.float32)),
        ),
        core.ShapedArray(
            tuple(getattr(edges[1], "shape", ())),
            np.dtype(getattr(edges[1], "dtype", np.float32)),
        ),
    )


def _gather_sample_column(
    ctx: LoweringContextProtocol,
    sample: ir.Value,
    *,
    column: int,
    value_count: int,
    name_hint: str,
) -> ir.Value:
    index = _const_i64(
        ctx,
        np.asarray(column, dtype=np.int64),
        f"{name_hint}_index",
    )
    result = ctx.builder.Gather(
        sample,
        index,
        axis=1,
        _outputs=[ctx.fresh_name(name_hint)],
    )
    if getattr(sample, "type", None) is not None:
        result.type = sample.type
    _stamp_type_and_shape(result, (value_count,))
    _ensure_value_metadata(ctx, result)
    return result


@register_primitive(
    jaxpr_primitive=_HISTOGRAMDD_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogramdd.html",
    onnx=[
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
    component="histogramdd",
    testcases=[
        {
            "testcase": "jnp_histogramdd_2d_explicit_bins",
            "callable": lambda sample, x_edges, y_edges: jnp.histogramdd(
                sample,
                bins=(x_edges, y_edges),
            ),
            "input_values": [
                np.asarray(
                    [
                        [0.0, 0.0],
                        [0.5, 1.0],
                        [1.0, 1.0],
                        [2.0, 2.0],
                        [3.0, 3.0],
                        [4.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
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
            "testcase": "jnp_histogramdd_2d_out_of_range",
            "callable": lambda sample, x_edges, y_edges: jnp.histogramdd(
                sample,
                bins=(x_edges, y_edges),
            ),
            "input_values": [
                np.asarray(
                    [
                        [-1.0, 0.0],
                        [0.0, -1.0],
                        [1.0, 1.0],
                        [3.0, 4.0],
                        [5.0, 2.0],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32),
                np.asarray([0.0, 2.0, 4.0], dtype=np.float32),
            ],
            "expected_output_shapes": [(3, 2), (4,), (3,)],
        },
        {
            "testcase": "jnp_histogramdd_2d_integer_inputs",
            "callable": lambda sample, x_edges, y_edges: jnp.histogramdd(
                sample,
                bins=(x_edges, y_edges),
            ),
            "input_values": [
                np.asarray([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32),
                np.asarray([0, 2, 4], dtype=np.int32),
                np.asarray([0, 1, 4], dtype=np.int32),
            ],
            "expected_output_shapes": [(2, 2), (3,), (3,)],
        },
    ],
)
class JnpHistogramddPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _HISTOGRAMDD_PRIM
    _FUNC_NAME: ClassVar[str] = "histogramdd"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        sample: core.AbstractValue,
        x_edges: core.AbstractValue,
        y_edges: core.AbstractValue,
        *,
        density: bool | None = None,
    ) -> tuple[core.ShapedArray, ...]:
        if density not in (None, False):
            raise NotImplementedError(
                "jnp.histogramdd with density=True is not supported for ONNX export"
            )
        return _abstract_eval_via_orig(
            JnpHistogramddPlugin._PRIM,
            JnpHistogramddPlugin._FUNC_NAME,
            sample,
            x_edges,
            y_edges,
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        sample_var, x_edges_var, y_edges_var = eqn.invars
        hist_var, x_edges_out_var, y_edges_out_var = eqn.outvars
        params = getattr(eqn, "params", {})
        if params.get("density") not in (None, False):
            raise NotImplementedError(
                "jnp.histogramdd with density=True is not supported for ONNX export"
            )

        sample_shape = tuple(getattr(sample_var.aval, "shape", ()))
        x_edges_shape = tuple(getattr(x_edges_var.aval, "shape", ()))
        y_edges_shape = tuple(getattr(y_edges_var.aval, "shape", ()))
        value_count = _validate_sample_shape(sample_shape)
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
        sample_dtype: np.dtype[Any] = np.dtype(
            getattr(sample_var.aval, "dtype", np.float32)
        )
        x_edges_dtype: np.dtype[Any] = np.dtype(
            getattr(x_edges_var.aval, "dtype", x_edges_out_dtype)
        )
        y_edges_dtype: np.dtype[Any] = np.dtype(
            getattr(y_edges_var.aval, "dtype", y_edges_out_dtype)
        )
        compare_dtype: np.dtype[Any] = np.promote_types(
            sample_dtype,
            np.promote_types(x_edges_out_dtype, y_edges_out_dtype),
        )

        sample_val = ctx.get_value_for_var(
            sample_var,
            name_hint=ctx.fresh_name("histdd_sample"),
        )
        x_edges_val = ctx.get_value_for_var(
            x_edges_var,
            name_hint=ctx.fresh_name("histdd_x_edges"),
        )
        y_edges_val = ctx.get_value_for_var(
            y_edges_var,
            name_hint=ctx.fresh_name("histdd_y_edges"),
        )
        hist_spec = ctx.get_value_for_var(
            hist_var, name_hint=ctx.fresh_name("histdd_hist")
        )
        x_edges_spec = ctx.get_value_for_var(
            x_edges_out_var, name_hint=ctx.fresh_name("histdd_x_edges_out")
        )
        y_edges_spec = ctx.get_value_for_var(
            y_edges_out_var, name_hint=ctx.fresh_name("histdd_y_edges_out")
        )

        x_edges_ready = _as_dtype(
            ctx,
            x_edges_val,
            from_dtype=x_edges_dtype,
            to_dtype=x_edges_out_dtype,
            shape=x_edges_shape,
            name_hint="histdd_x_edges_cast",
        )
        y_edges_ready = _as_dtype(
            ctx,
            y_edges_val,
            from_dtype=y_edges_dtype,
            to_dtype=y_edges_out_dtype,
            shape=y_edges_shape,
            name_hint="histdd_y_edges_cast",
        )
        sample_ready = _as_dtype(
            ctx,
            sample_val,
            from_dtype=sample_dtype,
            to_dtype=compare_dtype,
            shape=sample_shape,
            name_hint="histdd_sample_cast",
        )
        x_compare_edges = _as_dtype(
            ctx,
            x_edges_ready,
            from_dtype=x_edges_out_dtype,
            to_dtype=compare_dtype,
            shape=x_edges_shape,
            name_hint="histdd_x_compare_edges_cast",
        )
        y_compare_edges = _as_dtype(
            ctx,
            y_edges_ready,
            from_dtype=y_edges_out_dtype,
            to_dtype=compare_dtype,
            shape=y_edges_shape,
            name_hint="histdd_y_compare_edges_cast",
        )

        flat_x = _gather_sample_column(
            ctx,
            sample_ready,
            column=0,
            value_count=value_count,
            name_hint="histdd_x_values",
        )
        flat_y = _gather_sample_column(
            ctx,
            sample_ready,
            column=1,
            value_count=value_count,
            name_hint="histdd_y_values",
        )
        x_values = ctx.builder.Unsqueeze(
            flat_x,
            _const_i64(ctx, np.asarray([0], dtype=np.int64), "histdd_x_axes"),
            _outputs=[ctx.fresh_name("histdd_x_values_unsqueeze")],
        )
        x_values.type = flat_x.type
        _stamp_type_and_shape(x_values, (1, value_count))
        _ensure_value_metadata(ctx, x_values)
        y_values = ctx.builder.Unsqueeze(
            flat_y,
            _const_i64(ctx, np.asarray([0], dtype=np.int64), "histdd_y_axes"),
            _outputs=[ctx.fresh_name("histdd_y_values_unsqueeze")],
        )
        y_values.type = flat_y.type
        _stamp_type_and_shape(y_values, (1, value_count))
        _ensure_value_metadata(ctx, y_values)

        x_membership = _interval_membership(
            ctx,
            x_values,
            x_compare_edges,
            bin_count=x_bin_count,
            value_count=value_count,
            name_hint="histdd_x",
        )
        y_membership = _interval_membership(
            ctx,
            y_values,
            y_compare_edges,
            bin_count=y_bin_count,
            value_count=value_count,
            name_hint="histdd_y",
        )
        x_membership_3d = ctx.builder.Unsqueeze(
            x_membership,
            _const_i64(ctx, np.asarray([1], dtype=np.int64), "histdd_x_member_axes"),
            _outputs=[ctx.fresh_name("histdd_x_membership_unsqueeze")],
        )
        x_membership_3d.type = x_membership.type
        _stamp_type_and_shape(x_membership_3d, (x_bin_count, 1, value_count))
        _ensure_value_metadata(ctx, x_membership_3d)
        y_membership_3d = ctx.builder.Unsqueeze(
            y_membership,
            _const_i64(ctx, np.asarray([0], dtype=np.int64), "histdd_y_member_axes"),
            _outputs=[ctx.fresh_name("histdd_y_membership_unsqueeze")],
        )
        y_membership_3d.type = y_membership.type
        _stamp_type_and_shape(y_membership_3d, (1, y_bin_count, value_count))
        _ensure_value_metadata(ctx, y_membership_3d)

        joint_shape = (x_bin_count, y_bin_count, value_count)
        joint = ctx.builder.And(
            x_membership_3d,
            y_membership_3d,
            _outputs=[ctx.fresh_name("histdd_joint_membership")],
        )
        joint.type = ir.TensorType(ir.DataType.BOOL)
        _stamp_type_and_shape(joint, joint_shape)
        _ensure_value_metadata(ctx, joint)

        hist_enum = _dtype_to_ir(hist_dtype, ctx.builder.enable_double_precision)
        hits = ctx.builder.Cast(
            joint,
            to=int(hist_enum.value),
            _outputs=[ctx.fresh_name("histdd_hits")],
        )
        hits.type = ir.TensorType(hist_enum)
        _stamp_type_and_shape(hits, joint_shape)
        _ensure_value_metadata(ctx, hits)

        reduce_axes = _const_i64(
            ctx,
            np.asarray([2], dtype=np.int64),
            "histdd_reduce_axes",
        )
        hist_name = getattr(hist_spec, "name", None) or ctx.fresh_name("histdd_hist")
        hist_producer = getattr(hist_spec, "producer", None)
        if callable(hist_producer) and hist_producer() is not None:
            hist_name = ctx.fresh_name("histdd_hist")
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
            name_hint="histdd_x_edges_out",
        )
        y_edges_result = self._identity_output(
            ctx,
            y_edges_ready,
            out_spec=y_edges_spec,
            output_shape=y_edges_out_shape,
            output_dtype=y_edges_out_dtype,
            name_hint="histdd_y_edges_out",
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
        result = ctx.builder.Identity(value, _outputs=[output_name])
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = ir.TensorType(
                _dtype_to_ir(output_dtype, ctx.builder.enable_double_precision)
            )
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        else:
            _stamp_type_and_shape(result, output_shape)
        _ensure_value_metadata(ctx, result)
        return result

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., tuple[jax.Array, list[jax.Array]]] | None,
        ) -> Callable[..., tuple[jax.Array, list[jax.Array]]]:
            if orig is None:
                raise RuntimeError(
                    "Original jnp.histogramdd not found for monkey patching"
                )
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                sample: ArrayLike,
                bins: ArrayLike | list[ArrayLike] = 10,
                range: Sequence[None | ArrayLike | Sequence[ArrayLike]] | None = None,
                weights: ArrayLike | None = None,
                density: bool | None = None,
            ) -> tuple[jax.Array, list[jax.Array]]:
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
                        "jnp.histogramdd lowering requires explicit 1-D bin edge arrays"
                    )
                flat = cls._PRIM.bind(
                    jnp.asarray(sample),
                    x_edges_array,
                    y_edges_array,
                    density=density,
                )
                hist, x_edges_out, y_edges_out = cast(tuple[jax.Array, ...], flat)
                return hist, [x_edges_out, y_edges_out]

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


@JnpHistogramddPlugin._PRIM.def_impl
def _histogramdd_impl(
    sample: object,
    x_edges: object,
    y_edges: object,
    *,
    density: bool | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    orig = get_orig_impl(JnpHistogramddPlugin._PRIM, JnpHistogramddPlugin._FUNC_NAME)
    hist, edges = orig(sample, bins=(x_edges, y_edges), density=density)
    return cast(
        tuple[jax.Array, jax.Array, jax.Array],
        (hist, edges[0], edges[1]),
    )


JnpHistogramddPlugin._PRIM.def_abstract_eval(JnpHistogramddPlugin.abstract_eval)


def _histogramdd_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    density: bool | None = None,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[BatchDim, BatchDim, BatchDim]]:
    del density
    if any(dim is not None for dim in batch_dims):
        raise NotImplementedError("vmap over jnp.histogramdd is not supported")
    hist, x_edges, y_edges = JnpHistogramddPlugin._PRIM.bind(*batched_args)
    return (hist, x_edges, y_edges), (None, None, None)


batching.primitive_batchers[JnpHistogramddPlugin._PRIM] = _histogramdd_batch_rule
