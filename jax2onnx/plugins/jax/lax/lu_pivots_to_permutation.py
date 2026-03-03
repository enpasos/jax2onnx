# jax2onnx/plugins/jax/lax/lu_pivots_to_permutation.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _stamp_like(value: Any, ref: Any) -> None:
    if getattr(ref, "type", None) is not None:
        value.type = ref.type
    if getattr(ref, "shape", None) is not None:
        value.shape = ref.shape


def _lower_single_permutation(
    ctx: "IRContext",
    pivots_1d: ir.Value,
    *,
    k: int,
    permutation_size: int,
    name_prefix: str,
) -> ir.Value:
    perm = _const_i64(
        ctx,
        np.arange(permutation_size, dtype=np.int64),
        f"{name_prefix}_perm_init",
    )
    perm.type = ir.TensorType(ir.DataType.INT64)
    perm.shape = ir.Shape((permutation_size,))

    for i in range(k):
        i_idx = _const_i64(
            ctx, np.asarray([i], dtype=np.int64), f"{name_prefix}_i_idx_{i}"
        )
        piv_i = ctx.builder.Gather(
            pivots_1d,
            i_idx,
            axis=0,
            _outputs=[ctx.fresh_name(f"{name_prefix}_piv_i_{i}")],
        )
        piv_i.type = ir.TensorType(ir.DataType.INT64)
        piv_i.shape = ir.Shape((1,))

        val_i = ctx.builder.Gather(
            perm,
            i_idx,
            axis=0,
            _outputs=[ctx.fresh_name(f"{name_prefix}_val_i_{i}")],
        )
        val_i.type = ir.TensorType(ir.DataType.INT64)
        val_i.shape = ir.Shape((1,))

        val_j = ctx.builder.Gather(
            perm,
            piv_i,
            axis=0,
            _outputs=[ctx.fresh_name(f"{name_prefix}_val_j_{i}")],
        )
        val_j.type = ir.TensorType(ir.DataType.INT64)
        val_j.shape = ir.Shape((1,))

        perm = ctx.builder.ScatterElements(
            perm,
            i_idx,
            val_j,
            axis=0,
            _outputs=[ctx.fresh_name(f"{name_prefix}_swap_i_{i}")],
        )
        perm.type = ir.TensorType(ir.DataType.INT64)
        perm.shape = ir.Shape((permutation_size,))

        perm = ctx.builder.ScatterElements(
            perm,
            piv_i,
            val_i,
            axis=0,
            _outputs=[ctx.fresh_name(f"{name_prefix}_swap_j_{i}")],
        )
        perm.type = ir.TensorType(ir.DataType.INT64)
        perm.shape = ir.Shape((permutation_size,))

    return perm


@register_primitive(
    jaxpr_primitive=jax.lax.linalg.lu_pivots_to_permutation_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.lu_pivots_to_permutation.html",
    onnx=[
        {
            "component": "ScatterElements",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterElements.html",
        },
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "Squeeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Squeeze.html",
        },
        {
            "component": "Unsqueeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unsqueeze.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.12.1",
    context="primitives.lax",
    component="lu_pivots_to_permutation",
    testcases=[
        {
            "testcase": "lu_pivots_to_permutation_basic",
            "callable": lambda pivots: jax.lax.linalg.lu_pivots_to_permutation(
                pivots, permutation_size=4
            ),
            "input_values": [np.asarray([2, 2, 3, 3], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Gather", "ScatterElements"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lu_pivots_to_permutation_identity",
            "callable": lambda pivots: jax.lax.linalg.lu_pivots_to_permutation(
                pivots, permutation_size=3
            ),
            "input_values": [np.asarray([0, 1, 2], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["ScatterElements"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "lu_pivots_to_permutation_batched",
            "callable": lambda pivots: jax.lax.linalg.lu_pivots_to_permutation(
                pivots, permutation_size=4
            ),
            "input_values": [np.asarray([[2, 2, 3, 3], [0, 2, 3, 3]], dtype=np.int32)],
            "expected_output_dtypes": [np.int32],
            "post_check_onnx_graph": EG(
                ["Squeeze", "Concat"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class LuPivotsToPermutationPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.linalg.lu_pivots_to_permutation`` with sequential swaps."""

    def lower(self, ctx: "IRContext", eqn: Any) -> None:
        (pivots_var,) = eqn.invars
        (out_var,) = eqn.outvars
        params = dict(getattr(eqn, "params", {}) or {})
        permutation_size = int(params.get("permutation_size", 0))
        if permutation_size < 0:
            raise ValueError(
                f"lu_pivots_to_permutation requires non-negative permutation_size, got {permutation_size}"
            )

        pivots = ctx.get_value_for_var(
            pivots_var, name_hint=ctx.fresh_name("lu_pivots")
        )
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("lu_perm"))

        pivots_shape = tuple(getattr(getattr(pivots_var, "aval", None), "shape", ()))
        if len(pivots_shape) not in {1, 2}:
            raise NotImplementedError(
                "lu_pivots_to_permutation currently supports rank-1 or rank-2 pivots"
            )

        pivots_i64 = pivots
        if getattr(getattr(pivots, "type", None), "dtype", None) != ir.DataType.INT64:
            pivots_i64 = ctx.builder.Cast(
                pivots,
                to=int(ir.DataType.INT64.value),
                _outputs=[ctx.fresh_name("lu_pivots_i64")],
            )
            pivots_i64.type = ir.TensorType(ir.DataType.INT64)
            if getattr(pivots, "shape", None) is not None:
                pivots_i64.shape = pivots.shape

        out_i64_shape: tuple[int, ...]
        if len(pivots_shape) == 1:
            k_raw = pivots_shape[0]
            if not isinstance(k_raw, (int, np.integer)):
                raise NotImplementedError(
                    "lu_pivots_to_permutation requires static pivots length"
                )
            k = int(k_raw)
            perm = _lower_single_permutation(
                ctx,
                pivots_i64,
                k=k,
                permutation_size=permutation_size,
                name_prefix="lu",
            )
            out_i64_shape = (permutation_size,)
        else:
            batch_raw, k_raw = pivots_shape
            if not isinstance(batch_raw, (int, np.integer)) or not isinstance(
                k_raw, (int, np.integer)
            ):
                raise NotImplementedError(
                    "lu_pivots_to_permutation requires static batch and pivots lengths"
                )
            batch = int(batch_raw)
            k = int(k_raw)
            if batch < 0:
                raise ValueError(
                    f"lu_pivots_to_permutation requires non-negative batch size, got {batch}"
                )

            if batch == 0:
                perm = _const_i64(
                    ctx,
                    np.zeros((0, permutation_size), dtype=np.int64),
                    "lu_perm_empty",
                )
                perm.type = ir.TensorType(ir.DataType.INT64)
                perm.shape = ir.Shape((0, permutation_size))
            else:
                squeeze_axes = _const_i64(
                    ctx,
                    np.asarray([0], dtype=np.int64),
                    "lu_batch_sq_axes",
                )
                unsqueeze_axes = _const_i64(
                    ctx,
                    np.asarray([0], dtype=np.int64),
                    "lu_batch_unsq_axes",
                )
                rows: list[ir.Value] = []
                for b in range(batch):
                    b_idx = _const_i64(
                        ctx, np.asarray([b], dtype=np.int64), f"lu_batch_idx_{b}"
                    )
                    row_2d = ctx.builder.Gather(
                        pivots_i64,
                        b_idx,
                        axis=0,
                        _outputs=[ctx.fresh_name(f"lu_batch_row2d_{b}")],
                    )
                    row_2d.type = ir.TensorType(ir.DataType.INT64)
                    row_2d.shape = ir.Shape((1, k))
                    row = ctx.builder.Squeeze(
                        row_2d,
                        squeeze_axes,
                        _outputs=[ctx.fresh_name(f"lu_batch_row_{b}")],
                    )
                    row.type = ir.TensorType(ir.DataType.INT64)
                    row.shape = ir.Shape((k,))

                    row_perm = _lower_single_permutation(
                        ctx,
                        row,
                        k=k,
                        permutation_size=permutation_size,
                        name_prefix=f"lu_b{b}",
                    )
                    row_perm_2d = ctx.builder.Unsqueeze(
                        row_perm,
                        unsqueeze_axes,
                        _outputs=[ctx.fresh_name(f"lu_row_unsq_{b}")],
                    )
                    row_perm_2d.type = ir.TensorType(ir.DataType.INT64)
                    row_perm_2d.shape = ir.Shape((1, permutation_size))
                    rows.append(row_perm_2d)
                perm = ctx.builder.Concat(
                    *rows,
                    axis=0,
                    _outputs=[ctx.fresh_name("lu_perm_batched_i64")],
                )
                perm.type = ir.TensorType(ir.DataType.INT64)
                perm.shape = ir.Shape((batch, permutation_size))
            out_i64_shape = (batch, permutation_size)

        out_dtype: np.dtype[Any] = np.dtype(
            getattr(getattr(out_var, "aval", None), "dtype", np.int32)
        )
        out_dtype_enum = _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        if out_dtype_enum is None:
            raise TypeError(
                f"Unsupported lu_pivots_to_permutation output dtype '{out_dtype}'"
            )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("lu_perm")
        result = perm
        if out_dtype_enum != ir.DataType.INT64:
            result = ctx.builder.Cast(
                perm,
                to=int(out_dtype_enum.value),
                _outputs=[desired_name],
            )
            result.type = ir.TensorType(out_dtype_enum)
            result.shape = ir.Shape(out_i64_shape)

        if result is perm:
            # Keep requested output name when no cast is needed.
            renamed = ctx.builder.Identity(perm, _outputs=[desired_name])
            renamed.type = ir.TensorType(ir.DataType.INT64)
            renamed.shape = ir.Shape(out_i64_shape)
            result = renamed

        _stamp_like(result, out_spec if getattr(out_spec, "type", None) else result)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        ctx.bind_value_for_var(out_var, result)
