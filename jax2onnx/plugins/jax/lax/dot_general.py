# jax2onnx/plugins/jax/lax/dot_general.py

from __future__ import annotations

import itertools
import string
from typing import TYPE_CHECKING

import jax
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


@register_primitive(
    jaxpr_primitive=jax.lax.dot_general_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html",
    onnx=[
        {
            "component": "MatMul/Gemm",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        },
    ],
    since="v0.2.0",
    context="primitives.lax",
    component="dot_general",
    testcases=[
        {
            "testcase": "dot_contract_nm",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ()))
            ),
            "input_shapes": [(3, 4), (4, 2)],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Gemm:3x2", "inputs": {2: {"const": 0.0}}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_contract_min",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (1,)), ((), ()))
            ),
            "input_shapes": [(3, 4), (2, 4)],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Gemm:3x2", "inputs": {2: {"const": 0.0}}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_general",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ()))
            ),
            "input_shapes": [(3, 3), (3, 3)],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Gemm:3x3", "inputs": {2: {"const": 0.0}}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_general_lhs1_rhs1",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (1,)), ((), ()))
            ),
            "input_shapes": [(3, 3), (3, 3)],
            "post_check_onnx_graph": EG(
                [
                    {"path": "Gemm:3x3", "inputs": {2: {"const": 0.0}}},
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_double_contract",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1, 2), (0, 1)), ((), ()))
            ),
            "input_shapes": [(3, 4, 5), (4, 5, 6)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_batched_double_contract",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((2, 3), (1, 2)), ((0,), (0,)))
            ),
            "input_shapes": [(2, 3, 4, 5), (2, 4, 5, 6)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_highrank_batch",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((3,), (2,)), ((0, 1), (0, 1)))
            ),
            "input_shapes": [(2, 2, 3, 4), (2, 2, 4, 5)],
            "post_check_onnx_graph": EG(
                ["MatMul"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_contract_inner_lhs_with_middle_rhs",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((1,), (0,)), ((), ()))
            ),
            "input_shapes": [(3, 4, 5), (4, 5, 6)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_outer_product",
            "callable": lambda x, y: jax.lax.dot_general(x, y, (((), ()), ((), ()))),
            "input_shapes": [(3,), (4,)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dot_full_contract_scalar",
            "callable": lambda x, y: jax.lax.dot_general(
                x, y, (((0, 1), (0, 1)), ((), ()))
            ),
            "input_shapes": [(2, 3), (2, 3)],
            "post_check_onnx_graph": EG(
                ["Einsum"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class DotGeneralPlugin(PrimitiveLeafPlugin):
    """Lower ``lax.dot_general`` via MatMul/Gemm fast-path or Einsum fallback."""

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        lhs_var, rhs_var = eqn.invars
        out_var = eqn.outvars[0]

        params = getattr(eqn, "params", {})
        ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = params[
            "dimension_numbers"
        ]

        lhs_contract = tuple(lhs_contract)
        rhs_contract = tuple(rhs_contract)
        lhs_batch = tuple(lhs_batch)
        rhs_batch = tuple(rhs_batch)

        lhs_val = ctx.get_value_for_var(lhs_var, name_hint=ctx.fresh_name("dot_lhs"))
        rhs_val = ctx.get_value_for_var(rhs_var, name_hint=ctx.fresh_name("dot_rhs"))
        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("dot_out"))

        lhs_shape = tuple(getattr(lhs_var.aval, "shape", ()))
        rhs_shape = tuple(getattr(rhs_var.aval, "shape", ()))
        out_shape = tuple(getattr(out_var.aval, "shape", ()))

        if self._try_lower_matmul(
            ctx,
            lhs_var,
            rhs_var,
            out_var,
            lhs_val,
            rhs_val,
            out_spec,
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_contract,
            rhs_contract,
            lhs_batch,
            rhs_batch,
        ):
            return

        self._lower_via_einsum(
            ctx,
            lhs_var,
            rhs_var,
            out_var,
            lhs_val,
            rhs_val,
            out_spec,
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_contract,
            rhs_contract,
            lhs_batch,
            rhs_batch,
        )

    def _try_lower_matmul(
        self,
        ctx: "IRContext",
        lhs_var,
        rhs_var,
        out_var,
        lhs_val,
        rhs_val,
        out_spec,
        lhs_shape,
        rhs_shape,
        out_shape,
        lhs_contract,
        rhs_contract,
        lhs_batch,
        rhs_batch,
    ) -> bool:
        if lhs_batch or rhs_batch:
            if tuple(lhs_batch) != tuple(rhs_batch):
                return False
            if len(lhs_contract) != 1 or len(rhs_contract) != 1:
                return False

            lhs_contract_axis = lhs_contract[0]
            rhs_contract_axis = rhs_contract[0]

            lhs_rank = len(lhs_shape)
            rhs_rank = len(rhs_shape)

            lhs_free_axes = [
                axis
                for axis in range(lhs_rank)
                if axis not in lhs_batch and axis != lhs_contract_axis
            ]
            rhs_free_axes = [
                axis
                for axis in range(rhs_rank)
                if axis not in rhs_batch and axis != rhs_contract_axis
            ]

            if len(lhs_free_axes) != 1 or len(rhs_free_axes) != 1:
                return False

            lhs_perm = list(lhs_batch) + lhs_free_axes + [lhs_contract_axis]
            rhs_perm = list(rhs_batch) + [rhs_contract_axis] + rhs_free_axes

            def _transpose_if_needed(
                value, perm, original_shape, name_hint: str
            ) -> tuple[ir.Value, tuple[int, ...]]:
                if perm == list(range(len(original_shape))):
                    return value, original_shape
                permuted = ctx.builder.Transpose(
                    value,
                    _outputs=[ctx.fresh_name(name_hint)],
                    perm=perm,
                )
                val_dtype = getattr(getattr(value, "type", None), "dtype", None)
                if val_dtype is not None:
                    permuted.type = ir.TensorType(val_dtype)
                perm_shape = tuple(original_shape[i] for i in perm)
                _stamp_type_and_shape(permuted, perm_shape)
                _ensure_value_metadata(ctx, permuted)
                return permuted, perm_shape

            lhs_prepped, _ = _transpose_if_needed(
                lhs_val, lhs_perm, lhs_shape, "dot_lhs_perm"
            )
            rhs_prepped, _ = _transpose_if_needed(
                rhs_val, rhs_perm, rhs_shape, "dot_rhs_perm"
            )

            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("MatMul")
            result = ctx.builder.MatMul(
                lhs_prepped,
                rhs_prepped,
                _outputs=[desired_name],
            )
            out_dtype = np.dtype(
                getattr(
                    out_var.aval,
                    "dtype",
                    getattr(lhs_var.aval, "dtype", np.float32),
                )
            )
            result.type = ir.TensorType(
                _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
            )
            _stamp_type_and_shape(result, out_shape)
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)
            return True

        if len(lhs_contract) != 1 or len(rhs_contract) != 1:
            return False
        if len(lhs_batch) != 0 or len(rhs_batch) != 0:
            return False
        if len(lhs_shape) != 2 or len(rhs_shape) != 2:
            return False

        rhs_contract_axis = rhs_contract[0]
        rhs_rank = len(rhs_shape)
        if rhs_contract_axis not in (0, rhs_rank - 1):
            return False

        transpose_rhs = rhs_contract_axis == rhs_rank - 1
        rhs_input = rhs_val
        if transpose_rhs:
            perm = list(range(rhs_rank))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            rhs_perm_shape = tuple(rhs_shape[i] for i in perm)
            transposed = ctx.builder.Transpose(
                rhs_val,
                _outputs=[ctx.fresh_name("dot_rhs_T")],
                perm=perm,
            )
            rhs_dtype = getattr(getattr(rhs_val, "type", None), "dtype", None)
            if rhs_dtype is not None:
                transposed.type = ir.TensorType(rhs_dtype)
            _stamp_type_and_shape(transposed, rhs_perm_shape)
            _ensure_value_metadata(ctx, transposed)
            rhs_input = transposed

        out_dtype = np.dtype(
            getattr(out_var.aval, "dtype", getattr(lhs_var.aval, "dtype", np.float32))
        )
        bias_val = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name("dot_bias"),
            value=np.array(0, dtype=out_dtype),
        )

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Gemm")
        result = ctx.builder.Gemm(
            lhs_val,
            rhs_input,
            bias_val,
            alpha=1.0,
            beta=0.0,
            _outputs=[desired_name],
        )

        _stamp_type_and_shape(result, out_shape)
        result.type = ir.TensorType(
            _dtype_to_ir(out_dtype, ctx.builder.enable_double_precision)
        )
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
        return True

    def _lower_via_einsum(
        self,
        ctx: "IRContext",
        lhs_var,
        rhs_var,
        out_var,
        lhs_val,
        rhs_val,
        out_spec,
        lhs_shape,
        rhs_shape,
        out_shape,
        lhs_contract,
        rhs_contract,
        lhs_batch,
        rhs_batch,
    ) -> None:
        lhs_shape = tuple(lhs_shape)
        rhs_shape = tuple(rhs_shape)
        lhs_contract = tuple(lhs_contract)
        rhs_contract = tuple(rhs_contract)
        lhs_batch = tuple(lhs_batch)
        rhs_batch = tuple(rhs_batch)

        target_batch_rank = max(len(lhs_batch), len(rhs_batch))

        if len(lhs_batch) < target_batch_rank:
            lhs_val, lhs_shape, lhs_batch, lhs_contract = self._pad_operand_front(
                ctx,
                lhs_val,
                lhs_shape,
                lhs_batch,
                lhs_contract,
                target_batch_rank - len(lhs_batch),
                "dot_lhs_batch_pad",
            )
        if len(rhs_batch) < target_batch_rank:
            rhs_val, rhs_shape, rhs_batch, rhs_contract = self._pad_operand_front(
                ctx,
                rhs_val,
                rhs_shape,
                rhs_batch,
                rhs_contract,
                target_batch_rank - len(rhs_batch),
                "dot_rhs_batch_pad",
            )

        if len(lhs_contract) != len(rhs_contract):
            raise TypeError(
                "dot_general requires equal numbers of contracting dims on LHS/RHS"
            )

        for i, j in zip(lhs_batch, rhs_batch):
            lhs_dim = lhs_shape[i]
            rhs_dim = rhs_shape[j]
            if (
                lhs_dim is not None
                and rhs_dim is not None
                and lhs_dim not in (rhs_dim, 1)
                and rhs_dim not in (lhs_dim, 1)
            ):
                raise ValueError(
                    f"Batch dim mismatch: lhs[{i}]={lhs_dim} vs rhs[{j}]={rhs_dim}"
                )

        for i, j in zip(lhs_contract, rhs_contract):
            lhs_dim = lhs_shape[i]
            rhs_dim = rhs_shape[j]
            if lhs_dim is not None and rhs_dim is not None and lhs_dim != rhs_dim:
                raise ValueError(
                    f"Contract dim mismatch: lhs[{i}]={lhs_dim} vs rhs[{j}]={rhs_dim}"
                )

        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)

        lhs_batch_set = set(lhs_batch)
        rhs_batch_set = set(rhs_batch)
        lhs_contract_set = set(lhs_contract)
        rhs_contract_set = set(rhs_contract)

        lhs_free = [
            axis
            for axis in range(lhs_rank)
            if axis not in lhs_batch_set | lhs_contract_set
        ]
        rhs_free = [
            axis
            for axis in range(rhs_rank)
            if axis not in rhs_batch_set | rhs_contract_set
        ]

        main_gen = self._label_stream(string.ascii_lowercase + string.ascii_uppercase)
        batch_gen = self._label_stream(
            string.ascii_lowercase[::-1] + string.ascii_uppercase[::-1]
        )

        lhs_lbl = [None] * lhs_rank  # type: ignore[list-item]
        rhs_lbl = [None] * rhs_rank  # type: ignore[list-item]

        batch_pairs = list(zip(lhs_batch, rhs_batch))
        contract_pairs = list(zip(lhs_contract, rhs_contract))
        lhs_for_rhs_contract = {
            rhs_axis: lhs_axis for lhs_axis, rhs_axis in contract_pairs
        }

        for i, j in batch_pairs:
            label = next(batch_gen)
            lhs_lbl[i] = label
            rhs_lbl[j] = label

        for i in range(lhs_rank):
            if i in lhs_batch_set:
                continue
            if lhs_lbl[i] is None:
                lhs_lbl[i] = next(main_gen)

        for j in range(rhs_rank):
            if j in rhs_batch_set:
                continue
            if j in rhs_contract_set:
                lhs_axis = lhs_for_rhs_contract[j]
                rhs_lbl[j] = lhs_lbl[lhs_axis]
            else:
                if rhs_lbl[j] is None:
                    rhs_lbl[j] = next(main_gen)

        rhs_out_order = []
        for axis in lhs_batch:
            rhs_out_order.append(lhs_lbl[axis])
        for axis in lhs_free:
            rhs_out_order.append(lhs_lbl[axis])
        for axis in rhs_free:
            rhs_out_order.append(rhs_lbl[axis])

        if not lhs_contract and not rhs_contract:
            rhs_out_order = lhs_lbl + rhs_lbl

        if not lhs_free and not rhs_free:
            if lhs_batch:
                rhs_out_order = [lhs_lbl[i] for i in lhs_batch]
            else:
                rhs_out_order = []

        if any(label is None for label in lhs_lbl):
            raise RuntimeError(f"Unlabeled LHS axes: {lhs_lbl}")
        if any(label is None for label in rhs_lbl):
            raise RuntimeError(f"Unlabeled RHS axes: {rhs_lbl}")

        lhs_labels = "".join(lhs_lbl)
        rhs_labels = "".join(rhs_lbl)
        out_labels = "".join(rhs_out_order)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("Einsum")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Einsum")

        result = ctx.builder.Einsum(
            lhs_val,
            rhs_val,
            equation=f"{lhs_labels},{rhs_labels}->{out_labels}",
            _outputs=[desired_name],
        )

        spec_type = getattr(out_spec, "type", None)
        if spec_type is not None:
            result.type = spec_type
        else:
            aval_dtype = getattr(out_var.aval, "dtype", None)
            if aval_dtype is not None:
                result.type = ir.TensorType(
                    _dtype_to_ir(
                        np.dtype(aval_dtype), ctx.builder.enable_double_precision
                    )
                )
            else:
                inferred_dtype = next(
                    (
                        getattr(getattr(v, "type", None), "dtype", None)
                        for v in (lhs_val, rhs_val)
                        if getattr(getattr(v, "type", None), "dtype", None) is not None
                    ),
                    None,
                )
                if inferred_dtype is not None:
                    result.type = ir.TensorType(inferred_dtype)

        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    def _pad_operand_front(
        self,
        ctx: "IRContext",
        value: ir.Value,
        shape: tuple[int, ...],
        batch_axes: tuple[int, ...],
        contract_axes: tuple[int, ...],
        pad_count: int,
        name_hint: str,
    ) -> tuple[ir.Value, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if pad_count <= 0:
            return value, shape, batch_axes, contract_axes
        axes = tuple(range(pad_count))
        padded, new_shape = self._unsqueeze_value(ctx, value, shape, axes, name_hint)
        batch_axes = tuple(range(pad_count)) + tuple(
            axis + pad_count for axis in batch_axes
        )
        contract_axes = tuple(axis + pad_count for axis in contract_axes)
        return padded, new_shape, batch_axes, contract_axes

    @staticmethod
    def _unsqueeze_value(
        ctx: "IRContext",
        value: ir.Value,
        shape: tuple[int, ...],
        axes: tuple[int, ...],
        name_hint: str,
    ) -> tuple[ir.Value, tuple[int, ...]]:
        if not axes:
            return value, shape
        axes = tuple(sorted(axes))
        axes_const = _const_i64(
            ctx,
            np.asarray(axes, dtype=np.int64),
            ctx.fresh_name(f"{name_hint}_axes"),
        )
        unsqueezed = ctx.builder.Unsqueeze(
            value,
            axes_const,
            _outputs=[ctx.fresh_name(name_hint)],
        )
        val_dtype = getattr(getattr(value, "type", None), "dtype", None)
        if val_dtype is not None:
            unsqueezed.type = ir.TensorType(val_dtype)
        new_shape = list(shape)
        for axis in axes:
            new_shape.insert(axis, 1)
        _stamp_type_and_shape(unsqueezed, tuple(new_shape))
        _ensure_value_metadata(ctx, unsqueezed)
        return unsqueezed, tuple(new_shape)

    @staticmethod
    def _label_stream(alphabet: str):
        length = 1
        while True:
            for combo in itertools.product(alphabet, repeat=length):
                yield "".join(combo)
            length += 1
