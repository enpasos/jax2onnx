# jax2onnx/plugins/jax/nn/dot_product_attention.py

from __future__ import annotations

from collections import defaultdict
from typing import (
    Callable,
    ClassVar,
    DefaultDict,
    Dict,
    Final,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn
from jax.extend.core import Primitive
import onnx_ir as ir
from numpy.typing import ArrayLike

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape, _to_ir_dim_for_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.lax.scatter_utils import (
    _ensure_value_metadata as _ensure_value_metadata,
    _gather_int_scalar,
    _scalar_i64,
    _shape_of,
)
from jax2onnx.plugins.jax.lax._index_utils import _builder_op
from jax2onnx.converter.typing_support import LoweringContextProtocol


_DPA_PRIM: Final[Primitive] = Primitive("jax.nn.dot_product_attention")
_DPA_PRIM.multiple_results = False
_ORIG_DOT_PRODUCT_ATTENTION: Final = jax_nn.dot_product_attention


def _expect_dpa_mask_where_impl(model) -> bool:
    graph = model.graph
    consumers: DefaultDict[str, list[ir.Node]] = defaultdict(list)
    producers: Dict[str, ir.Node] = {}
    for node in graph.node:
        for out in node.output:
            producers[out] = node
        for inp in node.input:
            consumers[inp].append(node)

    for node in graph.node:
        if node.op_type != "Softmax" or not node.output:
            continue
        softmax_out = node.output[0]
        for consumer in consumers.get(softmax_out, []):
            if consumer.op_type != "Mul":
                continue
            if len(consumer.input) < 2:
                continue
            other_inputs = [inp for inp in consumer.input if inp != softmax_out]
            if not other_inputs:
                continue
            producer = producers.get(other_inputs[0])
            if producer and producer.op_type in {"Cast", "Mul", "Constant"}:
                return True
    return False


def _expect_dpa_mask_where(model) -> bool:
    return _expect_dpa_mask_where_impl(model)


_EXPECT_DPA_MASK_WHERE: Final = _expect_dpa_mask_where


def _dtype_enum_from_value(val: ir.Value) -> ir.DataType:
    dtype = getattr(getattr(val, "type", None), "dtype", None)
    if dtype is not None:
        return dtype
    raise TypeError("Missing dtype on value; ensure inputs are typed.")


def _numpy_dtype_from_aval(var) -> np.dtype:
    aval_dtype = getattr(getattr(var, "aval", None), "dtype", None)
    if aval_dtype is None:
        return np.dtype(np.float32)
    return np.dtype(aval_dtype)


DimLike = Union[int, str]


def _coerce_dim(dim: object, name: str) -> Tuple[DimLike, bool]:
    if isinstance(dim, (int, np.integer)):
        return int(dim), True
    if hasattr(dim, "_is_constant") and callable(getattr(dim, "_is_constant")):
        try:
            if dim._is_constant():  # type: ignore[attr-defined]
                const = dim._to_constant()  # type: ignore[attr-defined]
                if isinstance(const, (int, np.integer)):
                    return int(const), True
        except Exception:
            pass
    return str(dim), False


def _cast_to_int64(
    ctx: LoweringContextProtocol, value: ir.Value, *, base: str
) -> ir.Value:
    casted = ctx.builder.Cast(
        value,
        _outputs=[ctx.fresh_name(base)],
        to=int(ir.DataType.INT64.value),
    )
    casted.type = ir.TensorType(ir.DataType.INT64)
    casted.shape = value.shape
    _ensure_value_metadata(ctx, casted)
    return casted


def _make_range_value(
    ctx: LoweringContextProtocol, limit: ir.Value, *, base: str
) -> ir.Value:
    start = _scalar_i64(ctx, 0, f"{base}_start")
    step = _scalar_i64(ctx, 1, f"{base}_step")
    rng = _builder_op(
        ctx,
        "Range",
        [start, limit, step],
        name_hint=base,
        dtype=ir.DataType.INT64,
        shape=(None,),
    )
    return rng


def _logical_and(
    ctx: LoweringContextProtocol,
    lhs: ir.Value,
    rhs: ir.Value,
    *,
    base: str,
    shape_hint,
) -> ir.Value:
    out = _builder_op(
        ctx,
        "And",
        [lhs, rhs],
        name_hint=base,
        dtype=ir.DataType.BOOL,
        shape=tuple(_to_ir_dim_for_shape(d) for d in shape_hint),
    )
    return out


def _builder_tensor_op(
    ctx: LoweringContextProtocol,
    op_type: str,
    inputs: list[ir.Value | ir.Value],
    *,
    base: str,
    dtype: ir.DataType | None = None,
    dtype_like: ir.Value | None = None,
    shape: Tuple[object, ...] | None = None,
    attributes: Dict[str, object] | None = None,
) -> ir.Value:
    dtype_enum = dtype
    if dtype_enum is None and dtype_like is not None:
        dtype_enum = _dtype_enum_from_value(dtype_like)
    shape_dims = (
        tuple(_to_ir_dim_for_shape(dim) for dim in shape) if shape is not None else None
    )
    return _builder_op(
        ctx,
        op_type,
        list(inputs),
        name_hint=base,
        dtype=dtype_enum,
        shape=shape_dims,
        attributes=attributes,
    )


def _symbolic_or_dim(symbol: str, dim: DimLike) -> DimLike:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    if hasattr(dim, "_is_constant") and callable(getattr(dim, "_is_constant")):
        try:
            if dim._is_constant():  # type: ignore[attr-defined]
                const = dim._to_constant()  # type: ignore[attr-defined]
                if isinstance(const, (int, np.integer)):
                    return int(const)
        except Exception:
            pass
    if isinstance(dim, str) and dim:
        return dim
    text = str(dim)
    if text:
        return text
    return symbol


@register_primitive(
    jaxpr_primitive=_DPA_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.dot_product_attention.html",
    onnx=[
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "Softmax",
            "doc": "https://onnx.ai/onnx/operators/onnx__Softmax.html",
        },
    ],
    since="v0.8.0",
    context="primitives.nn",
    component="dot_product_attention",
    testcases=[
        {
            "testcase": "dpa_basic",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:2x8x4x32 -> MatMul:2x8x4x4 -> Mul:2x8x4x4 -> "
                    "Softmax:2x8x4x4 -> MatMul:2x8x4x32 -> Transpose:2x4x8x32"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_positional_bias_mask",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(
                q, k, v, None, None
            ),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:2x8x4x32 -> MatMul:2x8x4x4 -> Mul:2x8x4x4 -> "
                    "Softmax:2x8x4x4 -> MatMul:2x8x4x32 -> Transpose:2x4x8x32"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_diff_heads_embed",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(1, 2, 4, 16), (1, 2, 4, 16), (1, 2, 4, 16)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:1x4x2x16 -> MatMul:1x4x2x2 -> Mul:1x4x2x2 -> "
                    "Softmax:1x4x2x2 -> MatMul:1x4x2x16 -> Transpose:1x2x4x16"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_batch4_seq16",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(4, 2, 16, 8), (4, 2, 16, 8), (4, 2, 16, 8)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:4x16x2x8 -> MatMul:4x16x2x2 -> Mul:4x16x2x2 -> "
                    "Softmax:4x16x2x2 -> MatMul:4x16x2x8 -> Transpose:4x2x16x8"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_float64",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "input_dtypes": [np.float64, np.float64, np.float64],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:2x8x4x32 -> MatMul:2x8x4x4 -> Mul:2x8x4x4 -> "
                    "Softmax:2x8x4x4 -> MatMul:2x8x4x32 -> Transpose:2x4x8x32"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_heads1_embed4",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 1, 8, 4), (2, 1, 8, 4), (2, 1, 8, 4)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:2x8x1x4 -> MatMul:2x8x1x1 -> Mul:2x8x1x1 -> "
                    "Softmax:2x8x1x1 -> MatMul:2x8x1x4 -> Transpose:2x1x8x4"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_heads8_embed8",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 8, 8, 8), (2, 8, 8, 8), (2, 8, 8, 8)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:2x8x8x8 -> MatMul:2x8x8x8 -> Mul:2x8x8x8 -> "
                    "Softmax:2x8x8x8 -> MatMul:2x8x8x8 -> Transpose:2x8x8x8"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_batch1_seq2",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(1, 2, 2, 8), (1, 2, 2, 8), (1, 2, 2, 8)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:1x2x2x8 -> MatMul:1x2x2x2 -> Mul:1x2x2x2 -> "
                    "Softmax:1x2x2x2 -> MatMul:1x2x2x8 -> Transpose:1x2x2x8"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_batch8_seq4",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(8, 2, 4, 16), (8, 2, 4, 16), (8, 2, 4, 16)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:8x4x2x16 -> MatMul:8x4x2x2 -> Mul:8x4x2x2 -> "
                    "Softmax:8x4x2x2 -> MatMul:8x4x2x16 -> Transpose:8x2x4x16"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_axis1",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:2x8x4x32 -> MatMul:2x8x4x4 -> Mul:2x8x4x4 -> "
                    "Softmax:2x8x4x4 -> MatMul:2x8x4x32 -> Transpose:2x4x8x32"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_with_tensor_mask",
            "callable": lambda q, k, v, mask: jax_nn.dot_product_attention(
                q, k, v, mask=mask
            ),
            "input_shapes": [
                (2, 8, 4, 16),
                (2, 16, 4, 16),
                (2, 16, 4, 16),
                (2, 4, 8, 16),
            ],
            "input_dtypes": [np.float32, np.float32, np.float32, np.bool_],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": _EXPECT_DPA_MASK_WHERE,
        },
        {
            "testcase": "dpa_tiny_mask_all_valid",
            "callable": lambda q, k, v, mask: jax_nn.dot_product_attention(
                q, k, v, mask=mask
            ),
            "input_values": [
                np.arange(8, dtype=np.float32).reshape((1, 2, 1, 4)),
                np.arange(12, dtype=np.float32).reshape((1, 3, 1, 4)),
                np.arange(12, dtype=np.float32).reshape((1, 3, 1, 4)),
                np.ones((1, 1, 2, 3), dtype=bool),
            ],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Softmax -> Mul -> ReduceSum -> Greater -> Where -> Div -> Where "
                    "-> MatMul:1x1x2x4 -> Transpose:1x2x1x4"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_tiny_mask_mixed",
            "callable": lambda q, k, v, mask: jax_nn.dot_product_attention(
                q, k, v, mask=mask
            ),
            "input_values": [
                np.arange(8, dtype=np.float32).reshape((1, 2, 1, 4)),
                np.arange(12, dtype=np.float32).reshape((1, 3, 1, 4)),
                np.arange(12, dtype=np.float32).reshape((1, 3, 1, 4)),
                np.array([[[[True, False, True], [False, True, False]]]], dtype=bool),
            ],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Softmax -> Mul -> ReduceSum -> Greater -> Where -> Div -> Where "
                    "-> MatMul:1x1x2x4 -> Transpose:1x2x1x4"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_one_false",
            "callable": lambda q, k, v, mask: jax_nn.dot_product_attention(
                q, k, v, mask=mask
            ),
            "input_values": [
                np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32),
                np.array(
                    [[[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]]],
                    dtype=np.float32,
                ),
                np.array(
                    [[[[10.0, 20.0, 30.0, 40.0]], [[50.0, 60.0, 70.0, 80.0]]]],
                    dtype=np.float32,
                ),
                np.array([[[[True, False]]]], dtype=bool),
            ],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Softmax -> Mul -> ReduceSum -> Greater -> Where -> Div -> Where "
                    "-> MatMul:1x1x1x4 -> Transpose:1x1x1x4"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_mostly_false",
            "callable": lambda q, k, v, mask: jax_nn.dot_product_attention(
                q, k, v, mask=mask
            ),
            "input_values": [
                np.ones((1, 1, 1, 4), dtype=np.float32),
                np.ones((1, 2, 1, 4), dtype=np.float32),
                np.ones((1, 2, 1, 4), dtype=np.float32) * 7.0,
                np.array([[[[False, True]]]], dtype=bool),
            ],
            "expected_output_numpy": [np.zeros((1, 1, 1, 4), dtype=np.float32)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Softmax -> Mul -> ReduceSum -> Greater -> Where -> Div -> Where "
                    "-> MatMul:1x1x1x4 -> Transpose:1x1x1x4"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_with_causal_mask",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(
                q, k, v, is_causal=True
            ),
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:2x4x8x16 -> MatMul:2x4x8x8 -> Mul:2x4x8x8 -> "
                    "Where:2x4x8x8 -> Softmax:2x4x8x8 -> MatMul:2x4x8x16 -> "
                    "Transpose:2x8x4x16"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_with_padding_mask",
            "callable": lambda q, k, v, q_len, kv_len: jax_nn.dot_product_attention(
                q, k, v, query_seq_lengths=q_len, key_value_seq_lengths=kv_len
            ),
            "input_values": [
                np.linspace(-1.0, 1.0, num=2 * 8 * 4 * 16, dtype=np.float32).reshape(
                    2, 8, 4, 16
                ),
                np.linspace(-0.5, 0.5, num=2 * 8 * 4 * 16, dtype=np.float32).reshape(
                    2, 8, 4, 16
                ),
                np.linspace(0.25, 1.25, num=2 * 8 * 4 * 16, dtype=np.float32).reshape(
                    2, 8, 4, 16
                ),
                np.array([8, 4], dtype=np.int32),
                np.array([8, 7], dtype=np.int32),
            ],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EG(
                [
                    "Softmax -> Mul -> ReduceSum -> Greater -> Where -> Div -> Where "
                    "-> MatMul:2x4x8x16 -> Transpose:2x8x4x16"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_with_local_window_mask",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(
                q, k, v, local_window_size=(1, 1)
            ),
            "input_shapes": [(1, 16, 1, 4), (1, 16, 1, 4), (1, 16, 1, 4)],
            "rtol": 5e-1,
            "atol": 5e-1,
            "rtol_f64": 5e-1,
            "atol_f64": 5e-1,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:1x1x16x4 -> MatMul:1x1x16x16 -> Mul:1x1x16x16 -> "
                    "Softmax:1x1x16x16 -> MatMul:1x1x16x4 -> Transpose:1x16x1x4"
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "dpa_mask_none",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(
                q, k, v, mask=None
            ),
            "input_shapes": [
                (2, 4, 8, 32),
                (2, 4, 8, 32),
                (2, 4, 8, 32),
            ],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Transpose:2x8x4x32 -> MatMul:2x8x4x4 -> Mul:2x8x4x4 -> "
                    "Softmax:2x8x4x4 -> MatMul:2x8x4x32 -> Transpose:2x4x8x32"
                ],
                no_unused_inputs=True,
            ),
        },
    ],
)
class DotProductAttentionPlugin(PrimitiveLeafPlugin):
    """IR-only lowering of ``jax.nn.dot_product_attention``."""

    _PRIM: ClassVar[Primitive] = _DPA_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(q, k, v, *_, **__):
        return jax.core.ShapedArray(q.shape, q.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        invars = list(eqn.invars)
        out_var = eqn.outvars[0]

        has_mask = bool(eqn.params.get("has_mask", False))
        has_query_lengths = bool(eqn.params.get("has_query_lengths", False))
        has_key_value_lengths = bool(eqn.params.get("has_key_value_lengths", False))
        is_causal = bool(eqn.params.get("is_causal", False))

        idx = 0
        q_var = invars[idx]
        idx += 1
        k_var = invars[idx]
        idx += 1
        v_var = invars[idx]
        idx += 1
        mask_var = invars[idx] if has_mask else None
        if has_mask:
            idx += 1
        query_len_var = invars[idx] if has_query_lengths else None
        if has_query_lengths:
            idx += 1
        key_value_len_var = invars[idx] if has_key_value_lengths else None

        q_val = ctx.get_value_for_var(q_var, name_hint=ctx.fresh_name("dpa_q"))
        k_val = ctx.get_value_for_var(k_var, name_hint=ctx.fresh_name("dpa_k"))
        v_val = ctx.get_value_for_var(v_var, name_hint=ctx.fresh_name("dpa_v"))

        q_shape = tuple(getattr(getattr(q_var, "aval", None), "shape", ()))
        k_shape = tuple(getattr(getattr(k_var, "aval", None), "shape", ()))
        np_dtype = _numpy_dtype_from_aval(q_var)

        if len(q_shape) != 4 or len(k_shape) != 4:
            raise NotImplementedError(
                "jax.nn.dot_product_attention expects 4D query/key inputs"
            )

        batch_dim, q_len, num_heads, head_dim = q_shape
        k_len = k_shape[1]

        batch_dim_v, _ = _coerce_dim(batch_dim, "batch")
        num_heads_v, _ = _coerce_dim(num_heads, "num_heads")
        head_dim_v, head_static = _coerce_dim(head_dim, "head")
        if not head_static:
            raise NotImplementedError(
                "jax.nn.dot_product_attention requires a static head dimension"
            )
        q_len_v, _ = _coerce_dim(q_len, "query length")
        k_len_v, _ = _coerce_dim(k_len, "key length")

        head_dim_i = cast(int, head_dim_v)
        batch_dim_i: DimLike = _symbolic_or_dim("B", batch_dim_v)
        num_heads_i: DimLike = _symbolic_or_dim("N", num_heads_v)
        q_len_i: DimLike = _symbolic_or_dim("T", q_len_v)
        k_len_i: DimLike = _symbolic_or_dim("S", k_len_v)
        head_dim_sym: DimLike = _symbolic_or_dim("H", head_dim_i)

        q_t = _builder_tensor_op(
            ctx,
            "Transpose",
            [q_val],
            base="dpa_qT",
            dtype_like=q_val,
            shape=(batch_dim, num_heads, q_len, head_dim),
            attributes={"perm": (0, 2, 1, 3)},
        )
        _stamp_type_and_shape(q_t, (batch_dim_i, num_heads_i, q_len_i, head_dim_sym))

        k_t = _builder_tensor_op(
            ctx,
            "Transpose",
            [k_val],
            base="dpa_kT",
            dtype_like=k_val,
            shape=(batch_dim, num_heads, head_dim, k_len),
            attributes={"perm": (0, 2, 3, 1)},
        )
        _stamp_type_and_shape(k_t, (batch_dim_i, num_heads_i, head_dim_sym, k_len_i))

        logits = _builder_tensor_op(
            ctx,
            "MatMul",
            [q_t, k_t],
            base="dpa_logits",
            dtype_like=q_val,
            shape=(batch_dim, num_heads, q_len, k_len),
        )
        _stamp_type_and_shape(logits, (batch_dim_i, num_heads_i, q_len_i, k_len_i))

        scale = ctx.builder.add_initializer_from_scalar(
            name=ctx.fresh_name("dpa_scale"),
            value=np.asarray(1.0 / np.sqrt(float(head_dim_i)), dtype=np_dtype),
        )

        current_logits = _builder_tensor_op(
            ctx,
            "Mul",
            [logits, scale],
            base="dpa_scaled",
            dtype_like=logits,
            shape=(batch_dim, num_heads, q_len, k_len),
        )
        _stamp_type_and_shape(
            current_logits, (batch_dim_i, num_heads_i, q_len_i, k_len_i)
        )

        if is_causal:
            if not (
                isinstance(num_heads_v, (int, np.integer))
                and isinstance(q_len_v, (int, np.integer))
                and isinstance(k_len_v, (int, np.integer))
            ):
                raise NotImplementedError(
                    "jax.nn.dot_product_attention with is_causal=True "
                    "requires static num_heads/query/key dimensions"
                )
            num_heads_static = int(num_heads_v)
            q_len_static = int(q_len_v)
            k_len_static = int(k_len_v)
            tril = np.tril(np.ones((q_len_static, k_len_static), dtype=bool))
            tril_broadcast = np.broadcast_to(
                tril, (num_heads_static, q_len_static, k_len_static)
            ).reshape(1, num_heads_static, q_len_static, k_len_static)
            mask_const = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("dpa_causal_mask"), array=tril_broadcast
            )
            neg_inf = ctx.builder.add_initializer_from_scalar(
                name=ctx.fresh_name("dpa_neg_inf"),
                value=np.asarray(-np.inf, dtype=np_dtype),
            )
            causal_logits = _builder_tensor_op(
                ctx,
                "Where",
                [mask_const, current_logits, neg_inf],
                base="dpa_causal",
                dtype_like=current_logits,
                shape=(batch_dim, num_heads, q_len, k_len),
            )
            _stamp_type_and_shape(
                causal_logits, (batch_dim_i, num_heads_i, q_len_i, k_len_i)
            )
            current_logits = causal_logits

        length_mask_bool: ir.Value | None = None
        if has_query_lengths or has_key_value_lengths:
            q_shape_val = _shape_of(ctx, q_val, "dpa_q_shape")
            q_len_scalar = _gather_int_scalar(
                ctx, q_shape_val, axis=1, name_hint="dpa_q_len"
            )
            k_shape_val = _shape_of(ctx, k_val, "dpa_k_shape")
            k_len_scalar = _gather_int_scalar(
                ctx, k_shape_val, axis=1, name_hint="dpa_k_len"
            )

            q_idx_vec = _make_range_value(ctx, q_len_scalar, base="dpa_q_idx_vec")
            q_idx_shape = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("dpa_q_idx_shape"),
                array=np.asarray([1, 1, -1, 1], dtype=np.int64),
            )
            q_idx_broadcast = _builder_tensor_op(
                ctx,
                "Reshape",
                [q_idx_vec, q_idx_shape],
                base="dpa_q_idx",
                dtype=ir.DataType.INT64,
                shape=(1, 1, q_len, 1),
            )
            _stamp_type_and_shape(q_idx_broadcast, (1, 1, q_len_i, 1))

            k_idx_vec = _make_range_value(ctx, k_len_scalar, base="dpa_k_idx_vec")
            k_idx_shape = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("dpa_k_idx_shape"),
                array=np.asarray([1, 1, 1, -1], dtype=np.int64),
            )
            k_idx_broadcast = _builder_tensor_op(
                ctx,
                "Reshape",
                [k_idx_vec, k_idx_shape],
                base="dpa_k_idx",
                dtype=ir.DataType.INT64,
                shape=(1, 1, 1, k_len),
            )
            _stamp_type_and_shape(k_idx_broadcast, (1, 1, 1, k_len_i))

            q_mask: ir.Value | None = None
            if query_len_var is not None:
                query_len_val = ctx.get_value_for_var(
                    query_len_var, name_hint=ctx.fresh_name("dpa_query_len")
                )
                query_len_val = _cast_to_int64(
                    ctx, query_len_val, base="dpa_query_len_i64"
                )
                query_len_shape = ctx.builder.add_initializer_from_array(
                    name=ctx.fresh_name("dpa_query_len_shape"),
                    array=np.asarray([-1, 1, 1, 1], dtype=np.int64),
                )
                query_len_broadcast = _builder_tensor_op(
                    ctx,
                    "Reshape",
                    [query_len_val, query_len_shape],
                    base="dpa_query_len_bc",
                    dtype=ir.DataType.INT64,
                    shape=(batch_dim, 1, 1, 1),
                )
                _stamp_type_and_shape(query_len_broadcast, (batch_dim_i, 1, 1, 1))
                q_mask = _builder_tensor_op(
                    ctx,
                    "Less",
                    [q_idx_broadcast, query_len_broadcast],
                    base="dpa_query_mask",
                    dtype=ir.DataType.BOOL,
                    shape=(batch_dim, 1, q_len, 1),
                )
                _stamp_type_and_shape(q_mask, (batch_dim_i, 1, q_len_i, 1))

            k_mask: ir.Value | None = None
            if key_value_len_var is not None:
                key_len_val = ctx.get_value_for_var(
                    key_value_len_var, name_hint=ctx.fresh_name("dpa_key_len")
                )
                key_len_val = _cast_to_int64(ctx, key_len_val, base="dpa_key_len_i64")
                key_len_shape = ctx.builder.add_initializer_from_array(
                    name=ctx.fresh_name("dpa_key_len_shape"),
                    array=np.asarray([-1, 1, 1, 1], dtype=np.int64),
                )
                key_len_broadcast = _builder_tensor_op(
                    ctx,
                    "Reshape",
                    [key_len_val, key_len_shape],
                    base="dpa_key_len_bc",
                    dtype=ir.DataType.INT64,
                    shape=(batch_dim, 1, 1, 1),
                )
                _stamp_type_and_shape(key_len_broadcast, (batch_dim_i, 1, 1, 1))
                k_mask = _builder_tensor_op(
                    ctx,
                    "Less",
                    [k_idx_broadcast, key_len_broadcast],
                    base="dpa_key_mask",
                    dtype=ir.DataType.BOOL,
                    shape=(batch_dim, 1, 1, k_len),
                )
                _stamp_type_and_shape(k_mask, (batch_dim_i, 1, 1, k_len_i))

            if q_mask is not None and k_mask is not None:
                length_mask_bool = _logical_and(
                    ctx,
                    q_mask,
                    k_mask,
                    base="dpa_length_mask",
                    shape_hint=(batch_dim_i, 1, q_len_i, k_len_i),
                )
                _stamp_type_and_shape(
                    length_mask_bool, (batch_dim_i, 1, q_len_i, k_len_i)
                )
            elif q_mask is not None:
                length_mask_bool = q_mask
            elif k_mask is not None:
                length_mask_bool = k_mask

        mask_bool: ir.Value | None = None
        if has_mask and mask_var is not None:
            mask_val = ctx.get_value_for_var(
                mask_var, name_hint=ctx.fresh_name("dpa_mask")
            )
            mask_shape = tuple(getattr(getattr(mask_var, "aval", None), "shape", ()))
            mask_dims: list[DimLike] = []
            for idx, dim in enumerate(mask_shape):
                dim_val, _ = _coerce_dim(dim, f"mask_dim_{idx}")
                mask_dims.append(dim_val)
            mask_dims_tuple = tuple(mask_dims)
            symbol_bases = ("B", "N", "T", "S")
            mask_dims_sym = tuple(
                _symbolic_or_dim(
                    symbol_bases[idx] if idx < len(symbol_bases) else f"M{idx}", dim
                )
                for idx, dim in enumerate(mask_dims_tuple)
            )
            if mask_shape:
                _stamp_type_and_shape(mask_val, mask_dims_sym)
            mask_dtype = np.dtype(getattr(mask_var.aval, "dtype", np.bool_))
            if mask_dtype != np.bool_:
                mask_bool = _builder_tensor_op(
                    ctx,
                    "Cast",
                    [mask_val],
                    base="dpa_mask_bool",
                    dtype=ir.DataType.BOOL,
                    shape=mask_dims_tuple,
                    attributes={"to": int(ir.DataType.BOOL.value)},
                )
                _stamp_type_and_shape(mask_bool, mask_dims_sym)
            else:
                mask_bool = mask_val

        if length_mask_bool is not None:
            if mask_bool is None:
                mask_bool = length_mask_bool
            else:
                mask_bool = _logical_and(
                    ctx,
                    mask_bool,
                    length_mask_bool,
                    base="dpa_combined_mask",
                    shape_hint=(batch_dim_i, num_heads_i, q_len_i, k_len_i),
                )
                _stamp_type_and_shape(
                    mask_bool, (batch_dim_i, num_heads_i, q_len_i, k_len_i)
                )

        weights = _builder_tensor_op(
            ctx,
            "Softmax",
            [current_logits],
            base="dpa_weights",
            dtype_like=current_logits,
            shape=(batch_dim, num_heads, q_len, k_len),
            attributes={"axis": 3},
        )
        _stamp_type_and_shape(weights, (batch_dim_i, num_heads_i, q_len_i, k_len_i))

        if mask_bool is not None:
            weights_dtype = _dtype_enum_from_value(weights)
            mask_float = _builder_tensor_op(
                ctx,
                "Cast",
                [mask_bool],
                base="dpa_mask_float",
                dtype=weights_dtype,
                shape=(batch_dim, num_heads, q_len, k_len),
                attributes={"to": int(weights_dtype.value)},
            )
            _stamp_type_and_shape(
                mask_float, (batch_dim_i, num_heads_i, q_len_i, k_len_i)
            )

            masked_weights = _builder_tensor_op(
                ctx,
                "Mul",
                [weights, mask_float],
                base="dpa_weights_masked",
                dtype_like=weights,
                shape=(batch_dim, num_heads, q_len, k_len),
            )
            _stamp_type_and_shape(
                masked_weights, (batch_dim_i, num_heads_i, q_len_i, k_len_i)
            )

            axes_tensor = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("dpa_weights_axes"),
                array=np.asarray([3], dtype=np.int64),
            )
            sum_weights = _builder_tensor_op(
                ctx,
                "ReduceSum",
                [masked_weights, axes_tensor],
                base="dpa_weights_sum",
                dtype_like=weights,
                shape=(batch_dim, num_heads, q_len, 1),
                attributes={"keepdims": 1},
            )
            _stamp_type_and_shape(sum_weights, (batch_dim_i, num_heads_i, q_len_i, 1))

            zero_scalar = ctx.builder.add_initializer_from_scalar(
                name=ctx.fresh_name("dpa_zero"),
                value=np.asarray(0.0, dtype=np_dtype),
            )
            denom_nonzero = _builder_tensor_op(
                ctx,
                "Greater",
                [sum_weights, zero_scalar],
                base="dpa_weights_sum_nz",
                dtype=ir.DataType.BOOL,
                shape=(batch_dim, num_heads, q_len, 1),
            )
            _stamp_type_and_shape(denom_nonzero, (batch_dim_i, num_heads_i, q_len_i, 1))

            one_scalar = ctx.builder.add_initializer_from_scalar(
                name=ctx.fresh_name("dpa_one"),
                value=np.asarray(1.0, dtype=np_dtype),
            )
            safe_denom = _builder_tensor_op(
                ctx,
                "Where",
                [denom_nonzero, sum_weights, one_scalar],
                base="dpa_weights_denom",
                dtype_like=sum_weights,
                shape=(batch_dim, num_heads, q_len, 1),
            )
            _stamp_type_and_shape(safe_denom, (batch_dim_i, num_heads_i, q_len_i, 1))

            normalized_weights = _builder_tensor_op(
                ctx,
                "Div",
                [masked_weights, safe_denom],
                base="dpa_weights_norm",
                dtype_like=weights,
                shape=(batch_dim, num_heads, q_len, k_len),
            )
            _stamp_type_and_shape(
                normalized_weights, (batch_dim_i, num_heads_i, q_len_i, k_len_i)
            )

            nan_value = np.nan if np_dtype == np.float64 else 0.0
            nan_scalar = ctx.builder.add_initializer_from_scalar(
                name=ctx.fresh_name("dpa_nan"),
                value=np.asarray(nan_value, dtype=np_dtype),
            )
            weights = _builder_tensor_op(
                ctx,
                "Where",
                [denom_nonzero, normalized_weights, nan_scalar],
                base="dpa_weights_norm_nan",
                dtype_like=weights,
                shape=(batch_dim, num_heads, q_len, k_len),
            )
            _stamp_type_and_shape(weights, (batch_dim_i, num_heads_i, q_len_i, k_len_i))

        v_t = _builder_tensor_op(
            ctx,
            "Transpose",
            [v_val],
            base="dpa_vT",
            dtype_like=v_val,
            shape=(batch_dim, num_heads, k_len, head_dim),
            attributes={"perm": (0, 2, 1, 3)},
        )
        _stamp_type_and_shape(v_t, (batch_dim_i, num_heads_i, k_len_i, head_dim_sym))

        out_t = _builder_tensor_op(
            ctx,
            "MatMul",
            [weights, v_t],
            base="dpa_outT",
            dtype_like=v_val,
            shape=(batch_dim, num_heads, q_len, head_dim),
        )
        _stamp_type_and_shape(out_t, (batch_dim_i, num_heads_i, q_len_i, head_dim_sym))

        result = _builder_tensor_op(
            ctx,
            "Transpose",
            [out_t],
            base="dpa_out",
            dtype_like=out_t,
            shape=(batch_dim, q_len, num_heads, head_dim),
            attributes={"perm": (0, 2, 1, 3)},
        )
        _stamp_type_and_shape(result, (batch_dim_i, q_len_i, num_heads_i, head_dim_sym))
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_patched(
            orig_fn: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig_fn is None:
                raise RuntimeError("Original jax.nn.dot_product_attention not found")

            def _bind(q, k, v, *args, **kwargs):
                bias_arg: Optional[jax.Array] = None
                mask_arg: Optional[jax.Array] = None

                if args:
                    if len(args) >= 1:
                        bias_arg = args[0]
                    if len(args) >= 2:
                        mask_arg = args[1]

                if "bias" in kwargs:
                    bias_kw = kwargs.pop("bias")
                    bias_arg = bias_kw if bias_kw is not None else bias_arg
                if "mask" in kwargs:
                    mask_kw = kwargs.pop("mask")
                    mask_arg = mask_kw if mask_kw is not None else mask_arg

                is_causal = bool(kwargs.pop("is_causal", False))
                query_seq_lengths = kwargs.pop("query_seq_lengths", None)
                key_value_seq_lengths = kwargs.pop("key_value_seq_lengths", None)

                if isinstance(mask_arg, tuple) and len(mask_arg) == 2:
                    query_lengths, key_lengths = mask_arg
                    query_lengths = jnp.asarray(query_lengths, dtype=jnp.int32)
                    key_lengths = jnp.asarray(key_lengths, dtype=jnp.int32)
                    query_seq_lengths = query_lengths
                    key_value_seq_lengths = key_lengths
                    mask_arg = None

                if query_seq_lengths is not None or key_value_seq_lengths is not None:
                    batch_size = q.shape[0]
                    q_len = q.shape[2]
                    k_len = k.shape[2]
                    if query_seq_lengths is None:
                        query_seq_lengths = jnp.full(
                            (batch_size,), q_len, dtype=jnp.int32
                        )
                    else:
                        query_seq_lengths = jnp.asarray(
                            query_seq_lengths, dtype=jnp.int32
                        )
                    if key_value_seq_lengths is None:
                        key_value_seq_lengths = jnp.full(
                            (batch_size,), k_len, dtype=jnp.int32
                        )
                    else:
                        key_value_seq_lengths = jnp.asarray(
                            key_value_seq_lengths, dtype=jnp.int32
                        )

                unsupported = {
                    key: kwargs.get(key)
                    for key in (
                        "broadcast_dropout",
                        "dropout_rng",
                        "dropout_rate",
                        "deterministic",
                        "dtype",
                        "precision",
                    )
                    if kwargs.get(key) not in (None, False, 0.0)
                }
                if unsupported:
                    raise NotImplementedError(
                        "jax.nn.dot_product_attention converter plugin does not "
                        f"support arguments {tuple(unsupported.keys())}"
                    )

                if bias_arg is not None:
                    raise NotImplementedError(
                        "jax.nn.dot_product_attention bias argument is not yet supported"
                    )

                params = {
                    "is_causal": is_causal,
                    "has_mask": mask_arg is not None,
                    "has_query_lengths": query_seq_lengths is not None,
                    "has_key_value_lengths": key_value_seq_lengths is not None,
                }
                operands = [q, k, v]
                if mask_arg is not None:
                    operands.append(mask_arg)
                if query_seq_lengths is not None:
                    operands.append(query_seq_lengths)
                if key_value_seq_lengths is not None:
                    operands.append(key_value_seq_lengths)

                return cls._PRIM.bind(*operands, **params)

            return _bind

        return [
            AssignSpec(
                "jax.nn", "dot_product_attention_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="dot_product_attention",
                make_value=_make_patched,
                delete_if_missing=False,
            ),
        ]


@DotProductAttentionPlugin._PRIM.def_impl
def _impl(*args, **kwargs) -> ArrayLike:
    return _ORIG_DOT_PRODUCT_ATTENTION(*args, **kwargs)
