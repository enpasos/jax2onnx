from __future__ import annotations

from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    ClassVar,
    DefaultDict,
    Dict,
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
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import (
    _ensure_value_info as _add_value_info,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2.jax.lax.scatter_utils import (
    _ensure_value_info as _ensure_ir_value_info,
    _gather_int_scalar,
    _scalar_i64,
    _shape_of,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_DPA_PRIM = Primitive("jax.nn.dot_product_attention")
_DPA_PRIM.multiple_results = False
_ORIG_DOT_PRODUCT_ATTENTION = jax_nn.dot_product_attention


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


_EXPECT_DPA_MASK_WHERE = _expect_dpa_mask_where


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


def _make_tensor_value(
    ctx: "IRContext", like: ir.Value, shape, *, base: str
) -> ir.Value:
    dtype = _dtype_enum_from_value(like)
    dims = tuple(_to_ir_dim_for_shape(d) for d in shape)
    return ir.Value(
        name=ctx.fresh_name(base),
        type=ir.TensorType(dtype),
        shape=ir.Shape(dims),
    )


def _make_bool_tensor_value(ctx: "IRContext", shape, *, base: str) -> ir.Value:
    dims = tuple(_to_ir_dim_for_shape(d) for d in shape)
    return ir.Value(
        name=ctx.fresh_name(base),
        type=ir.TensorType(ir.DataType.BOOL),
        shape=ir.Shape(dims),
    )


def _cast_to_int64(ctx: "IRContext", value: ir.Value, *, base: str) -> ir.Value:
    casted = ir.Value(
        name=ctx.fresh_name(base),
        type=ir.TensorType(ir.DataType.INT64),
        shape=value.shape,
    )
    ctx.add_node(
        ir.Node(
            op_type="Cast",
            domain="",
            inputs=[value],
            outputs=[casted],
            name=ctx.fresh_name("Cast"),
            attributes=[IRAttr("to", IRAttrType.INT, int(ir.DataType.INT64.value))],
        )
    )
    _ensure_ir_value_info(ctx, casted)
    return casted


def _make_range_value(ctx: "IRContext", limit: ir.Value, *, base: str) -> ir.Value:
    start = _scalar_i64(ctx, 0, f"{base}_start")
    step = _scalar_i64(ctx, 1, f"{base}_step")
    rng = ir.Value(
        name=ctx.fresh_name(base),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape((None,)),
    )
    ctx.add_node(
        ir.Node(
            op_type="Range",
            domain="",
            inputs=[start, limit, step],
            outputs=[rng],
            name=ctx.fresh_name("Range"),
        )
    )
    _ensure_ir_value_info(ctx, rng)
    return rng


def _logical_and(
    ctx: "IRContext", lhs: ir.Value, rhs: ir.Value, *, base: str, shape_hint
) -> ir.Value:
    out = ir.Value(
        name=ctx.fresh_name(base),
        type=ir.TensorType(ir.DataType.BOOL),
        shape=ir.Shape(tuple(_to_ir_dim_for_shape(d) for d in shape_hint)),
    )
    ctx.add_node(
        ir.Node(
            op_type="And",
            domain="",
            inputs=[lhs, rhs],
            outputs=[out],
            name=ctx.fresh_name("And"),
        )
    )
    _ensure_ir_value_info(ctx, out)
    return out


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
    context="primitives2.nn",
    component="dot_product_attention",
    testcases=[
        {
            "testcase": "dpa_basic",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
        },
        {
            "testcase": "dpa_positional_bias_mask",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(
                q, k, v, None, None
            ),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
        },
        {
            "testcase": "dpa_diff_heads_embed",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(1, 2, 4, 16), (1, 2, 4, 16), (1, 2, 4, 16)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
        },
        {
            "testcase": "dpa_batch4_seq16",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(4, 2, 16, 8), (4, 2, 16, 8), (4, 2, 16, 8)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
        },
        {
            "testcase": "dpa_float64",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "input_dtypes": [np.float64, np.float64, np.float64],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "run_only_f64_variant": True,
        },
        {
            "testcase": "dpa_heads1_embed4",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 1, 8, 4), (2, 1, 8, 4), (2, 1, 8, 4)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
        },
        {
            "testcase": "dpa_heads8_embed8",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 8, 8, 8), (2, 8, 8, 8), (2, 8, 8, 8)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
        },
        {
            "testcase": "dpa_batch1_seq2",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(1, 2, 2, 8), (1, 2, 2, 8), (1, 2, 2, 8)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
        },
        {
            "testcase": "dpa_batch8_seq4",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(8, 2, 4, 16), (8, 2, 4, 16), (8, 2, 4, 16)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
        },
        {
            "testcase": "dpa_axis1",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
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
        },
        {
            "testcase": "dpa_with_causal_mask",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(
                q, k, v, is_causal=True
            ),
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
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
        },
        {
            "testcase": "dpa_with_local_window_mask",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(
                q, k, v, local_window_size=(1, 1)
            ),
            "input_shapes": [(1, 16, 1, 4), (1, 16, 1, 4), (1, 16, 1, 4)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
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

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
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
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("dpa_out"))

        q_shape = tuple(getattr(getattr(q_var, "aval", None), "shape", ()))
        k_shape = tuple(getattr(getattr(k_var, "aval", None), "shape", ()))
        np_dtype = _numpy_dtype_from_aval(q_var)

        batch_dim, q_len, num_heads, head_dim = q_shape
        k_len = k_shape[1]

        batch_dim_v, _ = _coerce_dim(batch_dim, "batch")
        num_heads_v, _ = _coerce_dim(num_heads, "num_heads")
        head_dim_v, head_static = _coerce_dim(head_dim, "head")
        if not head_static:
            raise NotImplementedError(
                "jax.nn.dot_product_attention requires static head dimension"
            )
        q_len_v, _ = _coerce_dim(q_len, "query length")
        k_len_v, _ = _coerce_dim(k_len, "key length")

        head_dim_i = cast(int, head_dim_v)
        batch_dim_i: DimLike = _symbolic_or_dim("B", batch_dim_v)
        num_heads_i: DimLike = _symbolic_or_dim("N", num_heads_v)
        q_len_i: DimLike = _symbolic_or_dim("T", q_len_v)
        k_len_i: DimLike = _symbolic_or_dim("S", k_len_v)
        head_dim_sym: DimLike = _symbolic_or_dim("H", head_dim_i)

        q_t = _make_tensor_value(
            ctx,
            q_val,
            (batch_dim, q_len, num_heads, head_dim),
            base="dpa_qT",
        )
        ctx.add_node(
            ir.Node(
                op_type="Transpose",
                domain="",
                inputs=[q_val],
                outputs=[q_t],
                name=ctx.fresh_name("Transpose"),
                attributes=[IRAttr("perm", IRAttrType.INTS, [0, 2, 1, 3])],
            )
        )
        _stamp_type_and_shape(q_t, (batch_dim_i, num_heads_i, q_len_i, head_dim_sym))
        _add_value_info(ctx, q_t)

        k_t = _make_tensor_value(
            ctx, k_val, (batch_dim, k_len, num_heads, head_dim), base="dpa_kT"
        )
        ctx.add_node(
            ir.Node(
                op_type="Transpose",
                domain="",
                inputs=[k_val],
                outputs=[k_t],
                name=ctx.fresh_name("Transpose"),
                attributes=[IRAttr("perm", IRAttrType.INTS, [0, 2, 3, 1])],
            )
        )
        _stamp_type_and_shape(k_t, (batch_dim_i, num_heads_i, head_dim_sym, k_len_i))
        _add_value_info(ctx, k_t)

        logits = _make_tensor_value(
            ctx, q_val, (batch_dim, num_heads, q_len, k_len), base="dpa_logits"
        )
        ctx.add_node(
            ir.Node(
                op_type="MatMul",
                domain="",
                inputs=[q_t, k_t],
                outputs=[logits],
                name=ctx.fresh_name("MatMul"),
            )
        )
        _stamp_type_and_shape(logits, (batch_dim_i, num_heads_i, q_len_i, k_len_i))
        _add_value_info(ctx, logits)

        scale = ctx.builder.add_initializer_from_scalar(
            ctx.fresh_name("dpa_scale"),
            np.asarray(1.0 / np.sqrt(float(head_dim_i)), dtype=np_dtype),
        )

        scaled = _make_tensor_value(
            ctx, logits, (batch_dim, num_heads, q_len, k_len), base="dpa_scaled"
        )
        ctx.add_node(
            ir.Node(
                op_type="Mul",
                domain="",
                inputs=[logits, scale],
                outputs=[scaled],
                name=ctx.fresh_name("Mul"),
            )
        )
        _stamp_type_and_shape(scaled, (batch_dim_i, num_heads_i, q_len_i, k_len_i))
        _add_value_info(ctx, scaled)

        current_logits: ir.Value = scaled

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
                ctx.fresh_name("dpa_causal_mask"), tril_broadcast
            )
            neg_inf = ctx.builder.add_initializer_from_scalar(
                ctx.fresh_name("dpa_neg_inf"),
                np.asarray(-np.inf, dtype=np_dtype),
            )
            masked = _make_tensor_value(
                ctx,
                current_logits,
                (batch_dim, num_heads, q_len, k_len),
                base="dpa_causal",
            )
            ctx.add_node(
                ir.Node(
                    op_type="Where",
                    domain="",
                    inputs=[mask_const, current_logits, neg_inf],
                    outputs=[masked],
                    name=ctx.fresh_name("Where"),
                )
            )
            _stamp_type_and_shape(masked, (batch_dim_i, num_heads_i, q_len_i, k_len_i))
            _add_value_info(ctx, masked)
            current_logits = masked

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
                ctx.fresh_name("dpa_q_idx_shape"),
                np.asarray([1, 1, -1, 1], dtype=np.int64),
            )
            q_idx_broadcast = ir.Value(
                name=ctx.fresh_name("dpa_q_idx"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((None,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Reshape",
                    domain="",
                    inputs=[q_idx_vec, q_idx_shape],
                    outputs=[q_idx_broadcast],
                    name=ctx.fresh_name("Reshape"),
                )
            )
            _ensure_ir_value_info(ctx, q_idx_broadcast)
            _stamp_type_and_shape(q_idx_broadcast, (1, 1, q_len_i, 1))

            k_idx_vec = _make_range_value(ctx, k_len_scalar, base="dpa_k_idx_vec")
            k_idx_shape = ctx.builder.add_initializer_from_array(
                ctx.fresh_name("dpa_k_idx_shape"),
                np.asarray([1, 1, 1, -1], dtype=np.int64),
            )
            k_idx_broadcast = ir.Value(
                name=ctx.fresh_name("dpa_k_idx"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((None,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Reshape",
                    domain="",
                    inputs=[k_idx_vec, k_idx_shape],
                    outputs=[k_idx_broadcast],
                    name=ctx.fresh_name("Reshape"),
                )
            )
            _ensure_ir_value_info(ctx, k_idx_broadcast)
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
                    ctx.fresh_name("dpa_query_len_shape"),
                    np.asarray([-1, 1, 1, 1], dtype=np.int64),
                )
                query_len_broadcast = ir.Value(
                    name=ctx.fresh_name("dpa_query_len_bc"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((None,)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Reshape",
                        domain="",
                        inputs=[query_len_val, query_len_shape],
                        outputs=[query_len_broadcast],
                        name=ctx.fresh_name("Reshape"),
                    )
                )
                _ensure_ir_value_info(ctx, query_len_broadcast)
                q_mask = _make_bool_tensor_value(
                    ctx, (batch_dim, 1, q_len, 1), base="dpa_query_mask"
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Less",
                        domain="",
                        inputs=[q_idx_broadcast, query_len_broadcast],
                        outputs=[q_mask],
                        name=ctx.fresh_name("Less"),
                    )
                )
                _stamp_type_and_shape(q_mask, (batch_dim_i, 1, q_len_i, 1))
                _add_value_info(ctx, q_mask)

            k_mask: ir.Value | None = None
            if key_value_len_var is not None:
                key_len_val = ctx.get_value_for_var(
                    key_value_len_var, name_hint=ctx.fresh_name("dpa_key_len")
                )
                key_len_val = _cast_to_int64(ctx, key_len_val, base="dpa_key_len_i64")
                key_len_shape = ctx.builder.add_initializer_from_array(
                    ctx.fresh_name("dpa_key_len_shape"),
                    np.asarray([-1, 1, 1, 1], dtype=np.int64),
                )
                key_len_broadcast = ir.Value(
                    name=ctx.fresh_name("dpa_key_len_bc"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((None,)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Reshape",
                        domain="",
                        inputs=[key_len_val, key_len_shape],
                        outputs=[key_len_broadcast],
                        name=ctx.fresh_name("Reshape"),
                    )
                )
                _ensure_ir_value_info(ctx, key_len_broadcast)
                k_mask = _make_bool_tensor_value(
                    ctx, (batch_dim, 1, 1, k_len), base="dpa_key_mask"
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Less",
                        domain="",
                        inputs=[k_idx_broadcast, key_len_broadcast],
                        outputs=[k_mask],
                        name=ctx.fresh_name("Less"),
                    )
                )
                _stamp_type_and_shape(k_mask, (batch_dim_i, 1, 1, k_len_i))
                _add_value_info(ctx, k_mask)

            if q_mask is not None and k_mask is not None:
                length_mask_bool = _logical_and(
                    ctx,
                    q_mask,
                    k_mask,
                    base="dpa_length_mask",
                    shape_hint=(batch_dim_i, 1, q_len_i, k_len_i),
                )
            elif q_mask is not None:
                length_mask_bool = q_mask
            elif k_mask is not None:
                length_mask_bool = k_mask

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
            if mask_shape:
                mask_dims_sym = tuple(
                    _symbolic_or_dim(sym, dim)
                    for sym, dim in zip(("B", "N", "T", "S"), mask_dims_tuple)
                )
                _stamp_type_and_shape(mask_val, mask_dims_sym)
                _add_value_info(ctx, mask_val)
            mask_dtype = np.dtype(getattr(mask_var.aval, "dtype", np.bool_))
            if mask_dtype != np.bool_:
                mask_bool = _make_bool_tensor_value(
                    ctx,
                    mask_dims_tuple,
                    base="dpa_mask_bool",
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Cast",
                        domain="",
                        inputs=[mask_val],
                        outputs=[mask_bool],
                        name=ctx.fresh_name("Cast"),
                        attributes=[
                            IRAttr("to", IRAttrType.INT, int(ir.DataType.BOOL.value))
                        ],
                    )
                )
                _stamp_type_and_shape(mask_bool, mask_dims_sym)
                _add_value_info(ctx, mask_bool)
            else:
                mask_bool = mask_val
        else:
            mask_bool = None

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
                _add_value_info(ctx, mask_bool)

        weights = _make_tensor_value(
            ctx,
            current_logits,
            (batch_dim, num_heads, q_len, k_len),
            base="dpa_weights",
        )
        ctx.add_node(
            ir.Node(
                op_type="Softmax",
                domain="",
                inputs=[current_logits],
                outputs=[weights],
                name=ctx.fresh_name("Softmax"),
                attributes=[IRAttr("axis", IRAttrType.INT, 3)],
            )
        )
        _stamp_type_and_shape(weights, (batch_dim_i, num_heads_i, q_len_i, k_len_i))
        _add_value_info(ctx, weights)

        if mask_bool is not None:
            mask_float = _make_tensor_value(
                ctx,
                weights,
                (batch_dim, num_heads, q_len, k_len),
                base="dpa_mask_float",
            )
            ctx.add_node(
                ir.Node(
                    op_type="Cast",
                    domain="",
                    inputs=[mask_bool],
                    outputs=[mask_float],
                    name=ctx.fresh_name("Cast"),
                    attributes=[
                        IRAttr(
                            "to",
                            IRAttrType.INT,
                            int(_dtype_enum_from_value(weights).value),
                        )
                    ],
                )
            )
            _stamp_type_and_shape(
                mask_float, (batch_dim_i, num_heads_i, q_len_i, k_len_i)
            )
            _add_value_info(ctx, mask_float)

            masked_weights = _make_tensor_value(
                ctx,
                weights,
                (batch_dim, num_heads, q_len, k_len),
                base="dpa_weights_masked",
            )
            ctx.add_node(
                ir.Node(
                    op_type="Mul",
                    domain="",
                    inputs=[weights, mask_float],
                    outputs=[masked_weights],
                    name=ctx.fresh_name("Mul"),
                )
            )
            _stamp_type_and_shape(
                masked_weights, (batch_dim_i, num_heads_i, q_len_i, k_len_i)
            )
            _add_value_info(ctx, masked_weights)

            sum_weights = _make_tensor_value(
                ctx,
                weights,
                (batch_dim, num_heads, q_len, 1),
                base="dpa_weights_sum",
            )
            axes_tensor = ctx.builder.add_initializer_from_array(
                ctx.fresh_name("dpa_weights_axes"),
                np.asarray([3], dtype=np.int64),
            )
            ctx.add_node(
                ir.Node(
                    op_type="ReduceSum",
                    domain="",
                    inputs=[masked_weights, axes_tensor],
                    outputs=[sum_weights],
                    name=ctx.fresh_name("ReduceSum"),
                    attributes=[
                        IRAttr("keepdims", IRAttrType.INT, 1),
                    ],
                )
            )
            _stamp_type_and_shape(sum_weights, (batch_dim_i, num_heads_i, q_len_i, 1))
            _add_value_info(ctx, sum_weights)

            zero_scalar = ctx.builder.add_initializer_from_scalar(
                ctx.fresh_name("dpa_zero"),
                np.asarray(0.0, dtype=np_dtype),
            )
            denom_nonzero = ir.Value(
                name=ctx.fresh_name("dpa_weights_sum_nz"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=ir.Shape(
                    (
                        _to_ir_dim_for_shape(batch_dim),
                        _to_ir_dim_for_shape(num_heads),
                        _to_ir_dim_for_shape(q_len),
                        _to_ir_dim_for_shape(1),
                    )
                ),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Greater",
                    domain="",
                    inputs=[sum_weights, zero_scalar],
                    outputs=[denom_nonzero],
                    name=ctx.fresh_name("Greater"),
                )
            )
            _add_value_info(ctx, denom_nonzero)

            one_scalar = ctx.builder.add_initializer_from_scalar(
                ctx.fresh_name("dpa_one"),
                np.asarray(1.0, dtype=np_dtype),
            )
            safe_denom = _make_tensor_value(
                ctx,
                sum_weights,
                (batch_dim, num_heads, q_len, 1),
                base="dpa_weights_denom",
            )
            ctx.add_node(
                ir.Node(
                    op_type="Where",
                    domain="",
                    inputs=[denom_nonzero, sum_weights, one_scalar],
                    outputs=[safe_denom],
                    name=ctx.fresh_name("Where"),
                )
            )
            _stamp_type_and_shape(safe_denom, (batch_dim_i, num_heads_i, q_len_i, 1))
            _add_value_info(ctx, safe_denom)

            normalized_weights = _make_tensor_value(
                ctx,
                weights,
                (batch_dim, num_heads, q_len, k_len),
                base="dpa_weights_norm",
            )
            ctx.add_node(
                ir.Node(
                    op_type="Div",
                    domain="",
                    inputs=[masked_weights, safe_denom],
                    outputs=[normalized_weights],
                    name=ctx.fresh_name("Div"),
                )
            )
            _stamp_type_and_shape(
                normalized_weights, (batch_dim_i, num_heads_i, q_len_i, k_len_i)
            )
            _add_value_info(ctx, normalized_weights)

            nan_value = np.nan if np_dtype == np.float64 else 0.0
            nan_scalar = ctx.builder.add_initializer_from_scalar(
                ctx.fresh_name("dpa_nan"),
                np.asarray(nan_value, dtype=np_dtype),
            )
            weights = _make_tensor_value(
                ctx,
                normalized_weights,
                (batch_dim, num_heads, q_len, k_len),
                base="dpa_weights_norm_nan",
            )
            ctx.add_node(
                ir.Node(
                    op_type="Where",
                    domain="",
                    inputs=[denom_nonzero, normalized_weights, nan_scalar],
                    outputs=[weights],
                    name=ctx.fresh_name("Where"),
                )
            )
            _stamp_type_and_shape(weights, (batch_dim_i, num_heads_i, q_len_i, k_len_i))
            _add_value_info(ctx, weights)
        # if no additional mask, retain softmax weights as-is

        v_t = _make_tensor_value(
            ctx, v_val, (batch_dim, num_heads, k_len, head_dim), base="dpa_vT"
        )
        ctx.add_node(
            ir.Node(
                op_type="Transpose",
                domain="",
                inputs=[v_val],
                outputs=[v_t],
                name=ctx.fresh_name("Transpose"),
                attributes=[IRAttr("perm", IRAttrType.INTS, [0, 2, 1, 3])],
            )
        )
        _stamp_type_and_shape(v_t, (batch_dim_i, num_heads_i, k_len_i, head_dim_sym))
        _add_value_info(ctx, v_t)

        out_t = _make_tensor_value(
            ctx, v_val, (batch_dim, num_heads, q_len, head_dim), base="dpa_outT"
        )
        ctx.add_node(
            ir.Node(
                op_type="MatMul",
                domain="",
                inputs=[weights, v_t],
                outputs=[out_t],
                name=ctx.fresh_name("MatMul"),
            )
        )
        _stamp_type_and_shape(out_t, (batch_dim_i, num_heads_i, q_len_i, head_dim_sym))
        _add_value_info(ctx, out_t)

        ctx.add_node(
            ir.Node(
                op_type="Transpose",
                domain="",
                inputs=[out_t],
                outputs=[out_val],
                name=ctx.fresh_name("Transpose"),
                attributes=[IRAttr("perm", IRAttrType.INTS, [0, 2, 1, 3])],
            )
        )
        _stamp_type_and_shape(
            out_val, (batch_dim_i, q_len_i, num_heads_i, head_dim_sym)
        )
        _add_value_info(ctx, out_val)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls):
        def _make_patched(orig_fn):
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
                        "jax.nn.dot_product_attention converter2 plugin does not "
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
def _impl(*args, **kwargs):
    return _ORIG_DOT_PRODUCT_ATTENTION(*args, **kwargs)
