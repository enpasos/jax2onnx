from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional, Tuple, Union, cast

import numpy as np
import jax
from jax import nn as jax_nn
from jax.extend.core import Primitive
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import (
    _ensure_value_info as _add_value_info,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins2._post_check_onnx_graph import expect_graph as EG2
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_DPA_PRIM = Primitive("jax.nn.dot_product_attention")
_DPA_PRIM.multiple_results = False

_EXPECT_DPA_MASK_WHERE = EG2(
    ["MatMul -> Mul -> Where -> Softmax -> MatMul"],
    symbols={"B": None, "N": None, "T": None, "S": None, "H": None},
)


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
            "use_onnx_ir": True,
        },
        {
            "testcase": "dpa_positional_bias_mask",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(
                q, k, v, None, None
            ),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "use_onnx_ir": True,
        },
        {
            "testcase": "dpa_diff_heads_embed",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(1, 2, 4, 16), (1, 2, 4, 16), (1, 2, 4, 16)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "use_onnx_ir": True,
        },
        {
            "testcase": "dpa_batch4_seq16",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(4, 2, 16, 8), (4, 2, 16, 8), (4, 2, 16, 8)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "use_onnx_ir": True,
        },
        {
            "testcase": "dpa_float64",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "input_dtypes": [np.float64, np.float64, np.float64],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "run_only_f64_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "dpa_heads1_embed4",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 1, 8, 4), (2, 1, 8, 4), (2, 1, 8, 4)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "use_onnx_ir": True,
        },
        {
            "testcase": "dpa_heads8_embed8",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 8, 8, 8), (2, 8, 8, 8), (2, 8, 8, 8)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "use_onnx_ir": True,
        },
        {
            "testcase": "dpa_batch1_seq2",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(1, 2, 2, 8), (1, 2, 2, 8), (1, 2, 2, 8)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "use_onnx_ir": True,
        },
        {
            "testcase": "dpa_batch8_seq4",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(8, 2, 4, 16), (8, 2, 4, 16), (8, 2, 4, 16)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "use_onnx_ir": True,
        },
        {
            "testcase": "dpa_axis1",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "use_onnx_ir": True,
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
            "use_onnx_ir": True,
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
            "use_onnx_ir": True,
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
            "use_onnx_ir": True,
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
            "use_onnx_ir": True,
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
            "use_onnx_ir": True,
        },
        {
            "testcase": "dpa_with_causal_mask",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(
                q, k, v, is_causal=True
            ),
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "use_onnx_ir": True,
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
            "use_onnx_ir": True,
        },
        {
            "testcase": "dpa_with_local_window_mask",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(
                q, k, v, local_window_size=(1, 1)
            ),
            "input_shapes": [(1, 16, 1, 4), (1, 16, 1, 4), (1, 16, 1, 4)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "use_onnx_ir": True,
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
            "use_onnx_ir": True,
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
        is_causal = bool(eqn.params.get("is_causal", False))

        q_var = invars[0]
        k_var = invars[1]
        v_var = invars[2]
        mask_var = invars[3] if has_mask else None

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
            (batch_dim, num_heads, q_len, head_dim),
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
            ctx, k_val, (batch_dim, num_heads, head_dim, k_len), base="dpa_kT"
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

            fill_value = ctx.builder.add_initializer_from_scalar(
                ctx.fresh_name("dpa_mask_fill"),
                np.asarray(np.finfo(np_dtype).min, dtype=np_dtype),
            )
            masked_logits = _make_tensor_value(
                ctx,
                current_logits,
                (batch_dim, num_heads, q_len, k_len),
                base="dpa_masked",
            )
            ctx.add_node(
                ir.Node(
                    op_type="Where",
                    domain="",
                    inputs=[mask_bool, current_logits, fill_value],
                    outputs=[masked_logits],
                    name=ctx.fresh_name("Where"),
                )
            )
            _stamp_type_and_shape(
                masked_logits, (batch_dim_i, num_heads_i, q_len_i, k_len_i)
            )
            _add_value_info(ctx, masked_logits)
            current_logits = masked_logits

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

                params = {"is_causal": is_causal}
                operands = [q, k, v]
                if mask_arg is not None:
                    operands.append(mask_arg)
                    params["has_mask"] = True

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
    return jax_nn.dot_product_attention(*args, **kwargs)
