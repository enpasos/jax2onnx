from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np
import jax
from jax import nn as jax_nn
from jax.extend.core import Primitive
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins2._ir_shapes import (
    _ensure_value_info as _add_value_info,
    _stamp_type_and_shape,
)
from jax2onnx.plugins2._post_check_onnx_graph import expect_graph as EG2
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter2.ir_context import IRContext


_DPA_PRIM = Primitive("jax.nn.dot_product_attention")
_DPA_PRIM.multiple_results = False

_EXPECT_DPA_MASK_WHERE = EG2(
    [
        "MatMul:BxNxTxS -> Mul:BxNxTxS -> Where:BxNxTxS -> Softmax:BxNxTxS -> MatMul:BxNxTxH"
    ],
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


def _require_static_dim(dim, name: str) -> int:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    raise NotImplementedError(
        f"jax.nn.dot_product_attention converter2 requires static {name} dimension;"
        " symbolic dimensions are not yet supported."
    )



def _make_tensor_value(
    ctx: "IRContext", like: ir.Value, shape, *, base: str
) -> ir.Value:
    dtype = _dtype_enum_from_value(like)
    return ir.Value(
        name=ctx.fresh_name(base),
        type=ir.TensorType(dtype),
        shape=ir.Shape(tuple(shape)),
    )


def _make_bool_tensor_value(ctx: "IRContext", shape, *, base: str) -> ir.Value:
    return ir.Value(
        name=ctx.fresh_name(base),
        type=ir.TensorType(ir.DataType.BOOL),
        shape=ir.Shape(tuple(shape)),
    )


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
            "testcase": "jaxnn_dpa_basic",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(q, k, v),
            "input_shapes": [(2, 4, 8, 32), (2, 4, 8, 32), (2, 4, 8, 32)],
            "rtol_f64": 5e-7,
            "atol_f64": 5e-7,
            "use_onnx_ir": True,
        },
        {
            "testcase": "jaxnn_dpa_mask",
            "callable": lambda q, k, v, mask: jax_nn.dot_product_attention(
                q, k, v, mask=mask
            ),
            "input_values": [
                np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4),
                np.linspace(0.5, 1.5, num=1 * 2 * 3 * 4, dtype=np.float32).reshape(
                    1, 2, 3, 4
                ),
                np.linspace(1.5, 2.5, num=1 * 2 * 3 * 4, dtype=np.float32).reshape(
                    1, 2, 3, 4
                ),
                np.array(
                    [
                        [
                            [True, False],
                            [False, True],
                        ],
                        [
                            [True, True],
                            [True, True],
                        ],
                        [
                            [True, False],
                            [True, True],
                        ],
                    ],
                    dtype=bool,
                ).reshape(1, 3, 2, 2),
            ],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": _EXPECT_DPA_MASK_WHERE,
        },
        {
            "testcase": "jaxnn_dpa_causal",
            "callable": lambda q, k, v: jax_nn.dot_product_attention(
                q, k, v, is_causal=True
            ),
            "input_shapes": [(1, 3, 2, 5), (1, 3, 2, 5), (1, 3, 2, 5)],
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

        batch_dim_i = _require_static_dim(batch_dim, "batch")
        num_heads_i = _require_static_dim(num_heads, "num_heads")
        head_dim_i = _require_static_dim(head_dim, "head")
        q_len_i = _require_static_dim(q_len, "query length")
        k_len_i = _require_static_dim(k_len, "key length")

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
        _stamp_type_and_shape(q_t, (batch_dim_i, num_heads_i, q_len_i, head_dim_i))
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
        _stamp_type_and_shape(k_t, (batch_dim_i, num_heads_i, head_dim_i, k_len_i))
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
            tril = np.tril(np.ones((q_len_i, k_len_i), dtype=bool))
            tril_broadcast = np.broadcast_to(
                tril, (num_heads_i, q_len_i, k_len_i)
            ).reshape(1, num_heads_i, q_len_i, k_len_i)
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
            mask_shape_ints = tuple(
                _require_static_dim(dim, f"mask_dim_{idx}")
                for idx, dim in enumerate(mask_shape)
            )
            if mask_shape:
                _stamp_type_and_shape(mask_val, mask_shape_ints)
                _add_value_info(ctx, mask_val)
            mask_dtype = np.dtype(getattr(mask_var.aval, "dtype", np.bool_))
            if mask_dtype != np.bool_:
                mask_bool = _make_bool_tensor_value(
                    ctx,
                    getattr(mask_val.shape, "dims", mask_val.shape),
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
                _stamp_type_and_shape(mask_bool, mask_shape_ints)
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
        _stamp_type_and_shape(v_t, (batch_dim_i, num_heads_i, k_len_i, head_dim_i))
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
        _stamp_type_and_shape(out_t, (batch_dim_i, num_heads_i, q_len_i, head_dim_i))
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
        _stamp_type_and_shape(out_val, (batch_dim_i, q_len_i, num_heads_i, head_dim_i))
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
