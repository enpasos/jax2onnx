# jax2onnx/plugins/flax/nnx/dot_product_attention.py

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, Tuple, Union, cast

import numpy as np
from flax import nnx
from jax import core
from jax.extend.core import Primitive
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.plugins._ir_shapes import (
    _ensure_value_info as _add_value_info,
    _stamp_type_and_shape,
    _to_ir_dim_for_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    _DynamicParamWrapper,
    register_primitive,
)

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.plugins.plugin_system import _IRBuildContext as IRContext  # type: ignore


DimLike = Union[int, str]


def _coerce_dim(dim: object, name: str) -> Tuple[DimLike, bool]:
    if isinstance(dim, (int, np.integer)):
        return int(dim), True
    # JAX symbolic dims may still encode a constant value; prefer the concrete int when available.
    if hasattr(dim, "_is_constant") and callable(getattr(dim, "_is_constant")):
        try:
            if dim._is_constant():  # type: ignore[attr-defined]
                const = dim._to_constant()  # type: ignore[attr-defined]
                if isinstance(const, (int, np.integer)):
                    return int(const), True
        except Exception:
            pass
    return str(dim), False


def _dtype_enum_from_value(val: ir.Value) -> ir.DataType:
    dtype = getattr(getattr(val, "type", None), "dtype", None)
    if dtype is None:
        raise TypeError("Missing dtype on value; ensure inputs are typed.")
    return dtype


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


DPA_PRIM = Primitive("nnx.dot_product_attention")
DPA_PRIM.multiple_results = False


EXPECT_DPA_BASE = EG(
    [
        (
            "Transpose -> MatMul -> Mul -> Softmax -> MatMul -> Transpose",
            {
                "counts": {
                    "Transpose": 4,
                    "MatMul": 2,
                    "Mul": 1,
                    "Softmax": 1,
                }
            },
        )
    ]
)


EXPECT_DPA_WITH_MASK = EG([("Where", {"counts": {"Where": 1}})])
EXPECT_DPA_WITH_BIAS = EG([("Add", {"counts": {"Add": 1}})])


@register_primitive(
    jaxpr_primitive=DPA_PRIM.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/attention.html#flax.nnx.dot_product_attention",
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
    since="v0.1.0",
    context="primitives.nnx",
    component="dot_product_attention",
    testcases=[
        {
            "testcase": "dpa_basic",
            "callable": lambda q, k, v: nnx.dot_product_attention(q, k, v),
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16)],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": EXPECT_DPA_BASE,
        },
        {
            "testcase": "dpa_with_tensor_mask",
            "callable": lambda q, k, v, mask: nnx.dot_product_attention(
                q, k, v, mask=mask
            ),
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16), (2, 4, 8, 8)],
            "input_dtypes": [np.float32, np.float32, np.float32, np.bool_],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: EXPECT_DPA_BASE(m)
            and EXPECT_DPA_WITH_MASK(m),
        },
        {
            "testcase": "dpa_with_bias",
            "callable": lambda q, k, v, bias: nnx.dot_product_attention(
                q, k, v, bias=bias
            ),
            "input_shapes": [(2, 8, 4, 16), (2, 8, 4, 16), (2, 8, 4, 16), (2, 4, 8, 8)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: EXPECT_DPA_BASE(m)
            and EXPECT_DPA_WITH_BIAS(m),
        },
        {
            "testcase": "dpa_with_causal_mask",
            "callable": lambda q, k, v, mask: nnx.dot_product_attention(
                q, k, v, mask=mask
            ),
            "input_values": [
                np.random.randn(1, 8, 4, 16).astype(np.float32),
                np.random.randn(1, 8, 4, 16).astype(np.float32),
                np.random.randn(1, 8, 4, 16).astype(np.float32),
                np.tril(np.ones((1, 4, 8, 8), dtype=bool)),
            ],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "post_check_onnx_graph": lambda m: EXPECT_DPA_BASE(m)
            and EXPECT_DPA_WITH_MASK(m),
        },
        {
            "testcase": "dpa_with_mask_and_bias",
            "callable": lambda q, k, v, mask, bias: nnx.dot_product_attention(
                q, k, v, mask=mask, bias=bias
            ),
            "input_values": [
                *(
                    lambda rng: (
                        rng.randn(2, 8, 4, 16).astype(np.float64),
                        rng.randn(2, 8, 4, 16).astype(np.float64),
                        rng.randn(2, 8, 4, 16).astype(np.float64),
                        np.where(
                            np.eye(8, dtype=bool)[None, None],
                            False,
                            True,
                        )
                        .repeat(2, axis=0)
                        .repeat(4, axis=1),
                        np.zeros((2, 4, 8, 8), dtype=np.float64),
                    )
                )(np.random.RandomState(0))
            ],
            "expected_output_shapes": [(2, 8, 4, 16)],
            "expected_output_dtypes": [np.float64],
            "rtol_f64": 1e-6,
            "atol_f64": 1e-6,
            "run_only_f64_variant": True,
            "post_check_onnx_graph": lambda m: EXPECT_DPA_BASE(m)
            and EXPECT_DPA_WITH_MASK(m)
            and EXPECT_DPA_WITH_BIAS(m),
        },
    ],
)
class DotProductAttentionPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = DPA_PRIM
    _ORIG_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(q, k, v, *_, **__):
        return core.ShapedArray(q.shape, q.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[override]
        params_raw = dict(getattr(eqn, "params", {}) or {})
        params = {
            key: (val.value if isinstance(val, _DynamicParamWrapper) else val)
            for key, val in params_raw.items()
        }
        has_mask = bool(params.get("has_mask", False))
        has_bias = bool(params.get("has_bias", False))

        invars = list(eqn.invars)
        q_var, k_var, v_var = invars[:3]
        idx = 3
        mask_var = invars[idx] if has_mask else None
        idx += 1 if has_mask else 0
        bias_var = invars[idx] if has_bias else None

        q_val = ctx.get_value_for_var(q_var, name_hint=ctx.fresh_name("dpa_q"))
        k_val = ctx.get_value_for_var(k_var, name_hint=ctx.fresh_name("dpa_k"))
        v_val = ctx.get_value_for_var(v_var, name_hint=ctx.fresh_name("dpa_v"))
        out_var = eqn.outvars[0]
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("dpa_out"))

        q_shape = tuple(getattr(getattr(q_var, "aval", None), "shape", ()))
        k_shape = tuple(getattr(getattr(k_var, "aval", None), "shape", ()))
        np_dtype = np.dtype(getattr(getattr(q_var, "aval", None), "dtype", np.float32))

        batch_dim, q_len, num_heads, head_dim = q_shape
        k_len = k_shape[1]

        batch_dim_v, _ = _coerce_dim(batch_dim, "batch")
        q_len_v, _ = _coerce_dim(q_len, "query length")
        num_heads_v, _ = _coerce_dim(num_heads, "num_heads")
        head_dim_v, head_static = _coerce_dim(head_dim, "head")
        if not head_static:
            raise NotImplementedError(
                "nnx.dot_product_attention requires static head dimension"
            )
        k_len_v, _ = _coerce_dim(k_len, "key length")

        head_dim_i = cast(int, head_dim_v)
        num_heads_i: DimLike = num_heads_v
        batch_dim_i: DimLike = batch_dim_v
        q_len_i: DimLike = q_len_v
        k_len_i: DimLike = k_len_v

        q_t = _make_tensor_value(
            ctx, q_val, (batch_dim, num_heads, q_len, head_dim), base="dpa_qT"
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
            ctx,
            q_val,
            (batch_dim, num_heads, q_len, k_len),
            base="dpa_logits",
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
            name=ctx.fresh_name("dpa_scale"),
            value=np.asarray(1.0 / np.sqrt(float(head_dim_i)), dtype=np_dtype),
        )

        scaled = _make_tensor_value(
            ctx,
            logits,
            (batch_dim, num_heads, q_len, k_len),
            base="dpa_scaled",
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

        if has_bias and bias_var is not None:
            bias_val = ctx.get_value_for_var(
                bias_var, name_hint=ctx.fresh_name("dpa_bias")
            )
            bias_val = cast_param_like(
                ctx, bias_val, current_logits, name_hint="dpa_bias_cast"
            )
            bias_dims = tuple(getattr(getattr(bias_var, "aval", None), "shape", ()))
            _stamp_type_and_shape(bias_val, bias_dims)
            biased = _make_tensor_value(
                ctx,
                current_logits,
                (batch_dim, num_heads, q_len, k_len),
                base="dpa_bias_add",
            )
            ctx.add_node(
                ir.Node(
                    op_type="Add",
                    domain="",
                    inputs=[current_logits, bias_val],
                    outputs=[biased],
                    name=ctx.fresh_name("Add"),
                )
            )
            _stamp_type_and_shape(biased, (batch_dim_i, num_heads_i, q_len_i, k_len_i))
            _add_value_info(ctx, biased)
            current_logits = biased

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
            mask_dtype = np.dtype(
                getattr(getattr(mask_var, "aval", None), "dtype", np.bool_)
            )
            if mask_dtype != np.bool_:
                mask_bool = _make_tensor_value(
                    ctx,
                    mask_val,
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
                mask_val = mask_bool
                _stamp_type_and_shape(mask_val, mask_dims_tuple)
                _add_value_info(ctx, mask_val)
            else:
                _stamp_type_and_shape(mask_val, mask_dims_tuple)
                _add_value_info(ctx, mask_val)

            fill_value = ctx.builder.add_initializer_from_scalar(
                name=ctx.fresh_name("dpa_mask_fill"),
                value=np.asarray(np.finfo(np_dtype).min, dtype=np_dtype),
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
                    inputs=[mask_val, current_logits, fill_value],
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

        call_param_values = getattr(ctx, "_call_param_value_by_name", {}) or {}
        det_input = call_param_values.get("deterministic")

        dropout_rate = float(params.get("dropout_rate", 0.0) or 0.0)
        if det_input is not None:
            training_mode = ir.Value(
                name=ctx.fresh_name("dpa_training_mode"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=ir.Shape(()),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Not",
                    domain="",
                    inputs=[det_input],
                    outputs=[training_mode],
                    name=ctx.fresh_name("Not"),
                )
            )
            ratio_val = ctx.builder.add_initializer_from_scalar(
                name=ctx.fresh_name("dpa_dropout_ratio"),
                value=np.asarray(dropout_rate, dtype=np.float32),
            )
            dropped_weights = _make_tensor_value(
                ctx,
                weights,
                (batch_dim, num_heads, q_len, k_len),
                base="dpa_dropout_weights",
            )
            ctx.add_node(
                ir.Node(
                    op_type="Dropout",
                    domain="",
                    inputs=[weights, ratio_val, training_mode],
                    outputs=[dropped_weights],
                    name=ctx.fresh_name("Dropout"),
                )
            )
            _stamp_type_and_shape(
                dropped_weights, (batch_dim_i, num_heads_i, q_len_i, k_len_i)
            )
            _add_value_info(ctx, dropped_weights)
            weights = dropped_weights

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
    def binding_specs(cls):
        def _make_patch(orig):
            cls._ORIG_CALL = orig

            def patched(q, k, v, mask=None, bias=None, **kwargs):
                operands = [q, k, v]
                has_mask = mask is not None
                has_bias = bias is not None
                if has_mask:
                    operands.append(mask)
                if has_bias:
                    operands.append(bias)
                safe_kwargs = {}
                for key, value in kwargs.items():
                    if hasattr(value, "aval"):
                        safe_kwargs[key] = _DynamicParamWrapper(value)
                    else:
                        safe_kwargs[key] = value
                return cls._PRIM.bind(
                    *operands,
                    has_mask=has_mask,
                    has_bias=has_bias,
                    **safe_kwargs,
                )

            return patched

        return [
            AssignSpec(
                "flax.nnx", "dot_product_attention_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="dot_product_attention",
                make_value=_make_patch,
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@DotProductAttentionPlugin._PRIM.def_impl
def _dpa_impl(q, k, v, *extra, **params):
    has_mask = bool(params.pop("has_mask", False))
    has_bias = bool(params.pop("has_bias", False))
    idx = 0
    mask = extra[idx] if has_mask else None
    idx += 1 if has_mask else 0
    bias = extra[idx] if has_bias else None

    if DotProductAttentionPlugin._ORIG_CALL is None:
        raise RuntimeError("Original nnx.dot_product_attention not captured")

    call_kwargs = dict(params)
    if mask is not None:
        call_kwargs["mask"] = mask
    if bias is not None:
        call_kwargs["bias"] = bias
    return DotProductAttentionPlugin._ORIG_CALL(q, k, v, **call_kwargs)


DotProductAttentionPlugin.ensure_abstract_eval_bound()
