# jax2onnx/plugins/equinox/eqx/nn/linear.py

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax.extend.core import Primitive
from jax.core import ShapedArray
from jax.interpreters import batching

from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _ensure_value_info,
    _is_static_int,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG2
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_has_gemm = EG2(["Gemm"], mode="any", search_functions=False)


def _const_i64(ctx, data, *, name: str) -> ir.Value:
    arr = np.asarray(data, dtype=np.int64)
    if hasattr(ir, "tensor"):
        tensor_obj = ir.tensor(arr)
    else:
        tensor_obj = arr
    value = ir.Value(
        name=ctx.fresh_name(name),
        type=ir.TensorType(ir.DataType.INT64),
        shape=ir.Shape(arr.shape if arr.shape else ()),
        const_value=tensor_obj if hasattr(ir.Value, "const_value") else None,
    )
    try:
        ctx._initializers.append(value)
    except Exception:
        attrs = []
        Attr = getattr(ir, "Attr", None)
        AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
        if Attr is not None:
            try:
                if hasattr(Attr, "t"):
                    attrs = [Attr.t("value", tensor_obj)]
                elif AttrType is not None:
                    attrs = [Attr("value", AttrType.TENSOR, tensor_obj)]
                else:
                    attrs = [Attr("value", tensor_obj)]
            except Exception:
                attrs = []
        node = ir.Node(
            op_type="Constant",
            domain="",
            inputs=[],
            outputs=[value],
            name=ctx.fresh_name("Constant"),
            attributes=attrs,
            num_outputs=1,
        )
        ctx.add_node(node)
    return value


_eqx_linear_symbolic = eqx.nn.Linear(128, 64, key=jax.random.PRNGKey(0))
_eqx_linear_highrank = eqx.nn.Linear(128, 64, key=jax.random.PRNGKey(42))
_eqx_linear_no_bias = eqx.nn.Linear(128, 64, use_bias=False, key=jax.random.PRNGKey(7))


@register_primitive(
    jaxpr_primitive="eqx.nn.linear",
    jax_doc="https://docs.kidger.site/equinox/api/eqx/nn/linear/",
    onnx=[
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
    ],
    since="v0.8.0",
    context="primitives.eqx",
    component="linear",
    testcases=[
        {
            "testcase": "eqx_linear_symbolic_batch",
            "callable": lambda x, _mod=_eqx_linear_symbolic: jax.vmap(_mod)(x),
            "input_shapes": [("B", 128)],
            "post_check_onnx_graph": _has_gemm,
        },
        {
            "testcase": "eqx_linear_no_bias_symbolic_batch",
            "callable": lambda x, _mod=_eqx_linear_no_bias: jax.vmap(_mod)(x),
            "input_shapes": [("B", 128)],
            "post_check_onnx_graph": _has_gemm,
        },
        {
            "testcase": "eqx_linear_no_bias_vector",
            "callable": _eqx_linear_no_bias,
            "input_shapes": [(128,)],
            "post_check_onnx_graph": _has_gemm,
        },
        {
            "testcase": "eqx_linear_high_rank",
            "callable": lambda x, _mod=_eqx_linear_highrank: jax.vmap(jax.vmap(_mod))(
                x
            ),
            "input_shapes": [(32, 10, 128)],
            "post_check_onnx_graph": _has_gemm,
        },
        {
            "testcase": "eqx_linear_vector",
            "callable": _eqx_linear_symbolic,
            "input_shapes": [(128,)],
            "post_check_onnx_graph": _has_gemm,
        },
    ],
)
class LinearPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.linear")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, weight, bias):
        del bias
        out_features = weight.shape[0]
        if x.ndim <= 1:
            out_shape = (out_features,)
        else:
            out_shape = (*x.shape[:-1], out_features)
        return ShapedArray(out_shape, x.dtype)

    def lower(self, ctx, eqn):
        x_var, weight_var, bias_var = eqn.invars
        out_var = eqn.outvars[0]

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("linear_x"))
        w_val = ctx.get_value_for_var(weight_var, name_hint=ctx.fresh_name("linear_w"))
        b_val = ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("linear_b"))

        w_val = cast_param_like(ctx, w_val, x_val, name_hint="linear_w_cast")
        b_val = cast_param_like(ctx, b_val, x_val, name_hint="linear_b_cast")

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        weight_shape = tuple(getattr(getattr(weight_var, "aval", None), "shape", ()))
        in_features = int(weight_shape[1]) if len(weight_shape) > 1 else 0
        out_features = int(weight_shape[0]) if weight_shape else 0

        need_flatten = len(x_shape) != 2
        if need_flatten:
            reshape_shape = _const_i64(ctx, [-1, in_features], name="linear_in_shape")
            flat_val = ir.Value(
                name=ctx.fresh_name("linear_flat"),
                type=x_val.type,
                shape=ir.Shape((None, in_features)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Reshape",
                    domain="",
                    inputs=[x_val, reshape_shape],
                    outputs=[flat_val],
                    name=ctx.fresh_name("Reshape"),
                )
            )
            gemm_input = flat_val
        else:
            gemm_input = x_val

        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("linear_out"))
        if need_flatten:
            gemm_out = ir.Value(
                name=ctx.fresh_name("linear_gemm"),
                type=x_val.type,
                shape=ir.Shape((None, out_features)),
            )
        else:
            gemm_out = out_val

        inputs = [gemm_input, w_val, b_val]
        gemm_node = ir.Node(
            op_type="Gemm",
            domain="",
            inputs=inputs,
            outputs=[gemm_out],
            name=ctx.fresh_name("Gemm"),
        )
        ctx.add_node(gemm_node)
        ctx.set_node_attrs(
            gemm_node,
            {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1},
        )

        if need_flatten:
            batch_dims = x_shape[:-1]
            all_static = all(_is_static_int(d) for d in batch_dims)
            if all_static:
                final_vals = [int(d) for d in batch_dims] + [out_features]
                final_shape = _const_i64(ctx, final_vals, name="linear_out_shape")
                ctx.add_node(
                    ir.Node(
                        op_type="Reshape",
                        domain="",
                        inputs=[gemm_out, final_shape],
                        outputs=[out_val],
                        name=ctx.fresh_name("Reshape"),
                    )
                )
                _stamp_type_and_shape(out_val, tuple(final_vals))
            else:
                rank = len(x_shape)
                shape_val = ir.Value(
                    name=ctx.fresh_name("linear_shape"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((rank,)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Shape",
                        domain="",
                        inputs=[x_val],
                        outputs=[shape_val],
                        name=ctx.fresh_name("Shape"),
                    )
                )
                starts = _const_i64(ctx, [0], name="linear_slice_start")
                ends = _const_i64(ctx, [max(rank - 1, 0)], name="linear_slice_end")
                axes = _const_i64(ctx, [0], name="linear_slice_axes")
                batch_dims_val = ir.Value(
                    name=ctx.fresh_name("linear_batch_dims"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((max(rank - 1, 0),)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Slice",
                        domain="",
                        inputs=[shape_val, starts, ends, axes],
                        outputs=[batch_dims_val],
                        name=ctx.fresh_name("Slice"),
                    )
                )
                out_feat = _const_i64(ctx, [out_features], name="linear_out_feature")
                final_shape = ir.Value(
                    name=ctx.fresh_name("linear_final_shape"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((rank if rank else 1,)),
                )
                concat_node = ir.Node(
                    op_type="Concat",
                    domain="",
                    inputs=[batch_dims_val, out_feat],
                    outputs=[final_shape],
                    name=ctx.fresh_name("Concat"),
                )
                ctx.add_node(concat_node)
                ctx.set_node_attrs(concat_node, {"axis": 0})
                ctx.add_node(
                    ir.Node(
                        op_type="Reshape",
                        domain="",
                        inputs=[gemm_out, final_shape],
                        outputs=[out_val],
                        name=ctx.fresh_name("Reshape"),
                    )
                )
                target_dims = []
                for idx in range(max(len(x_shape) - 1, 0)):
                    target_dims.append(
                        _dim_label_from_value_or_aval(x_val, x_shape, idx)
                    )
                target_dims.append(out_features)
                _stamp_type_and_shape(out_val, tuple(target_dims))
            _ensure_value_info(ctx, out_val)
        else:
            out_shape = (*x_shape[:-1], out_features) if x_shape else (out_features,)
            _stamp_type_and_shape(out_val, out_shape)
            _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("equinox.nn", "linear_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.Linear",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _patch_call(orig):
        del orig

        def wrapped(self, x):
            weight = jnp.asarray(self.weight)
            bias = self.bias
            if bias is None:
                out_features = weight.shape[0]
                bias = jnp.zeros((out_features,), dtype=x.dtype)
            else:
                bias = jnp.asarray(bias, dtype=x.dtype)
            return LinearPlugin._PRIM.bind(x, weight, bias)

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, weight, bias: cls.abstract_eval(x, weight, bias)
            )
            cls._ABSTRACT_EVAL_BOUND = True


@LinearPlugin._PRIM.def_impl
def _linear_impl(x, weight, bias):
    w = jnp.asarray(weight)
    b = jnp.asarray(bias)
    y = jnp.matmul(x, jnp.swapaxes(w, -1, -2))
    return y + b


def _linear_batch_rule(batched_args, batch_dims):
    x, weight, bias = batched_args
    x_bdim, w_bdim, b_bdim = batch_dims
    if w_bdim is not None or b_bdim is not None:
        raise NotImplementedError("Batching over Linear parameters is not supported.")
    out = LinearPlugin._PRIM.bind(x, weight, bias)
    return out, x_bdim


batching.primitive_batchers[LinearPlugin._PRIM] = _linear_batch_rule
