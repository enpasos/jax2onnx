# jax2onnx/plugins/equinox/eqx/nn/linear.py

from __future__ import annotations

from typing import Callable, ClassVar, Final, Optional

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import (
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
    _is_static_int,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.lax._index_utils import _const_i64


_eqx_linear_symbolic: Final = eqx.nn.Linear(128, 64, key=jax.random.PRNGKey(0))
_eqx_linear_highrank: Final = eqx.nn.Linear(128, 64, key=jax.random.PRNGKey(42))
_eqx_linear_no_bias: Final = eqx.nn.Linear(
    128, 64, use_bias=False, key=jax.random.PRNGKey(7)
)


def _ensure_static_int(dim: int | str | None) -> int:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    raise TypeError("Dimension is not a static integer")


def _value_to_numpy(val: ir.Value) -> Optional[np.ndarray]:
    for attr in ("const_value", "_const_value", "value", "data", "numpy"):
        payload = getattr(val, attr, None)
        if payload is None:
            continue
        try:
            return np.asarray(payload)
        except Exception:
            try:
                return np.asarray(payload())  # callable returning array-like
            except Exception:
                continue
    return None


def _set_value_const_payload(val: ir.Value, arr: np.ndarray) -> None:
    payload = ir.tensor(arr) if hasattr(ir, "tensor") else arr
    for attr in ("const_value", "_const_value", "value", "data", "numpy"):
        if hasattr(val, attr):
            try:
                setattr(val, attr, payload)
            except Exception:
                pass


def _inline_scalar_bias(
    ctx: LoweringContextProtocol, bias_val: ir.Value, out_features: int
) -> ir.Value:
    expand_node = getattr(bias_val, "producer", None)
    if callable(expand_node):
        try:
            expand_node = expand_node()
        except Exception:
            expand_node = None
    if expand_node is None or getattr(expand_node, "op_type", "") != "Expand":
        return bias_val

    expand_inputs = list(
        getattr(expand_node, "inputs", getattr(expand_node, "input", []))
    )
    if not expand_inputs:
        return bias_val

    const_input = expand_inputs[0]
    arr = _value_to_numpy(const_input)
    # For debugging - disable in production if needed
    if arr is None:
        for init in getattr(ctx.builder, "initializers", []):
            if getattr(init, "name", None) == getattr(const_input, "name", None):
                arr = _value_to_numpy(init)
                if arr is not None:
                    break
    if arr is None:
        return bias_val

    arr = np.asarray(arr)

    if arr.size == 1:
        broadcast = np.broadcast_to(arr.reshape(()), (out_features,))
    elif arr.size == out_features:
        broadcast = arr.reshape((out_features,))
    else:
        return bias_val
    bias_type = getattr(bias_val, "type", None)
    if isinstance(bias_type, ir.TensorType):
        new_type = ir.TensorType(bias_type.dtype)
    else:
        new_type = bias_type

    # Route through builder so function-mode + duplicate policy apply.
    new_val = ctx.builder.add_initializer_from_array(
        name=ctx.fresh_name("linear_bias_inline"), array=np.asarray(broadcast)
    )
    # Preserve the desired dtype metadata if available
    if new_type is not None:
        new_val.type = new_type

    try:
        mapping = getattr(ctx.builder, "_var2val", None)
        if isinstance(mapping, dict):
            for var, val in list(mapping.items()):
                if val is bias_val:
                    mapping[var] = new_val
    except Exception:
        pass

    try:
        nodes = getattr(ctx.builder, "nodes", None)
        if isinstance(nodes, list):
            if expand_node in nodes:
                nodes.remove(expand_node)
            if len(expand_inputs) > 1:
                shape_val = expand_inputs[1]
                shape_producer = getattr(shape_val, "producer", None)
                if callable(shape_producer):
                    try:
                        shape_producer = shape_producer()
                    except Exception:
                        shape_producer = None
                if shape_producer and getattr(shape_producer, "op_type", "") in {
                    "Concat",
                    "Reshape",
                }:
                    output_names = {
                        getattr(v, "name", None)
                        for v in getattr(
                            shape_producer,
                            "outputs",
                            getattr(shape_producer, "output", []),
                        )
                    }
                    if output_names:
                        still_used = False
                        for node in nodes:
                            if node is shape_producer:
                                continue
                            for iv in getattr(
                                node, "inputs", getattr(node, "input", [])
                            ):
                                if getattr(iv, "name", None) in output_names:
                                    still_used = True
                                    break
                            if still_used:
                                break
                        if not still_used and shape_producer in nodes:
                            nodes.remove(shape_producer)
    except Exception:
        pass

    return new_val


def _flatten_leading_dim_label(
    x_val, x_shape: tuple[int | str | None, ...]
) -> int | str | None:
    batch_dims = x_shape[:-1]
    if not batch_dims:
        return 1

    labels: list[int | str | None] = []
    all_static = True
    for idx, dim in enumerate(batch_dims):
        label = _dim_label_from_value_or_aval(x_val, x_shape, idx)
        if label is None:
            if _is_static_int(dim):
                label = _ensure_static_int(dim)
            else:
                label = None
        else:
            if isinstance(label, str) and not label:
                label = None
        labels.append(label)
        if not _is_static_int(dim):
            all_static = False

    non_null = [lb for lb in labels if lb is not None]
    if len(non_null) == 1 and len(labels) == 1:
        return non_null[0]

    if len(non_null) == len(labels) and all(
        isinstance(lb, (int, np.integer)) for lb in non_null
    ):
        prod = 1
        for lb in non_null:
            prod *= int(lb)
        return prod

    if all_static and batch_dims:
        prod = 1
        for dim in batch_dims:
            prod *= _ensure_static_int(dim)
        return prod

    return None


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
            "post_check_onnx_graph": expect_graph(
                ["Gemm:Bx64"],
                no_unused_inputs=True,
                must_absent=["Expand"],
            ),
        },
        {
            "testcase": "eqx_linear_no_bias_symbolic_batch",
            "callable": lambda x, _mod=_eqx_linear_no_bias: jax.vmap(_mod)(x),
            "input_shapes": [("B", 128)],
            "post_check_onnx_graph": expect_graph(
                ["Gemm:Bx64"],
                no_unused_inputs=True,
                must_absent=["Expand"],
            ),
        },
        {
            "testcase": "eqx_linear_no_bias_vector",
            "callable": _eqx_linear_no_bias,
            "input_shapes": [(128,)],
            "post_check_onnx_graph": expect_graph(
                ["Reshape:1x128 -> Gemm:1x64 -> Reshape:64"],
                no_unused_inputs=True,
                must_absent=["Expand"],
            ),
        },
        {
            "testcase": "eqx_linear_high_rank",
            "callable": lambda x, _mod=_eqx_linear_highrank: jax.vmap(jax.vmap(_mod))(
                x
            ),
            "input_shapes": [(32, 10, 128)],
            "post_check_onnx_graph": expect_graph(
                ["Reshape:320x128 -> Gemm:320x64 -> Reshape:32x10x64"],
                no_unused_inputs=True,
                must_absent=["Expand"],
            ),
        },
        {
            "testcase": "eqx_linear_vector",
            "callable": _eqx_linear_symbolic,
            "input_shapes": [(128,)],
            "post_check_onnx_graph": expect_graph(
                ["Reshape:1x128 -> Gemm:1x64 -> Reshape:64"],
                no_unused_inputs=True,
                must_absent=["Expand"],
            ),
        },
    ],
)
class LinearPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.linear")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax_core.AbstractValue,
        weight: jax_core.AbstractValue,
        bias: jax_core.AbstractValue,
    ) -> ShapedArray:
        del bias
        out_features = weight.shape[0]
        if x.ndim <= 1:
            out_shape = (out_features,)
        else:
            out_shape = (*x.shape[:-1], out_features)
        return ShapedArray(out_shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax_core.JaxprEqn) -> None:
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for Eqx Linear lowering"
            )

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
        b_val = _inline_scalar_bias(ctx, b_val, out_features)

        need_flatten = len(x_shape) != 2
        flat_dim_label = None
        gemm_input = x_val

        if need_flatten:
            flat_dim_label = _flatten_leading_dim_label(x_val, x_shape)
            reshape_first_dim = -1
            if isinstance(flat_dim_label, (int, np.integer)) and flat_dim_label > 0:
                reshape_first_dim = int(flat_dim_label)
            elif flat_dim_label == 1:
                reshape_first_dim = 1
            reshape_shape = _const_i64(
                ctx,
                [reshape_first_dim, in_features],
                name="linear_in_shape",
            )
            gemm_input = builder.Reshape(
                x_val,
                reshape_shape,
                _outputs=[ctx.fresh_name("linear_flat")],
            )
            if getattr(x_val, "type", None) is not None:
                gemm_input.type = x_val.type
            reshape_dims = (
                flat_dim_label if flat_dim_label is not None else None,
                in_features,
            )
            _stamp_type_and_shape(gemm_input, reshape_dims)
            _ensure_value_metadata(ctx, gemm_input)

        gemm_output_name = ctx.fresh_name(
            "linear_gemm" if need_flatten else "linear_out"
        )
        gemm_inputs = [gemm_input, w_val]
        if b_val is not None:
            gemm_inputs.append(b_val)

        beta_attr = 1.0 if b_val is not None else 0.0
        gemm_result = builder.Gemm(
            *gemm_inputs,
            alpha=1.0,
            beta=beta_attr,
            transA=0,
            transB=1,
            _outputs=[gemm_output_name],
        )

        result_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if result_dtype is not None:
            gemm_result.type = ir.TensorType(result_dtype)

        if not need_flatten:
            out_dims: list[object] = []
            if x_shape:
                for idx in range(len(x_shape) - 1):
                    label = _dim_label_from_value_or_aval(x_val, x_shape, idx)
                    if label is None:
                        dim = x_shape[idx]
                        if _is_static_int(dim):
                            label = _ensure_static_int(dim)
                        else:
                            label = dim
                    out_dims.append(label)
            out_dims.append(out_features)
            _stamp_type_and_shape(gemm_result, tuple(out_dims))
            _ensure_value_metadata(ctx, gemm_result)
            ctx.bind_value_for_var(out_var, gemm_result)
            return

        leading_dim = flat_dim_label if flat_dim_label is not None else None
        _stamp_type_and_shape(gemm_result, (leading_dim, out_features))
        _ensure_value_metadata(ctx, gemm_result)

        batch_dims = x_shape[:-1]
        all_static = all(_is_static_int(d) for d in batch_dims)

        if all_static:
            final_vals = [_ensure_static_int(d) for d in batch_dims] + [out_features]
            final_shape = _const_i64(ctx, final_vals, name_hint="linear_out_shape")
            final_output = builder.Reshape(
                gemm_result,
                final_shape,
                _outputs=[ctx.fresh_name("linear_out")],
            )
            if result_dtype is not None:
                final_output.type = ir.TensorType(result_dtype)
            _stamp_type_and_shape(final_output, tuple(final_vals))
            _ensure_value_metadata(ctx, final_output)
            ctx.bind_value_for_var(out_var, final_output)
            return

        rank = len(x_shape)
        shape_val = builder.Shape(
            x_val,
            _outputs=[ctx.fresh_name("linear_shape")],
        )
        shape_val.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(shape_val, (rank if rank else 1,))
        _ensure_value_metadata(ctx, shape_val)

        starts = _const_i64(ctx, [0], name_hint="linear_slice_start")
        ends = _const_i64(ctx, [max(rank - 1, 0)], name_hint="linear_slice_end")
        axes = _const_i64(ctx, [0], name_hint="linear_slice_axes")
        steps = _const_i64(ctx, [1], name_hint="linear_slice_steps")

        batch_dims_val = builder.Slice(
            shape_val,
            starts,
            ends,
            axes,
            steps,
            _outputs=[ctx.fresh_name("linear_batch_dims")],
        )
        batch_dims_val.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(batch_dims_val, (max(rank - 1, 0),))
        _ensure_value_metadata(ctx, batch_dims_val)

        out_feat = _const_i64(ctx, [out_features], name_hint="linear_out_feature")
        final_shape = builder.Concat(
            batch_dims_val,
            out_feat,
            axis=0,
            _outputs=[ctx.fresh_name("linear_final_shape")],
        )
        final_shape.type = ir.TensorType(ir.DataType.INT64)
        final_rank = max(rank - 1, 0) + 1
        _stamp_type_and_shape(final_shape, (final_rank,))
        _ensure_value_metadata(ctx, final_shape)

        final_output = builder.Reshape(
            gemm_result,
            final_shape,
            _outputs=[ctx.fresh_name("linear_out")],
        )
        if result_dtype is not None:
            final_output.type = ir.TensorType(result_dtype)

        target_dims: list[object] = []
        for idx in range(max(len(x_shape) - 1, 0)):
            label = _dim_label_from_value_or_aval(x_val, x_shape, idx)
            if label is None:
                dim = x_shape[idx]
                if _is_static_int(dim):
                    label = _ensure_static_int(dim)
                else:
                    label = dim
            target_dims.append(label)
        target_dims.append(out_features)
        _stamp_type_and_shape(final_output, tuple(target_dims))
        _ensure_value_metadata(ctx, final_output)
        ctx.bind_value_for_var(out_var, final_output)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
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
    def _patch_call(
        orig: Callable[..., jax.Array] | None,
    ) -> Callable[[eqx.nn.Linear, jax.Array], jax.Array]:
        del orig

        def wrapped(self: eqx.nn.Linear, x: jax.Array) -> jax.Array:
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
def _linear_impl(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    w = jnp.asarray(weight)
    b = jnp.asarray(bias)
    y = jnp.matmul(x, jnp.swapaxes(w, -1, -2))
    return y + b


def _linear_batch_rule(
    batched_args: tuple[jax.Array, jax.Array, jax.Array],
    batch_dims: tuple[int | None, int | None, int | None],
) -> tuple[jax.Array, int | None]:
    x, weight, bias = batched_args
    x_bdim, w_bdim, b_bdim = batch_dims
    if w_bdim is not None or b_bdim is not None:
        raise NotImplementedError("Batching over Linear parameters is not supported.")
    out = LinearPlugin._PRIM.bind(x, weight, bias)
    return out, x_bdim


batching.primitive_batchers[LinearPlugin._PRIM] = _linear_batch_rule
