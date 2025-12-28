# jax2onnx/plugins/flax/linen/dense.py

from __future__ import annotations
from typing import Any, Callable, ClassVar, Final, Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import linen as nn
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_primitive,
    construct_and_call,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins.flax.test_utils import linen_to_nnx
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._utils import cast_param_like
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    _is_static_int,
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
    _as_ir_dim_label,
    is_shape_all_unknown,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64


# ------------------------------------------------------------------
# Graph-pattern expectations
# ------------------------------------------------------------------
_GEMM_ONLY_COUNTS: Final[dict[str, int]] = {
    "Gemm": 1,
    "Reshape": 0,
    "Shape": 0,
    "Slice": 0,
    "Concat": 0,
    "CastLike": 0,
    "Transpose": 0,
    "Add": 0,  # Optimization should remove Add
}

_RGR_COUNTS: Final[dict[str, int]] = {
    "Reshape": 2,
    "Gemm": 1,
    "Shape": 0,
    "Slice": 0,
    "Concat": 0,
    "CastLike": 0,
    "Transpose": 0,
    "Add": 0,
}

_DYNAMIC_RGR_COUNTS: Final[dict[str, int]] = {
    "Reshape": 2,
    "Gemm": 1,
    "Shape": 1,
    "Slice": 1,
    "Concat": 1,
    "CastLike": 0,
    "Transpose": 0,
    "Add": 0,
}


def _linear_expect(
    path: str,
    *,
    counts: dict[str, int],
    symbols: dict[str, Any] | None = None,
    extra_specs: tuple[str, ...] = (),
):
    specs: list[Any] = [(path, {"counts": dict(counts)})]
    specs.extend(extra_specs)
    return EG(specs, symbols=symbols, no_unused_inputs=True)


EXPECT_GEMM_ONLY: Final = _linear_expect(
    "Gemm:Bx64",
    symbols={"B": None},
    counts=_GEMM_ONLY_COUNTS,
)

EXPECT_RGR_STATIC_3: Final = _linear_expect(
    "Reshape:30x128 -> Gemm:30x64 -> Reshape:3x10x64",
    counts=_RGR_COUNTS,
)

EXPECT_RGR_STATIC_2: Final = _linear_expect(
    "Reshape:20x128 -> Gemm:20x64 -> Reshape:2x10x64",
    counts=_RGR_COUNTS,
)

EXPECT_DYNAMIC_RGR: Final = _linear_expect(
    "Reshape:?x128 -> Gemm:?x64 -> Reshape:Bx10x64",
    counts=_DYNAMIC_RGR_COUNTS,
    symbols={"B": None},
    extra_specs=("Shape -> Slice -> Concat -> Reshape",),
)


def _linear_output_dims(
    x_val: ir.Value,
    x_shape: tuple,
    out_val: ir.Value,
    out_shape: tuple,
    fallback_last: int,
):
    # Derive output dimension labels for a linear layer, preserving batch
    # dimensions and using fallback_last as the final dimension when metadata
    # is unavailable.
    out_rank = len(out_shape)
    batch_rank = max(len(x_shape) - 1, 0)
    if out_rank:
        batch_rank = min(batch_rank, max(out_rank - 1, 0))

    dims: list[Any] = []
    for idx in range(batch_rank):
        label = _dim_label_from_value_or_aval(x_val, x_shape, idx)
        if label is None and idx < len(x_shape):
            maybe_dim = x_shape[idx]
            fallback_label = _as_ir_dim_label(maybe_dim)
            if fallback_label is not None:
                label = fallback_label
        dims.append(label)

    last_dim = None
    if out_rank:
        last_dim = _dim_label_from_value_or_aval(out_val, out_shape, out_rank - 1)
        if last_dim is None and (out_rank - 1) < len(out_shape):
            maybe_dim = out_shape[out_rank - 1]
            fallback_label = _as_ir_dim_label(maybe_dim)
            if fallback_label is not None:
                last_dim = fallback_label
    if last_dim is None:
        last_dim = fallback_last

    dims.append(last_dim)
    return tuple(dims)


@register_primitive(
    jaxpr_primitive="linen.dense",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.Dense",
    onnx=[
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"}
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="dense",
    testcases=[
        {
            "testcase": "dense_basic",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Dense,
                input_shape=(1, 32),
                features=64,
                dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 32)],
            "expected_output_shapes": [("B", 64)],
            "post_check_onnx_graph": EXPECT_GEMM_ONLY,
        },
        {
            "testcase": "dense_high_rank_dynamic",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Dense,
                input_shape=(1, 10, 128),
                features=64,
                dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 10, 128)],
            "expected_output_shapes": [("B", 10, 64)],
            "run_only_dynamic": True,
            "post_check_onnx_graph": EXPECT_DYNAMIC_RGR,
        },
        {
            "testcase": "dense_high_rank_static",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Dense,
                input_shape=(3, 10, 128),
                features=64,
                dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 10, 128)],
            "expected_output_shapes": [(3, 10, 64)],
            "post_check_onnx_graph": EXPECT_RGR_STATIC_3,
        },
        {
            "testcase": "dense_high_rank_no_bias",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Dense,
                input_shape=(2, 10, 128),
                features=64,
                use_bias=False,
                dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 10, 128)],
            "expected_output_shapes": [(2, 10, 64)],
            "post_check_onnx_graph": EXPECT_RGR_STATIC_2,
        },
        {
            "testcase": "dense_no_bias",
            "callable": construct_and_call(
                linen_to_nnx,
                module_cls=nn.Dense,
                input_shape=(1, 32),
                features=64,
                use_bias=False,
                dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 32)],
            "expected_output_shapes": [("B", 64)],
            "post_check_onnx_graph": EXPECT_GEMM_ONLY,
        },
    ],
)
class DensePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("linen.dense")
    _PRIM.multiple_results = False
    _ORIGINAL_DENSE_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, kernel, bias, *, use_bias: bool, dimension_numbers=None):
        k0, k1 = kernel.shape
        out_shape = (*x.shape[:-1], k1)
        return jax.core.ShapedArray(out_shape, x.dtype)

    def lower(self, ctx: Any, eqn):
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError("IR build context missing builder")

        x_var, kernel_var, bias_var = eqn.invars
        out_var = eqn.outvars[0]
        use_bias = bool(eqn.params["use_bias"])

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        k_val = ctx.get_value_for_var(kernel_var, name_hint=ctx.fresh_name("kernel"))
        b_val = (
            ctx.get_value_for_var(bias_var, name_hint=ctx.fresh_name("bias"))
            if use_bias
            else None
        )

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if is_shape_all_unknown(getattr(x_val, "shape", None)) and any(
            d is not None for d in x_shape
        ):
            _stamp_type_and_shape(x_val, x_shape)

        k_val = cast_param_like(ctx, k_val, x_val, "kernel_cast")
        if use_bias and b_val is not None:
            b_val = cast_param_like(ctx, b_val, x_val, "bias_cast")

        k_shape = tuple(getattr(getattr(kernel_var, "aval", None), "shape", ()))
        in_features = int(k_shape[0])
        out_features = int(k_shape[1])

        need_flatten = len(x_shape) > 2
        gemm_input = x_val
        batch_dim_vals: list[Any] = []
        all_batch_static = True

        if need_flatten:
            x_batch_idx = list(range(max(len(x_shape) - 1, 0)))
            batch_dim_vals = [x_shape[i] for i in x_batch_idx]
            all_batch_static = all(_is_static_int(d) for d in batch_dim_vals)
            if all_batch_static:
                m_size = int(np.prod([int(d) for d in batch_dim_vals]) or 1)
                x2d_dims: tuple[Optional[int], int] = (m_size, in_features)
            else:
                x2d_dims = (None, in_features)

            x2d_shape_c = _const_i64(ctx, [-1, in_features], name_hint="x2d_shape")
            gemm_input = builder.Reshape(
                x_val,
                x2d_shape_c,
                _outputs=[ctx.fresh_name("input_reshape")],
            )
            if getattr(x_val, "type", None) is not None:
                gemm_input.type = x_val.type
            _stamp_type_and_shape(gemm_input, x2d_dims)
            _ensure_value_metadata(ctx, gemm_input)

        gemm_output_name = ctx.fresh_name("gemm_output" if need_flatten else "out")
        gemm_inputs = [gemm_input, k_val]
        if use_bias and b_val is not None:
            gemm_inputs.append(b_val)

        gemm_result = builder.Gemm(
            *gemm_inputs,
            alpha=1.0,
            beta=0.0 if not use_bias else 1.0,
            transA=0,
            transB=0,
            _outputs=[gemm_output_name],
        )

        result_dtype = getattr(getattr(x_val, "type", None), "dtype", None)
        if result_dtype is not None:
            gemm_result.type = ir.TensorType(result_dtype)

        if need_flatten:
            if all_batch_static:
                m_size = int(np.prod([int(d) for d in batch_dim_vals]) or 1)
                gemm_dims: tuple[Optional[int], int] = (m_size, out_features)
            else:
                gemm_dims = (None, out_features)
            _stamp_type_and_shape(gemm_result, gemm_dims)
            _ensure_value_metadata(ctx, gemm_result)

        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", ()) or ())

        if not need_flatten:
            y_meta = _linear_output_dims(
                x_val,
                x_shape,
                gemm_result,
                out_shape,
                int(out_features),
            )
            _stamp_type_and_shape(gemm_result, y_meta)
            _ensure_value_metadata(ctx, gemm_result)
            ctx.bind_value_for_var(out_var, gemm_result)
            return

        if all_batch_static:
            final_vals = [int(d) for d in batch_dim_vals] + [int(out_features)]
            final_shape_c = _const_i64(ctx, final_vals, name_hint="final_shape_c")
            final_output = builder.Reshape(
                gemm_result,
                final_shape_c,
                _outputs=[ctx.fresh_name("out")],
            )
            if result_dtype is not None:
                final_output.type = ir.TensorType(result_dtype)
            y_meta = _linear_output_dims(
                x_val,
                x_shape,
                final_output,
                out_shape,
                int(out_features),
            )
            _stamp_type_and_shape(final_output, y_meta)
            _ensure_value_metadata(ctx, final_output)
            ctx.bind_value_for_var(out_var, final_output)
            return

        shp = builder.Shape(x_val, _outputs=[ctx.fresh_name("x_shape")])
        shp.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(shp, (len(x_shape),))
        _ensure_value_metadata(ctx, shp)

        starts = _const_i64(ctx, [0], name_hint="slice_starts")
        ends = _const_i64(ctx, [len(x_shape) - 1], name_hint="slice_ends")
        axes_val = _const_i64(ctx, [0], name_hint="slice_axes")
        steps = _const_i64(ctx, [1], name_hint="slice_steps")

        batch_dims = builder.Slice(
            shp,
            starts,
            ends,
            axes_val,
            steps,
            _outputs=[ctx.fresh_name("batch_dims")],
        )
        batch_dims.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(batch_dims, (len(x_shape) - 1,))
        _ensure_value_metadata(ctx, batch_dims)

        of = _const_i64(ctx, [out_features], name_hint="out_features_c")
        final_shape = builder.Concat(
            batch_dims,
            of,
            axis=0,
            _outputs=[ctx.fresh_name("final_shape")],
        )
        final_shape.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(final_shape, (len(x_shape),))
        _ensure_value_metadata(ctx, final_shape)

        final_output = builder.Reshape(
            gemm_result,
            final_shape,
            _outputs=[ctx.fresh_name("out")],
        )
        if result_dtype is not None:
            final_output.type = ir.TensorType(result_dtype)
        y_meta = _linear_output_dims(
            x_val,
            x_shape,
            final_output,
            out_shape,
            int(out_features),
        )
        _stamp_type_and_shape(final_output, y_meta)
        _ensure_value_metadata(ctx, final_output)
        ctx.bind_value_for_var(out_var, final_output)

    @staticmethod
    def get_monkey_patch(orig_fn):
        DensePlugin._ORIGINAL_DENSE_CALL = orig_fn
        prim = DensePlugin._PRIM

        def patched(self, x):
            # Access pre-initialized parameters directly from the module's scope
            # This avoids calling self.param() which would try to compute shapes
            # from traced inputs during JAX tracing.

            # Check if we can access variables directly (apply mode)
            scope = getattr(self, "scope", None)
            if scope is not None and hasattr(scope, "variables"):
                variables = scope.variables()
                params = variables.get("params", {})
                kernel = params.get("kernel")
                bias = params.get("bias") if self.use_bias else None

                if kernel is not None:
                    # We have pre-initialized params - use them directly
                    inputs = x
                    if bias is None and self.use_bias:
                        bias = jnp.zeros((self.features,), dtype=inputs.dtype)
                    elif not self.use_bias:
                        # Use a consistently shaped fallback bias even when not applied.
                        bias = jnp.zeros((self.features,), dtype=inputs.dtype)

                    dn = (((inputs.ndim - 1,), (0,)), ((), ()))
                    return prim.bind(
                        inputs,
                        kernel,
                        bias,
                        use_bias=self.use_bias,
                        dimension_numbers=dn,
                    )

            # Fallback to original behavior for init or when params not available
            return orig_fn(self, x)

        return patched

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.linen", "dense_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.linen.Dense",
                attr="__call__",
                make_value=lambda orig: cls.get_monkey_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True


@DensePlugin._PRIM.def_impl
def _impl(x, kernel, bias, *, use_bias, dimension_numbers):
    y = jax.lax.dot_general(x, kernel, dimension_numbers=dimension_numbers)
    if use_bias and bias is not None:
        y = y + bias
    return y
