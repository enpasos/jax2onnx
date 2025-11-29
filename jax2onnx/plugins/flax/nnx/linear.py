# jax2onnx/plugins/flax/nnx/linear.py

from __future__ import annotations
from typing import Any, Callable, ClassVar, Final, Optional
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
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


def _linear_output_dims(
    x_val: ir.Value,
    x_shape: tuple,
    out_val: ir.Value,
    out_shape: tuple,
    fallback_last: int,
):
    """Return dims tuple for linear outputs preserving batch labels when known."""
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


# ------------------------------------------------------------------
# Graph-pattern expectations used by tests
# ------------------------------------------------------------------
_GEMM_ONLY_COUNTS: Final[dict[str, int]] = {
    "Gemm": 1,
    "Reshape": 0,
    "Shape": 0,
    "Slice": 0,
    "Concat": 0,
    "CastLike": 0,
    "Transpose": 0,
}

_RGR_COUNTS: Final[dict[str, int]] = {
    "Reshape": 2,
    "Gemm": 1,
    "Shape": 0,
    "Slice": 0,
    "Concat": 0,
    "CastLike": 0,
    "Transpose": 0,
}

_DYNAMIC_RGR_COUNTS: Final[dict[str, int]] = {
    "Reshape": 2,
    "Gemm": 1,
    "Shape": 1,
    "Slice": 1,
    "Concat": 1,
    "CastLike": 0,
    "Transpose": 0,
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


@register_primitive(
    jaxpr_primitive="nnx.linear",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html",
    onnx=[
        {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Shape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Shape.html",
        },
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
        {
            "component": "CastLike",
            "doc": "https://onnx.ai/onnx/operators/onnx__CastLike.html",
        },
    ],
    since="v0.1.0",
    context="primitives.nnx",
    component="linear",
    testcases=[
        {
            "testcase": "linear_symbolic_batch",
            "callable": construct_and_call(
                nnx.Linear,
                128,
                64,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 128)],
            "expected_output_shapes": [("B", 64)],
            "post_check_onnx_graph": EXPECT_GEMM_ONLY,
        },
        {
            "testcase": "linear_high_rank",
            "callable": construct_and_call(
                nnx.Linear,
                128,
                64,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 10, 128)],
            "run_only_dynamic": True,
            "expected_output_shapes": [("B", 10, 64)],
            "post_check_onnx_graph": EXPECT_DYNAMIC_RGR,
        },
        {
            "testcase": "linear_high_rank_static",
            "callable": construct_and_call(
                nnx.Linear,
                128,
                64,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 10, 128)],
            "expected_output_shapes": [(3, 10, 64)],
            "post_check_onnx_graph": EXPECT_RGR_STATIC_3,
        },
        {
            "testcase": "linear_no_bias",
            "callable": construct_and_call(
                nnx.Linear,
                128,
                64,
                use_bias=False,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 128)],
            "expected_output_shapes": [("B", 64)],
            "post_check_onnx_graph": EXPECT_GEMM_ONLY,
        },
        {
            "testcase": "linear_high_rank_no_bias",
            "callable": construct_and_call(
                nnx.Linear,
                128,
                64,
                use_bias=False,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 10, 128)],
            "run_only_dynamic": True,
            "expected_output_shapes": [("B", 10, 64)],
            "post_check_onnx_graph": EXPECT_DYNAMIC_RGR,
        },
        {
            "testcase": "linear_high_rank_no_bias",
            "callable": construct_and_call(
                nnx.Linear,
                128,
                64,
                use_bias=False,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 10, 128)],
            "expected_output_shapes": [(2, 10, 64)],
            "post_check_onnx_graph": EXPECT_RGR_STATIC_2,
        },
        {
            "testcase": "linear_merge_symbolic_dim",
            "callable": construct_and_call(
                nnx.Linear,
                128,
                64,
                dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 10, 128)],
            "run_only_dynamic": True,
            "run_only_f32_variant": True,
            "expected_output_shapes": [("B", 10, 64)],
            "post_check_onnx_graph": EXPECT_DYNAMIC_RGR,
        },
    ],
)
class LinearPlugin(PrimitiveLeafPlugin):
    # Private primitive for this world (no import-time global assignment)
    _PRIM: ClassVar[Primitive] = Primitive("nnx.linear")
    _PRIM.multiple_results = False
    _ORIGINAL_LINEAR_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(x, kernel, bias, *, use_bias: bool, dimension_numbers=None):
        # If we don't have the original __call__, fall back to shape math.
        if LinearPlugin._ORIGINAL_LINEAR_CALL is None:
            if dimension_numbers is None:
                lhs, rhs = ((x.ndim - 1,), (0,))
                dimension_numbers = ((lhs, rhs), ((), ()))
            k0, k1 = kernel.shape
            need_flat = (k0 != x.shape[-1]) or (x.ndim > 2)
            out_shape = (x.shape[0], k1) if need_flat else (*x.shape[:-1], k1)
            return jax.core.ShapedArray(out_shape, x.dtype)

        if dimension_numbers is None:
            lhs, rhs = ((x.ndim - 1,), (0,))
            dimension_numbers = ((lhs, rhs), ((), ()))

        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        k_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        b_spec = jax.ShapeDtypeStruct(bias.shape, bias.dtype) if use_bias else None

        def _helper(xv, kv, bv):
            from types import SimpleNamespace

            def promote_dtype(args, dtype=None):
                return args

            def dot_general(a, b, dimension_numbers=None, precision=None, **kwargs):
                return jax.lax.dot_general(a, b, dimension_numbers)

            class MockParam:
                def __init__(self, value):
                    self.value = value

                def __getitem__(self, item):
                    return self.value[item]

            dummy = SimpleNamespace(
                kernel=MockParam(kv),
                bias=MockParam(bv) if use_bias else None,
                use_bias=use_bias,
                axis=-1,
                in_features=kv.shape[0],
                out_features=kv.shape[1],
                promote_dtype=promote_dtype,
                dtype=xv.dtype,
                dot_general=dot_general,
                precision=None,
                preferred_element_type=None,
            )
            return LinearPlugin._ORIGINAL_LINEAR_CALL(dummy, xv)

        out = jax.eval_shape(_helper, x_spec, k_spec, b_spec)
        out = jax.tree_util.tree_leaves(out)[0]
        return jax.core.ShapedArray(out.shape, out.dtype)

    # ---------- lowering (IR) ----------

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

    # ---------- monkey-patch helper (single, non-duplicated) ----------
    @staticmethod
    def get_monkey_patch(orig_fn):
        LinearPlugin._ORIGINAL_LINEAR_CALL = orig_fn
        prim = LinearPlugin._PRIM

        def patched(self, x):
            dn = (((x.ndim - 1,), (0,)), ((), ()))
            kernel = self.kernel.value
            use_bias = self.bias is not None
            bias = self.bias.value if use_bias else jnp.zeros((), dtype=x.dtype)
            return prim.bind(x, kernel, bias, use_bias=use_bias, dimension_numbers=dn)

        return patched

    @classmethod
    def binding_specs(cls):
        """Patch bindings while active."""
        return [
            AssignSpec("flax.nnx", "linear_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx.Linear",
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


@LinearPlugin._PRIM.def_impl
def _impl(x, kernel, bias, *, use_bias, dimension_numbers):
    y = jax.lax.dot_general(x, kernel, dimension_numbers=dimension_numbers)
    if use_bias and bias is not None:
        y = y + bias
    return y
