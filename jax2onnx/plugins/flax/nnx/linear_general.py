# jax2onnx/plugins/flax/nnx/linear_general.py


from __future__ import annotations
from typing import Any, Callable, ClassVar, Final
import numpy as np
import jax
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins._utils import cast_param_like, inline_reshape_initializer
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    _prod,
    _to_ir_dim_for_shape,
    _is_static_int,
    _dim_label_from_value_or_aval,
    _ensure_value_metadata,
    _as_ir_dim_label,
)


# ------------------------------------------------------------------
# Helpers used by a testcase's post_check_onnx_graph (shape asserts)
# ------------------------------------------------------------------
def _shape_of(coll, name: str):
    """Return tuple of dims (int | str | None) for the tensor named `name`.
    Test utilities historically used 'var_0' for the first input. If that exact
    name is not found, fall back sensibly:
      • if asking for 'var_0' or 'in_0' → use the first graph input
      • if asking for 'var_3' or 'out_0' → use the first graph output
    This keeps checks working across both historical and current naming schemes."""
    for vi in coll:
        if vi.name == name:
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_param") and d.dim_param:
                    dims.append(d.dim_param)
                elif d.HasField("dim_value"):
                    dims.append(d.dim_value)
                else:
                    dims.append(None)
            return tuple(dims)
    # Compatibility across name changes ('var_0'/'in0', 'var_3'/'out_0'):
    if name in {"var_0", "in_0", "var_3", "out_0"} and len(coll) >= 1:
        vi = coll[0]
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_param") and d.dim_param:
                dims.append(d.dim_param)
            elif d.HasField("dim_value"):
                dims.append(d.dim_value)
            else:
                dims.append(None)
        return tuple(dims)
    raise KeyError(f"Cannot find '{name}' in ValueInfo collection")


def _shape_prefix_of(coll, prefix: str):
    """Return dims for the first tensor whose name starts with `prefix`.
    Prefer exact/shortest match to avoid picking e.g. 'input_reshape_shape'."""
    candidates = [vi for vi in coll if vi.name.startswith(prefix)]
    if not candidates:
        raise KeyError(f"No tensor name starting with '{prefix}'")
    # prefer the exact name, or otherwise the shortest prefixed one
    vi = min(candidates, key=lambda v: len(v.name))
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_param") and d.dim_param:
            dims.append(d.dim_param)
        elif d.HasField("dim_value"):
            dims.append(d.dim_value)
        else:
            dims.append(None)
    return tuple(dims)


def _linear_general_output_dims(
    x_val: ir.Value,
    x_shape: tuple,
    batch_indices: list[int],
    out_val: ir.Value,
    out_shape: tuple,
    fallback_tail: list[int],
):
    """Derive output dims preserving batch labels and using aval metadata."""
    dims: list = []
    for idx in batch_indices:
        label = _dim_label_from_value_or_aval(x_val, x_shape, idx)
        if label is None and idx < len(x_shape):
            maybe_dim = x_shape[idx]
            fallback_label = _as_ir_dim_label(maybe_dim)
            if fallback_label is not None:
                label = fallback_label
        dims.append(label)

    tail_start = len(dims)
    total_rank = len(out_shape)
    tail_count = max(total_rank - tail_start, len(fallback_tail))

    for offset in range(max(tail_count, 0)):
        pos = tail_start + offset
        label = None
        if pos < total_rank:
            label = _dim_label_from_value_or_aval(out_val, out_shape, pos)
            if label is None:
                maybe_dim = out_shape[pos]
                fallback_label = _as_ir_dim_label(maybe_dim)
                if fallback_label is not None:
                    label = fallback_label
        if label is None and offset < len(fallback_tail):
            label = fallback_tail[offset]
        dims.append(label)

    if total_rank and len(dims) > total_rank:
        dims = dims[:total_rank]
    return tuple(dims)


def _b_matches(x):
    """Batch dim 'B' is considered equivalent to anonymous dynamic (None)."""
    return x in ("B", None)


def _eq_oldworld_input(s):
    """
    Old-world input: B×8×4×16.
    IR may anonymize the non-batch dims; accept 8|None, 4|None, 16|None.
    """
    return (
        len(s) == 4
        and _b_matches(s[0])
        and (s[1] in (8, None))
        and (s[2] in (4, None))
        and (s[3] in (16, None))
    )


def _eq_oldworld_output(s):
    """Require the old-world output shape B×8×32 (allow 'B'~None only for batch)."""
    return len(s) == 3 and _b_matches(s[0]) and s[1] == 8 and s[2] == 32


def _is_qxK(s, K):
    """Internal reshape should be ?×K."""
    return len(s) == 2 and s[0] in (None, "B") and (s[1] == K or s[1] is None)


def _first_node(m, op):
    return next(n for n in m.graph.node if n.op_type == op)


def _init_dims(m, name):
    t = next((t for t in m.graph.initializer if t.name == name), None)
    return list(t.dims) if t is not None else None


# ------------------------------------------------------------------
# Graph-pattern expectations used by tests
# ------------------------------------------------------------------
# Basic presence of a single Gemm (no flatten/reshape path needed).
EXPECT_GEMM_ONLY: Final = EG([("Gemm", {"counts": {"Gemm": 1}})])
# Static flatten path: Reshape -> Gemm -> Reshape (no dynamic shape ops).
EXPECT_RGR: Final = EG(
    [
        (
            "Reshape -> Gemm -> Reshape",
            {
                "counts": {
                    "Gemm": 1,
                    "CastLike": 0,
                    "Transpose": 0,
                }
            },
        )
    ]
)
# Dynamic flatten path: input Reshape to Gemm, and separate dynamic-shape chain
# (Shape->Slice->Concat) that feeds the final Reshape's shape, plus Gemm->Reshape.
EXPECT_DYNAMIC_RGR: Final = EG(
    [
        (
            "Reshape -> Gemm -> Reshape",
            {
                "counts": {
                    "Gemm": 1,
                    "CastLike": 0,
                    "Transpose": 0,
                }
            },
        ),
        "Shape -> Slice -> Concat -> Reshape",
        "Gemm -> Reshape",
    ]
)


# ------------------------------------------------------------------
# ONNX primitive registration and plugin for LinearGeneral
# ------------------------------------------------------------------
@register_primitive(
    jaxpr_primitive="nnx.linear_general",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.LinearGeneral",
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
    component="linear_general",
    testcases=[
        {
            "testcase": "linear_general_merge_symbolic_dim",
            "callable": construct_and_call(
                nnx.LinearGeneral,
                in_features=(4, 16),  # ⟨4,16⟩ are contracting dims
                out_features=32,
                axis=(-2, -1),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 8, 4, 16)],
            "expected_output_shapes": [("B", 8, 32)],
            "run_only_dynamic": True,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: (
                (_eq_oldworld_input(_shape_of(m.graph.input, "in_0")))
                and (lambda ok_ir: True if ok_ir is True else _is_qxK(ok_ir, 64))(
                    # If value_info is present, enforce ?×64; if missing, treat as True.
                    (lambda: _shape_prefix_of(m.graph.value_info, "input_reshape"))()
                    if any(
                        vi.name.startswith("input_reshape") for vi in m.graph.value_info
                    )
                    else True
                )
                and (lambda ok_go: True if ok_go is True else _is_qxK(ok_go, 32))(
                    (lambda: _shape_prefix_of(m.graph.value_info, "gemm_output"))()
                    if any(
                        vi.name.startswith("gemm_output") for vi in m.graph.value_info
                    )
                    else True
                )
                and _eq_oldworld_output(_shape_of(m.graph.output, "out_0"))
            ),
        },
        {
            "testcase": "linear_general",
            "callable": construct_and_call(
                nnx.LinearGeneral,
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4, 8, 32)],
            "expected_output_shapes": [("B", 4, 256)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": lambda m: (
                EXPECT_RGR(m)
                and (
                    lambda gemm:
                    # C is present and is a 1-D initializer of length 256
                    (len(gemm.input) >= 3)
                    and (_init_dims(m, gemm.input[2]) == [256])
                )(_first_node(m, "Gemm"))
            ),
        },
        {
            "testcase": "linear_general_2",
            "callable": construct_and_call(
                nnx.LinearGeneral,
                in_features=(30,),
                out_features=(20,),
                axis=(-1,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 30)],
            "expected_output_shapes": [(3, 20)],
            "run_only_f32_variant": True,
            # rank-2 input -> no flatten needed, just Gemm
            "post_check_onnx_graph": EXPECT_GEMM_ONLY,
        },
        {
            "testcase": "linear_general_3",
            "callable": construct_and_call(
                nnx.LinearGeneral,
                in_features=(256,),
                out_features=(8, 32),
                axis=(-1,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 4, 256)],
            "expected_output_shapes": [(2, 4, 8, 32)],
            "run_only_f32_variant": True,
            # static 3D -> static flatten: Reshape -> Gemm -> Reshape
            "post_check_onnx_graph": EXPECT_RGR,
        },
        {
            "testcase": "linear_general_4",
            "callable": construct_and_call(
                nnx.LinearGeneral,
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(2, 4, 8, 32)],
            "expected_output_shapes": [(2, 4, 256)],
            "run_only_f32_variant": True,
            # Static case: ensure we eliminated Shape/Slice/Concat and inlined
            # the final Reshape's shape as a constant of length 3 (3×4×256).
            "post_check_onnx_graph": EXPECT_RGR,
        },
        {
            "testcase": "linear_general_abstract_eval_axes",
            "callable": construct_and_call(
                nnx.LinearGeneral,
                in_features=(256,),
                out_features=(8, 32),
                axis=(-1,),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 10, 256)],
            "expected_output_shape": (3, 10, 8, 32),
            "expected_output_shapes": [(3, 10, 8, 32)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_RGR,
        },
        {
            "testcase": "linear_general_abstract_eval_axes_pair",
            "callable": construct_and_call(
                nnx.LinearGeneral,
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [(3, 10, 8, 32)],
            "expected_output_shape": (3, 10, 256),
            "expected_output_shapes": [(3, 10, 256)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_RGR,
        },
        {
            "testcase": "dynamic_batch_and_feature_dims",
            # Bind our primitive directly with explicit dimension_numbers.
            "callable": lambda x, k, b: LinearGeneralPlugin._linear_general(
                x, k, b, dimension_numbers=(((2,), (0,)), ((), ()))
            ),
            "input_shapes": [("B", "H", 16), (16, 4, 4), (4, 4)],
            "expected_output_shapes": [("B", "H", 4, 4)],
            "run_only_dynamic": True,
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EXPECT_DYNAMIC_RGR,
        },
    ],
)
class LinearGeneralPlugin(PrimitiveLeafPlugin):
    """
    IR-only plugin for flax.nnx.LinearGeneral:
      Reshape([-1,K]) → Gemm → Reshape-back.
    Maintains the original value-name prefixes ("input_reshape", "gemm_output")
    so post-checks from the existing tests continue to work.
    """

    _PRIM: ClassVar[Primitive] = Primitive("nnx.linear_general")
    _PRIM.multiple_results = False
    _ORIGINAL_CALL: ClassVar[Callable | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(x, kernel, bias, *, dimension_numbers):
        if LinearGeneralPlugin._ORIGINAL_CALL is None:
            # Pure shape math fallback.
            ((lhs_contract, rhs_contract), _) = dimension_numbers
            x_batch = tuple(i for i in range(x.ndim) if i not in lhs_contract)
            k_out = tuple(i for i in range(kernel.ndim) if i not in rhs_contract)
            out_shape = tuple(x.shape[i] for i in x_batch) + tuple(
                kernel.shape[i] for i in k_out
            )
            return jax.core.ShapedArray(out_shape, x.dtype)

        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        k_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)
        b_spec = (
            jax.ShapeDtypeStruct(bias.shape, bias.dtype) if bias is not None else None
        )

        def _helper(xv, kv, bv):
            rhs_contract = dimension_numbers[0][1]
            out_dims = [i for i in range(kv.ndim) if i not in rhs_contract]
            out_features = tuple(kv.shape[i] for i in out_dims)

            # match Flax API surface the __call__ expects
            def promote_dtype(args, dtype=None):
                # shape-only path: return as-is
                return args

            def dot_general(a, b, dimension_numbers=None, precision=None, **_):
                return jax.lax.dot_general(
                    a, b, dimension_numbers=dimension_numbers, precision=precision
                )

            from types import SimpleNamespace

            class MockParam:
                def __init__(self, value):
                    self.value = value

                def __getitem__(self, item):
                    return self.value[item]

            dummy = SimpleNamespace(
                kernel=MockParam(kv),
                bias=None if bv is None else MockParam(bv),
                dimension_numbers=dimension_numbers,
                batch_axis={},  # len(self.batch_axis) is used
                axis=dimension_numbers[0][0],  # lhs contracting axes
                in_features=tuple(kv.shape[: len(rhs_contract)]),
                out_features=out_features,
                promote_dtype=promote_dtype,
                dtype=None,  # match Flax default
                dot_general=dot_general,  # function path
                dot_general_cls=None,
                precision=None,
                preferred_element_type=None,
            )
            return LinearGeneralPlugin._ORIGINAL_CALL(dummy, xv)

        out = jax.eval_shape(_helper, x_spec, k_spec, b_spec)
        out = jax.tree_util.tree_leaves(out)[0]
        return jax.core.ShapedArray(out.shape, out.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: Any, eqn):
        x_var, k_var, b_var = eqn.invars[:3]
        y_var = eqn.outvars[0]
        ((lhs_contract, rhs_contract), _) = eqn.params["dimension_numbers"]

        # Values
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        k_val = ctx.get_value_for_var(k_var, name_hint=ctx.fresh_name("kernel"))
        out_spec = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))
        out_name = getattr(out_spec, "name", None) or ctx.fresh_name("out")
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for LinearGeneral lowering"
            )

        # ---------- ensure graph.input shows the original, unfused shape ----------
        # Preserve symbolic labels like "B" and the literal dims 8,4,16.
        # IMPORTANT: don't overwrite shape labels if the binder already set them.
        def _all_unknown(shp):
            if shp is None:
                return True
            dims = getattr(shp, "dims", None)
            if dims is None:
                try:
                    dims = list(shp)
                except Exception:
                    return True
            for d in dims:
                if _as_ir_dim_label(d) is not None:
                    return False
            return True

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if _all_unknown(getattr(x_val, "shape", None)):
            x_meta = tuple(_to_ir_dim_for_shape(d) for d in x_shape)
            if any(v is not None for v in x_meta):
                _stamp_type_and_shape(x_val, x_shape)
        k_shape = tuple(getattr(getattr(k_var, "aval", None), "shape", ()))
        rhs_contract = tuple((a % len(k_shape)) for a in rhs_contract)
        lhs_contract = tuple((a % max(len(x_shape), 1)) for a in lhs_contract)

        K = _prod(int(k_shape[i]) for i in rhs_contract)
        k_out_idx = [i for i in range(len(k_shape)) if i not in rhs_contract]
        k_out_dims = [int(k_shape[i]) for i in k_out_idx]
        Cout = _prod(k_out_dims)

        # --- KERNEL (B) ---
        desired_k_shape = (int(K), int(Cout))

        # Try to inline the kernel reshape if it's a constant initializer
        k2d = inline_reshape_initializer(ctx, k_val, desired_k_shape, "kernel2d")

        if k2d is k_val:
            kshape_c = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("kshape"),
                array=np.asarray(desired_k_shape, dtype=np.int64),
            )
            k2d = ctx.builder.Reshape(
                k_val,
                kshape_c,
                _outputs=[ctx.fresh_name("kernel2d")],
            )
            if getattr(k_val, "type", None) is not None:
                k2d.type = k_val.type
            _stamp_type_and_shape(k2d, desired_k_shape)
            _ensure_value_metadata(ctx, k2d)

        # IMPORTANT: cast *after* shaping so the Gemm input has the final dtype
        k2d = cast_param_like(ctx, k2d, x_val, "kernel_cast")

        # --- INPUT (A) flatten remains unchanged ...
        # (keep producing 'input_reshape' for tests)
        x_batch_idx = [i for i in range(len(x_shape)) if i not in lhs_contract]
        need_flatten = len(x_shape) > 2

        gemm_in = x_val
        if need_flatten:
            x2d_shape_c = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("x2d_shape"),
                array=np.asarray([-1, int(K)], dtype=np.int64),
            )
            x2d = ctx.builder.Reshape(
                x_val,
                x2d_shape_c,
                _outputs=[ctx.fresh_name("input_reshape")],
            )
            if getattr(x_val, "type", None) is not None:
                x2d.type = x_val.type
            _stamp_type_and_shape(x2d, (None, int(K)))
            _ensure_value_metadata(ctx, x2d)
            gemm_in = x2d
        # Bias: ensure 1-D [Cout] for Gemm.C, preferring build-time inline reshape
        use_bias = (b_var is not None) and (getattr(b_var, "aval", None) is not None)
        if use_bias:
            b_val = ctx.get_value_for_var(b_var, name_hint=ctx.fresh_name("bias"))
            desired_b_shape = (int(Cout),)

            # If not already 1-D [Cout], inline if constant, else insert runtime Reshape.
            b2d = b_val
            b_aval_shape = tuple(getattr(getattr(b_var, "aval", None), "shape", ()))
            if len(b_aval_shape) != 1 or int(b_aval_shape[0]) != int(Cout):
                b_inline = inline_reshape_initializer(
                    ctx, b_val, desired_b_shape, "bias_vec"
                )
                if b_inline is b_val:
                    bshape_c = ctx.builder.add_initializer_from_array(
                        name=ctx.fresh_name("bshape"),
                        array=np.asarray([int(Cout)], dtype=np.int64),
                    )
                    b2d = ctx.builder.Reshape(
                        b_val,
                        bshape_c,
                        _outputs=[ctx.fresh_name("bias2d")],
                    )
                    if getattr(b_val, "type", None) is not None:
                        b2d.type = b_val.type
                    _stamp_type_and_shape(b2d, desired_b_shape)
                    _ensure_value_metadata(ctx, b2d)
                else:
                    b2d = b_inline

            # Cast AFTER shaping to match JAX's promotion: params → input dtype
            b2d = cast_param_like(ctx, b2d, x_val, "bias_cast")
        else:
            b2d = None

        # Gemm
        gemm_output_name = ctx.fresh_name("gemm_output") if need_flatten else out_name
        if use_bias:
            inputs = [gemm_in, k2d, b2d]
            gemm_out = builder.Gemm(
                *inputs,
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=0,
                _outputs=[gemm_output_name],
            )
        else:
            gemm_out = builder.Gemm(
                gemm_in,
                k2d,
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=0,
                _outputs=[gemm_output_name],
            )
        if getattr(x_val, "type", None) is not None:
            gemm_out.type = x_val.type
        if need_flatten:
            _stamp_type_and_shape(gemm_out, (None, int(Cout)))
            _ensure_value_metadata(ctx, gemm_out)
        else:
            out_aval_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))
            y_meta = _linear_general_output_dims(
                x_val,
                x_shape,
                x_batch_idx,
                gemm_out,
                out_aval_shape,
                [int(v) for v in k_out_dims],
            )
            _stamp_type_and_shape(gemm_out, y_meta)
            _ensure_value_metadata(ctx, gemm_out)
            ctx.bind_value_for_var(y_var, gemm_out)

        if not need_flatten:
            return

        # Reshape back if needed
        x_batch_idx = [i for i in range(len(x_shape)) if i not in lhs_contract]
        batch_dim_vals = [x_shape[i] for i in x_batch_idx]
        all_batch_static = all(_is_static_int(d) for d in batch_dim_vals)

        if all_batch_static:
            final_vals = [int(d) for d in batch_dim_vals] + [int(v) for v in k_out_dims]
            final_shape_c = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("final_shape_c"),
                array=np.asarray(final_vals, dtype=np.int64),
            )

            out_aval_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))
            y_meta = _linear_general_output_dims(
                x_val,
                x_shape,
                x_batch_idx,
                out_spec,
                out_aval_shape,
                [int(v) for v in k_out_dims],
            )
            final_val = ctx.builder.Reshape(
                gemm_out,
                final_shape_c,
                _outputs=[out_name],
            )
            if getattr(out_spec, "type", None) is not None:
                final_val.type = out_spec.type
            _stamp_type_and_shape(final_val, y_meta)
            _ensure_value_metadata(ctx, final_val)
            ctx.bind_value_for_var(y_var, final_val)
            return

        # --- dynamic path: only create Shape/Slice/Concat if a dynamic batch dim exists ---
        shp = ctx.builder.Shape(
            x_val,
            _outputs=[ctx.fresh_name("x_shape")],
        )
        shp.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(shp, (len(x_shape),))
        _ensure_value_metadata(ctx, shp)
        starts = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("slice_starts"),
            array=np.asarray([0], dtype=np.int64),
        )
        ends = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("slice_ends"),
            array=np.asarray([len(x_shape) - len(lhs_contract)], dtype=np.int64),
        )
        batch_dims = ctx.builder.Slice(
            shp,
            starts,
            ends,
            _outputs=[ctx.fresh_name("batch_dims")],
        )
        batch_dims.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(batch_dims, (len(x_shape) - len(lhs_contract),))
        _ensure_value_metadata(ctx, batch_dims)
        of = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("out_features_c"),
            array=np.asarray(k_out_dims, dtype=np.int64),
        )
        # Dynamic path: just reuse the sliced batch vector directly.
        # This matches the “old world” behavior and avoids extra Concat.
        batch_mixed = batch_dims
        final_shape = ctx.builder.Concat(
            batch_mixed,
            of,
            axis=0,
            _outputs=[ctx.fresh_name("final_shape")],
        )
        final_shape.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(final_shape, (len(x_batch_idx) + len(k_out_dims),))
        _ensure_value_metadata(ctx, final_shape)
        out_aval_shape = tuple(getattr(getattr(y_var, "aval", None), "shape", ()))
        y_meta = _linear_general_output_dims(
            x_val,
            x_shape,
            x_batch_idx,
            out_spec,
            out_aval_shape,
            [int(v) for v in k_out_dims],
        )
        final_val = ctx.builder.Reshape(
            gemm_out,
            final_shape,
            _outputs=[out_name],
        )
        if getattr(out_spec, "type", None) is not None:
            final_val.type = out_spec.type
        _stamp_type_and_shape(final_val, y_meta)
        _ensure_value_metadata(ctx, final_val)
        ctx.bind_value_for_var(y_var, final_val)
        # When no flatten was needed, Gemm already wrote directly to y_val, so no extra Reshape.

    # ---------- explicit binding helper for a testcase ----------
    @staticmethod
    def _linear_general(x, kernel, bias, *, dimension_numbers):
        """Direct bind for tests that want to call the primitive without nnx module."""
        return LinearGeneralPlugin._PRIM.bind(
            x, kernel, bias, dimension_numbers=dimension_numbers
        )

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(orig_fn: Callable):
        LinearGeneralPlugin._ORIGINAL_CALL = orig_fn
        prim = LinearGeneralPlugin._PRIM

        def patched(self, x):
            # normalize possibly-negative axes to positive indices
            rank = max(getattr(x, "ndim", len(x.shape)), 1)
            if isinstance(self.axis, int):
                lhs = (self.axis % rank,)
            else:
                lhs = tuple((a % rank) for a in self.axis)
            rhs = tuple(range(len(self.in_features)))  # kernel contracting dims
            dn = ((lhs, rhs), ((), ()))
            kernel = self.kernel.value
            bias = self.bias.value if self.bias is not None else None
            return prim.bind(x, kernel, bias, dimension_numbers=dn)

        return patched

    @classmethod
    def binding_specs(cls):
        return [
            # Make/override flax.nnx.linear_general_p to point to our private Primitive
            AssignSpec(
                "flax.nnx", "linear_general_p", cls._PRIM, delete_if_missing=True
            ),
            # Monkey-patch nnx.LinearGeneral.__call__ while tracing
            MonkeyPatchSpec(
                target="flax.nnx.LinearGeneral",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, kernel, bias, dimension_numbers=None: cls.abstract_eval(
                    x, kernel, bias, dimension_numbers=dimension_numbers
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@LinearGeneralPlugin._PRIM.def_impl
def _impl(x, kernel, bias, *, dimension_numbers):
    y = jax.lax.dot_general(x, kernel, dimension_numbers=dimension_numbers)
    if bias is not None:
        y = y + bias
    return y
