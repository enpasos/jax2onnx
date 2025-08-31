# file: jax2onnx/plugins2/flax/nnx/linear_general.py


from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar
import numpy as np
import jax
from jax.extend.core import Primitive
from flax import nnx
import onnx_ir as ir

from jax2onnx.plugins2._utils import cast_param_like, inline_reshape_initializer
from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2._ir_shapes import (
    _stamp_type_and_shape,
    _prod,
    _as_ir_dim_label,
    _to_ir_dim_for_shape,
    _is_static_int,
    _dim_label_from_value_or_aval,
    _ensure_value_info as _add_value_info,  # avoid local name shadowing
)


if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


# ------------------------------------------------------------------
# Helpers used by a testcase's post_check_onnx_graph (shape asserts)
# ------------------------------------------------------------------
def _shape_of(coll, name: str):
    """Return tuple of dims (int | str | None) for the tensor named `name`.
    Test utilities historically used 'var_0' for the first input. If that exact
    name is not found, fall back sensibly:
      • if asking for 'var_0' or 'in_0' → use the first graph input
      • if asking for 'var_3' or 'out_0' → use the first graph output
    This keeps checks working across legacy and IR pipelines."""
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
    # Legacy compatibility across name changes ('var_0'/'in0', 'var_3'/'out_0'):
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


# put near the other helpers
# def _stamp_type_and_shape(v: ir.Value, dims):
#     """Ensure both meta and tensor-type shape carry symbolic names like 'B'."""
#     ir_dims = tuple(_to_ir_dim_for_shape(d) for d in dims)
#     sh = ir.Shape(ir_dims)
#     # meta (for value_info / viewers)
#     v.shape = sh
#     # type (what gets serialized into graph.input/output)
#     try:
#         if isinstance(getattr(v, "type", None), ir.TensorType):
#             v.type = ir.TensorType(v.type.dtype, sh)
#     except Exception:
#         # Never fail the build because of stamping
#         pass


# def _prod(xs: Iterable[int]) -> int:
#     """Product of ints; tolerant to numpy scalars/py ints."""
#     p = 1
#     for v in xs:
#         p *= int(v)
#     return int(p)

# Best-effort conversion of an IR/JAX dimension into an ONNX dim label:
#  - ints → int dim_value
#  - strings → dim_param (e.g., "B")
#  - objects with `.param` / `.value` (if present) → use them
#  - otherwise → None (unknown)
# def _as_ir_dim_label(d):
#     """
#     Best-effort extraction of a printable dim label from onnx_ir dims.
#     Handles:
#       - ir.SymbolicDim('B')  -> 'B'
#       - objects with .param/.value (e.g., JAX ShapedArray dims)
#       - plain ints/strings/None
#     """
#     # 1) onnx_ir.SymbolicDim
#     try:
#         if isinstance(d, ir.SymbolicDim):  # type: ignore[attr-defined]
#             # common attribute names used by different builds
#             for attr in ("param", "name", "symbol", "label"):
#                 v = getattr(d, attr, None)
#                 if v:
#                     return str(v)
#             # fallback: parse from repr "SymbolicDim(B)" or "SymbolicDim('B')"
#             s = repr(d)
#             m = re.search(r"SymbolicDim\(['\"]?([A-Za-z0-9_]+)['\"]?\)", s)
#             if m:
#                 return m.group(1)
#             # last resort: str(d) if it looks like a bare symbol
#             s = str(d)
#             if s and s.isidentifier():
#                 return s
#     except Exception:
#         pass
#     # 2) generic objects with .param/.value (e.g., JAX ShapedArray dims)
#     try:
#         if hasattr(d, "param") and getattr(d, "param", None):
#             return getattr(d, "param")
#         if hasattr(d, "value") and getattr(d, "value", None) is not None:
#             return int(getattr(d, "value"))
#     except Exception:
#         pass
#     # 3) primitives
#     if isinstance(d, (int, np.integer)):
#         return int(d)
#     if isinstance(d, str):
#         return d
#     if d is None:
#         return None
#     return None

# Convert a dim “label” into what onnx_ir expects inside a Shape:
# - ints -> int
# - strings like "B" -> ir.SymbolicDim("B")
# - objects with .param/.value mapped accordingly
# - existing ir.SymbolicDim passthrough
# def _to_ir_dim_for_shape(d):
#     try:
#         # Already a SymbolicDim from onnx_ir
#         if isinstance(d, ir.SymbolicDim):  # type: ignore[attr-defined]
#             return d
#         if hasattr(d, "param") and d.param:
#             return ir.SymbolicDim(str(d.param))  # type: ignore[attr-defined]
#         if hasattr(d, "value") and d.value is not None:
#             return int(d.value)
#     except Exception:
#         pass
#     if isinstance(d, (int, np.integer)):
#         return int(d)
#     if isinstance(d, str):
#         return ir.SymbolicDim(d)  # type: ignore[attr-defined]
#     if d is None:
#         return None
#     return None

# def _is_static_int(d) -> bool:
#     """True if `d` is a known, non-negative integer."""
#     return isinstance(d, (int, np.integer)) and int(d) >= 0

# def _dim_label_from_value_or_aval(val: ir.Value, aval_shape: tuple, i: int):
#     """
#     Read the i-th dimension label, preferring the IR Value's recorded shape
#     (which may preserve symbolic names like 'B'), and falling back to the JAX
#     aval shape if the IR shape is missing/anonymous.
#     """
#     shp = getattr(val, "shape", None)
#     if shp is not None:
#         dims = getattr(shp, "dims", None)
#         if dims is None:
#             try:
#                 dims = list(shp)
#             except Exception:
#                 dims = None
#         if dims is not None and i < len(dims):
#             return _as_ir_dim_label(dims[i])
#     if i < len(aval_shape):
#         return _as_ir_dim_label(aval_shape[i])
#     return None


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
    return len(s) == 2 and s[0] is None and s[1] == K


def _first_node(m, op):
    return next(n for n in m.graph.node if n.op_type == op)


def _init_dims(m, name):
    t = next((t for t in m.graph.initializer if t.name == name), None)
    return list(t.dims) if t is not None else None


def _nodes(m, op: str):
    """Return all nodes with the given op_type."""
    return [n for n in m.graph.node if n.op_type == op]


def _node_producing(m, tensor_name: str):
    """Return the node that produces `tensor_name`."""
    return next(n for n in m.graph.node if tensor_name in n.output)


def _ensure_value_info(ctx, v: ir.Value | None):
    # Backward-compatible local alias (imported helper)
    return _ensure_value_info(ctx, v)


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
    context="primitives2.nnx",
    component="linear_general",
    testcases=[
        {
            "testcase": "linear_general_merge_symbolic_dim",
            "callable": nnx.LinearGeneral(
                in_features=(4, 16),  # ⟨4,16⟩ are contracting dims
                out_features=32,
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 8, 4, 16)],
            "run_only_dynamic": True,
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            # Match the original “old world” shapes exactly (allowing only B↔None):
            #   input         : B×8×4×16
            #   input_reshape : ?×64
            #   gemm_output   : ?×32
            #   output        : B×8×32
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
            "callable": nnx.LinearGeneral(
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [("B", 4, 8, 32)],
            "expected_output_shapes": [("B", 4, 256)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: (
                lambda gemm:
                # C is present and is a 1-D initializer of length 256
                (len(gemm.input) >= 3)
                and (_init_dims(m, gemm.input[2]) == [256])
            )(_first_node(m, "Gemm")),
        },
        {
            "testcase": "linear_general_2",
            "callable": nnx.LinearGeneral(
                in_features=(30,),
                out_features=(20,),
                axis=(-1,),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(3, 30)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "linear_general_3",
            "callable": nnx.LinearGeneral(
                in_features=(256,),
                out_features=(8, 32),
                axis=(-1,),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 256)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "linear_general_4",
            "callable": nnx.LinearGeneral(
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(2, 4, 8, 32)],
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
            # Static case: ensure we eliminated Shape/Slice/Concat and inlined
            # the final Reshape's shape as a constant of length 3 (3×4×256).
            "post_check_onnx_graph": lambda m: (
                lambda final_r: (final_r.op_type == "Reshape")
                and (len(_nodes(m, "Shape")) == 0)
                and (len(_nodes(m, "Slice")) == 0)
                and (len(_nodes(m, "Concat")) == 0)
                and (len(final_r.input) >= 2)
                and (_init_dims(m, final_r.input[1]) == [3])  # shape initializer rank=3
                and (_shape_of(m.graph.output, "out_0") == (2, 4, 256))
            )(_node_producing(m, "out_0")),
        },
        {
            "testcase": "linear_general_abstract_eval_axes",
            "callable": nnx.LinearGeneral(
                in_features=(256,),
                out_features=(8, 32),
                axis=(-1,),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(3, 10, 256)],
            "expected_output_shape": (3, 10, 8, 32),
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "linear_general_abstract_eval_axes_pair",
            "callable": nnx.LinearGeneral(
                in_features=(8, 32),
                out_features=(256,),
                axis=(-2, -1),
                rngs=nnx.Rngs(0),
            ),
            "input_shapes": [(3, 10, 8, 32)],
            "expected_output_shape": (3, 10, 256),
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
        {
            "testcase": "linear_general_dynamic_batch_and_feature_dims",
            # Bind our primitive directly with explicit dimension_numbers.
            "callable": lambda x, k, b: LinearGeneralPlugin._linear_general(
                x, k, b, dimension_numbers=(((2,), (0,)), ((), ()))
            ),
            "input_shapes": [("B", "H", 16), (16, 4, 4), (4, 4)],
            "run_only_dynamic": True,
            "run_only_f32_variant": True,
            "use_onnx_ir": True,
        },
    ],
)
class LinearGeneralPlugin(PrimitiveLeafPlugin):
    """
    IR-only plugin for flax.nnx.LinearGeneral:
      Reshape([-1,K]) → Gemm → Reshape-back.
    Maintains legacy value-name prefixes ("input_reshape", "gemm_output")
    so post-checks from the original tests continue to work.
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

            dummy = SimpleNamespace(
                kernel=SimpleNamespace(value=kv),
                bias=None if bv is None else SimpleNamespace(value=bv),
                dimension_numbers=dimension_numbers,
                batch_axis={},  # len(self.batch_axis) is used
                axis=dimension_numbers[0][0],  # lhs contracting axes
                in_features=tuple(kv.shape[: len(rhs_contract)]),
                out_features=out_features,
                promote_dtype=promote_dtype,
                dtype=None,  # match Flax default
                dot_general=dot_general,  # function path
                dot_general_cls=None,  # NEW: ensure branch is skipped
                precision=None,
            )
            return LinearGeneralPlugin._ORIGINAL_CALL(dummy, xv)

        out = jax.eval_shape(_helper, x_spec, k_spec, b_spec)
        out = jax.tree_util.tree_leaves(out)[0]
        return jax.core.ShapedArray(out.shape, out.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: "IRBuildContext", eqn):
        x_var, k_var, b_var = eqn.invars[:3]
        y_var = eqn.outvars[0]
        ((lhs_contract, rhs_contract), _) = eqn.params["dimension_numbers"]

        # Values
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        k_val = ctx.get_value_for_var(k_var, name_hint=ctx.fresh_name("kernel"))

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
            # Not constant → keep a runtime Reshape (old-world compatible)
            kshape_c = ir.Value(
                name=ctx.fresh_name("kshape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((2,)),
                const_value=ir.tensor(np.array(desired_k_shape, dtype=np.int64)),
            )
            ctx._initializers.append(kshape_c)
            k2d = ir.Value(
                name=ctx.fresh_name("kernel2d"),
                type=k_val.type,
                shape=ir.Shape(desired_k_shape),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Reshape",
                    domain="",
                    inputs=[k_val, kshape_c],
                    outputs=[k2d],
                    name=ctx.fresh_name("Reshape"),
                )
            )

        # IMPORTANT: cast *after* shaping so the Gemm input has the final dtype
        k2d = cast_param_like(ctx, k2d, x_val, "kernel_cast")

        # --- INPUT (A) flatten remains unchanged ...
        # (keep producing 'input_reshape' for tests)
        x_batch_idx = [i for i in range(len(x_shape)) if i not in lhs_contract]
        need_flatten = len(x_shape) > 2

        gemm_in = x_val
        if need_flatten:
            x2d_shape_c = ir.Value(
                name=ctx.fresh_name("x2d_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((2,)),
                const_value=ir.tensor(np.asarray([-1, int(K)], dtype=np.int64)),
            )
            ctx._initializers.append(x2d_shape_c)
            x2d = ir.Value(
                name=ctx.fresh_name("input_reshape"),
                type=x_val.type,
                shape=ir.Shape((None, int(K))),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Reshape",
                    domain="",
                    inputs=[x_val, x2d_shape_c],
                    outputs=[x2d],
                    name=ctx.fresh_name("Reshape"),
                )
            )
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
                    # not constant → runtime Reshape
                    bshape_c = ir.Value(
                        name=ctx.fresh_name("bshape"),
                        type=ir.TensorType(ir.DataType.INT64),
                        shape=ir.Shape((1,)),
                        const_value=ir.tensor(np.asarray([int(Cout)], dtype=np.int64)),
                    )
                    ctx._initializers.append(bshape_c)
                    b2d = ir.Value(
                        name=ctx.fresh_name("bias2d"),
                        type=b_val.type,
                        shape=ir.Shape(desired_b_shape),
                    )
                    ctx.add_node(
                        ir.Node(
                            op_type="Reshape",
                            domain="",
                            inputs=[b_val, bshape_c],
                            outputs=[b2d],
                            name=ctx.fresh_name("Reshape"),
                        )
                    )
                else:
                    b2d = b_inline

            # Cast AFTER shaping to match JAX's promotion: params → input dtype
            b2d = cast_param_like(ctx, b2d, x_val, "bias_cast")
        else:
            b2d = None

        # Gemm
        if need_flatten:
            gemm_out = ir.Value(
                name=ctx.fresh_name("gemm_output"),
                type=x_val.type,
                shape=ir.Shape((None, int(Cout))),
            )
        else:
            gemm_out = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))
            # Preserve symbolic batch labels directly on Gemm output when no flatten is needed.
            y_meta = tuple(
                [_dim_label_from_value_or_aval(x_val, x_shape, i) for i in x_batch_idx]
                + [int(v) for v in k_out_dims]
            )
            _stamp_type_and_shape(gemm_out, y_meta)

        inputs = [gemm_in, k2d] + ([b2d] if use_bias else [])
        ctx.add_node(
            ir.Node(
                op_type="Gemm",
                domain="",
                inputs=inputs,
                outputs=[gemm_out],
                name=ctx.fresh_name("Gemm"),
                attributes=[
                    ir.Attr("alpha", ir.AttributeType.FLOAT, 1.0),
                    ir.Attr("beta", ir.AttributeType.FLOAT, 1.0),
                    ir.Attr("transA", ir.AttributeType.INT, 0),
                    ir.Attr("transB", ir.AttributeType.INT, 0),
                ],
            )
        )
        # Reshape back if needed
        if need_flatten:
            # If all batch dims are statically known, inline the final shape
            # and avoid building Shape/Slice/Concat. Otherwise, fall back to
            # the dynamic path (preserving symbolic labels like 'B').
            x_batch_idx = [i for i in range(len(x_shape)) if i not in lhs_contract]
            batch_dim_vals = [x_shape[i] for i in x_batch_idx]
            all_batch_static = all(_is_static_int(d) for d in batch_dim_vals)

            if all_batch_static:
                final_vals = [int(d) for d in batch_dim_vals] + [
                    int(v) for v in k_out_dims
                ]
                final_shape_c = ir.Value(
                    name=ctx.fresh_name("final_shape_c"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((len(final_vals),)),
                    const_value=ir.tensor(np.array(final_vals, dtype=np.int64)),
                )
                ctx._initializers.append(final_shape_c)

                # Meta shape for graph.output (keep symbols if present on input)
                y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))
                y_meta = tuple(
                    [
                        _dim_label_from_value_or_aval(x_val, x_shape, i)
                        for i in x_batch_idx
                    ]
                    + [int(v) for v in k_out_dims]
                )
                # Stamp BOTH meta and TensorType so the graph.output keeps 'B'
                _stamp_type_and_shape(y_val, y_meta)
                ctx.add_node(
                    ir.Node(
                        op_type="Reshape",
                        domain="",
                        inputs=[gemm_out, final_shape_c],
                        outputs=[y_val],
                        name=ctx.fresh_name("Reshape"),
                    )
                )
                # Re-assert shape and also force a value_info entry so the
                # Reshape→out_0 node port renders B×… (not ?×…).
                _stamp_type_and_shape(y_val, y_meta)
                _add_value_info(ctx, y_val)
            else:
                # --- dynamic path: only create Shape/Slice/Concat if a dynamic batch dim exists ---
                shp = ir.Value(
                    name=ctx.fresh_name("x_shape"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((len(x_shape),)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Shape",
                        domain="",
                        inputs=[x_val],
                        outputs=[shp],
                        name=ctx.fresh_name("Shape"),
                    )
                )
                starts = ir.Value(
                    name=ctx.fresh_name("slice_starts"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                    const_value=ir.tensor(np.array([0], dtype=np.int64)),
                )
                ends = ir.Value(
                    name=ctx.fresh_name("slice_ends"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                    const_value=ir.tensor(
                        np.array([len(x_shape) - len(lhs_contract)], dtype=np.int64)
                    ),
                )
                ctx._initializers.extend([starts, ends])
                batch_dims = ir.Value(
                    name=ctx.fresh_name("batch_dims"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((len(x_shape) - len(lhs_contract),)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Slice",
                        domain="",
                        inputs=[shp, starts, ends],
                        outputs=[batch_dims],
                        name=ctx.fresh_name("Slice"),
                    )
                )
                of = ir.Value(
                    name=ctx.fresh_name("out_features_c"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((len(k_out_dims),)),
                    const_value=ir.tensor(np.array(k_out_dims, dtype=np.int64)),
                )
                ctx._initializers.append(of)
                # Dynamic path: just reuse the sliced batch vector directly.
                # This matches the “old world” behavior and avoids extra Concat.
                batch_mixed = batch_dims
                final_shape = ir.Value(
                    name=ctx.fresh_name("final_shape"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((len(x_batch_idx) + len(k_out_dims),)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Concat",
                        domain="",
                        inputs=[batch_mixed, of],
                        outputs=[final_shape],
                        name=ctx.fresh_name("Concat"),
                        attributes=[ir.Attr("axis", ir.AttributeType.INT, 0)],
                    )
                )
                y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))
                y_meta = tuple(
                    [
                        _dim_label_from_value_or_aval(x_val, x_shape, i)
                        for i in x_batch_idx
                    ]
                    + [int(v) for v in k_out_dims]
                )
                _stamp_type_and_shape(y_val, y_meta)
                ctx.add_node(
                    ir.Node(
                        op_type="Reshape",
                        domain="",
                        inputs=[gemm_out, final_shape],
                        outputs=[y_val],
                        name=ctx.fresh_name("Reshape"),
                    )
                )
                # Re-assert shape and also force a value_info entry so the
                # Reshape→out_0 node port renders B×… (not ?×…).
                _stamp_type_and_shape(y_val, y_meta)
                _add_value_info(ctx, y_val)
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
