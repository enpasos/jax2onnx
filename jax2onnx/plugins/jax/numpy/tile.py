# jax2onnx/plugins/jax/numpy/tile.py

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr as IRAttr, AttributeType as IRAttrType

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

try:  # pragma: no cover
    from jax._src.export.shape_poly import _DimExpr as DimExpr
except Exception:  # pragma: no cover
    DimExpr = ()  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


def _is_dim_expr(x: object) -> bool:
    if isinstance(DimExpr, tuple):  # fallback if the import above failed
        return False
    return isinstance(x, DimExpr)


_ORIG_TILE = jnp.tile
_TILE_PRIM = make_jnp_primitive("jax.numpy.tile")


def _tile_param(a):
    """Legacy helper: tile a learnable parameter by the batch dimension."""
    b = a.shape[0]
    param = jnp.zeros((1, 1, 4), dtype=a.dtype)
    return jnp.tile(param, (b, 1, 1))


def _tile_with_symbolic_repeats(a):
    """Repeat using the leading dimension for both dynamic/static variants."""
    b = a.shape[0]
    return jnp.tile(a, (b, 1, 1))


@register_primitive(
    jaxpr_primitive=_TILE_PRIM.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tile.html",
    onnx=[
        {"component": "Tile", "doc": "https://onnx.ai/onnx/operators/onnx__Tile.html"}
    ],
    since="v0.8.0",
    context="primitives.jnp",
    component="tile",
    testcases=[
        {
            "testcase": "tile_repeats",
            "callable": lambda a: jnp.tile(a, (3, 1, 1)),
            "input_shapes": [(1, 1, 8)],
        },
        {
            "testcase": "tile_a",
            "callable": lambda a: jnp.tile(a, (1, 2)),
            "input_shapes": [(2, 3)],
        },
        {
            "testcase": "tile_b",
            "callable": lambda a: jnp.tile(a, (1, 2, 1)),
            "input_shapes": [(1, 5, 5)],
        },
        {
            "testcase": "tile_c",
            "callable": lambda a: jnp.tile(a, (1, 4)),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "tile_d",
            "callable": lambda a: jnp.tile(a, 2),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "tile_dynamic_input_static",
            "callable": lambda a: jnp.tile(a, (2, 1)),
            "input_shapes": [(7, 3)],
        },
        {
            "testcase": "tile_dynamic_input",
            "callable": lambda a: jnp.tile(a, (2, 1)),
            "input_shapes": [("B", 3)],
        },
        {
            "testcase": "tile_pad",
            "callable": lambda a: jnp.tile(a, (2, 3, 4)),
            "input_shapes": [(4, 5)],
        },
        {
            "testcase": "tile_param_symbolic",
            "callable": _tile_param,
            "input_shapes": [("B", 5)],
        },
        {
            "testcase": "tile_with_symbolic_repeats_static",
            "callable": _tile_with_symbolic_repeats,
            "input_shapes": [(11, 1, 256)],
        },
        {
            "testcase": "tile_with_symbolic_repeats",
            "callable": _tile_with_symbolic_repeats,
            "input_shapes": [("B", 1, 256)],
        },
        {
            "testcase": "jnp_tile_basic",
            "callable": lambda a: jnp.tile(a, (2, 1)),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "jnp_tile_scalar_repeats",
            "callable": lambda a: jnp.tile(a, 3),
            "input_shapes": [(4,)],
        },
        {
            "testcase": "jnp_tile_pad_rank",
            "callable": lambda a: jnp.tile(a, (2, 3, 4)),
            "input_shapes": [(5, 6)],
        },
        {
            "testcase": "jnp_tile_symbolic",
            "callable": lambda a: jnp.tile(a, (a.shape[0], 1, 1)),
            "input_shapes": [("B", 1, 8)],
        },
    ],
)
class JnpTilePlugin(PrimitiveLeafPlugin):
    """Lower ``jax.numpy.tile`` to ONNX ``Tile``."""

    _PRIM: ClassVar = _TILE_PRIM
    _FUNC_NAME: ClassVar[str] = "tile"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(a, *, repeats, **_):
        # Defer to the original implementation for robust symbolic-shape handling.
        spec = jax.ShapeDtypeStruct(a.shape, a.dtype)
        out = jax.eval_shape(lambda arr: _ORIG_TILE(arr, repeats), spec)
        return jax.core.ShapedArray(out.shape, out.dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        input_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        repeats_var = eqn.invars[1] if len(eqn.invars) > 1 else None
        static_repeats = getattr(eqn, "params", {}).get("repeats")

        input_val = ctx.get_value_for_var(
            input_var, name_hint=ctx.fresh_name("tile_in")
        )
        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("tile_out"))

        input_shape = tuple(getattr(input_var.aval, "shape", ()))
        input_dtype = np.dtype(getattr(input_var.aval, "dtype", np.float32))

        repeats_val: ir.Value
        repeats_rank: int
        if repeats_var is not None:
            repeats_val = ctx.get_value_for_var(
                repeats_var, name_hint=ctx.fresh_name("tile_repeats_dyn")
            )
            repeats_dtype = np.dtype(getattr(repeats_var.aval, "dtype", np.int64))
            if repeats_dtype != np.int64:
                cast_val = ir.Value(
                    name=ctx.fresh_name("tile_repeats_cast"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=repeats_val.shape,
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Cast",
                        domain="",
                        inputs=[repeats_val],
                        outputs=[cast_val],
                        name=ctx.fresh_name("Cast"),
                        attributes=[
                            IRAttr("to", IRAttrType.INT, int(ir.DataType.INT64.value))
                        ],
                    )
                )
                _stamp_type_and_shape(
                    cast_val, tuple(getattr(repeats_var.aval, "shape", ()))
                )
                _ensure_value_info(ctx, cast_val)
                repeats_val = cast_val
            repeats_rank_dim = getattr(repeats_var.aval, "shape", (None,))[0]
            if not isinstance(repeats_rank_dim, int):
                raise ValueError("Dynamic tile repeats must have concrete length")
            repeats_rank = repeats_rank_dim
        elif static_repeats is not None:
            repeats_val, repeats_rank = self._build_static_repeats(ctx, static_repeats)
        else:
            raise ValueError("jnp.tile lowering requires repeats information")

        input_val, repeats_vec = self._align_repeats_rank(
            ctx,
            input_val,
            repeats_val,
            input_shape,
            input_dtype,
            repeats_rank,
        )

        ctx.add_node(
            ir.Node(
                op_type="Tile",
                domain="",
                inputs=[input_val, repeats_vec],
                outputs=[out_val],
                name=ctx.fresh_name("Tile"),
            )
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        out_dtype = _dtype_to_ir(
            np.dtype(getattr(out_var.aval, "dtype", input_dtype)),
            ctx.builder.enable_double_precision,
        )
        out_val.type = ir.TensorType(out_dtype)
        _stamp_type_and_shape(out_val, out_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.tile not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a, reps):
                if isinstance(reps, jax.Array) and not isinstance(
                    reps, jax.core.Tracer
                ):
                    reps = np.asarray(reps)
                if isinstance(reps, np.ndarray):
                    if reps.ndim == 0:
                        return cls._PRIM.bind(a, repeats=(int(reps),))
                    if reps.ndim == 1:
                        return cls._PRIM.bind(a, repeats=tuple(int(x) for x in reps))
                    raise ValueError("tile repeats array must be 0- or 1-D")
                if isinstance(reps, (int, np.integer)):
                    return cls._PRIM.bind(a, repeats=(int(reps),))
                if isinstance(reps, (tuple, list)):
                    return cls._PRIM.bind(a, repeats=tuple(reps))
                return cls._PRIM.bind(a, reps)

            return _patched

        return [
            AssignSpec(
                "jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _shape_vector(
        ctx: "IRContext", cache: dict[ir.Value, ir.Value], src: ir.Value
    ) -> ir.Value:
        cached = cache.get(src)
        if cached is not None:
            return cached
        rank = len(getattr(getattr(src, "shape", None), "dims", ()) or ())
        shape_val = ir.Value(
            name=ctx.fresh_name("tile_shape"),
            type=ir.TensorType(ir.DataType.INT64),
            shape=ir.Shape((rank,)),
        )
        ctx.add_node(
            ir.Node(
                op_type="Shape",
                domain="",
                inputs=[src],
                outputs=[shape_val],
                name=ctx.fresh_name("Shape"),
            )
        )
        cache[src] = shape_val
        return shape_val

    def _build_static_repeats(self, ctx: "IRContext", repeats_param):
        repeats_tuple = (
            (repeats_param,)
            if isinstance(repeats_param, (int, np.integer))
            else tuple(repeats_param)
        )
        origin_getter = getattr(ctx, "get_symbolic_dim_origin", None)

        def _origin(dim):
            if origin_getter is None:
                return None
            return origin_getter(dim) or origin_getter(str(dim))

        shape_cache: dict[ir.Value, ir.Value] = {}
        pieces: list[ir.Value] = []
        for idx, rep in enumerate(repeats_tuple):
            if isinstance(rep, (int, np.integer)):
                pieces.append(_const_i64(ctx, [int(rep)], f"tile_rep_{idx}"))
            elif _is_dim_expr(rep) or isinstance(rep, str):
                origin = _origin(rep)
                if origin is None:
                    raise ValueError(f"Symbolic repeat '{rep}' has no origin")
                src_val, axis = origin
                shape_vec = self._shape_vector(ctx, shape_cache, src_val)
                gather_idx = _const_i64(ctx, [int(axis)], f"tile_sym_idx_{idx}")
                gathered = ir.Value(
                    name=ctx.fresh_name("tile_rep_dyn"),
                    type=ir.TensorType(ir.DataType.INT64),
                    shape=ir.Shape((1,)),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Gather",
                        domain="",
                        inputs=[shape_vec, gather_idx],
                        outputs=[gathered],
                        name=ctx.fresh_name("Gather"),
                        attributes=[IRAttr("axis", IRAttrType.INT, 0)],
                    )
                )
                _stamp_type_and_shape(gathered, (1,))
                _ensure_value_info(ctx, gathered)
                pieces.append(gathered)
            else:
                raise TypeError(f"Unsupported repeats element type: {type(rep)}")

        if len(pieces) == 1:
            repeats_vec = pieces[0]
        else:
            repeats_vec = ir.Value(
                name=ctx.fresh_name("tile_repeats"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((len(pieces),)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Concat",
                    domain="",
                    inputs=pieces,
                    outputs=[repeats_vec],
                    name=ctx.fresh_name("Concat"),
                    attributes=[IRAttr("axis", IRAttrType.INT, 0)],
                )
            )
            _stamp_type_and_shape(repeats_vec, (len(pieces),))
            _ensure_value_info(ctx, repeats_vec)
        repeats_rank = len(pieces)
        return repeats_vec, repeats_rank

    def _align_repeats_rank(
        self,
        ctx: "IRContext",
        input_val: ir.Value,
        repeats_val: ir.Value,
        input_shape: tuple,
        input_np_dtype: np.dtype,
        repeats_rank: int,
    ) -> tuple[ir.Value, ir.Value]:
        aligned = repeats_val
        current_shape = input_shape
        input_rank = len(current_shape)

        if input_rank > repeats_rank:
            pad = _const_i64(
                ctx, np.ones(input_rank - repeats_rank, dtype=np.int64), "tile_pad"
            )
            new_vec = ir.Value(
                name=ctx.fresh_name("tile_repeats_pad"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((input_rank,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Concat",
                    domain="",
                    inputs=[pad, aligned],
                    outputs=[new_vec],
                    name=ctx.fresh_name("Concat"),
                    attributes=[IRAttr("axis", IRAttrType.INT, 0)],
                )
            )
            _stamp_type_and_shape(new_vec, (input_rank,))
            _ensure_value_info(ctx, new_vec)
            aligned = new_vec
            repeats_rank = input_rank

        if repeats_rank > input_rank:
            num_new = repeats_rank - input_rank
            ones_vec = _const_i64(
                ctx, np.ones(num_new, dtype=np.int64), "tile_pre_shape"
            )

            input_shape_val = ir.Value(
                name=ctx.fresh_name("tile_input_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((input_rank,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Shape",
                    domain="",
                    inputs=[input_val],
                    outputs=[input_shape_val],
                    name=ctx.fresh_name("Shape"),
                )
            )

            target_shape = ir.Value(
                name=ctx.fresh_name("tile_target_shape"),
                type=ir.TensorType(ir.DataType.INT64),
                shape=ir.Shape((repeats_rank,)),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Concat",
                    domain="",
                    inputs=[ones_vec, input_shape_val],
                    outputs=[target_shape],
                    name=ctx.fresh_name("Concat"),
                    attributes=[IRAttr("axis", IRAttrType.INT, 0)],
                )
            )
            _stamp_type_and_shape(target_shape, (repeats_rank,))
            _ensure_value_info(ctx, target_shape)

            dtype_enum = _dtype_to_ir(
                input_np_dtype, ctx.builder.enable_double_precision
            )
            new_shape = tuple([1] * num_new + list(current_shape))
            reshaped = ir.Value(
                name=ctx.fresh_name("tile_input_pad"),
                type=ir.TensorType(dtype_enum),
                shape=ir.Shape(new_shape),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Reshape",
                    domain="",
                    inputs=[input_val, target_shape],
                    outputs=[reshaped],
                    name=ctx.fresh_name("Reshape"),
                )
            )
            _stamp_type_and_shape(reshaped, new_shape)
            _ensure_value_info(ctx, reshaped)
            input_val = reshaped
            current_shape = new_shape
            input_rank = repeats_rank

        return input_val, aligned


@JnpTilePlugin._PRIM.def_impl
def _tile_impl(a, *, repeats):
    orig = get_orig_impl(JnpTilePlugin._PRIM, JnpTilePlugin._FUNC_NAME)
    return orig(a, repeats)


JnpTilePlugin._PRIM.def_abstract_eval(JnpTilePlugin.abstract_eval)
