# jax2onnx/plugins/jax/numpy/tile.py

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
import onnx_ir as ir
from jax.interpreters import batching

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import (
    LoweringContextProtocol,
    SymbolicDimOrigin,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.utils.shape_poly import is_dim_expr


_ORIG_TILE: Final = jnp.tile
_TILE_PRIM: Final = make_jnp_primitive("jax.numpy.tile")


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
            "post_check_onnx_graph": EG(
                ["Tile:3x1x8"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tile_a",
            "callable": lambda a: jnp.tile(a, (1, 2)),
            "input_shapes": [(2, 3)],
            "post_check_onnx_graph": EG(
                ["Tile:2x6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tile_b",
            "callable": lambda a: jnp.tile(a, (1, 2, 1)),
            "input_shapes": [(1, 5, 5)],
            "post_check_onnx_graph": EG(
                ["Tile:1x10x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tile_c",
            "callable": lambda a: jnp.tile(a, (1, 4)),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["Tile:3x12"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tile_d",
            "callable": lambda a: jnp.tile(a, 2),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["Tile:3x6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tile_dynamic_input_static",
            "callable": lambda a: jnp.tile(a, (2, 1)),
            "input_shapes": [(7, 3)],
            "post_check_onnx_graph": EG(
                ["Tile:14x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tile_dynamic_input",
            "callable": lambda a: jnp.tile(a, (2, 1)),
            "input_shapes": [("B", 3)],
            "post_check_onnx_graph": EG(
                ["Tile:Bx3"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tile_pad",
            "callable": lambda a: jnp.tile(a, (2, 3, 4)),
            "input_shapes": [(4, 5)],
            "post_check_onnx_graph": EG(
                ["Reshape:1x4x5 -> Tile:2x12x20"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tile_param_symbolic",
            "callable": _tile_param,
            "input_shapes": [("B", 5)],
            "post_check_onnx_graph": EG(
                ["Concat -> Tile:Bx1x4"],
                symbols={"B": None},
            ),
        },
        {
            "testcase": "tile_with_symbolic_repeats_static",
            "callable": _tile_with_symbolic_repeats,
            "input_shapes": [(11, 1, 256)],
            "post_check_onnx_graph": EG(
                ["Tile:121x1x256"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tile_with_symbolic_repeats",
            "callable": _tile_with_symbolic_repeats,
            "input_shapes": [("B", 1, 256)],
            "post_check_onnx_graph": EG(
                ["Tile:Bx1x256"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_tile_basic",
            "callable": lambda a: jnp.tile(a, (2, 1)),
            "input_shapes": [(3, 4)],
            "post_check_onnx_graph": EG(
                ["Tile:6x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_tile_scalar_repeats",
            "callable": lambda a: jnp.tile(a, 3),
            "input_shapes": [(4,)],
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 3.0}},
                        "path": "Tile:12",
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_tile_pad_rank",
            "callable": lambda a: jnp.tile(a, (2, 3, 4)),
            "input_shapes": [(5, 6)],
            "post_check_onnx_graph": EG(
                ["Reshape:1x5x6 -> Tile:2x15x24"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_tile_symbolic",
            "callable": lambda a: jnp.tile(a, (a.shape[0], 1, 1)),
            "input_shapes": [("B", 1, 8)],
            "post_check_onnx_graph": EG(
                ["Tile:Bx1x8"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "tile_vmap_batching",
            "callable": lambda x: jax.vmap(lambda y: jnp.tile(y, 2))(x),
            "input_shapes": [(3, 4)],
        },
    ],
)
class JnpTilePlugin(PrimitiveLeafPlugin):
    """Lower ``jax.numpy.tile`` to ONNX ``Tile``."""

    _PRIM: ClassVar = _TILE_PRIM
    _FUNC_NAME: ClassVar[str] = "tile"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        a: core.AbstractValue,
        *,
        repeats: ArrayLike,
        **_: object,
    ) -> core.ShapedArray:
        # Defer to the original implementation for robust symbolic-shape handling.
        spec = jax.ShapeDtypeStruct(a.shape, a.dtype)
        out = jax.eval_shape(lambda arr: _ORIG_TILE(arr, repeats), spec)
        return jax.core.ShapedArray(out.shape, out.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
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
                cast_val = ctx.builder.Cast(
                    repeats_val,
                    _outputs=[ctx.fresh_name("tile_repeats_cast")],
                    to=int(ir.DataType.INT64.value),
                )
                _stamp_type_and_shape(
                    cast_val, tuple(getattr(repeats_var.aval, "shape", ()))
                )
                cast_val.type = ir.TensorType(ir.DataType.INT64)
                _ensure_value_metadata(ctx, cast_val)
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

        desired_name = getattr(out_val, "name", None) or ctx.fresh_name("Tile")
        producer = getattr(out_val, "producer", lambda: None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("Tile")

        result = ctx.builder.Tile(
            input_val,
            repeats_vec,
            _outputs=[desired_name],
        )

        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        out_dtype = _dtype_to_ir(
            np.dtype(getattr(out_var.aval, "dtype", input_dtype)),
            ctx.builder.enable_double_precision,
        )
        result.type = ir.TensorType(out_dtype)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.tile not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(a: ArrayLike, reps: ArrayLike) -> jax.Array:
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
        ctx: LoweringContextProtocol, cache: dict[ir.Value, ir.Value], src: ir.Value
    ) -> ir.Value:
        cached = cache.get(src)
        if cached is not None:
            return cached
        rank = len(getattr(getattr(src, "shape", None), "dims", ()) or ())
        shape_val = ctx.builder.Shape(src, _outputs=[ctx.fresh_name("tile_shape")])
        shape_val.type = ir.TensorType(ir.DataType.INT64)
        _stamp_type_and_shape(shape_val, (rank,))
        _ensure_value_metadata(ctx, shape_val)
        cache[src] = shape_val
        return shape_val

    def _build_static_repeats(
        self,
        ctx: LoweringContextProtocol,
        repeats_param: Sequence[int | np.integer] | int | np.integer,
    ) -> tuple[ir.Value, int]:
        repeats_tuple = (
            (repeats_param,)
            if isinstance(repeats_param, (int, np.integer))
            else tuple(repeats_param)
        )
        origin_getter = getattr(ctx, "get_symbolic_dim_origin", None)

        def _origin(dim) -> SymbolicDimOrigin | None:
            return SymbolicDimOrigin.resolve(origin_getter, dim)

        shape_cache: dict[ir.Value, ir.Value] = {}
        pieces: list[ir.Value] = []
        for idx, rep in enumerate(repeats_tuple):
            if isinstance(rep, (int, np.integer)):
                pieces.append(_const_i64(ctx, [int(rep)], f"tile_rep_{idx}"))
            elif is_dim_expr(rep) or isinstance(rep, str):
                origin = _origin(rep)
                if origin is None:
                    raise ValueError(f"Symbolic repeat '{rep}' has no origin")
                shape_vec = self._shape_vector(ctx, shape_cache, origin.value)
                gather_idx = _const_i64(ctx, [int(origin.axis)], f"tile_sym_idx_{idx}")
                gathered = ctx.builder.Gather(
                    shape_vec,
                    gather_idx,
                    axis=0,
                    _outputs=[ctx.fresh_name("tile_rep_dyn")],
                )
                _stamp_type_and_shape(gathered, (1,))
                gathered.type = ir.TensorType(ir.DataType.INT64)
                _ensure_value_metadata(ctx, gathered)
                pieces.append(gathered)
            else:
                raise TypeError(f"Unsupported repeats element type: {type(rep)}")

        if len(pieces) == 1:
            repeats_vec = pieces[0]
        else:
            repeats_vec = ctx.builder.Concat(
                *pieces,
                axis=0,
                _outputs=[ctx.fresh_name("tile_repeats")],
            )
            repeats_vec.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(repeats_vec, (len(pieces),))
            _ensure_value_metadata(ctx, repeats_vec)
        repeats_rank = len(pieces)
        return repeats_vec, repeats_rank

    def _align_repeats_rank(
        self,
        ctx: LoweringContextProtocol,
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
            new_vec = ctx.builder.Concat(
                pad,
                aligned,
                axis=0,
                _outputs=[ctx.fresh_name("tile_repeats_pad")],
            )
            new_vec.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(new_vec, (input_rank,))
            _ensure_value_metadata(ctx, new_vec)
            aligned = new_vec
            repeats_rank = input_rank

        if repeats_rank > input_rank:
            num_new = repeats_rank - input_rank
            ones_vec = _const_i64(
                ctx, np.ones(num_new, dtype=np.int64), "tile_pre_shape"
            )

            input_shape_val = ctx.builder.Shape(
                input_val, _outputs=[ctx.fresh_name("tile_input_shape")]
            )
            input_shape_val.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(input_shape_val, (input_rank,))
            _ensure_value_metadata(ctx, input_shape_val)

            target_shape = ctx.builder.Concat(
                ones_vec,
                input_shape_val,
                axis=0,
                _outputs=[ctx.fresh_name("tile_target_shape")],
            )
            target_shape.type = ir.TensorType(ir.DataType.INT64)
            _stamp_type_and_shape(target_shape, (repeats_rank,))
            _ensure_value_metadata(ctx, target_shape)

            dtype_enum = _dtype_to_ir(
                input_np_dtype, ctx.builder.enable_double_precision
            )
            new_shape = tuple([1] * num_new + list(current_shape))
            reshaped = ctx.builder.Reshape(
                input_val,
                target_shape,
                _outputs=[ctx.fresh_name("tile_input_pad")],
            )
            reshaped.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(reshaped, new_shape)
            _ensure_value_metadata(ctx, reshaped)
            input_val = reshaped
            current_shape = new_shape
            input_rank = repeats_rank

        return input_val, aligned


@JnpTilePlugin._PRIM.def_impl
def _tile_impl(a, *, repeats):
    orig = get_orig_impl(JnpTilePlugin._PRIM, JnpTilePlugin._FUNC_NAME)
    return orig(a, repeats)


JnpTilePlugin._PRIM.def_abstract_eval(JnpTilePlugin.abstract_eval)


BatchDim = int | type(batching.not_mapped)


def _tile_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    repeats: ArrayLike,
    **_: object,
) -> tuple[jax.Array, BatchDim]:
    if len(batched_args) != 1:
        raise NotImplementedError(
            "vmap batching for dynamic tile repeats is not supported"
        )

    (operand,), (bdim,) = batched_args, batch_dims
    repeats_tuple = tuple(repeats)

    if bdim is batching.not_mapped:
        out = JnpTilePlugin._PRIM.bind(operand, repeats=repeats_tuple)
        return out, batching.not_mapped

    batch_size = operand.shape[bdim]
    operand = batching.bdim_at_front(operand, bdim, batch_size)

    slice_rank = operand.ndim - 1
    repeats_rank = len(repeats_tuple)

    if repeats_rank > slice_rank:
        extra_axes = repeats_rank - slice_rank
        new_shape = (operand.shape[0],) + (1,) * extra_axes + operand.shape[1:]
        operand = jnp.reshape(operand, new_shape)
        repeats_full = (1,) + repeats_tuple
    else:
        padded = (1,) * (slice_rank - repeats_rank) + repeats_tuple
        repeats_full = (1,) + padded

    out = JnpTilePlugin._PRIM.bind(operand, repeats=repeats_full)
    return out, 0


batching.primitive_batchers[JnpTilePlugin._PRIM] = _tile_batch_rule
