# jax2onnx/plugins/dm_pix/space_to_depth.py

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any, ClassVar, Final, cast

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax import core
from jax.extend.core import Primitive
from jax.interpreters import batching
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

pix: ModuleType | None
try:
    import dm_pix as pix
except Exception:  # pragma: no cover - optional dependency
    pix = cast(ModuleType | None, None)


def _make_dm_pix_primitive(name: str) -> Primitive:
    prim = Primitive(name)
    prim.multiple_results = False
    return prim


_SPACE_TO_DEPTH_PRIM: Final = _make_dm_pix_primitive("dm_pix.space_to_depth")


def _check_block_size(block_size: int) -> int:
    bs = int(block_size)
    if bs < 2:
        raise ValueError("dm_pix.space_to_depth requires block_size >= 2")
    return bs


def _space_to_depth_impl(
    inputs: ArrayLike,
    *,
    block_size: int,
) -> jax.Array:
    bs = _check_block_size(block_size)
    x = jnp.asarray(inputs)
    rank = int(x.ndim)

    if rank == 3:
        h, w, c = x.shape
        if isinstance(h, (int, np.integer)) and int(h) % bs != 0:
            raise ValueError(
                f"Input height {int(h)} must be divisible by block_size ({bs})."
            )
        if isinstance(w, (int, np.integer)) and int(w) % bs != 0:
            raise ValueError(
                f"Input width {int(w)} must be divisible by block_size ({bs})."
            )
        h_out = h // bs
        w_out = w // bs
        y = jnp.reshape(x, (h_out, bs, w_out, bs, c))
        y = jnp.transpose(y, (0, 2, 1, 3, 4))
        return jnp.reshape(y, (h_out, w_out, c * bs * bs))

    if rank == 4:
        n, h, w, c = x.shape
        if isinstance(h, (int, np.integer)) and int(h) % bs != 0:
            raise ValueError(
                f"Input height {int(h)} must be divisible by block_size ({bs})."
            )
        if isinstance(w, (int, np.integer)) and int(w) % bs != 0:
            raise ValueError(
                f"Input width {int(w)} must be divisible by block_size ({bs})."
            )
        h_out = h // bs
        w_out = w // bs
        y = jnp.reshape(x, (n, h_out, bs, w_out, bs, c))
        y = jnp.transpose(y, (0, 1, 3, 2, 4, 5))
        return jnp.reshape(y, (n, h_out, w_out, c * bs * bs))

    raise ValueError(
        f"dm_pix.space_to_depth expects rank 3 or 4, received rank {rank}."
    )


_TESTCASES: list[dict[str, Any]] = (
    []
    if pix is None
    else [
        {
            "testcase": "space_to_depth_nhwc",
            "callable": lambda x: pix.space_to_depth(x, 2),
            "input_shapes": [(1, 8, 8, 1)],
            "expected_output_shapes": [(1, 4, 4, 4)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Transpose:1x1x8x8 -> SpaceToDepth:1x4x4x4 -> Transpose:1x4x4x4"],
                no_unused_inputs=True,
            ),
        }
    ]
)


@register_primitive(
    jaxpr_primitive=_SPACE_TO_DEPTH_PRIM.name,
    jax_doc="https://dm-pix.readthedocs.io/en/latest/api.html#space-to-depth",
    onnx=[
        {
            "component": "SpaceToDepth",
            "doc": "https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html",
        }
    ],
    since="0.12.1",
    context="primitives.dm_pix",
    component="space_to_depth",
    testcases=_TESTCASES,
)
class DmPixSpaceToDepthPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _SPACE_TO_DEPTH_PRIM
    _FUNC_NAME: ClassVar[str] = "space_to_depth"

    @staticmethod
    def abstract_eval(
        x: core.AbstractValue,
        *,
        block_size: int,
    ) -> core.ShapedArray:
        bs = _check_block_size(block_size)
        spec = jax.ShapeDtypeStruct(getattr(x, "shape", ()), getattr(x, "dtype", None))
        result = jax.eval_shape(
            lambda arr: _space_to_depth_impl(arr, block_size=bs),
            spec,
        )
        return core.ShapedArray(result.shape, result.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        params = dict(getattr(eqn, "params", {}))
        block_size_param = params.get("block_size")
        if block_size_param is None:
            raise KeyError("dm_pix.space_to_depth lowering missing 'block_size'")
        block_size = _check_block_size(int(block_size_param))

        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        x_shape = tuple(getattr(x_var.aval, "shape", ()))
        out_shape = tuple(getattr(out_var.aval, "shape", ()))
        rank = len(x_shape)
        if rank not in (3, 4):
            raise NotImplementedError(
                f"dm_pix.space_to_depth lowering supports rank 3/4, received rank {rank}"
            )

        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for dm_pix.space_to_depth lowering"
            )

        x_val = ctx.get_value_for_var(
            x_var, name_hint=ctx.fresh_name("space_to_depth_in")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("space_to_depth_out")
        )
        x_dtype = getattr(getattr(x_val, "type", None), "dtype", None)

        nhwc4 = x_val
        nhwc4_shape: tuple[object, ...]
        if rank == 3:
            unsq_axes = _const_i64(
                ctx,
                np.asarray([0], dtype=np.int64),
                "space_to_depth_unsqueeze_axes",
            )
            nhwc4 = builder.Unsqueeze(
                x_val,
                unsq_axes,
                _outputs=[ctx.fresh_name("space_to_depth_nhwc4")],
            )
            if x_dtype is not None:
                nhwc4.type = ir.TensorType(x_dtype)
            nhwc4_shape = (1, *x_shape)
            _stamp_type_and_shape(nhwc4, nhwc4_shape)
            _ensure_value_metadata(ctx, nhwc4)
        else:
            nhwc4_shape = x_shape

        to_nchw = builder.Transpose(
            nhwc4,
            perm=[0, 3, 1, 2],
            _outputs=[ctx.fresh_name("space_to_depth_to_nchw")],
        )
        if x_dtype is not None:
            to_nchw.type = ir.TensorType(x_dtype)
        _stamp_type_and_shape(
            to_nchw,
            (
                nhwc4_shape[0],
                nhwc4_shape[3],
                nhwc4_shape[1],
                nhwc4_shape[2],
            ),
        )
        _ensure_value_metadata(ctx, to_nchw)

        s2d_nchw = builder.SpaceToDepth(
            to_nchw,
            blocksize=int(block_size),
            _outputs=[ctx.fresh_name("SpaceToDepth")],
        )
        if x_dtype is not None:
            s2d_nchw.type = ir.TensorType(x_dtype)
        out_nhwc4_shape = out_shape if rank == 4 else (1, *out_shape)
        _stamp_type_and_shape(
            s2d_nchw,
            (
                out_nhwc4_shape[0],
                out_nhwc4_shape[3],
                out_nhwc4_shape[1],
                out_nhwc4_shape[2],
            ),
        )
        _ensure_value_metadata(ctx, s2d_nchw)

        out_nhwc4 = builder.Transpose(
            s2d_nchw,
            perm=[0, 2, 3, 1],
            _outputs=[ctx.fresh_name("space_to_depth_to_nhwc")],
        )
        if x_dtype is not None:
            out_nhwc4.type = ir.TensorType(x_dtype)
        _stamp_type_and_shape(out_nhwc4, out_nhwc4_shape)
        _ensure_value_metadata(ctx, out_nhwc4)

        if rank == 3:
            sq_axes = _const_i64(
                ctx,
                np.asarray([0], dtype=np.int64),
                "space_to_depth_squeeze_axes",
            )
            result = builder.Squeeze(
                out_nhwc4,
                sq_axes,
                _outputs=[
                    getattr(out_spec, "name", None)
                    or ctx.fresh_name("space_to_depth_out_hwc")
                ],
            )
        else:
            result = out_nhwc4
            result.name = (
                getattr(out_spec, "name", None)
                or getattr(result, "name", None)
                or ctx.fresh_name("space_to_depth_out_nhwc")
            )

        out_spec_type = getattr(out_spec, "type", None)
        if out_spec_type is not None:
            result.type = out_spec_type
        elif x_dtype is not None:
            result.type = ir.TensorType(x_dtype)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        if pix is None:
            return []

        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original dm_pix.space_to_depth not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(inputs: ArrayLike, block_size: int) -> jax.Array:
                try:
                    bs = _check_block_size(int(block_size))
                except Exception:
                    return orig(inputs, block_size)
                return cls._PRIM.bind(jnp.asarray(inputs), block_size=bs)

            return _patched

        return [
            AssignSpec(
                target=pix,
                attr=f"{cls._FUNC_NAME}_p",
                value=cls._PRIM,
                delete_if_missing=True,
            ),
            MonkeyPatchSpec(
                target=pix,
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@DmPixSpaceToDepthPlugin._PRIM.def_impl
def _space_to_depth_prim_impl(inputs: ArrayLike, *, block_size: int) -> jax.Array:
    return _space_to_depth_impl(inputs, block_size=int(block_size))


DmPixSpaceToDepthPlugin._PRIM.def_abstract_eval(DmPixSpaceToDepthPlugin.abstract_eval)


BatchDim = int | type(batching.not_mapped)


def _space_to_depth_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[BatchDim, ...],
    *,
    block_size: int,
) -> tuple[jax.Array, BatchDim]:
    (inputs,), (bdim,) = batched_args, batch_dims
    if bdim is batching.not_mapped:
        out = DmPixSpaceToDepthPlugin._PRIM.bind(inputs, block_size=block_size)
        return out, batching.not_mapped

    moved = batching.bdim_at_front(inputs, bdim, inputs.shape[bdim])
    out = jax.vmap(
        lambda x: DmPixSpaceToDepthPlugin._PRIM.bind(x, block_size=block_size)
    )(moved)
    return out, 0


batching.primitive_batchers[DmPixSpaceToDepthPlugin._PRIM] = _space_to_depth_batch_rule
