# jax2onnx/plugins/jax/image/resize.py

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar, Final

import jax
import jax.image as jimage
import numpy as np
import onnx_ir as ir
from jax import core
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.image._common import get_orig_impl, make_image_primitive
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _normalized_dim(dim: object) -> int | None:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    return None


_RESIZE_PRIM: Final = make_image_primitive("jax.image.resize")


def _normalize_method(method: str | jimage.ResizeMethod) -> str:
    if isinstance(method, jimage.ResizeMethod):
        method = method.name.lower()
    if not isinstance(method, str):
        raise TypeError("resize 'method' must be a string or ResizeMethod enum")
    return method.lower()


def _canonical_method(method: str) -> str:
    alias_map = {
        "bilinear": "linear",
        "trilinear": "linear",
        "triangle": "linear",
        "bicubic": "cubic",
        "tricubic": "cubic",
    }
    return alias_map.get(method, method)


@register_primitive(
    jaxpr_primitive=_RESIZE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.image.resize.html",
    onnx=[
        {
            "component": "Resize",
            "doc": "https://onnx.ai/onnx/operators/onnx__Resize.html",
        }
    ],
    since="v0.10.0",
    context="primitives.jax_image",
    component="resize",
    testcases=[
        {
            "testcase": "resize_linear",
            "callable": lambda x: jimage.resize(
                x, (4, 4), method="linear", antialias=False
            ),
            "input_shapes": [(2, 2)],
            "expected_output_shapes": [(4, 4)],
            "post_check_onnx_graph": EG(["Resize"], no_unused_inputs=True),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "resize_nearest",
            "callable": lambda x: jimage.resize(
                x, (2, 2, 3), method="nearest", antialias=False
            ),
            "input_shapes": [(1, 1, 3)],
            "expected_output_shapes": [(2, 2, 3)],
            "post_check_onnx_graph": EG(["Resize"], no_unused_inputs=True),
            "run_only_f32_variant": True,
        },
    ],
)
class ImageResizePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _RESIZE_PRIM
    _FUNC_NAME: ClassVar[str] = "resize"

    _SUPPORTED_MODES: ClassVar[dict[str, str]] = {
        "nearest": "nearest",
        "linear": "linear",
        "cubic": "cubic",
    }

    @staticmethod
    def abstract_eval(
        image: core.AbstractValue,
        *,
        shape: Sequence[int | np.integer],
        method: str | jimage.ResizeMethod = "linear",
        **params: object,
    ) -> core.ShapedArray:
        if shape is None:
            raise TypeError("resize requires a target shape")
        try:
            out_shape = tuple(int(dim) for dim in shape)
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError("resize shape must be a sequence of integers") from exc
        dtype = getattr(image, "dtype", None)
        if dtype is None:
            dtype = image.dtype  # type: ignore[attr-defined]
        return core.ShapedArray(out_shape, dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        image_var = eqn.invars[0]
        out_var = eqn.outvars[0]

        params = dict(eqn.params)
        shape = tuple(int(dim) for dim in params["shape"])
        method = _canonical_method(_normalize_method(params.get("method", "linear")))
        antialias = bool(params.get("antialias", False))
        precision = params.get("precision", None)

        if method not in self._SUPPORTED_MODES:
            raise NotImplementedError(f"resize method '{method}' is not supported")
        if antialias:
            raise NotImplementedError("resize with antialias=True is not supported yet")
        if precision not in (None, jax.lax.Precision.DEFAULT):
            raise NotImplementedError("resize precision overrides are not supported")

        image_val = ctx.get_value_for_var(
            image_var, name_hint=ctx.fresh_name("resize_input")
        )

        empty_f32 = np.asarray([], dtype=np.float32)
        roi = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("resize_roi"), array=empty_f32
        )
        scales = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("resize_scales"), array=empty_f32
        )
        sizes = _const_i64(
            ctx,
            np.asarray(shape, dtype=np.int64),
            ctx.fresh_name("resize_sizes"),
        )

        resize_kwargs = {
            "mode": self._SUPPORTED_MODES[method],
            "coordinate_transformation_mode": "half_pixel",
        }
        if method == "nearest":
            resize_kwargs["nearest_mode"] = "round_prefer_floor"
        if method == "cubic":
            resize_kwargs["cubic_coeff_a"] = -0.5
        if antialias:
            resize_kwargs["antialias"] = 1

        resize_out = ctx.builder.Resize(
            image_val,
            roi,
            scales,
            sizes,
            _outputs=[ctx.fresh_name("resize_out")],
            **resize_kwargs,
        )

        image_dtype = getattr(getattr(image_val, "type", None), "dtype", None)
        if image_dtype is not None:
            resize_out.type = ir.TensorType(image_dtype)
        _stamp_type_and_shape(resize_out, tuple(_normalized_dim(dim) for dim in shape))
        _ensure_value_metadata(ctx, resize_out)
        ctx.bind_value_for_var(out_var, resize_out)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jax.image.resize not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                image: ArrayLike,
                shape: Sequence[int | np.integer],
                method: str | jimage.ResizeMethod = "linear",
                antialias: bool = True,
                precision: object = None,
            ) -> jax.Array:
                return cls._PRIM.bind(
                    image,
                    shape=tuple(shape),
                    method=method,
                    antialias=antialias,
                    precision=precision,
                )

            return _patched

        return [
            AssignSpec(
                "jax.image",
                f"{cls._FUNC_NAME}_p",
                cls._PRIM,
                delete_if_missing=True,
            ),
            MonkeyPatchSpec(
                target="jax.image",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@ImageResizePlugin._PRIM.def_impl
def _resize_impl(
    image: ArrayLike,
    *,
    shape: Sequence[int | np.integer],
    method: str | jimage.ResizeMethod = "linear",
    antialias: bool = True,
    precision: object = None,
) -> jax.Array:
    orig = get_orig_impl(ImageResizePlugin._PRIM, ImageResizePlugin._FUNC_NAME)
    return orig(image, shape, method=method, antialias=antialias, precision=precision)


ImageResizePlugin._PRIM.def_abstract_eval(ImageResizePlugin.abstract_eval)
