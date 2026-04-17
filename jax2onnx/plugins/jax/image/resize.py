# jax2onnx/plugins/jax/image/resize.py

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Final

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
_MAX_EXACT_LINEAR_OPSET9_WEIGHTS: Final[int] = 1_000_000


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


def _compute_exact_linear_resize_weights(
    *,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    dtype: np.dtype[Any],
    precision: object,
) -> np.ndarray:
    """Build an exact static linear resize matrix from the original JAX op."""
    try:
        from jax._src.image.scale import _resize as jax_resize_impl  # noqa: PLC0415
    except Exception:
        orig_resize = get_orig_impl(_RESIZE_PRIM, "resize")

        def jax_resize_impl(
            image: ArrayLike,
            shape: Sequence[int],
            method: str,
            antialias: bool,
            precision: object,
        ) -> jax.Array:
            return orig_resize(
                image,
                shape,
                method=method,
                antialias=antialias,
                precision=precision,
            )

    input_shape_tuple = tuple(int(dim) for dim in input_shape)
    output_shape_tuple = tuple(int(dim) for dim in output_shape)
    input_size = int(np.prod(input_shape_tuple, dtype=np.int64))
    output_size = int(np.prod(output_shape_tuple, dtype=np.int64))
    basis = np.eye(input_size, dtype=dtype).reshape((input_size,) + input_shape_tuple)

    resized_basis = jax.vmap(
        lambda x: jax_resize_impl(x, output_shape_tuple, "linear", False, precision)
    )(basis)
    return np.asarray(resized_basis, dtype=dtype).reshape((input_size, output_size))


@register_primitive(
    jaxpr_primitive=_RESIZE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.image.resize.html",
    onnx=[
        {
            "component": "Upsample",
            "doc": "https://onnx.ai/onnx/operators/onnx__Upsample.html",
        },
        {
            "component": "Resize",
            "doc": "https://onnx.ai/onnx/operators/onnx__Resize.html",
        },
    ],
    since="0.10.0",
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
        {
            "testcase": "resize_nearest_antialias_ignored",
            "callable": lambda x: jimage.resize(
                x, (2, 2, 3), method="nearest", antialias=True
            ),
            "input_shapes": [(1, 1, 3)],
            "expected_output_shapes": [(2, 2, 3)],
            "post_check_onnx_graph": EG(["Resize"], no_unused_inputs=True),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "resize_nearest_opset9_upsample",
            "callable": lambda x: jimage.resize(
                x, (4, 4), method="nearest", antialias=False
            ),
            "input_shapes": [(2, 2)],
            "expected_output_shapes": [(4, 4)],
            "opset_version": 9,
            "post_check_onnx_graph": EG(["Upsample"], no_unused_inputs=True),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "resize_linear_opset9_upsample",
            "callable": lambda x: jimage.resize(
                x, (3, 5), method="linear", antialias=False
            ),
            "input_shapes": [(2, 2)],
            "expected_output_shapes": [(3, 5)],
            "opset_version": 9,
            "post_check_onnx_graph": EG(
                ["Reshape:4 -> MatMul:15 -> Reshape:3x5"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "resize_nearest_rank3_opset9_upsample",
            "callable": lambda x: jimage.resize(
                x, (2, 4, 6), method="nearest", antialias=False
            ),
            "input_shapes": [(1, 2, 3)],
            "expected_output_shapes": [(2, 4, 6)],
            "opset_version": 9,
            "post_check_onnx_graph": EG(["Upsample"], no_unused_inputs=True),
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
            raise TypeError("resize abstract value is missing dtype")
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
        if method == "nearest":
            antialias = False
            precision = None
        if antialias:
            raise NotImplementedError("resize with antialias=True is not supported yet")
        if precision not in (None, jax.lax.Precision.DEFAULT):
            raise NotImplementedError("resize precision overrides are not supported")

        image_val = ctx.get_value_for_var(
            image_var, name_hint=ctx.fresh_name("resize_input")
        )
        image_dtype = getattr(getattr(image_val, "type", None), "dtype", None)

        opset = int(getattr(ctx.builder, "opset", 21))
        if opset <= 9:
            if method not in {"nearest", "linear"}:
                raise NotImplementedError(
                    "resize with opset<=9 supports nearest/linear only"
                )
            input_shape = tuple(getattr(getattr(image_var, "aval", None), "shape", ()))
            if len(input_shape) != len(shape):
                raise NotImplementedError(
                    "resize opset<=9 requires static input rank matching output rank"
                )

            scales_list: list[float] = []
            for in_dim_raw, out_dim in zip(input_shape, shape):
                in_dim = _normalized_dim(in_dim_raw)
                if in_dim is None or in_dim <= 0:
                    raise NotImplementedError(
                        "resize opset<=9 requires static positive input dimensions"
                    )
                scales_list.append(float(out_dim) / float(in_dim))

            if method == "linear":
                input_shape_ints = tuple(int(dim) for dim in input_shape)
                input_size = int(np.prod(input_shape_ints, dtype=np.int64))
                output_size = int(np.prod(shape, dtype=np.int64))
                if input_size * output_size <= _MAX_EXACT_LINEAR_OPSET9_WEIGHTS:
                    input_np_dtype = np.dtype(
                        getattr(getattr(image_var, "aval", None), "dtype", np.float32)
                    )
                    weight_dtype = (
                        input_np_dtype
                        if np.issubdtype(input_np_dtype, np.floating)
                        else np.dtype(np.float32)
                    )
                    weights = _compute_exact_linear_resize_weights(
                        input_shape=input_shape_ints,
                        output_shape=shape,
                        dtype=weight_dtype,
                        precision=precision,
                    )
                    weight_dtype_enum = (
                        ir.DataType.DOUBLE
                        if weight_dtype == np.float64
                        else ir.DataType.FLOAT
                    )

                    flat_shape = _const_i64(
                        ctx,
                        np.asarray([input_size], dtype=np.int64),
                        ctx.fresh_name("resize_flat_shape"),
                    )
                    flat_val = ctx.builder.Reshape(
                        image_val,
                        flat_shape,
                        _outputs=[ctx.fresh_name("resize_flattened")],
                    )
                    if image_dtype is not None:
                        flat_val.type = ir.TensorType(image_dtype)
                    _stamp_type_and_shape(flat_val, (input_size,))
                    _ensure_value_metadata(ctx, flat_val)

                    matmul_input = flat_val
                    if input_np_dtype != weight_dtype:
                        matmul_input = ctx.builder.Cast(
                            flat_val,
                            _outputs=[ctx.fresh_name("resize_linear_cast")],
                            to=int(weight_dtype_enum.value),
                        )
                        matmul_input.type = ir.TensorType(weight_dtype_enum)
                        _stamp_type_and_shape(matmul_input, (input_size,))
                        _ensure_value_metadata(ctx, matmul_input)

                    weights_const = ctx.builder.add_initializer_from_array(
                        name=ctx.fresh_name("resize_linear_weights"),
                        array=weights,
                    )
                    matmul_out = ctx.builder.MatMul(
                        matmul_input,
                        weights_const,
                        _outputs=[ctx.fresh_name("resize_linear_flat_out")],
                    )
                    matmul_out.type = ir.TensorType(weight_dtype_enum)
                    _stamp_type_and_shape(matmul_out, (output_size,))
                    _ensure_value_metadata(ctx, matmul_out)

                    out_shape_const = _const_i64(
                        ctx,
                        np.asarray(shape, dtype=np.int64),
                        ctx.fresh_name("resize_out_shape"),
                    )
                    exact_out = ctx.builder.Reshape(
                        matmul_out,
                        out_shape_const,
                        _outputs=[ctx.fresh_name("resize_out")],
                    )
                    exact_out.type = ir.TensorType(weight_dtype_enum)
                    _stamp_type_and_shape(
                        exact_out, tuple(_normalized_dim(dim) for dim in shape)
                    )
                    _ensure_value_metadata(ctx, exact_out)
                    ctx.bind_value_for_var(out_var, exact_out)
                    return

            scales_const = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("upsample_scales"),
                array=np.asarray(scales_list, dtype=np.float32),
            )
            upsample_out = ctx.builder.Upsample(
                image_val,
                scales_const,
                _outputs=[ctx.fresh_name("upsample_out")],
                mode=method,
            )
            if image_dtype is not None:
                upsample_out.type = ir.TensorType(image_dtype)
            _stamp_type_and_shape(
                upsample_out, tuple(_normalized_dim(dim) for dim in shape)
            )
            _ensure_value_metadata(ctx, upsample_out)
            ctx.bind_value_for_var(out_var, upsample_out)
            return

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

        resize_kwargs: dict[str, object] = {
            "mode": self._SUPPORTED_MODES[method],
            "coordinate_transformation_mode": "half_pixel",
        }
        if method == "nearest":
            resize_kwargs["nearest_mode"] = "round_prefer_floor"
        if method == "cubic":
            resize_kwargs["cubic_coeff_a"] = -0.5

        resize_out = ctx.builder.Resize(
            image_val,
            roi,
            scales,
            sizes,
            _outputs=[ctx.fresh_name("resize_out")],
            **resize_kwargs,
        )

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
