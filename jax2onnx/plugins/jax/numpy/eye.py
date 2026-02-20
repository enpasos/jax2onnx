# jax2onnx/plugins/jax/numpy/eye.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_EYE_PRIM: Final = make_jnp_primitive("jax.numpy.eye")


def _normalize_dims(n: Any, m: Any | None) -> tuple[int, int]:
    n_i = int(n)
    m_i = n_i if m is None else int(m)
    if n_i < 0 or m_i < 0:
        raise ValueError("jnp.eye dimensions must be non-negative")
    return n_i, m_i


def _np_dtype_from_ir(dtype: ir.DataType) -> np.dtype[Any]:
    mapping: dict[ir.DataType, np.dtype[Any]] = {
        ir.DataType.BOOL: np.dtype(np.bool_),
        ir.DataType.INT8: np.dtype(np.int8),
        ir.DataType.INT16: np.dtype(np.int16),
        ir.DataType.INT32: np.dtype(np.int32),
        ir.DataType.INT64: np.dtype(np.int64),
        ir.DataType.UINT8: np.dtype(np.uint8),
        ir.DataType.UINT16: np.dtype(np.uint16),
        ir.DataType.UINT32: np.dtype(np.uint32),
        ir.DataType.UINT64: np.dtype(np.uint64),
        ir.DataType.FLOAT16: np.dtype(np.float16),
        ir.DataType.FLOAT: np.dtype(np.float32),
        ir.DataType.DOUBLE: np.dtype(np.float64),
        ir.DataType.BFLOAT16: np.dtype(np.float32),
    }
    return mapping.get(dtype, np.dtype(np.float32))


@register_primitive(
    jaxpr_primitive=_EYE_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.eye.html",
    onnx=[
        {
            "component": "EyeLike",
            "doc": "https://onnx.ai/onnx/operators/onnx__EyeLike.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="eye",
    testcases=[
        {
            "testcase": "jnp_eye_square",
            "callable": lambda: jnp.eye(4, dtype=jnp.float32),
            "input_values": [],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["ConstantOfShape:4x4 -> EyeLike:4x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_eye_rect_k1",
            "callable": lambda: jnp.eye(3, 5, k=1, dtype=jnp.float32),
            "input_values": [],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["ConstantOfShape:3x5 -> EyeLike:3x5"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpEyePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _EYE_PRIM
    _FUNC_NAME: ClassVar[str] = "eye"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        *,
        n: int,
        m: int | None = None,
        k: int = 0,
        dtype: np.dtype[Any] | type | None = None,
    ) -> core.ShapedArray:
        del k
        n_i, m_i = _normalize_dims(n, m)
        out_dtype = np.dtype(jnp.float32 if dtype is None else dtype)
        return core.ShapedArray((n_i, m_i), out_dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (out_var,) = eqn.outvars
        params = dict(getattr(eqn, "params", {}) or {})

        n_i, m_i = _normalize_dims(params.get("n"), params.get("m"))
        k_i = int(params.get("k", 0))
        dtype_param = params.get("dtype", np.float32)
        req_dtype = np.dtype(dtype_param)

        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("eye_out"))
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", (n_i, m_i)))
        out_spec_type = getattr(out_spec, "type", None)
        target_enum = _dtype_to_ir(req_dtype, ctx.builder.enable_double_precision)
        if out_spec_type is not None:
            target_enum = out_spec_type.dtype

        shape_tensor = _const_i64(
            ctx,
            np.asarray([n_i, m_i], dtype=np.int64),
            "eye_shape",
        )
        _stamp_type_and_shape(shape_tensor, (2,))
        _ensure_value_metadata(ctx, shape_tensor)

        dummy_dtype = _np_dtype_from_ir(target_enum)
        dummy = ctx.builder.ConstantOfShape(
            shape_tensor,
            value=ir.tensor(np.asarray([0], dtype=dummy_dtype)),
            _outputs=[ctx.fresh_name("eye_like_in")],
        )
        dummy.type = ir.TensorType(target_enum)
        _stamp_type_and_shape(dummy, (n_i, m_i))
        _ensure_value_metadata(ctx, dummy)

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("eye_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("eye_out")

        result = ctx.builder.EyeLike(
            dummy,
            k=k_i,
            dtype=int(target_enum.value),
            _outputs=[desired_name],
        )
        if out_spec_type is not None:
            result.type = out_spec_type
        else:
            result.type = ir.TensorType(target_enum)
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
                raise RuntimeError("Original jnp.eye not found for monkey patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                N: Any,
                M: Any | None = None,
                k: int = 0,
                dtype: np.dtype[Any] | type | None = None,
                *,
                device: Any | None = None,
            ) -> jax.Array:
                if device is not None:
                    return orig(N, M=M, k=k, dtype=dtype, device=device)
                try:
                    n_i, m_i = _normalize_dims(N, M)
                    k_i = int(k)
                    if dtype is None:
                        resolved_dtype = np.dtype(orig(1, M=1).dtype)
                    else:
                        resolved_dtype = np.dtype(dtype)
                except Exception:
                    return orig(N, M=M, k=k, dtype=dtype)
                return cls._PRIM.bind(n=n_i, m=m_i, k=k_i, dtype=resolved_dtype)

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


@JnpEyePlugin._PRIM.def_impl
def _eye_impl(
    *,
    n: int,
    m: int | None = None,
    k: int = 0,
    dtype: np.dtype[Any] | type | None = None,
) -> jax.Array:
    orig = get_orig_impl(JnpEyePlugin._PRIM, JnpEyePlugin._FUNC_NAME)
    return orig(n, M=m, k=k, dtype=dtype)


JnpEyePlugin._PRIM.def_abstract_eval(JnpEyePlugin.abstract_eval)
