# jax2onnx/plugins/jax/numpy/windows.py

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
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_HANNING_PRIM: Final = make_jnp_primitive("jax.numpy.hanning")
_HAMMING_PRIM: Final = make_jnp_primitive("jax.numpy.hamming")
_BLACKMAN_PRIM: Final = make_jnp_primitive("jax.numpy.blackman")


def _window_size(n: Any) -> int:
    n_i = int(n)
    if n_i < 0:
        raise ValueError("window size must be non-negative")
    return n_i


def _window_impl(
    prim,
    func_name: str,
    *,
    n: int,
    dtype: np.dtype[Any] | type = np.float32,
) -> jax.Array:
    orig = get_orig_impl(prim, func_name)
    out = orig(n)
    out_dtype = np.dtype(dtype)
    if np.dtype(out.dtype) != out_dtype:
        out = jnp.asarray(out, dtype=out_dtype)
    return out


class _WindowBasePlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar
    _FUNC_NAME: ClassVar[str]
    _ONNX_OP: ClassVar[str]
    _AFFINE_SCALE_BIAS: ClassVar[tuple[float, float] | None] = None
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @classmethod
    def abstract_eval(
        cls,
        *,
        n: int,
        dtype: np.dtype[Any] | type = np.float32,
    ) -> core.ShapedArray:
        n_i = _window_size(n)
        return core.ShapedArray((n_i,), np.dtype(dtype))

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (out_var,) = eqn.outvars
        params = dict(getattr(eqn, "params", {}) or {})

        n_i = _window_size(params.get("n"))
        dtype_param = np.dtype(params.get("dtype", np.float32))

        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name(f"{self._FUNC_NAME}_out")
        )
        out_shape = tuple(getattr(getattr(out_var, "aval", None), "shape", (n_i,)))

        out_spec_type = getattr(out_spec, "type", None)
        target_enum = _dtype_to_ir(dtype_param, ctx.builder.enable_double_precision)
        if out_spec_type is not None:
            target_enum = out_spec_type.dtype

        size_tensor = ctx.bind_const_for_var(object(), np.asarray(n_i, dtype=np.int64))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            f"{self._FUNC_NAME}_out"
        )
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name(f"{self._FUNC_NAME}_out")

        window_out_name = (
            desired_name
            if self._AFFINE_SCALE_BIAS is None
            else ctx.fresh_name(f"{self._FUNC_NAME}_window")
        )
        if self._ONNX_OP == "HannWindow":
            result = ctx.builder.HannWindow(
                size_tensor,
                periodic=0,
                output_datatype=int(target_enum.value),
                _outputs=[window_out_name],
            )
        elif self._ONNX_OP == "HammingWindow":
            result = ctx.builder.HammingWindow(
                size_tensor,
                periodic=0,
                output_datatype=int(target_enum.value),
                _outputs=[window_out_name],
            )
        elif self._ONNX_OP == "BlackmanWindow":
            result = ctx.builder.BlackmanWindow(
                size_tensor,
                periodic=0,
                output_datatype=int(target_enum.value),
                _outputs=[window_out_name],
            )
        else:  # pragma: no cover - defensive for subclass misuse
            raise ValueError(f"Unsupported window op '{self._ONNX_OP}'")
        result.type = ir.TensorType(target_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)

        final_result = result
        if self._AFFINE_SCALE_BIAS is not None:
            scale_val, bias_val = self._AFFINE_SCALE_BIAS
            scale_const = ctx.builder.add_initializer_from_scalar(
                name=ctx.fresh_name(f"{self._FUNC_NAME}_scale"),
                value=np.asarray(scale_val, dtype=np.float32),
            )
            bias_const = ctx.builder.add_initializer_from_scalar(
                name=ctx.fresh_name(f"{self._FUNC_NAME}_bias"),
                value=np.asarray(bias_val, dtype=np.float32),
            )
            scaled = ctx.builder.Mul(
                result,
                scale_const,
                _outputs=[ctx.fresh_name(f"{self._FUNC_NAME}_scaled")],
            )
            final_result = ctx.builder.Add(
                scaled,
                bias_const,
                _outputs=[desired_name],
            )

        if out_spec_type is not None:
            final_result.type = out_spec_type
        else:
            final_result.type = ir.TensorType(target_enum)
        _stamp_type_and_shape(final_result, out_shape)
        _ensure_value_metadata(ctx, final_result)
        ctx.bind_value_for_var(out_var, final_result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError(f"Original jnp.{cls._FUNC_NAME} not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(m: Any) -> jax.Array:
                try:
                    n_i = _window_size(m)
                    resolved_dtype = np.dtype(orig(1).dtype)
                except Exception:
                    return orig(m)
                return cls._PRIM.bind(n=n_i, dtype=resolved_dtype)

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


@register_primitive(
    jaxpr_primitive=_HANNING_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hanning.html",
    onnx=[
        {
            "component": "HannWindow",
            "doc": "https://onnx.ai/onnx/operators/onnx__HannWindow.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="hanning",
    testcases=[
        {
            "testcase": "jnp_hanning_basic",
            "callable": lambda: jnp.hanning(5),
            "input_values": [],
            "expected_output_shapes": [(5,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["HannWindow:5"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class JnpHanningPlugin(_WindowBasePlugin):
    _PRIM: ClassVar = _HANNING_PRIM
    _FUNC_NAME: ClassVar[str] = "hanning"
    _ONNX_OP: ClassVar[str] = "HannWindow"


@register_primitive(
    jaxpr_primitive=_HAMMING_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.hamming.html",
    onnx=[
        {
            "component": "HammingWindow",
            "doc": "https://onnx.ai/onnx/operators/onnx__HammingWindow.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="hamming",
    testcases=[
        {
            "testcase": "jnp_hamming_basic",
            "callable": lambda: jnp.hamming(5),
            "input_values": [],
            "expected_output_shapes": [(5,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["HammingWindow:5 -> Mul:5 -> Add:5"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class JnpHammingPlugin(_WindowBasePlugin):
    _PRIM: ClassVar = _HAMMING_PRIM
    _FUNC_NAME: ClassVar[str] = "hamming"
    _ONNX_OP: ClassVar[str] = "HammingWindow"
    # ONNX HammingWindow uses alpha=25/46, beta=21/46 while jnp.hamming uses
    # alpha=0.54, beta=0.46. Apply affine correction to match JAX numerics.
    _AFFINE_SCALE_BIAS: ClassVar[tuple[float, float]] = (
        0.46 / (21.0 / 46.0),
        0.54 - (0.46 / (21.0 / 46.0)) * (25.0 / 46.0),
    )


@register_primitive(
    jaxpr_primitive=_BLACKMAN_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.blackman.html",
    onnx=[
        {
            "component": "BlackmanWindow",
            "doc": "https://onnx.ai/onnx/operators/onnx__BlackmanWindow.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="blackman",
    testcases=[
        {
            "testcase": "jnp_blackman_basic",
            "callable": lambda: jnp.blackman(5),
            "input_values": [],
            "expected_output_shapes": [(5,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["BlackmanWindow:5"],
                no_unused_inputs=True,
            ),
        }
    ],
)
class JnpBlackmanPlugin(_WindowBasePlugin):
    _PRIM: ClassVar = _BLACKMAN_PRIM
    _FUNC_NAME: ClassVar[str] = "blackman"
    _ONNX_OP: ClassVar[str] = "BlackmanWindow"


@JnpHanningPlugin._PRIM.def_impl
def _hanning_impl(
    *,
    n: int,
    dtype: np.dtype[Any] | type = np.float32,
) -> jax.Array:
    return _window_impl(
        JnpHanningPlugin._PRIM, JnpHanningPlugin._FUNC_NAME, n=n, dtype=dtype
    )


@JnpHammingPlugin._PRIM.def_impl
def _hamming_impl(
    *,
    n: int,
    dtype: np.dtype[Any] | type = np.float32,
) -> jax.Array:
    return _window_impl(
        JnpHammingPlugin._PRIM, JnpHammingPlugin._FUNC_NAME, n=n, dtype=dtype
    )


@JnpBlackmanPlugin._PRIM.def_impl
def _blackman_impl(
    *,
    n: int,
    dtype: np.dtype[Any] | type = np.float32,
) -> jax.Array:
    return _window_impl(
        JnpBlackmanPlugin._PRIM, JnpBlackmanPlugin._FUNC_NAME, n=n, dtype=dtype
    )


JnpHanningPlugin._PRIM.def_abstract_eval(JnpHanningPlugin.abstract_eval)
JnpHammingPlugin._PRIM.def_abstract_eval(JnpHammingPlugin.abstract_eval)
JnpBlackmanPlugin._PRIM.def_abstract_eval(JnpBlackmanPlugin.abstract_eval)
