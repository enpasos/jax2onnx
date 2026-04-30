# jax2onnx/plugins/jax/numpy/diag.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Final, cast

import jax
from jax import core
from jax.interpreters import batching
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from numpy.typing import ArrayLike, NDArray

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax._autodiff_utils import register_jvp_via_jax_jvp
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.numpy._common import get_orig_impl, make_jnp_primitive
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_DIAG_PRIM: Final = make_jnp_primitive("jax.numpy.diag")


def normalize_k_scalar(k: int | np.integer | ArrayLike) -> int:
    arr = np.asarray(k)
    if arr.ndim != 0:
        raise ValueError("jnp.diag expects scalar k")
    return int(arr.item())


def is_static_int_shape(shape: tuple[Any, ...]) -> bool:
    return all(isinstance(d, (int, np.integer)) for d in shape)


def abstract_eval_via_orig_diag(
    prim: Any,
    func_name: str,
    x: core.AbstractValue,
    *,
    k: int,
) -> core.ShapedArray:
    x_shape = tuple(getattr(x, "shape", ()))
    x_dtype: np.dtype[Any] = np.dtype(getattr(x, "dtype", np.float32))
    x_spec = jax.ShapeDtypeStruct(x_shape, x_dtype)
    orig = get_orig_impl(prim, func_name)
    out = jax.eval_shape(lambda a: orig(a, k=k), x_spec)
    out_shape = tuple(getattr(out, "shape", ()))
    out_dtype = np.dtype(getattr(out, "dtype", x_dtype))
    return core.ShapedArray(out_shape, out_dtype)


def _diag_extract_indices(rows: int, cols: int, k: int) -> NDArray[np.int64]:
    if k >= 0:
        length = max(0, min(rows, cols - k))
        row_start = 0
        col_start = k
    else:
        length = max(0, min(rows + k, cols))
        row_start = -k
        col_start = 0

    if length <= 0:
        return cast(NDArray[np.int64], np.asarray([], dtype=np.int64))

    idx: NDArray[np.int64] = np.arange(length, dtype=np.int64)
    row_idx = idx + np.int64(row_start)
    col_idx = idx + np.int64(col_start)
    return cast(NDArray[np.int64], row_idx * np.int64(cols) + col_idx)


def lower_extract_diagonal_2d(
    ctx: LoweringContextProtocol,
    x_val: ir.Value,
    *,
    rows: int,
    cols: int,
    k: int,
    out_spec: ir.Value,
    output_hint: str,
) -> ir.Value:
    flat_shape = _const_i64(
        ctx,
        np.asarray([rows * cols], dtype=np.int64),
        f"{output_hint}_flat_shape",
    )

    flat = ctx.builder.Reshape(
        x_val,
        flat_shape,
        _outputs=[ctx.fresh_name(f"{output_hint}_flat")],
    )
    if getattr(x_val, "type", None) is not None:
        flat.type = x_val.type
    _stamp_type_and_shape(flat, (rows * cols,))
    _ensure_value_metadata(ctx, flat)

    diag_indices = _diag_extract_indices(rows, cols, k)
    idx_val = _const_i64(ctx, diag_indices, f"{output_hint}_indices")
    _stamp_type_and_shape(idx_val, (len(diag_indices),))
    _ensure_value_metadata(ctx, idx_val)

    desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(output_hint)
    producer = getattr(out_spec, "producer", None)
    if callable(producer) and producer() is not None:
        desired_name = ctx.fresh_name(output_hint)

    result = cast(
        ir.Value,
        ctx.builder.Gather(
            flat,
            idx_val,
            axis=0,
            _outputs=[desired_name],
        ),
    )
    if getattr(out_spec, "type", None) is not None:
        result.type = out_spec.type
    elif getattr(x_val, "type", None) is not None:
        result.type = x_val.type

    if getattr(out_spec, "shape", None) is not None:
        result.shape = out_spec.shape
    else:
        _stamp_type_and_shape(result, (len(diag_indices),))
    _ensure_value_metadata(ctx, result)
    return result


@register_primitive(
    jaxpr_primitive=_DIAG_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.diag.html",
    onnx=[
        {
            "component": "Mul",
            "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html",
        },
        {
            "component": "Reshape",
            "doc": "https://onnx.ai/onnx/operators/onnx__Reshape.html",
        },
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
    ],
    since="0.12.2",
    context="primitives.jnp",
    component="diag",
    testcases=[
        {
            "testcase": "jnp_diag_from_vector",
            "callable": lambda x: jnp.diag(x),
            "input_values": [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
            "expected_output_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["Mul:3x3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_diag_from_matrix",
            "callable": lambda x: jnp.diag(x),
            "input_values": [np.arange(12, dtype=np.float32).reshape(3, 4)],
            "expected_output_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Reshape:12 -> Gather:3"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jnp_diag_from_matrix_k1",
            "callable": lambda x: jnp.diag(x, k=1),
            "input_values": [np.arange(12, dtype=np.float32).reshape(3, 4)],
            "expected_output_shapes": [(3,)],
            "post_check_onnx_graph": EG(
                ["Reshape:12 -> Gather:3"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class JnpDiagPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar = _DIAG_PRIM
    _FUNC_NAME: ClassVar[str] = "diag"
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: core.AbstractValue, *, k: int = 0) -> core.ShapedArray:
        return abstract_eval_via_orig_diag(
            JnpDiagPlugin._PRIM,
            JnpDiagPlugin._FUNC_NAME,
            x,
            k=int(k),
        )

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        (x_var,) = eqn.invars
        (out_var,) = eqn.outvars

        params = dict(getattr(eqn, "params", {}) or {})
        k = normalize_k_scalar(params.get("k", 0))

        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        rank = len(x_shape)

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("jnp_diag_in"))
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("jnp_diag_out")
        )

        if rank == 1:
            if k != 0:
                raise NotImplementedError("jnp.diag vector mode currently supports k=0")
            if not is_static_int_shape(x_shape):
                raise NotImplementedError("jnp.diag vector mode requires static shape")

            n = int(x_shape[0])
            x_dtype: np.dtype[Any] = np.dtype(
                getattr(getattr(x_var, "aval", None), "dtype", np.float32)
            )
            eye_const = ctx.bind_const_for_var(object(), np.eye(n, dtype=x_dtype))
            _stamp_type_and_shape(eye_const, (n, n))
            _ensure_value_metadata(ctx, eye_const)

            desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
                "jnp_diag_out"
            )
            producer = getattr(out_spec, "producer", None)
            if callable(producer) and producer() is not None:
                desired_name = ctx.fresh_name("jnp_diag_out")

            result = ctx.builder.Mul(eye_const, x_val, _outputs=[desired_name])
            if getattr(out_spec, "type", None) is not None:
                result.type = out_spec.type
            if getattr(out_spec, "shape", None) is not None:
                result.shape = out_spec.shape
            _ensure_value_metadata(ctx, result)
            ctx.bind_value_for_var(out_var, result)
            return

        if rank == 2:
            if not is_static_int_shape(x_shape):
                raise NotImplementedError("jnp.diag matrix mode requires static shape")

            rows = int(x_shape[0])
            cols = int(x_shape[1])
            result = lower_extract_diagonal_2d(
                ctx,
                x_val,
                rows=rows,
                cols=cols,
                k=k,
                out_spec=out_spec,
                output_hint="jnp_diag_out",
            )
            ctx.bind_value_for_var(out_var, result)
            return

        raise NotImplementedError(
            "jnp.diag currently supports rank-1 and rank-2 inputs"
        )

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(
            orig: Callable[..., jax.Array] | None,
        ) -> Callable[..., jax.Array]:
            if orig is None:
                raise RuntimeError("Original jnp.diag not found for patching")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(v: ArrayLike, k: int = 0) -> jax.Array:
                k_norm = normalize_k_scalar(k)
                arr = jnp.asarray(v)
                arr_shape = tuple(arr.shape)
                arr_rank = len(arr_shape)

                if arr_rank == 1 and k_norm == 0 and is_static_int_shape(arr_shape):
                    return cls._PRIM.bind(arr, k=k_norm)

                if arr_rank == 2 and is_static_int_shape(arr_shape):
                    return cls._PRIM.bind(arr, k=k_norm)

                return orig(v, k=k)

            return _patched

        return [
            AssignSpec("jax.numpy", "diag_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr="diag",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@JnpDiagPlugin._PRIM.def_impl
def _diag_impl(v: ArrayLike, *, k: int = 0) -> jax.Array:
    orig = get_orig_impl(JnpDiagPlugin._PRIM, JnpDiagPlugin._FUNC_NAME)
    return orig(v, k=k)


register_jvp_via_jax_jvp(JnpDiagPlugin._PRIM, _diag_impl)


def _diag_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[Any, ...],
    *,
    k: int = 0,
) -> tuple[jax.Array, Any]:
    (x,), (bdim,) = batched_args, batch_dims

    if bdim is batching.not_mapped:
        out = JnpDiagPlugin._PRIM.bind(x, k=k)
        return out, batching.not_mapped

    batch_size = x.shape[int(bdim)]
    x_front = batching.bdim_at_front(x, int(bdim), batch_size)
    orig = get_orig_impl(JnpDiagPlugin._PRIM, JnpDiagPlugin._FUNC_NAME)
    out = jax.vmap(lambda a: orig(a, k=k))(x_front)
    return out, 0


batching.primitive_batchers[JnpDiagPlugin._PRIM] = _diag_batch_rule
