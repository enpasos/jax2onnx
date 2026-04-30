# jax2onnx/plugins/jax/lax/reduce_window.py

from __future__ import annotations

from typing import Any, cast

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir

from jax2onnx.converter.ir_builder import _dtype_to_ir
from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import (
    DimInput,
    _ensure_value_metadata,
    _stamp_type_and_shape,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax._index_utils import _const_i64
from jax2onnx.plugins.jax.lax.reduce_window_sum import (
    ReduceWindowSumPlugin,
    _flatten_padding,
    _normalize_int_tuple,
)
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _canonical_reducer_name(name: str | None) -> str | None:
    if name is None:
        return None

    base = name.split(".")[-1]
    alias_map = {
        "maximum": "max",
        "minimum": "min",
        "logical_and": "and",
        "logical_or": "or",
        "logical_xor": "xor",
    }
    return alias_map.get(base, base)


def _extract_reducer_primitive_name(jaxpr_obj: Any) -> str | None:
    jaxpr = getattr(jaxpr_obj, "jaxpr", jaxpr_obj)
    eqns = tuple(getattr(jaxpr, "eqns", ()) or ())
    outvars = tuple(getattr(jaxpr, "outvars", ()) or ())
    if len(eqns) != 1 or len(outvars) != 1:
        return None
    inner_eqn = eqns[0]
    inner_outvars = tuple(getattr(inner_eqn, "outvars", ()) or ())
    if len(inner_outvars) != 1 or inner_outvars[0] != outvars[0]:
        return None
    primitive = getattr(inner_eqn, "primitive", None)
    return _canonical_reducer_name(getattr(primitive, "name", None))


def _extract_scalar_literal(invar: Any) -> np.ndarray | None:
    if hasattr(invar, "val"):
        value = np.asarray(getattr(invar, "val"))
        if value.ndim == 0:
            return cast(np.ndarray, value)
    return None


def _is_identity_init(
    reducer: str, init_arr: np.ndarray[Any, np.dtype[Any]], dtype: np.dtype[Any]
) -> bool:
    val: Any = init_arr.item()
    if reducer == "add":
        return bool(np.asarray(val == 0))
    if reducer == "max":
        if np.issubdtype(dtype, np.floating):
            return bool(np.isneginf(val))
        if np.issubdtype(dtype, np.integer):
            return int(val) == np.iinfo(dtype).min
        return False
    if reducer == "min":
        if np.issubdtype(dtype, np.floating):
            return bool(np.isposinf(val))
        if np.issubdtype(dtype, np.integer):
            return int(val) == np.iinfo(dtype).max
        return False
    return False


@register_primitive(
    jaxpr_primitive=lax.reduce_window_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_window.html",
    onnx=[
        {
            "component": "Conv",
            "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html",
        },
        {
            "component": "MaxPool",
            "doc": "https://onnx.ai/onnx/operators/onnx__MaxPool.html",
        },
        {"component": "Neg", "doc": "https://onnx.ai/onnx/operators/onnx__Neg.html"},
    ],
    since="0.12.1",
    context="primitives.lax",
    component="reduce_window",
    testcases=[
        {
            "testcase": "reduce_window_add_lambda",
            "callable": lambda x: jax.lax.reduce_window(
                x,
                0.0,
                lambda a, b: a + b,
                window_dimensions=(2, 2),
                window_strides=(1, 1),
                padding=((0, 1), (0, 1)),
            ),
            "input_shapes": [(3, 3)],
            "post_check_onnx_graph": EG(
                ["Unsqueeze -> Unsqueeze -> Conv -> Squeeze"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "reduce_window_max_lambda",
            "callable": lambda x: jax.lax.reduce_window(
                x,
                -jnp.inf,
                lambda a, b: jnp.maximum(a, b),
                window_dimensions=(2, 2),
                window_strides=(1, 1),
                padding=((0, 1), (0, 1)),
            ),
            "input_values": [
                np.asarray(
                    [[1.0, -2.0, 3.0], [0.5, 4.0, -1.0], [-3.0, 2.0, 0.0]],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["Unsqueeze -> Unsqueeze -> MaxPool -> Squeeze"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "reduce_window_min_lambda_stride",
            "callable": lambda x: jax.lax.reduce_window(
                x,
                jnp.inf,
                lambda a, b: jnp.minimum(a, b),
                window_dimensions=(2, 2),
                window_strides=(2, 1),
                padding=((0, 1), (1, 0)),
            ),
            "input_values": [
                np.asarray(
                    [[1.0, -2.0, 3.0], [0.5, 4.0, -1.0], [-3.0, 2.0, 0.0]],
                    dtype=np.float32,
                )
            ],
            "post_check_onnx_graph": EG(
                ["Unsqueeze -> Unsqueeze -> Neg -> MaxPool -> Neg -> Squeeze"],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
    ],
)
class ReduceWindowPlugin(PrimitiveLeafPlugin):
    """Lower `lax.reduce_window` for `add`, `max`, and `min` reducers."""

    _MAXPOOL_DTYPES = {
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    }
    if hasattr(np, "bfloat16"):
        _MAXPOOL_DTYPES.add(np.dtype(np.bfloat16))

    def lower(self, ctx: LoweringContextProtocol, eqn: Any) -> None:
        params = getattr(eqn, "params", {})
        reducer_name = _extract_reducer_primitive_name(params.get("jaxpr"))
        if reducer_name not in {"add", "max", "min"}:
            raise NotImplementedError(
                "reduce_window currently supports reducer jaxpr primitives: add/max/min"
            )

        consts = tuple(params.get("consts", ()) or ())
        if consts:
            raise NotImplementedError(
                "reduce_window with captured reducer constants is not yet supported"
            )

        if len(eqn.invars) < 2:
            raise ValueError("reduce_window expects operand and init value inputs")
        init_lit = _extract_scalar_literal(eqn.invars[1])
        if init_lit is None:
            raise NotImplementedError(
                "reduce_window currently requires a scalar literal init value"
            )

        operand_var = eqn.invars[0]
        aval_dtype = getattr(getattr(operand_var, "aval", None), "dtype", None)
        if aval_dtype is None:
            raise TypeError("reduce_window operand dtype is unknown")
        np_dtype = np.dtype(aval_dtype)

        if not _is_identity_init(reducer_name, init_lit, np_dtype):
            raise NotImplementedError(
                f"reduce_window with reducer='{reducer_name}' currently requires its identity init value"
            )

        if reducer_name == "add":
            # Reuse the already-featured Conv/LpPool reduce_window_sum lowering.
            ReduceWindowSumPlugin().lower(ctx, eqn)
            return

        window_dims = _normalize_int_tuple(
            params.get("window_dimensions", ()), "window_dimensions"
        )
        if not window_dims:
            raise ValueError("reduce_window requires non-empty window_dimensions")
        window_strides = _normalize_int_tuple(
            params.get("window_strides", (1,) * len(window_dims)), "window_strides"
        )
        padding_raw = params.get("padding", tuple((0, 0) for _ in window_dims))
        if isinstance(padding_raw, str):
            raise ValueError(
                "reduce_window lowering expects concrete padding pairs inside JAX eqn"
            )
        padding_pairs = tuple(tuple(int(v) for v in pair) for pair in padding_raw)
        if len(padding_pairs) != len(window_dims):
            raise ValueError(
                f"Padding rank mismatch: expected {len(window_dims)}, received {padding_pairs}"
            )
        base_dilation = _normalize_int_tuple(
            params.get("base_dilation", (1,) * len(window_dims)), "base_dilation"
        )
        if any(d != 1 for d in base_dilation):
            raise NotImplementedError(
                "reduce_window max/min with base_dilation != 1 is not yet supported"
            )
        window_dilation = _normalize_int_tuple(
            params.get("window_dilation", (1,) * len(window_dims)), "window_dilation"
        )

        if np_dtype not in self._MAXPOOL_DTYPES:
            raise TypeError(
                f"reduce_window reducer='{reducer_name}' currently supports floating operand dtypes only, got {np_dtype}"
            )

        operand_val = ctx.get_value_for_var(
            operand_var, name_hint=ctx.fresh_name("reduce_window_in")
        )
        out_var = eqn.outvars[0]
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("reduce_window_out")
        )

        operand_shape: tuple[DimInput, ...] = tuple(
            getattr(getattr(operand_var, "aval", None), "shape", ())
        )
        out_shape: tuple[DimInput, ...] = tuple(
            getattr(getattr(out_var, "aval", None), "shape", ())
        )
        dtype_enum = _dtype_to_ir(np_dtype, ctx.builder.enable_double_precision)
        if dtype_enum is None:
            raise TypeError(f"Unsupported dtype for reduce_window: {np_dtype}")

        axes0 = _const_i64(ctx, np.asarray([0], dtype=np.int64), "rw_axes0")
        axes1 = _const_i64(ctx, np.asarray([1], dtype=np.int64), "rw_axes1")

        unsq0 = cast(
            ir.Value,
            ctx.builder.Unsqueeze(
                operand_val,
                axes0,
                _outputs=[ctx.fresh_name("reduce_window_unsq0")],
            ),
        )
        unsq0.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(unsq0, (1,) + operand_shape)
        _ensure_value_metadata(ctx, unsq0)

        unsq1 = cast(
            ir.Value,
            ctx.builder.Unsqueeze(
                unsq0,
                axes1,
                _outputs=[ctx.fresh_name("reduce_window_unsq1")],
            ),
        )
        unsq1.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(unsq1, (1, 1) + operand_shape)
        _ensure_value_metadata(ctx, unsq1)

        pooled_input = unsq1
        if reducer_name == "min":
            pooled_input = cast(
                ir.Value,
                ctx.builder.Neg(
                    unsq1,
                    _outputs=[ctx.fresh_name("reduce_window_min_neg_in")],
                ),
            )
            pooled_input.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(pooled_input, (1, 1) + operand_shape)
            _ensure_value_metadata(ctx, pooled_input)

        maxpool_kwargs: dict[str, object] = {
            "kernel_shape": tuple(window_dims),
            "strides": tuple(window_strides),
        }
        pads_flat = _flatten_padding(padding_pairs)
        if any(pads_flat):
            maxpool_kwargs["pads"] = pads_flat
        if any(d != 1 for d in window_dilation):
            maxpool_kwargs["dilations"] = tuple(window_dilation)

        pooled = cast(
            ir.Value,
            ctx.builder.MaxPool(
                pooled_input,
                _outputs=[ctx.fresh_name("reduce_window_pool")],
                **maxpool_kwargs,
            ),
        )
        pooled.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(pooled, (1, 1) + out_shape)
        _ensure_value_metadata(ctx, pooled)

        restored = pooled
        if reducer_name == "min":
            restored = cast(
                ir.Value,
                ctx.builder.Neg(
                    pooled,
                    _outputs=[ctx.fresh_name("reduce_window_min_neg_out")],
                ),
            )
            restored.type = ir.TensorType(dtype_enum)
            _stamp_type_and_shape(restored, (1, 1) + out_shape)
            _ensure_value_metadata(ctx, restored)

        squeeze_axes = _const_i64(
            ctx, np.asarray([0, 1], dtype=np.int64), "rw_squeeze_axes"
        )
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "reduce_window"
        )
        result = cast(
            ir.Value,
            ctx.builder.Squeeze(
                restored,
                squeeze_axes,
                _outputs=[desired_name],
            ),
        )
        result.type = ir.TensorType(dtype_enum)
        _stamp_type_and_shape(result, out_shape)
        _ensure_value_metadata(ctx, result)
        ctx.bind_value_for_var(out_var, result)
