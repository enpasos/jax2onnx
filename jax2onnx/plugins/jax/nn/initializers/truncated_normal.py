# jax2onnx/plugins/jax/nn/initializers/truncated_normal.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Final

import jax
from jax import core
import jax.numpy as jnp
from jax.extend.core import Primitive
import numpy as np

from jax2onnx.plugins._ir_shapes import _ensure_value_metadata
from jax2onnx.plugins._ir_shapes import _stamp_type_and_shape
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:  # pragma: no cover
    from jax2onnx.converter.ir_context import IRContext


_TRUNCATED_NORMAL_PRIM: Final[Primitive] = Primitive(
    "jax.nn.initializers.truncated_normal"
)
_TRUNCATED_NORMAL_PRIM.multiple_results = False


def _initializer_callable(key):
    init = jax.nn.initializers.truncated_normal(stddev=1.0, dtype=jnp.float32)
    return init(key, (4, 5), jnp.float32)


@register_primitive(
    jaxpr_primitive=_TRUNCATED_NORMAL_PRIM.name,
    context="primitives.nn",
    component="truncated_normal",
    since="v0.7.1",
    testcases=[
        {
            "testcase": "initializer",
            "callable": _initializer_callable,
            "input_values": [jax.random.PRNGKey(0)],
            "expected_output_shapes": [(4, 5)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [{"inputs": {1: {"const": 1.0}}, "path": "Mul:4x5"}],
            ),
        },
        {
            "testcase": "random_truncated_normal_positional",
            "callable": lambda key: jax.random.truncated_normal(
                key, -2.0, 2.0, (3, 3), jnp.float32
            ),
            "input_values": [jax.random.PRNGKey(0)],
            "expected_output_shapes": [(3, 3)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [],
            ),
        },
        {
            "testcase": "flax_dense_like_init",
            "callable": lambda key, x: jax.random.truncated_normal(
                key, -2.0, 2.0, (x.shape[-1], 128), jnp.float32
            ),
            "input_values": [
                jax.random.PRNGKey(0),
                jnp.ones((1, 10), dtype=jnp.float32),
            ],
            "expected_output_shapes": [(10, 128)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [],
            ),
        },
    ],
)
class TruncatedNormalPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.random.truncated_normal`` to a zero initializer."""

    _PRIM: ClassVar[Primitive] = _TRUNCATED_NORMAL_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False
    # No eager impl; runtime evaluation falls back to zeros directly in patch

    @staticmethod
    def abstract_eval(
        key_av: Any,
        lower_av: Any,
        upper_av: Any,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        **_: Any,
    ) -> core.ShapedArray:
        del key_av, lower_av, upper_av

        def _safe_dim(dim: Any) -> Any:
            try:
                hash(dim)
                return dim
            except TypeError:
                return None

        safe_shape = tuple(_safe_dim(dim) for dim in shape)
        return core.ShapedArray(safe_shape, dtype)

    def lower(self, ctx: "IRContext", eqn):  # type: ignore[name-defined]
        (out_var,) = eqn.outvars
        params = eqn.params or {}
        shape_param = tuple(params.get("shape", ()))
        dtype_param = params.get("dtype")

        aval = getattr(out_var, "aval", None)
        shape = tuple(getattr(aval, "shape", shape_param))
        dtype = getattr(aval, "dtype", dtype_param or np.float32)

        np_dtype = np.dtype(dtype)
        zeros = np.zeros(shape, dtype=np_dtype)
        out_val = ctx.bind_const_for_var(out_var, zeros)
        _stamp_type_and_shape(out_val, shape)
        _ensure_value_metadata(ctx, out_val)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @staticmethod
    def _to_int(dim: Any) -> int:
        try:
            return int(dim)
        except Exception as exc:  # pragma: no cover - best-effort coercion
            aval = getattr(dim, "aval", None)
            val = getattr(aval, "val", None) if aval is not None else None
            if val is not None:
                return int(val)
            raise TypeError(f"Dimension {dim!r} is not statically known") from exc

    @classmethod
    def _make_random_patch(cls, _orig):
        def _patched(key, lower, upper, *pos, **kwargs):
            shape = kwargs.pop("shape", None)
            dtype = kwargs.pop("dtype", None)
            if pos:
                if len(pos) >= 1 and shape is None:
                    shape = pos[0]
                if len(pos) >= 2 and dtype is None:
                    dtype = pos[1]

            shape = shape or ()
            dtype = dtype or jnp.float_

            shape_tuple_raw = shape if isinstance(shape, (tuple, list)) else (shape,)
            static_shape: list[int] = []
            for dim in shape_tuple_raw:
                static_shape.append(cls._to_int(dim))

            return jnp.zeros(tuple(static_shape), dtype)

        return _patched

    @classmethod
    def _make_initializer_patch(cls, orig):
        if orig is None:  # pragma: no cover - safety guard
            return lambda *args, **kwargs: cls._make_random_patch(None)(*args, **kwargs)

        patched_random = cls._make_random_patch(None)

        def _patched(*args, **kwargs):
            if (
                args
                and hasattr(args[0], "shape")
                and getattr(args[0], "shape", None) == (2,)
            ):
                return patched_random(*args, **kwargs)

            def _wrapped(key, shape, dtype=None):
                dtype_local = dtype or kwargs.get("dtype") or jnp.float_
                lower = kwargs.get("lower", -2.0)
                upper = kwargs.get("upper", 2.0)
                samples = patched_random(
                    key,
                    lower,
                    upper,
                    shape,
                    dtype_local,
                )
                stddev = args[0] if args else kwargs.get("stddev", 1e-2)
                stddev_arr = jnp.array(stddev, dtype_local)
                return samples * stddev_arr

            return _wrapped

        return _patched

    @classmethod
    def binding_specs(cls):
        return [
            MonkeyPatchSpec(
                target="jax.random",
                attr="truncated_normal",
                make_value=cls._make_random_patch,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="jax.nn.initializers",
                attr="truncated_normal",
                make_value=cls._make_initializer_patch,
                delete_if_missing=False,
            ),
        ]
